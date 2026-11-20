"""Command line entry point (``gesturelink`` or ``python -m gesturelink``)."""

from __future__ import annotations

import argparse
import platform
from datetime import datetime
from pathlib import Path

from gesturelink import __version__


def probe_cameras(max_index: int = 4) -> list[tuple[int, int, int]]:
    """Indexes of cameras that open and return a frame, with the frame size."""
    import cv2

    found = []
    for i in range(max_index):
        # CAP_DSHOW opens much faster than the default MSMF backend on Windows
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        cap = cv2.VideoCapture(i, backend)
        try:
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            if ok and frame is not None:
                h, w = frame.shape[:2]
                found.append((i, w, h))
        finally:
            cap.release()
    return found


def probe_serial_ports() -> list[str]:
    from serial.tools import list_ports

    return [f"{p.device} ({p.description})" for p in list_ports.comports()]


def _info() -> int:
    import cv2
    import mediapipe as mp

    print(f"gesturelink {__version__}")
    print(f"python      {platform.python_version()}")
    print(f"opencv      {cv2.__version__}")
    print(f"mediapipe   {mp.__version__}")
    cams = probe_cameras()
    print("cameras     " + (", ".join(f"#{i} {w}x{h}" for i, w, h in cams) or "none"))
    ports = probe_serial_ports()
    print("serial      " + (", ".join(ports) or "none (plug in the Arduino)"))
    return 0


def _collect(args: argparse.Namespace) -> int:
    from gesturelink.data.collect import image_frames, run_camera_session
    from gesturelink.data.labels import load_labels

    labels = load_labels(args.labels)
    session = args.session or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.out_dir) / f"{session}.csv"  # one CSV per session, labels appended
    if args.images:
        # offline mode: landmarks from a folder of photos instead of the webcam
        from gesturelink.data.collect import collect_samples
        from gesturelink.data.io import SampleWriter
        from gesturelink.data.landmarks import HandExtractor

        with HandExtractor(args.model) as extractor, SampleWriter(out_path) as writer:
            stats = collect_samples(
                image_frames(args.images), extractor, writer, args.label, labels, session,
                max_samples=args.samples, min_interval_ms=0,
            )
    else:
        stats = run_camera_session(
            args.label, labels, out_path, args.model, session,
            camera_index=args.camera, max_samples=args.samples,
            min_interval_ms=args.interval_ms, countdown_s=args.countdown,
            preview=not args.no_preview,
        )
    print(
        f"{args.label}: saved {stats.saved} samples "
        f"({stats.frames_with_hand}/{stats.frames_seen} frames had a hand) -> {out_path}"
    )
    return 0 if stats.saved else 1


def _prepare(args: argparse.Namespace) -> int:
    from gesturelink.data.labels import load_labels
    from gesturelink.data.prepare import prepare_dataset

    meta = prepare_dataset(
        args.input, args.out_dir, load_labels(args.labels),
        val_fraction=args.val_fraction, seed=args.seed, mirror_left=args.mirror_left,
    )
    train, val = meta["counts"]["train"], meta["counts"]["val"]
    dropped = meta["num_dropped_degenerate"]
    print(f"{meta['num_samples']} samples ({dropped} dropped) -> {args.out_dir}")
    print(f"train {sum(train.values())} / val {sum(val.values())}")
    for name in meta["labels"]:
        print(f"  {name:<10} train {train[name]:>4}  val {val[name]:>4}")
    return 0


def _synthesize(args: argparse.Namespace) -> int:
    from gesturelink.data.labels import load_labels
    from gesturelink.data.synthesize import synthesize_csv

    labels = load_labels(args.labels)
    n = synthesize_csv(args.out, labels, args.per_class, args.seed, args.noise)
    print(f"wrote {n} synthetic samples ({args.per_class} per class) -> {args.out}")
    return 0


def _train(args: argparse.Namespace) -> int:
    from gesturelink.data.labels import load_labels
    from gesturelink.model.eval import format_report
    from gesturelink.model.train import load_config, train

    cfg = load_config(
        args.config, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed,
        data=args.data, out_dir=args.out_dir,
    )
    labels = load_labels(args.labels)
    metrics = train(cfg, labels)
    print(format_report(metrics, labels.names))
    print(
        f"best epoch {metrics['best_epoch']} of {metrics['epochs_run']}, "
        f"{metrics['train_seconds']}s -> {metrics['checkpoint']}"
    )
    return 0


def _predict(args: argparse.Namespace) -> int:
    from gesturelink.data.io import find_csv_files, iter_samples
    from gesturelink.model.predict import GesturePredictor

    predictor = GesturePredictor(args.checkpoint, mirror_left=args.mirror_left)
    samples = [s for f in find_csv_files(args.input) for s in iter_samples(f)]
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        print("no samples to classify")
        return 1
    preds = predictor.predict_batch(
        [s.landmarks for s in samples], [s.handedness for s in samples]
    )
    correct = skipped = 0
    for s, p in zip(samples, preds, strict=True):
        if p is None:  # degenerate hand, nothing to classify
            skipped += 1
            continue
        correct += p.label == s.label
        if args.verbose:
            mark = "ok " if p.label == s.label else "BAD"
            print(f"{mark} true={s.label:<10} pred={p.label:<10} conf={p.confidence:.2f}")
    used = len(samples) - skipped
    print(f"{correct}/{used} correct ({correct / max(used, 1):.1%}), {skipped} skipped")
    return 0


def _download_model(args: argparse.Namespace) -> int:
    from gesturelink.data.landmarks import MODEL_URL, download_model

    path = download_model(args.dest, args.url or MODEL_URL, force=args.force)
    print(f"model ready: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


def _board(args: argparse.Namespace) -> int:
    from gesturelink.serial_client import (
        BoardClient,
        BoardError,
        BoardTimeout,
        ProtocolError,
        find_board_port,
    )

    if args.board_cmd is None:
        print("pick a board command: ping, servo, led, lcd, status")
        return 2
    port = args.port or find_board_port()
    if port is None:
        print("no Arduino found, plug it in or pass --port (COM3, rfc2217://localhost:4000)")
        return 1
    try:
        with BoardClient(port, baud=args.baud, timeout=args.timeout) as board:
            if args.board_cmd == "ping":
                print(board.ping().raw)
            elif args.board_cmd == "servo":
                print(board.servo(args.degrees).raw)
            elif args.board_cmd == "led":
                print(board.led(args.value).raw)
            elif args.board_cmd == "lcd":
                print(board.lcd(args.row, " ".join(args.text)).raw)
            elif args.board_cmd == "status":
                s = board.status()
                print(f"servo={s.servo} led={s.led} uptime_ms={s.uptime_ms}")
    except ProtocolError as exc:  # board sent something we can't parse
        print(f"unexpected reply: {exc}")
        return 1
    except ValueError as exc:  # bad argument, nothing was sent (sai tham số, chưa gửi gì)
        print(f"invalid argument: {exc}")
        return 2
    except BoardError as exc:
        print(exc)
        return 1
    except BoardTimeout as exc:
        print(f"{exc} (is the right port selected and the firmware flashed?)")
        return 1
    except OSError as exc:  # port busy / missing
        print(f"cannot open {port}: {exc}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gesturelink", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("info", help="show versions, cameras and serial ports")

    p = sub.add_parser("collect", help="record labelled landmark samples from the webcam")
    p.add_argument("--label", required=True, help="gesture name from configs/labels.json")
    p.add_argument("--samples", type=int, default=200, help="how many samples to save")
    p.add_argument("--session", help="session id (default: timestamp)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--interval-ms", type=int, default=100, help="min gap between saved samples")
    p.add_argument("--countdown", type=int, default=3, help="seconds before recording starts")
    p.add_argument("--no-preview", action="store_true", help="don't open the preview window")
    p.add_argument("--images", help="use a folder of images instead of the webcam")
    p.add_argument("--out-dir", default="data/raw")
    p.add_argument("--model", default="models/hand_landmarker.task")
    p.add_argument("--labels", default=None, help="labels json (default configs/labels.json)")

    p = sub.add_parser("prepare", help="normalize raw CSVs and split into train/val")
    p.add_argument("--input", default="data/raw", help="a CSV file or a folder of CSVs")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mirror-left", action="store_true", help="mirror left hands to look right")
    p.add_argument("--labels", default=None)

    p = sub.add_parser("synthesize", help="generate synthetic landmark samples (no camera)")
    p.add_argument("--out", default="data/raw/synthetic.csv")
    p.add_argument("--per-class", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise", type=float, default=0.02, help="landmark jitter (palm size units)")
    p.add_argument("--labels", default=None)

    p = sub.add_parser("download-model", help="download the free hand_landmarker.task model")
    p.add_argument("--dest", default="models/hand_landmarker.task")
    p.add_argument("--url", default=None, help="override the default Google storage URL")
    p.add_argument("--force", action="store_true", help="download again even if it exists")

    p = sub.add_parser("train", help="train the gesture classifier on data/processed")
    p.add_argument("--config", default=None, help="train config json (default configs/train.json)")
    p.add_argument("--data", default=None, help="dataset.npz (overrides the config)")
    p.add_argument("--out-dir", default=None, help="where to write the .pt and metrics.json")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--labels", default=None)

    p = sub.add_parser("predict", help="classify raw landmark CSV rows with a trained model")
    p.add_argument("--input", required=True, help="a raw landmark CSV or a folder of them")
    p.add_argument("--checkpoint", default="models/gesturelink.pt")
    p.add_argument("--limit", type=int, default=0, help="only the first N rows")
    p.add_argument("--mirror-left", action="store_true", help="use if prepare ran with it")
    p.add_argument("--verbose", "-v", action="store_true", help="print every row")

    p = sub.add_parser("board", help="send commands to the Arduino over serial")
    p.add_argument(
        "--port", default=None, help="COM3, /dev/ttyACM0 or rfc2217://host:port (default: auto)"
    )
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--timeout", type=float, default=2.0, help="seconds to wait for a reply")
    bsub = p.add_subparsers(dest="board_cmd")
    bsub.add_parser("ping", help="check the firmware answers")
    bp = bsub.add_parser("servo", help="move the servo")
    bp.add_argument("degrees", type=int, help="0-180")
    bp = bsub.add_parser("led", help="set the LEDs")
    bp.add_argument("value", help="on, off or a bit mask 0-15 (e.g. 5 or 0x5)")
    bp = bsub.add_parser("lcd", help="write text on one LCD row")
    bp.add_argument("row", type=int, help="0 or 1")
    bp.add_argument("text", nargs="*", help="up to 16 characters")
    bsub.add_parser("status", help="read back servo, LED mask and uptime")

    args = parser.parse_args(argv)
    handlers = {
        "info": lambda a: _info(),
        "collect": _collect,
        "prepare": _prepare,
        "synthesize": _synthesize,
        "download-model": _download_model,
        "train": _train,
        "predict": _predict,
        "board": _board,
    }
    if args.command in handlers:
        return handlers[args.command](args)
    parser.print_help()
    return 0
