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


def _download_model(args: argparse.Namespace) -> int:
    from gesturelink.data.landmarks import MODEL_URL, download_model

    path = download_model(args.dest, args.url or MODEL_URL, force=args.force)
    print(f"model ready: {path} ({path.stat().st_size / 1e6:.1f} MB)")
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

    args = parser.parse_args(argv)
    handlers = {
        "info": lambda a: _info(),
        "collect": _collect,
        "prepare": _prepare,
        "synthesize": _synthesize,
        "download-model": _download_model,
    }
    if args.command in handlers:
        return handlers[args.command](args)
    parser.print_help()
    return 0
