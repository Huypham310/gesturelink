"""Collect labelled landmark samples from a webcam (or from a folder of images).

The core loop (``collect_samples``) only needs an iterable of frames and a callable that turns a
frame into landmarks, so it can be tested without a camera or MediaPipe.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gesturelink.data.io import SampleWriter
from gesturelink.data.labels import Labels
from gesturelink.data.schema import Sample

# (timestamp_ms, BGR frame)
FrameSource = Iterable[tuple[int, np.ndarray]]
# frame -> (landmarks (21, 3), handedness) or None when no hand was found
Extractor = Callable[[np.ndarray], "tuple[np.ndarray, str] | None"]
# called after every frame with (frame, detection, saved_so_far); return False to stop early
FrameCallback = Callable[[np.ndarray, "tuple[np.ndarray, str] | None", int], bool]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class CollectStats:
    frames_seen: int = 0
    frames_with_hand: int = 0
    saved: int = 0


def collect_samples(
    frames: FrameSource,
    extract: Extractor,
    writer: SampleWriter,
    label: str,
    labels: Labels,
    session_id: str,
    max_samples: int = 200,
    min_interval_ms: int = 100,
    on_frame: FrameCallback | None = None,
) -> CollectStats:
    """Run the extractor on frames and save up to ``max_samples`` detections.

    ``min_interval_ms`` spaces the saved samples out so we don't store 30 near-identical frames
    per second of the same pose.
    """
    label_id = labels.id_of(label)  # fail early on a typo in the label name
    stats = CollectStats()
    last_saved_ms: int | None = None

    for timestamp_ms, frame in frames:
        stats.frames_seen += 1
        detection = extract(frame)
        if detection is not None:
            stats.frames_with_hand += 1
            far_enough = last_saved_ms is None or timestamp_ms - last_saved_ms >= min_interval_ms
            if far_enough:
                landmarks, handedness = detection
                writer.write(
                    Sample(
                        label=label,
                        label_id=label_id,
                        session_id=session_id,
                        timestamp_ms=int(timestamp_ms),
                        handedness=handedness,
                        landmarks=landmarks,
                    )
                )
                stats.saved += 1
                last_saved_ms = timestamp_ms
        if on_frame is not None and on_frame(frame, detection, stats.saved) is False:
            break
        if stats.saved >= max_samples:
            break
    return stats


def image_frames(folder: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    """Frames from a folder of images (sorted by name). Timestamps are just 33 ms apart."""
    import cv2

    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"no images in {folder}")
    for i, path in enumerate(files):
        frame = cv2.imread(str(path))
        if frame is None:  # unreadable file, skip it
            continue
        yield i * 33, frame


def camera_frames(camera_index: int = 0) -> Iterator[tuple[int, np.ndarray]]:
    """Frames from a webcam, mirrored like a selfie view. Needs a real camera."""
    import cv2

    # CAP_DSHOW is much faster to open than MSMF on Windows
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") and _is_windows() else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera #{camera_index}")
    start = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield int((time.monotonic() - start) * 1000), cv2.flip(frame, 1)
    finally:
        cap.release()


def _is_windows() -> bool:
    import platform

    return platform.system() == "Windows"


def run_camera_session(
    label: str,
    labels: Labels,
    out_path: str | Path,
    model_path: str | Path,
    session_id: str,
    camera_index: int = 0,
    max_samples: int = 200,
    min_interval_ms: int = 100,
    countdown_s: int = 3,
    preview: bool = True,
) -> CollectStats:
    """Record one gesture interactively: countdown, then save samples (press q to stop)."""
    import cv2

    from gesturelink.data.landmarks import HandExtractor

    window = "gesturelink collect (q to stop)"
    with HandExtractor(model_path) as extractor, SampleWriter(out_path) as writer:
        frames = camera_frames(camera_index)
        started = time.monotonic()

        def extract(frame: np.ndarray):
            # trong lúc đếm ngược thì chưa lưu gì, chỉ cho người dùng vào tư thế
            if time.monotonic() - started < countdown_s:
                return None
            return extractor(frame)

        def on_frame(frame, detection, saved) -> bool:
            if not preview:
                return True
            shown = frame.copy()
            left = countdown_s - (time.monotonic() - started)
            text = f"get ready: {left:.0f}" if left > 0 else f"{label}: {saved}/{max_samples}"
            cv2.putText(shown, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if detection is not None:
                h, w = shown.shape[:2]
                for x, y, _ in detection[0]:
                    cv2.circle(shown, (int(x * w), int(y * h)), 3, (0, 0, 255), -1)
            cv2.imshow(window, shown)
            return (cv2.waitKey(1) & 0xFF) != ord("q")

        try:
            return collect_samples(
                frames, extract, writer, label, labels, session_id,
                max_samples=max_samples, min_interval_ms=min_interval_ms, on_frame=on_frame,
            )
        finally:
            frames.close()
            if preview:
                cv2.destroyAllWindows()
