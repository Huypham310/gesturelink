"""MediaPipe hand landmark extraction (Tasks API, ``HandLandmarker``).

mediapipe 1.x no longer ships the old ``mp.solutions`` API, so everything here goes through
``mediapipe.tasks``. The ``hand_landmarker.task`` model is a free download, see
``gesturelink download-model`` and ``models/README.md``.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from gesturelink.data.schema import NUM_COORDS, NUM_LANDMARKS

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path("models") / "hand_landmarker.task"

# landmark indices of the MediaPipe hand model (handy for the synthetic generator and docs)
WRIST = 0
THUMB = (1, 2, 3, 4)  # CMC, MCP, IP, TIP
INDEX = (5, 6, 7, 8)  # MCP, PIP, DIP, TIP
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
FINGERTIPS = (4, 8, 12, 16, 20)


def download_model(
    dest: str | Path = DEFAULT_MODEL_PATH, url: str = MODEL_URL, force: bool = False
) -> Path:
    """Download the hand landmarker model if it is not there yet. Returns the local path."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    # tải về file tạm trước, xong mới đổi tên để tránh file hỏng nếu mất mạng giữa chừng
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:  # noqa: S310
        while chunk := resp.read(1 << 16):
            fh.write(chunk)
    if tmp.stat().st_size == 0:
        tmp.unlink()
        raise RuntimeError(f"downloaded an empty file from {url}")
    tmp.replace(dest)
    return dest


class HandExtractor:
    """Thin wrapper around ``HandLandmarker`` in IMAGE mode (one frame in, one hand out)."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, min_confidence: float = 0.5):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found, run `gesturelink download-model` first"
            )
        # imports are lazy so the rest of the data package works without loading mediapipe
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def __call__(self, bgr_frame: np.ndarray) -> tuple[np.ndarray, str] | None:
        """Detect a hand in an OpenCV BGR frame.

        Returns ``(landmarks (21, 3), handedness)`` or None when no hand is found.
        """
        rgb = np.ascontiguousarray(bgr_frame[:, :, ::-1])  # OpenCV is BGR, MediaPipe wants RGB
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        if not result.hand_landmarks:
            return None
        pts = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
        if pts.shape != (NUM_LANDMARKS, NUM_COORDS):
            return None
        # note: MediaPipe assumes a mirrored (selfie) image when it names the hand
        handedness = result.handedness[0][0].category_name
        return pts, handedness

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> HandExtractor:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
