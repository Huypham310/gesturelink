"""Synthetic hand landmarks, so tests/CI have data without a webcam.

A tiny kinematic hand: each finger is three segments that bend towards the camera by a "curl"
amount in [0, 1]. Every gesture is just a set of curls. Then we add random rotation, scale,
position, jitter and handedness so the samples are not all identical.

This is NOT a replacement for real data. It only has to be (a) well formed, (b) separable
enough that the pipeline and a classifier can be sanity-checked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gesturelink.data.io import SampleWriter
from gesturelink.data.labels import Labels
from gesturelink.data.schema import NUM_COORDS, NUM_LANDMARKS, Sample

# finger setup in the hand frame: x right, y up (towards fingertips), z towards the camera.
# palm size (wrist -> middle MCP) is exactly 1.0 here.
_FINGERS = {
    #          base position (MCP)      fan angle (deg), segment lengths
    "index": ((-0.40, 0.95, 0.0), -5.0, (0.50, 0.30, 0.25)),
    "middle": ((0.00, 1.00, 0.0), 0.0, (0.55, 0.33, 0.27)),
    "ring": ((0.38, 0.93, 0.0), 5.0, (0.50, 0.30, 0.25)),
    "pinky": ((0.72, 0.80, 0.0), 12.0, (0.40, 0.22, 0.20)),
}
_FINGER_ORDER = ("index", "middle", "ring", "pinky")
_FINGER_FIRST_IDX = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
# how far each joint bends at curl = 1 (degrees, MCP / PIP / DIP)
_BEND_DEG = (75.0, 95.0, 60.0)

_THUMB_CMC = np.array([-0.35, 0.20, 0.0])
_THUMB_LENGTHS = (0.45, 0.35, 0.30)
_THUMB_SIDE = np.array([-0.8, 0.6, 0.0])  # thumb stuck out to the side
_THUMB_UP = np.array([-0.15, 1.0, 0.0])  # thumb pointing up
_THUMB_CURLED = np.array([0.5, 0.3, -0.8])  # thumb folded across the palm
_THUMB_WEIGHTS = (0.5, 0.8, 1.0)

EXTENDED = (0.0, 0.12)  # curl ranges
CURLED = (0.85, 1.0)

# gesture -> (thumb curl range, thumb "up" amount, index, middle, ring, pinky curl ranges)
_GESTURES: dict[str, dict] = {
    "open_palm": {"thumb": EXTENDED, "up": 0.0, "fingers": (EXTENDED,) * 4},
    "fist": {"thumb": (0.7, 0.95), "up": 0.0, "fingers": (CURLED,) * 4},
    "thumbs_up": {"thumb": EXTENDED, "up": 1.0, "fingers": (CURLED,) * 4},
    "peace": {
        "thumb": (0.7, 0.95), "up": 0.0,
        "fingers": (EXTENDED, EXTENDED, CURLED, CURLED),
        "fan": {"index": -12.0, "middle": 8.0},  # V shape
    },
    "point": {"thumb": (0.7, 0.95), "up": 0.0, "fingers": (EXTENDED, CURLED, CURLED, CURLED)},
    "ok": {"thumb": (0.35, 0.5), "up": 0.0, "fingers": ((0.45, 0.6), EXTENDED, EXTENDED, EXTENDED)},
    # relaxed, half-open hand: not any of the commands
    "none": {"thumb": (0.2, 0.7), "up": 0.0, "fingers": ((0.25, 0.7),) * 4},
}


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _build_finger(name: str, curl: float, fan_deg: float) -> np.ndarray:
    """4 points (MCP, PIP, DIP, TIP) of one finger in the hand frame."""
    base, _, lengths = _FINGERS[name]
    pos = np.array(base, dtype=np.float64)
    fan = np.deg2rad(fan_deg)
    along = np.array([np.sin(fan), np.cos(fan), 0.0])  # finger direction when straight
    toward_palm = np.array([0.0, 0.0, -1.0])  # curling moves the finger towards the camera
    pts = [pos.copy()]
    angle = 0.0
    for bend, length in zip(_BEND_DEG, lengths, strict=True):
        angle += np.deg2rad(bend) * curl  # joint angles add up along the finger
        pos = pos + (along * np.cos(angle) + toward_palm * np.sin(angle)) * length
        pts.append(pos.copy())
    return np.array(pts)


def _build_thumb(curl: float, up: float) -> np.ndarray:
    """4 points (CMC, MCP, IP, TIP) of the thumb."""
    extended = _unit(_THUMB_SIDE * (1 - up) + _THUMB_UP * up)
    folded = _unit(_THUMB_CURLED)
    pos = _THUMB_CMC.copy()
    pts = [pos.copy()]
    for weight, length in zip(_THUMB_WEIGHTS, _THUMB_LENGTHS, strict=True):
        mix = min(curl * weight, 1.0)
        pos = pos + _unit(extended * (1 - mix) + folded * mix) * length
        pts.append(pos.copy())
    return np.array(pts)


def _rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """3D rotation from three angles in radians (z, x, y axes)."""
    cz, sz = np.cos(roll), np.sin(roll)
    cx, sx = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return rz @ rx @ ry


def hand_frame_pose(gesture: str, rng: np.random.Generator) -> np.ndarray:
    """Clean (21, 3) hand in the hand frame for a gesture, with random curl inside its ranges."""
    spec = _GESTURES[gesture]
    pts = np.zeros((NUM_LANDMARKS, NUM_COORDS))
    thumb_curl = rng.uniform(*spec["thumb"])
    pts[1:5] = _build_thumb(thumb_curl, spec["up"])
    fans = spec.get("fan", {})
    for name, curl_range in zip(_FINGER_ORDER, spec["fingers"], strict=True):
        fan = _FINGERS[name][1] + fans.get(name, 0.0) + rng.normal(0, 2.0)
        first = _FINGER_FIRST_IDX[name]
        pts[first : first + 4] = _build_finger(name, rng.uniform(*curl_range), fan)
    if gesture == "ok":
        # thumb tip touches the index tip: that is the whole point of the OK sign
        tip = pts[8] + rng.normal(0, 0.03, 3)
        pts[4] = tip
        pts[3] = (pts[2] + tip) / 2 + rng.normal(0, 0.02, 3)
    return pts


def make_pose(
    gesture: str, rng: np.random.Generator, noise: float = 0.02
) -> tuple[np.ndarray, str]:
    """One raw sample: (21, 3) in MediaPipe-like image coordinates and a handedness string."""
    pts = hand_frame_pose(gesture, rng)
    pts = pts + rng.normal(0, noise, pts.shape)  # per-landmark jitter, relative to palm size

    handedness = "Left" if rng.random() < 0.5 else "Right"
    if handedness == "Left":
        pts[:, 0] *= -1  # mirror the whole hand

    # small random 3D rotation, hands are mostly upright but never perfectly
    roll = np.deg2rad(np.clip(rng.normal(0, 12), -35, 35))
    pitch = np.deg2rad(np.clip(rng.normal(0, 8), -25, 25))
    yaw = np.deg2rad(np.clip(rng.normal(0, 10), -30, 30))
    pts = pts @ _rotation(roll, pitch, yaw).T

    # TODO: thử thêm lỗi mất landmark (occlusion) cho giống dữ liệu thật hơn
    scale = rng.uniform(0.12, 0.30)  # palm size as a fraction of the frame
    wrist_x, wrist_y = rng.uniform(0.3, 0.7), rng.uniform(0.55, 0.85)
    out = np.empty_like(pts)
    out[:, 0] = np.clip(wrist_x + pts[:, 0] * scale, 0.0, 1.0)
    out[:, 1] = np.clip(wrist_y - pts[:, 1] * scale, 0.0, 1.0)  # image y points down
    out[:, 2] = pts[:, 2] * scale
    return out.astype(np.float32), handedness


def synthesize_samples(
    labels: Labels,
    per_class: int = 200,
    seed: int = 42,
    noise: float = 0.02,
    session_id: str = "synthetic",
) -> list[Sample]:
    """Balanced list of synthetic samples, same order every time for a given seed."""
    rng = np.random.default_rng(seed)
    samples = []
    timestamp_ms = 0
    for label_id, name in enumerate(labels.names):
        if name not in _GESTURES:
            raise ValueError(f"no synthetic recipe for label {name!r}")
        for _ in range(per_class):
            landmarks, handedness = make_pose(name, rng, noise)
            samples.append(Sample(name, label_id, session_id, timestamp_ms, handedness, landmarks))
            timestamp_ms += 100
    return samples


def synthesize_csv(
    out_path: str | Path,
    labels: Labels,
    per_class: int = 200,
    seed: int = 42,
    noise: float = 0.02,
    session_id: str = "synthetic",
) -> int:
    """Write synthetic samples to a CSV (an existing file is replaced). Returns the row count."""
    out_path = Path(out_path)
    out_path.unlink(missing_ok=True)  # synthetic data is reproducible, so just overwrite
    samples = synthesize_samples(labels, per_class, seed, noise, session_id)
    with SampleWriter(out_path) as writer:
        for sample in samples:
            writer.write(sample)
        return writer.count
