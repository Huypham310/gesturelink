"""Landmark normalization: wrist-relative, scaled by palm size.

After this step a hand looks the same wherever it is in the frame and however far it is from
the camera, so the classifier only has to learn the finger shape.
"""

from __future__ import annotations

import numpy as np

from gesturelink.data.schema import FEATURE_DIM, NUM_COORDS, NUM_LANDMARKS

WRIST = 0
MIDDLE_FINGER_MCP = 9  # base knuckle of the middle finger
MIN_PALM_SIZE = 1e-6


class DegenerateHandError(ValueError):
    """The wrist and middle knuckle collapse into one point, so there is no scale to use."""


def palm_size(landmarks: np.ndarray) -> float:
    """3D distance wrist -> middle finger MCP. Used as the unit length."""
    pts = np.asarray(landmarks, dtype=np.float32).reshape(NUM_LANDMARKS, NUM_COORDS)
    return float(np.linalg.norm(pts[MIDDLE_FINGER_MCP] - pts[WRIST]))


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalize one hand. Accepts (21, 3) or (63,), returns a flat float32 (63,) vector."""
    pts = np.asarray(landmarks, dtype=np.float32).reshape(NUM_LANDMARKS, NUM_COORDS)
    size = palm_size(pts)
    if not np.isfinite(size) or size < MIN_PALM_SIZE:
        raise DegenerateHandError(f"palm size too small ({size})")
    rel = pts - pts[WRIST]  # wrist becomes the origin
    # chia cho palm size để tay gần/xa camera đều ra cùng một kích thước
    return (rel / size).reshape(FEATURE_DIM).astype(np.float32)


def normalize_batch(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised version for (N, 63) arrays.

    Returns ``(features, valid)`` where ``valid`` is a bool mask; rows with a degenerate palm
    (or NaN) are zeros in ``features`` and False in ``valid`` so the caller can drop them.
    """
    arr = np.asarray(batch, dtype=np.float32).reshape(-1, NUM_LANDMARKS, NUM_COORDS)
    wrist = arr[:, WRIST : WRIST + 1, :]
    sizes = np.linalg.norm(arr[:, MIDDLE_FINGER_MCP, :] - arr[:, WRIST, :], axis=1)
    valid = np.isfinite(sizes) & (sizes >= MIN_PALM_SIZE) & np.isfinite(arr).all(axis=(1, 2))
    safe = np.where(valid, sizes, 1.0)  # avoid dividing by 0 for the invalid rows
    out = (arr - wrist) / safe[:, None, None]
    out[~valid] = 0.0
    return out.reshape(-1, FEATURE_DIM).astype(np.float32), valid


def mirror_x(features: np.ndarray) -> np.ndarray:
    """Flip already-normalized features left<->right (negate every x).

    Handy to make left hands look like right hands (or as augmentation).
    """
    out = np.array(features, dtype=np.float32, copy=True)
    shaped = out.reshape(-1, NUM_LANDMARKS, NUM_COORDS)
    shaped[:, :, 0] *= -1.0
    return out
