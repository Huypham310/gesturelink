"""Row schema for the raw landmark CSV files.

One row = one detected hand in one frame:

    label,label_id,session_id,timestamp_ms,handedness,x0,y0,z0,...,x20,y20,z20

The coordinates are the *raw* MediaPipe values (x, y in [0, 1] image space, z relative depth).
Normalization happens later, in ``prepare``, so the CSV always keeps the original detections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NUM_LANDMARKS = 21
NUM_COORDS = 3  # x, y, z
FEATURE_DIM = NUM_LANDMARKS * NUM_COORDS  # 63

HANDEDNESS_VALUES = ("Left", "Right")

META_COLUMNS = ["label", "label_id", "session_id", "timestamp_ms", "handedness"]
# order is x0,y0,z0,x1,y1,z1,... so a (21, 3) array flattens straight into these columns
COORD_COLUMNS = [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in "xyz"]
CSV_COLUMNS = META_COLUMNS + COORD_COLUMNS


class SchemaError(ValueError):
    """Raised when a row or a file does not match the expected landmark schema."""


@dataclass
class Sample:
    """A single labelled hand detection (raw, not normalized)."""

    label: str
    label_id: int
    session_id: str
    timestamp_ms: int
    handedness: str
    landmarks: np.ndarray  # shape (21, 3)

    def __post_init__(self) -> None:
        # ép về float32 và kiểm tra đủ 63 số trước khi reshape
        self.landmarks = np.asarray(self.landmarks, dtype=np.float32)
        if self.landmarks.size != FEATURE_DIM:
            raise SchemaError(
                f"expected {FEATURE_DIM} landmark values, got {self.landmarks.size}"
            )
        self.landmarks = self.landmarks.reshape(NUM_LANDMARKS, NUM_COORDS)
        if self.handedness not in HANDEDNESS_VALUES:
            raise SchemaError(f"handedness must be Left or Right, got {self.handedness!r}")
        if not np.isfinite(self.landmarks).all():
            raise SchemaError("landmarks contain NaN or inf")
        if not self.label:
            raise SchemaError("empty label")

    def to_row(self) -> list[str]:
        """Flat list of strings in CSV_COLUMNS order."""
        coords = [f"{v:.6f}" for v in self.landmarks.reshape(-1)]
        return [
            self.label,
            str(self.label_id),
            self.session_id,
            str(self.timestamp_ms),
            self.handedness,
            *coords,
        ]

    @classmethod
    def from_row(cls, row: dict[str, str]) -> Sample:
        """Build a Sample from a csv.DictReader row, validating every field."""
        missing = [c for c in CSV_COLUMNS if c not in row or row[c] is None]
        if missing:
            raise SchemaError(f"missing columns: {missing[:5]}")
        try:
            coords = [float(row[c]) for c in COORD_COLUMNS]
            label_id = int(row["label_id"])
            timestamp_ms = int(row["timestamp_ms"])
        except ValueError as exc:
            raise SchemaError(f"bad numeric value: {exc}") from exc
        if any(math.isnan(v) or math.isinf(v) for v in coords):
            raise SchemaError("landmarks contain NaN or inf")
        return cls(
            label=row["label"],
            label_id=label_id,
            session_id=row["session_id"],
            timestamp_ms=timestamp_ms,
            handedness=row["handedness"],
            landmarks=np.array(coords, dtype=np.float32),
        )


def check_header(header: list[str]) -> None:
    """Raise SchemaError unless the header is exactly CSV_COLUMNS (same order)."""
    if list(header) != CSV_COLUMNS:
        extra = set(header) - set(CSV_COLUMNS)
        missing = set(CSV_COLUMNS) - set(header)
        raise SchemaError(
            f"unexpected CSV header (missing={sorted(missing)[:4]}, extra={sorted(extra)[:4]})"
        )
