"""Gesture label set, loaded from ``configs/labels.json``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from gesturelink.data.schema import SchemaError

# src/gesturelink/data/labels.py -> project root is 3 levels above the package dir
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABELS_PATH = PROJECT_ROOT / "configs" / "labels.json"


@dataclass(frozen=True)
class Labels:
    """Ordered gesture names; the index in ``names`` is the class id."""

    names: tuple[str, ...]
    descriptions: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.names)

    def id_of(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise SchemaError(
                f"unknown label {name!r}, expected one of {list(self.names)}"
            ) from None

    def name_of(self, label_id: int) -> str:
        if not 0 <= label_id < len(self.names):
            raise SchemaError(f"label id {label_id} out of range 0..{len(self.names) - 1}")
        return self.names[label_id]

    def check_pair(self, name: str, label_id: int) -> None:
        """Make sure a CSV row's label and label_id agree with this label set."""
        if self.id_of(name) != label_id:
            raise SchemaError(f"label {name!r} has id {self.id_of(name)}, row says {label_id}")


def load_labels(path: str | Path | None = None) -> Labels:
    """Read the label file. ids must be 0..N-1 without gaps."""
    path = Path(path) if path else DEFAULT_LABELS_PATH
    if not path.exists():
        # fallback: run from the project root without an editable install
        alt = Path.cwd() / "configs" / "labels.json"
        if path == DEFAULT_LABELS_PATH and alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"labels file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    gestures = sorted(data["gestures"], key=lambda g: g["id"])
    ids = [g["id"] for g in gestures]
    if ids != list(range(len(gestures))):
        raise SchemaError(f"label ids must be 0..N-1 with no gaps, got {ids}")
    names = tuple(g["name"] for g in gestures)
    if len(set(names)) != len(names):
        raise SchemaError("duplicate label names")
    return Labels(names=names, descriptions=tuple(g.get("description", "") for g in gestures))
