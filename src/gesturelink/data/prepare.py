"""Turn raw landmark CSVs into a normalized, split dataset (``dataset.npz`` + ``meta.json``)."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from gesturelink.data.io import find_csv_files, iter_samples
from gesturelink.data.labels import Labels
from gesturelink.data.normalize import mirror_x, normalize_batch
from gesturelink.data.schema import FEATURE_DIM, SchemaError

DEFAULT_SEED = 42
DEFAULT_VAL_FRACTION = 0.2


def stratified_split(
    labels: np.ndarray, val_fraction: float = DEFAULT_VAL_FRACTION, seed: int = DEFAULT_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class shuffled split. Returns sorted (train_idx, val_idx).

    Written with numpy only (no sklearn needed). A class with a single sample stays in train.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        if len(idx) >= 2:
            n_val = min(max(n_val, 1), len(idx) - 1)  # keep >= 1 sample on both sides
        else:
            n_val = 0
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(val_parts))


def load_raw(path: str | Path, labels: Labels) -> dict:
    """Load every CSV under ``path`` into arrays, checking labels against the label set."""
    files = find_csv_files(path)
    coords, ids, handed, sessions = [], [], [], []
    for file in files:
        for sample in iter_samples(file):
            labels.check_pair(sample.label, sample.label_id)
            coords.append(sample.landmarks.reshape(-1))
            ids.append(sample.label_id)
            handed.append(sample.handedness)
            sessions.append(sample.session_id)
    if not coords:
        raise SchemaError(f"no samples found in {path}")
    return {
        "landmarks": np.stack(coords).astype(np.float32),
        "label_ids": np.array(ids, dtype=np.int64),
        "handedness": np.array(handed),
        "sessions": np.array(sessions),
        "files": [f.name for f in files],
    }


def prepare_dataset(
    input_path: str | Path,
    out_dir: str | Path,
    labels: Labels,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
    mirror_left: bool = False,
) -> dict:
    """Normalize, split 80/20 (stratified) and write ``dataset.npz`` and ``meta.json``.

    Returns the metadata dict that was written to ``meta.json``.
    """
    raw = load_raw(input_path, labels)
    features, valid = normalize_batch(raw["landmarks"])
    n_dropped = int((~valid).sum())

    features = features[valid]
    y = raw["label_ids"][valid]
    handedness = raw["handedness"][valid]
    sessions = raw["sessions"][valid]
    if mirror_left:
        # biến tay trái thành tay phải để model chỉ cần học một phía
        left = handedness == "Left"
        features[left] = mirror_x(features[left])

    train_idx, val_idx = stratified_split(y, val_fraction, seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "dataset.npz",
        X_train=features[train_idx],
        y_train=y[train_idx],
        X_val=features[val_idx],
        y_val=y[val_idx],
    )

    def counts(idx: np.ndarray) -> dict[str, int]:
        c = Counter(int(v) for v in y[idx])
        return {name: c.get(i, 0) for i, name in enumerate(labels.names)}

    meta = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "feature_dim": FEATURE_DIM,
        "labels": list(labels.names),
        "num_samples": int(len(y)),
        "num_dropped_degenerate": n_dropped,
        "split": {"val_fraction": val_fraction, "seed": seed, "stratified": True},
        "counts": {"train": counts(train_idx), "val": counts(val_idx)},
        "handedness": dict(Counter(handedness.tolist())),
        "sessions": sorted(set(sessions.tolist())),
        "source_files": raw["files"],
        "mirror_left": mirror_left,
        "normalization": "subtract wrist (0), divide by wrist->middle MCP (9) distance",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Read ``dataset.npz`` back into a plain dict of arrays (used by training later)."""
    with np.load(path) as data:
        return {k: data[k] for k in data.files}
