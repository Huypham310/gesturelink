"""PyTorch dataset + loaders on top of ``dataset.npz`` from the prepare step."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from gesturelink.data.normalize import mirror_x
from gesturelink.data.prepare import load_dataset
from gesturelink.data.schema import FEATURE_DIM


class LandmarkDataset(Dataset):
    """(63,) float features and an int class id per item.

    ``noise_std`` and ``mirror_prob`` turn on training-time augmentation; leave them at 0 for
    validation. Features are already normalized (wrist = origin, palm size = 1), so noise of
    0.01 means about 1% of the palm size.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        noise_std: float = 0.0,
        mirror_prob: float = 0.0,
    ):
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError(f"features must have shape (N, {FEATURE_DIM}), got {features.shape}")
        if len(features) != len(labels):
            raise ValueError("features and labels have different lengths")
        self.x = torch.from_numpy(features)
        self.y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        self.noise_std = noise_std
        self.mirror_prob = mirror_prob

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x[i]
        if self.mirror_prob > 0 and torch.rand(()) < self.mirror_prob:
            # lật trái/phải: model không cần biết tay nào
            x = torch.from_numpy(mirror_x(x.numpy()[None])[0])
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x, self.y[i]


def load_splits(path: str | Path) -> dict[str, np.ndarray]:
    """Load ``dataset.npz`` and check that it has the four arrays training needs."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found, run `gesturelink prepare` first")
    data = load_dataset(path)
    missing = {"X_train", "y_train", "X_val", "y_val"} - set(data)
    if missing:
        raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
    return data


def make_loaders(
    data: dict[str, np.ndarray],
    batch_size: int = 32,
    noise_std: float = 0.0,
    mirror_prob: float = 0.0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Train loader (shuffled, augmented) and val loader (plain, in order)."""
    train_ds = LandmarkDataset(data["X_train"], data["y_train"], noise_std, mirror_prob)
    val_ds = LandmarkDataset(data["X_val"], data["y_val"])
    gen = torch.Generator().manual_seed(seed)  # fixed shuffle order for reproducible runs
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=max(batch_size, 256), shuffle=False)
    return train_loader, val_loader
