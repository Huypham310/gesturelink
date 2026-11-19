"""Inference: raw MediaPipe landmarks in, gesture name + confidence out."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from gesturelink.data.normalize import mirror_x, normalize_batch
from gesturelink.data.schema import FEATURE_DIM
from gesturelink.model.train import load_checkpoint


@dataclass(frozen=True)
class Prediction:
    label: str
    label_id: int
    confidence: float  # softmax probability of the winning class
    probabilities: dict[str, float]


class GesturePredictor:
    """Loads a checkpoint once and classifies hands.

    ``predict`` takes the *raw* landmarks (21x3 or 63 floats, as saved in the collect CSVs) and
    normalizes them itself, so callers never have to remember the preprocessing.
    """

    def __init__(self, checkpoint: str | Path, mirror_left: bool | None = None):
        self.model, ckpt = load_checkpoint(checkpoint)
        self.labels: list[str] = list(ckpt["labels"])
        self.config: dict = ckpt["config"]
        self.val_accuracy: float | None = ckpt.get("val_accuracy")
        # keep inference consistent with how the data was prepared; default = no mirroring
        self.mirror_left = bool(mirror_left)

    def predict_batch(
        self, landmarks: np.ndarray, handedness: list[str] | None = None
    ) -> list[Prediction | None]:
        """Classify N raw hands. A degenerate hand (no palm size) gives ``None`` in its slot."""
        raw = np.asarray(landmarks, dtype=np.float32).reshape(-1, FEATURE_DIM)
        feats, valid = normalize_batch(raw)
        if self.mirror_left and handedness is not None:
            left = np.array([h == "Left" for h in handedness])
            feats[left] = mirror_x(feats[left])
        results: list[Prediction | None] = [None] * len(raw)
        if not valid.any():
            return results
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.from_numpy(feats[valid])), dim=1).numpy()
        for slot, p in zip(np.flatnonzero(valid), probs, strict=True):
            best = int(p.argmax())
            results[slot] = Prediction(
                label=self.labels[best],
                label_id=best,
                confidence=float(p[best]),
                probabilities={n: round(float(v), 4) for n, v in zip(self.labels, p, strict=True)},
            )
        return results

    def predict(self, landmarks: np.ndarray, handedness: str | None = None) -> Prediction | None:
        """Classify one hand; ``None`` if the landmarks are degenerate."""
        hands = [handedness] if handedness else None
        return self.predict_batch(np.asarray(landmarks).reshape(1, -1), hands)[0]
