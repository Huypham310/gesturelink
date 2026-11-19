"""Evaluation helpers: accuracy, confusion matrix, per-class precision / recall / F1.

Plain numpy, no sklearn, so the dependency list stays short.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def predict_batches(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader. Returns (true ids, predicted ids)."""
    model.eval()  # tắt dropout khi đánh giá
    ys, preds = [], []
    for x, y in loader:
        ys.append(y.numpy())
        preds.append(model(x).argmax(dim=1).numpy())
    if not ys:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(ys), np.concatenate(preds)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """``cm[i, j]`` = samples of true class i predicted as class j."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)), 1)
    return cm


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def per_class_report(cm: np.ndarray, names: list[str] | tuple[str, ...]) -> dict[str, dict]:
    """Precision, recall, F1 and support for each class (0 when a denominator is 0)."""
    report = {}
    for i, name in enumerate(names):
        tp = int(cm[i, i])
        support = int(cm[i].sum())
        predicted = int(cm[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return report


def evaluate(
    model: nn.Module, loader: DataLoader, names: list[str] | tuple[str, ...]
) -> dict:
    """Full metrics dict (JSON friendly) for one loader."""
    y_true, y_pred = predict_batches(model, loader)
    cm = confusion_matrix(y_true, y_pred, len(names))
    per_class = per_class_report(cm, names)
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()])) if per_class else 0.0
    return {
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "macro_f1": round(macro_f1, 4),
        "num_samples": int(len(y_true)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def format_report(metrics: dict, names: list[str] | tuple[str, ...]) -> str:
    """Small text table for the CLI."""
    lines = [f"{'class':<12}{'prec':>7}{'recall':>8}{'f1':>7}{'n':>6}"]
    for name in names:
        m = metrics["per_class"][name]
        lines.append(
            f"{name:<12}{m['precision']:>7.2f}{m['recall']:>8.2f}{m['f1']:>7.2f}{m['support']:>6}"
        )
    lines.append(f"accuracy {metrics['accuracy']:.4f}  macro-f1 {metrics['macro_f1']:.4f}")
    return "\n".join(lines)
