"""Training loop, config loading and checkpoint save / load."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn

from gesturelink.data.labels import PROJECT_ROOT, Labels
from gesturelink.data.schema import FEATURE_DIM
from gesturelink.model.dataset import load_splits, make_loaders
from gesturelink.model.eval import evaluate
from gesturelink.model.net import DEFAULT_HIDDEN, GestureNet

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.json"
CHECKPOINT_VERSION = 1


@dataclass
class TrainConfig:
    """Hyperparameters and paths. Defaults match ``configs/train.json``."""

    seed: int = 42
    hidden: list[int] = field(default_factory=lambda: list(DEFAULT_HIDDEN))
    dropout: float = 0.1
    epochs: int = 80
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    noise_std: float = 0.01  # gaussian jitter on training features (palm size units)
    mirror_prob: float = 0.5  # chance to flip a training sample left<->right
    # TODO: thử learning rate scheduler (cosine) nếu có data thật
    patience: int = 15  # stop after this many epochs without a better val accuracy (0 = off)
    data: str = "data/processed/dataset.npz"
    meta: str = "data/processed/meta.json"
    out_dir: str = "models"
    checkpoint_name: str = "gesturelink.pt"

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be >= 1")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= self.mirror_prob <= 1.0:
            raise ValueError("mirror_prob must be in [0, 1]")
        if self.noise_std < 0:
            raise ValueError("noise_std must be >= 0")


def load_config(path: str | Path | None = None, **overrides) -> TrainConfig:
    """Read a JSON config, then apply non-None keyword overrides (used by CLI flags)."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        alt = Path.cwd() / "configs" / "train.json"  # same fallback as labels.json
        if path == DEFAULT_CONFIG_PATH and alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"train config not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(TrainConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    raw.update({k: v for k, v in overrides.items() if v is not None})
    cfg = TrainConfig(**raw)
    cfg.validate()
    return cfg


def set_seed(seed: int) -> None:
    # torch seeds init, dropout, shuffling and augmentation noise (numpy isn't used for those)
    torch.manual_seed(seed)


def save_checkpoint(
    path: str | Path, model: GestureNet, labels: Labels, config: TrainConfig, extra: dict
) -> Path:
    """Save weights + everything needed to rebuild the model. Only plain types go in the file,
    so it can be read back with ``torch.load(weights_only=True)``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "state_dict": model.state_dict(),
            "arch": model.arch(),
            "labels": list(labels.names),
            "config": asdict(config),
            **extra,
        },
        path,
    )
    return path


def load_checkpoint(path: str | Path) -> tuple[GestureNet, dict]:
    """Rebuild the model from a checkpoint. Returns (model in eval mode, raw checkpoint dict)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}, run `gesturelink train` first")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version {ckpt.get('version')}")
    model = GestureNet(**ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def train(config: TrainConfig, labels: Labels, verbose: bool = True) -> dict:
    """Train on ``config.data`` and write the best checkpoint + ``metrics.json`` to out_dir.

    "Best" = highest val accuracy (ties -> lower val loss). Returns the metrics dict.
    """
    config.validate()
    set_seed(config.seed)
    data = load_splits(config.data)
    train_loader, val_loader = make_loaders(
        data, config.batch_size, config.noise_std, config.mirror_prob, config.seed
    )
    if int(data["y_train"].max()) >= len(labels) or int(data["y_val"].max()) >= len(labels):
        raise ValueError("dataset has label ids outside the label set")

    model = GestureNet(FEATURE_DIM, config.hidden, len(labels), config.dropout)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    best = {"acc": -1.0, "loss": float("inf"), "epoch": 0, "state": None}
    history = []
    stale = 0
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            n += len(y)
        train_loss = total / max(n, 1)

        val_loss, val_acc = _val_scores(model, val_loader, loss_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "val_acc": round(val_acc, 4),
            }
        )
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(
                f"epoch {epoch:>3}  loss {train_loss:.4f}  "
                f"val_loss {val_loss:.4f}  acc {val_acc:.3f}"
            )

        improved = val_acc > best["acc"] or (val_acc == best["acc"] and val_loss < best["loss"])
        if improved:
            # deepcopy, otherwise the saved weights keep changing as training goes on
            best.update(
                acc=val_acc, loss=val_loss, epoch=epoch, state=copy.deepcopy(model.state_dict())
            )
            stale = 0
        else:
            stale += 1
            if config.patience and stale >= config.patience:
                if verbose:
                    print(f"early stop at epoch {epoch} (best was {best['epoch']})")
                break
    seconds = time.perf_counter() - start

    # quay lại epoch tốt nhất thay vì epoch cuối (cuối có thể bị overfit nhẹ)
    model.load_state_dict(best["state"])
    metrics = evaluate(model, val_loader, labels.names)
    metrics.update(
        {
            "best_epoch": best["epoch"],
            "epochs_run": len(history),
            "train_seconds": round(seconds, 2),
            "num_parameters": model.num_parameters(),
            "num_train": int(len(data["y_train"])),
            "labels": list(labels.names),
            "config": asdict(config),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "history": history,
        }
    )

    out_dir = Path(config.out_dir)
    ckpt_path = save_checkpoint(
        out_dir / config.checkpoint_name, model, labels, config,
        {"val_accuracy": metrics["accuracy"], "best_epoch": best["epoch"]},
    )
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics["checkpoint"] = str(ckpt_path)
    metrics["metrics_path"] = str(metrics_path)
    return metrics


@torch.no_grad()
def _val_scores(model: nn.Module, loader, loss_fn: nn.Module) -> tuple[float, float]:
    """(mean loss, accuracy) over a loader."""
    model.eval()
    total, correct, n = 0.0, 0, 0
    for x, y in loader:
        logits = model(x)
        total += loss_fn(logits, y).item() * len(y)
        correct += int((logits.argmax(dim=1) == y).sum())
        n += len(y)
    return total / max(n, 1), correct / max(n, 1)
