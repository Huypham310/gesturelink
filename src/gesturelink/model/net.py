"""The classifier: a small MLP on the 63 normalized landmark values."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from gesturelink.data.schema import FEATURE_DIM

DEFAULT_HIDDEN = (128, 64)


class GestureNet(nn.Module):
    """MLP ``63 -> 128 -> 64 -> num_classes`` (sizes come from ``hidden``).

    Input is the wrist-normalized landmark vector from ``prepare`` (already scale-free), so we
    don't need conv layers or a separate standardization step. Output is raw logits.
    """

    def __init__(
        self,
        in_dim: int = FEATURE_DIM,
        hidden: Sequence[int] = DEFAULT_HIDDEN,
        num_classes: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        # lưu lại các tham số để checkpoint dựng lại đúng kiến trúc
        self.in_dim = in_dim
        self.hidden = tuple(int(h) for h in hidden)
        self.num_classes = num_classes
        self.dropout = dropout

        layers: list[nn.Module] = []
        prev = in_dim
        for width in self.hidden:
            layers += [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)]
            prev = width
        layers.append(nn.Linear(prev, num_classes))  # no softmax here, CrossEntropyLoss does it
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def arch(self) -> dict:
        """Constructor kwargs, saved in the checkpoint so the model can be rebuilt on load."""
        return {
            "in_dim": self.in_dim,
            "hidden": list(self.hidden),
            "num_classes": self.num_classes,
            "dropout": self.dropout,
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
