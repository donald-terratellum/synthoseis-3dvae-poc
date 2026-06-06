from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSupervisionLoss(nn.Module):
    """Weighted loss wrapper for MONAI-style multi-scale decoder outputs."""

    def __init__(self, base_loss: nn.Module, weights: Sequence[float] = (1.0, 0.5, 0.25)):
        super().__init__()
        if len(weights) == 0:
            raise ValueError("weights must not be empty")
        self.base_loss = base_loss
        self.weights = tuple(float(w) for w in weights)
        if any(w < 0.0 for w in self.weights):
            raise ValueError("weights must be non-negative")

    def forward(self, outputs, target: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return self.base_loss(outputs, target)
        if outputs is None:
            raise ValueError("outputs must not be None")

        predictions = list(outputs)
        if len(predictions) == 0:
            raise ValueError("outputs must contain at least one tensor")
        if len(predictions) != len(self.weights):
            raise ValueError(
                f"outputs length ({len(predictions)}) must match weights length ({len(self.weights)})"
            )

        total = torch.zeros((), dtype=target.dtype, device=target.device)
        for weight, pred in zip(self.weights, predictions):
            target_for_scale = target
            if pred.shape[2:] != target.shape[2:]:
                target_for_scale = F.interpolate(target, size=pred.shape[2:], mode='trilinear', align_corners=False)
            total = total + (weight * self.base_loss(pred, target_for_scale))
        return total
