"""Targets and losses for the patch-level geology classifier decoder (WP3)."""

import numpy as np
import torch
import torch.nn.functional as F

from src.model import GEOLOGY_DIP_CLASSES, GEOLOGY_PRESENCE_CLASSES

PRESENCE_TARGET_KEYS = {c: f"label_presence_{c}" for c in GEOLOGY_PRESENCE_CLASSES}
DIP_TARGET_KEYS = {"dip_mean": "meta_dip_mean_class", "dip_range": "meta_dip_range_class"}
ALL_CLASSIFIER_TARGETS = GEOLOGY_PRESENCE_CLASSES + tuple(DIP_TARGET_KEYS)
POS_WEIGHT_MAX = 50.0


def classifier_target_keys(targets=ALL_CLASSIFIER_TARGETS):
    """Dataset array names needed for the selected classifier targets."""
    keys = []
    for t in targets:
        if t in PRESENCE_TARGET_KEYS:
            keys.append(PRESENCE_TARGET_KEYS[t])
        elif t in DIP_TARGET_KEYS:
            keys.append(DIP_TARGET_KEYS[t])
        else:
            raise ValueError(f"Unknown geology classifier target {t!r}; choose from {ALL_CLASSIFIER_TARGETS}")
    return tuple(keys)


def compute_pos_weight(presence, max_weight=POS_WEIGHT_MAX):
    """Per-class clip(n_neg / n_pos, 1, max_weight) from an (N, C) 0/1 presence matrix; 1 when a class has no positives."""
    y = np.asarray(presence, dtype=np.float64)
    n_pos = y.sum(axis=0)
    n_neg = y.shape[0] - n_pos
    ratio = np.where(n_pos > 0, n_neg / np.maximum(n_pos, 1.0), 1.0)
    return torch.as_tensor(np.clip(ratio, 1.0, max_weight), dtype=torch.float32)


def focal_bce_with_logits(logits, targets, pos_weight=None, gamma=2.0):
    ce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    return ce * (1.0 - p_t).pow(float(gamma))


def compute_geology_classifier_loss(
    logits,
    batch,
    targets=ALL_CLASSIFIER_TARGETS,
    loss_type="bce",
    focal_gamma=2.0,
    label_smoothing=0.05,
    pos_weight=None,
):
    """Return (loss, parts): mean presence BCE/focal over selected classes plus CE per selected dip target."""
    device = logits["presence"].device
    parts = {}
    total = logits["presence"].new_zeros(())
    presence_idx = [GEOLOGY_PRESENCE_CLASSES.index(t) for t in targets if t in PRESENCE_TARGET_KEYS]
    if presence_idx:
        y = torch.stack(
            [torch.as_tensor(batch[PRESENCE_TARGET_KEYS[GEOLOGY_PRESENCE_CLASSES[i]]]).to(device).float() for i in presence_idx],
            dim=1,
        )
        z = logits["presence"][:, presence_idx]
        pw = pos_weight[presence_idx].to(device) if pos_weight is not None else None
        if loss_type == "focal":
            per_elem = focal_bce_with_logits(z, y, pw, focal_gamma)
        elif loss_type == "bce":
            per_elem = F.binary_cross_entropy_with_logits(z, y, pos_weight=pw, reduction="none")
        else:
            raise ValueError(f"Unsupported geology classifier loss type {loss_type!r}")
        presence_loss = per_elem.mean(dim=0).mean()
        parts["presence"] = float(presence_loss.detach())
        total = total + presence_loss
    for name, key in DIP_TARGET_KEYS.items():
        if name not in targets:
            continue
        cls = torch.as_tensor(batch[key]).to(device).round().long().clamp(0, GEOLOGY_DIP_CLASSES - 1)
        dip_loss = F.cross_entropy(logits[name], cls, label_smoothing=float(label_smoothing))
        parts[name] = float(dip_loss.detach())
        total = total + dip_loss
    return total, parts
