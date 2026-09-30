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
        raw = torch.as_tensor(batch[key]).to(device).round().long()
        # Negative class = ignored target (dip label masked after a depth-warp augmentation).
        valid = raw >= 0
        if not bool(valid.any()):
            continue
        cls = raw.clamp(0, GEOLOGY_DIP_CLASSES - 1)
        dip_loss = F.cross_entropy(logits[name][valid], cls[valid], label_smoothing=float(label_smoothing))
        parts[name] = float(dip_loss.detach())
        total = total + dip_loss
    return total, parts


def binary_auroc(y_true, scores):
    """Mann-Whitney AUROC with average ranks for ties; None when a class is missing."""
    y = np.asarray(y_true).reshape(-1) > 0
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    uniq, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg_rank = cum - (counts - 1) / 2.0
    ranks = avg_rank[inverse]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _threshold_curve(y_true, scores):
    """Distinct thresholds (descending) with cumulative true/false positives at score >= threshold."""
    y = np.asarray(y_true).reshape(-1) > 0
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    last_of_group = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]
    tp = np.cumsum(y)[last_of_group]
    fp = (last_of_group + 1) - tp
    return s[last_of_group], tp.astype(np.float64), fp.astype(np.float64), int(y.sum())


def average_precision(y_true, scores):
    """Step-wise AP = sum_k (R_k - R_{k-1}) P_k over distinct thresholds; None without positives."""
    _, tp, fp, n_pos = _threshold_curve(y_true, scores)
    if n_pos == 0:
        return None
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def best_f1_threshold(y_true, scores, default=0.5):
    """Threshold on scores maximizing F1 (predict score >= threshold); default when there are no positives."""
    thresholds, tp, fp, n_pos = _threshold_curve(y_true, scores)
    if n_pos == 0:
        return float(default)
    f1 = 2.0 * tp / (tp + fp + n_pos)
    return float(thresholds[int(np.argmax(f1))])


def f1_at_threshold(y_true, scores, threshold):
    y = np.asarray(y_true).reshape(-1) > 0
    pred = np.asarray(scores, dtype=np.float64).reshape(-1) >= float(threshold)
    tp = float(np.sum(pred & y))
    denom = float(pred.sum() + y.sum())
    return 2.0 * tp / denom if denom > 0 else 0.0


def confusion_matrix(true_labels, pred_labels, n_classes):
    cm = np.zeros((int(n_classes), int(n_classes)), dtype=np.int64)
    np.add.at(cm, (np.asarray(true_labels, dtype=np.int64), np.asarray(pred_labels, dtype=np.int64)), 1)
    return cm


def _mean_or_none(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def evaluate_classifier_predictions(probs, targets, thresholds=None, auroc_gate=0.75):
    """Classifier report on a natural-prevalence set.

    probs: {"presence": (N, 7), "dip_mean": (N, 6), "dip_range": (N, 6)} probabilities.
    targets: {"presence": (N, 7) 0/1, "dip_mean": (N,), "dip_range": (N,)} labels.
    thresholds: optional per-class presence thresholds (tuned on train); 0.5 otherwise.
    """
    presence_p = np.asarray(probs["presence"], dtype=np.float64)
    presence_y = np.asarray(targets["presence"]) > 0
    thresholds = thresholds or {}
    presence = {}
    for i, name in enumerate(GEOLOGY_PRESENCE_CLASSES):
        t = float(thresholds.get(name, 0.5))
        presence[name] = {
            "prevalence": float(presence_y[:, i].mean()),
            "auroc": binary_auroc(presence_y[:, i], presence_p[:, i]),
            "average_precision": average_precision(presence_y[:, i], presence_p[:, i]),
            "threshold": t,
            "f1": f1_at_threshold(presence_y[:, i], presence_p[:, i], t) if presence_y[:, i].any() else None,
        }
    report = {
        "n": int(presence_p.shape[0]),
        "presence": presence,
        "macro_auroc": _mean_or_none(v["auroc"] for v in presence.values()),
        "macro_average_precision": _mean_or_none(v["average_precision"] for v in presence.values()),
        "macro_f1": _mean_or_none(v["f1"] for v in presence.values()),
    }
    gate = {}
    for name in DIP_TARGET_KEYS:
        true = np.clip(np.rint(np.asarray(targets[name], dtype=np.float64)).astype(np.int64), 0, GEOLOGY_DIP_CLASSES - 1)
        pred = np.argmax(np.asarray(probs[name]), axis=1)
        counts = np.bincount(true, minlength=GEOLOGY_DIP_CLASSES)
        accuracy = float(np.mean(pred == true))
        majority = float(counts.max() / max(counts.sum(), 1))
        report[name] = {
            "accuracy": accuracy,
            "majority_rate": majority,
            "class_counts": counts.tolist(),
            "confusion_matrix": confusion_matrix(true, pred, GEOLOGY_DIP_CLASSES).tolist(),
        }
        gate[f"{name}_above_majority"] = bool(accuracy > majority)
    gate[f"macro_auroc_ge_{auroc_gate}"] = bool(report["macro_auroc"] is not None and report["macro_auroc"] >= auroc_gate)
    gate["passed"] = bool(all(gate.values()))
    report["sanity_gate"] = gate
    return report
