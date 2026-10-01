"""Keep precomputed per-patch dip/azimuth labels consistent with geometric augmentations.

Dip is apparent dip in voxel-index space, computed in ``scripts/sample_patches.py`` from the
gradient of ``geologic_age_faulted``: ``dip = atan2(|grad_xy|, |grad_z|)`` and
``azimuth = atan2(grad_y, grad_x)``. A depth mapping that places ``s`` output samples per source
sample scales ``tan(dip)`` by ``s``; x/y flips and swaps reflect the azimuth and leave dip unchanged.
"""

import numpy as np


DIP_MEAN_CLASS_EDGES_DEG = (10.0, 20.0, 30.0, 40.0, 50.0)
DIP_RANGE_CLASS_EDGES_DEG = (8.0, 12.0, 16.0, 24.0, 32.0)
DEFAULT_DIP_SAMPLE_COUNT = 512
DIP_SAMPLE_KEYS = ("dip_samples_deg", "dip_samples_z")
DIP_DEGREE_KEYS = ("meta_dip_mean_deg", "meta_dip_std_deg", "meta_dip_range_deg")
DIP_CLASS_KEYS = ("meta_dip_mean_class", "meta_dip_range_class")
# Keys whose value changes when apparent dip changes.
DIP_DEPENDENT_KEYS = DIP_DEGREE_KEYS + DIP_CLASS_KEYS + ("meta_structural_complexity",)
AZIMUTH_KEY = "meta_azimuth_mean_deg"
AZIMUTH_VARIANCE_KEY = "meta_azimuth_circular_variance"
# Dip class value the classifier loss skips.
DIP_IGNORE_CLASS = -1.0
DIP_LABEL_POLICIES = ("adjust", "mask", "ignore")
# Weight of meta_dip_std_deg / 45 inside meta_structural_complexity (sample_patches.py).
STRUCTURAL_COMPLEXITY_DIP_STD_WEIGHT = 0.35


def dip_field_deg(structural_patch):
    """Per-voxel apparent dip (deg) plus the x/y gradients and their magnitude."""
    eps = 1e-8
    gx, gy, gz = np.gradient(structural_patch.astype(np.float32), edge_order=1)
    horizontal_mag = np.sqrt(gx * gx + gy * gy)
    dip_deg = np.degrees(np.arctan2(horizontal_mag, np.abs(gz) + eps))
    return dip_deg, gx, gy, horizontal_mag


def sample_dip_field(dip_deg, n_samples, rng):
    """Random voxel dips and their z indices, used later to re-derive dip stats after depth warps."""
    field = np.asarray(dip_deg)
    n = int(n_samples)
    z_dtype = np.uint8 if field.shape[-1] <= 256 else np.uint16
    if n <= 0 or field.size == 0:
        return np.zeros(max(n, 0), np.float16), np.zeros(max(n, 0), z_dtype)
    flat_idx = rng.integers(0, field.size, size=n)
    z_idx = flat_idx % field.shape[-1]
    return field.reshape(-1)[flat_idx].astype(np.float16), z_idx.astype(z_dtype)


def quantize(value, edges):
    return float(np.digitize(float(value), edges))


def adjust_azimuth_deg(azimuth_deg, swap_xy=False, flip_x=False, flip_y=False):
    """Azimuth after the swap -> flip_x -> flip_y order used by apply_pair_augmentations."""
    a = float(azimuth_deg)
    if swap_xy:
        a = 90.0 - a
    if flip_x:
        a = 180.0 - a
    if flip_y:
        a = -a
    return a % 360.0


def vertical_warp_stretch_at_source(target_indices, z_source):
    """Output samples per source sample (dz'/dz) at source depths for a vertical warp.

    ``target_indices[z']`` is the source depth read for output depth ``z'`` and is increasing.
    """
    t = np.asarray(target_indices, dtype=np.float64)
    if t.size < 2:
        return np.ones(np.shape(z_source), dtype=np.float64)
    stretch_out = 1.0 / np.maximum(np.gradient(t), 1e-6)
    return np.interp(np.asarray(z_source, dtype=np.float64), t, stretch_out)


def stretch_dip_deg(dip_deg, stretch):
    """Apparent dip after scaling depth by ``stretch``: tan(dip') = tan(dip) * stretch."""
    d = np.radians(np.asarray(dip_deg, dtype=np.float64))
    return np.degrees(np.arctan2(np.sin(d) * stretch, np.cos(d)))


def _weighted_percentiles(values, weights, percentiles):
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weights[order]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(np.asarray(percentiles, dtype=np.float64) / 100.0, cw, v)


def dip_stats(dip_deg, weights=None):
    """(mean, std, p90 - p10) of dips with optional voxel weights."""
    d = np.asarray(dip_deg, dtype=np.float64).reshape(-1)
    w = np.ones_like(d) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    keep = np.isfinite(d) & np.isfinite(w) & (w > 0)
    d, w = d[keep], w[keep]
    if d.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.sum(w * d) / np.sum(w))
    std = float(np.sqrt(np.sum(w * (d - mean) ** 2) / np.sum(w)))
    p10, p90 = _weighted_percentiles(d, w, (10.0, 90.0))
    return mean, std, float(p90 - p10)


def adjusted_dip_values_for_stretch(
    stored,
    dip_samples_deg,
    stretch_per_sample,
    density_weights=None,
    mean_edges=DIP_MEAN_CLASS_EDGES_DEG,
    range_edges=DIP_RANGE_CLASS_EDGES_DEG,
):
    """New values for DIP_DEPENDENT_KEYS after a depth stretch.

    ``stored`` holds the exact precomputed values (the three degree keys, optionally
    ``meta_structural_complexity``). The change measured on the voxel samples (stretched,
    density-weighted by the stretch, minus unstretched) is added to the exact values, so sampling
    error mostly cancels and an identity stretch returns the stored values.
    """
    samples = np.asarray(dip_samples_deg, dtype=np.float64)
    stretch = np.asarray(stretch_per_sample, dtype=np.float64)
    weights = stretch if density_weights is None else np.asarray(density_weights, dtype=np.float64)
    base = dip_stats(samples)
    warped = dip_stats(stretch_dip_deg(samples, stretch), weights=weights)
    mean = float(np.clip(float(stored["meta_dip_mean_deg"]) + warped[0] - base[0], 0.0, 90.0))
    std = float(max(0.0, float(stored["meta_dip_std_deg"]) + warped[1] - base[1]))
    dip_range = float(np.clip(float(stored["meta_dip_range_deg"]) + warped[2] - base[2], 0.0, 90.0))
    out = {
        "meta_dip_mean_deg": mean,
        "meta_dip_std_deg": std,
        "meta_dip_range_deg": dip_range,
        "meta_dip_mean_class": quantize(mean, mean_edges),
        "meta_dip_range_class": quantize(dip_range, range_edges),
    }
    if "meta_structural_complexity" in stored:
        delta = STRUCTURAL_COMPLEXITY_DIP_STD_WEIGHT * (std - float(stored["meta_dip_std_deg"])) / 45.0
        out["meta_structural_complexity"] = float(max(0.0, float(stored["meta_structural_complexity"]) + delta))
    return out
