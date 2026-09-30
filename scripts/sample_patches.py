#!/usr/bin/env python3

"""Sample 3D patches from existing model_data.zarr stores into a destination zarr store.

Usage:
    python scripts/sample_patches.py --source /path/to/fake_data --out data/train.zarr --patch_size 32 32 32 --n_patches 5000 \
        --seismic_key seismicCubes_cumsum_fullstack --geoscore_key geologic_score --n_per_volume 100 --exclude_dir validation
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import random
import secrets
import numpy as np
import zarr
import math
from typing import Any, cast

from src.geology_label_augment import (
    DEFAULT_DIP_SAMPLE_COUNT,
    DIP_MEAN_CLASS_EDGES_DEG,
    DIP_RANGE_CLASS_EDGES_DEG,
    dip_field_deg,
    quantize,
    sample_dip_field,
)


DERIVED_METADATA_KEYS = (
    "meta_dip_mean_deg",
    "meta_dip_std_deg",
    "meta_azimuth_mean_deg",
    "meta_azimuth_circular_variance",
    "meta_fault_fraction",
    "meta_fault_intersection_fraction",
    "meta_geologic_score_mean",
    "meta_sand_fraction",
    "meta_shale_fraction",
    "meta_flat_spot_fraction",
    "meta_onlap_fraction",
    "meta_onlap_variability",
    "meta_channel_fraction",
    "meta_channel_core_fraction",
    "meta_structural_complexity",
    "meta_dip_range_deg",
    "meta_dip_mean_class",
    "meta_dip_range_class",
    "meta_water_fraction",
    "meta_closure_fraction",
)

DEFAULT_SEISMIC_KEY = "seismicCubes_cumsum_fullstack"
# faulted_lithology: -1 water (above seabed), 0 shale, 1 sand; fractional at boundaries.
DEFAULT_SAND_THRESHOLD = 0.5
DEFAULT_ONLAP_THRESHOLD = 0.5

LABEL_CLASS_ORDER = ("fault", "fault_x", "channel", "closure", "onlap", "sand", "flat_spot")
LABEL_CLASS_SOURCES = {
    "fault": "fault_segments_id",
    "fault_x": "fault_intersection_segments",
    "channel": "faults/faulted_channel_labels",
    "closure": "closure_segments_id",
    "onlap": "onlap_segments",
    "sand": "faulted_lithology",
    "flat_spot": "flat_spot",
}
# Classes whose source array holds distinct object ids; others use a coarse spatial cell as object key.
OBJECT_ID_CLASSES = ("fault", "closure")
DEFAULT_CLASS_QUOTAS = {c: 0.125 for c in ("fault", "fault_x", "channel", "closure", "onlap", "flat_spot")}
DEFAULT_BACKGROUND_FRACTION = 0.25
DEFAULT_MAX_PATCHES_PER_OBJECT = 24
BACKGROUND = "background"


def normalize_patch_size(values):
    if len(values) == 1:
        v = int(values[0])
        dims = (v, v, v)
    elif len(values) == 3:
        dims = tuple(int(v) for v in values)
    else:
        raise ValueError("--patch_size expects either 1 value or 3 values: X Y Z")
    if any(v <= 0 for v in dims):
        raise ValueError("patch_size values must be positive")
    return dims


def candidate_positions(shape, patch_size, n_candidates=500):
    sx, sy, sz = patch_size
    max_x = shape[0] - sx
    max_y = shape[1] - sy
    max_z = shape[2] - sz
    if max_x < 0 or max_y < 0 or max_z < 0:
        return []
    candidates = []
    for _ in range(n_candidates):
        i = random.randint(0, max_x)
        j = random.randint(0, max_y)
        k = random.randint(0, max_z)
        candidates.append((i, j, k))
    return candidates


def pick_weighted_positions(geoscore, sampling_shape, patch_size, n_picks, n_candidates=1000, allow_overlap=True):
    # geoscore: numpy array shaped (X,Y,Z)
    candidates = candidate_positions(sampling_shape, patch_size, n_candidates=n_candidates)
    if not candidates:
        return []
    sx, sy, sz = patch_size
    scores = []
    for (i,j,k) in candidates:
        # geoscore can have different extents than seismic; out-of-range slices get zero weight.
        patch_score = geoscore[i:i+sx, j:j+sy, k:k+sz]
        score = float(patch_score.sum()) if patch_score.size > 0 else 0.0
        scores.append(score)
    scores = np.array(scores)
    if scores.sum() <= 0:
        # fallback to uniform picks; allow replacement when overlapping is enabled
        if allow_overlap:
            return random.choices(candidates, k=n_picks)
        chosen = random.sample(candidates, min(n_picks, len(candidates)))
        return chosen
    probs = scores / scores.sum()
    if allow_overlap:
        idx = np.random.choice(len(candidates), size=n_picks, replace=True, p=probs)
    else:
        # replace=False requires enough non-zero probability entries.
        nonzero = int(np.count_nonzero(probs))
        max_unique = min(len(candidates), nonzero)
        pick_count = min(n_picks, max_unique)
        idx = np.random.choice(len(candidates), size=pick_count, replace=False, p=probs)
    return [candidates[i] for i in idx]


def _safe_extract_patch(array, origin, patch_size):
    i, j, k = origin
    sx, sy, sz = patch_size
    patch = np.asarray(array[i:i+sx, j:j+sy, k:k+sz], dtype=np.float32)
    if patch.shape != (sx, sy, sz):
        return None
    return patch


def _safe_extract_patch_by_key(zvol, key, origin, patch_size):
    try:
        if key not in zvol:
            return None
        return _safe_extract_patch(zvol[key], origin, patch_size)
    except Exception:
        return None


def _compute_dip_azimuth_features(structural_patch):
    dip_deg, gx, gy, horizontal_mag = dip_field_deg(structural_patch)

    azimuth_rad = np.arctan2(gy, gx)
    valid_mask = horizontal_mag > 1e-6
    if np.any(valid_mask):
        az_valid = azimuth_rad[valid_mask]
        mean_sin = float(np.mean(np.sin(az_valid)))
        mean_cos = float(np.mean(np.cos(az_valid)))
        azimuth_mean_rad = float(np.arctan2(mean_sin, mean_cos))
        mean_resultant_length = float(np.sqrt(mean_sin * mean_sin + mean_cos * mean_cos))
        azimuth_circular_variance = float(max(0.0, 1.0 - mean_resultant_length))
    else:
        azimuth_mean_rad = 0.0
        azimuth_circular_variance = 1.0

    azimuth_mean_deg = (float(np.degrees(azimuth_mean_rad)) + 360.0) % 360.0
    dip_mean_deg = float(np.nanmean(dip_deg))
    dip_std_deg = float(np.nanstd(dip_deg))
    dip_p10_deg, dip_p90_deg = np.nanpercentile(dip_deg, [10.0, 90.0])
    dip_range_deg = float(dip_p90_deg - dip_p10_deg)
    return dip_mean_deg, dip_std_deg, dip_range_deg, azimuth_mean_deg, azimuth_circular_variance


def compute_patch_dip_samples(
    zvol, origin, patch_size, n_samples, rng, dip_source_key="geologic_age_faulted", label_z_offset=0
):
    """Voxel dip samples (deg, float16) and local z indices for the label window of a seismic patch."""
    origin = (origin[0], origin[1], origin[2] + int(label_z_offset))
    structural_patch = _safe_extract_patch_by_key(zvol, dip_source_key, origin, patch_size)
    if structural_patch is None or structural_patch.size == 0:
        structural_patch = np.zeros(tuple(patch_size), dtype=np.float32)
    structural_patch = np.nan_to_num(structural_patch, nan=0.0, posinf=0.0, neginf=0.0)
    return sample_dip_field(dip_field_deg(structural_patch)[0], n_samples, rng)


def write_dip_samples(dst, dip_samples_deg, dip_samples_z):
    """Write (N, K) dip sample arrays to an open zarr group, replacing existing ones."""
    n, k = dip_samples_deg.shape
    chunks = (min(max(n, 1), 4096), k)
    for name, data in (("dip_samples_deg", dip_samples_deg), ("dip_samples_z", dip_samples_z)):
        if name in dst:
            del dst[name]
        if hasattr(dst, "create_dataset"):
            arr = dst.create_dataset(name, shape=data.shape, dtype=data.dtype, chunks=chunks)
        else:
            arr = dst.create_array(name, shape=data.shape, dtype=data.dtype, chunks=chunks)
        arr[:] = data
    dst.attrs["dip_sample_count"] = int(k)


def compute_lithology_fractions(lith_patch, sand_threshold=DEFAULT_SAND_THRESHOLD):
    """Return (sand, shale, water) fractions; sand/shale are over rock voxels only."""
    lith = np.asarray(lith_patch, dtype=np.float32)
    rock = lith >= 0.0
    water_fraction = float(np.mean(~rock))
    n_rock = int(rock.sum())
    if n_rock == 0:
        return 0.0, 0.0, water_fraction
    sand_fraction = float(np.count_nonzero(lith[rock] >= sand_threshold) / n_rock)
    return sand_fraction, 1.0 - sand_fraction, water_fraction


def compute_patch_derived_metadata(
    zvol,
    origin,
    patch_size,
    geoscore_key,
    dip_source_key="geologic_age_faulted",
    sand_threshold=DEFAULT_SAND_THRESHOLD,
    label_z_offset=0,
):
    metadata = {k: 0.0 for k in DERIVED_METADATA_KEYS}
    # All metadata sources (including geologic_score) live in label depth space.
    origin = (origin[0], origin[1], origin[2] + int(label_z_offset))

    geoscore_patch = _safe_extract_patch_by_key(zvol, geoscore_key, origin, patch_size)
    if geoscore_patch is not None and geoscore_patch.size > 0:
        geoscore_patch = np.nan_to_num(geoscore_patch, nan=0.0, posinf=0.0, neginf=0.0)
        metadata["meta_geologic_score_mean"] = float(np.mean(geoscore_patch))

    structural_patch = _safe_extract_patch_by_key(zvol, dip_source_key, origin, patch_size)
    if structural_patch is not None and structural_patch.size > 0:
        structural_patch = np.nan_to_num(structural_patch, nan=0.0, posinf=0.0, neginf=0.0)
        dip_mean_deg, dip_std_deg, dip_range_deg, azimuth_mean_deg, azimuth_circular_variance = _compute_dip_azimuth_features(structural_patch)
        metadata["meta_dip_mean_deg"] = dip_mean_deg
        metadata["meta_dip_std_deg"] = dip_std_deg
        metadata["meta_dip_range_deg"] = dip_range_deg
        metadata["meta_dip_mean_class"] = quantize(dip_mean_deg, DIP_MEAN_CLASS_EDGES_DEG)
        metadata["meta_dip_range_class"] = quantize(dip_range_deg, DIP_RANGE_CLASS_EDGES_DEG)
        metadata["meta_azimuth_mean_deg"] = azimuth_mean_deg
        metadata["meta_azimuth_circular_variance"] = azimuth_circular_variance

    fault_segment_patch = _safe_extract_patch_by_key(zvol, "fault_segments_id", origin, patch_size)
    if fault_segment_patch is not None and fault_segment_patch.size > 0:
        fault_segment_patch = np.nan_to_num(fault_segment_patch, nan=0.0, posinf=0.0, neginf=0.0)
        metadata["meta_fault_fraction"] = float(np.mean(fault_segment_patch > 0.0))

    fault_intersection_patch = _safe_extract_patch_by_key(zvol, "fault_intersection_segments", origin, patch_size)
    if fault_intersection_patch is not None and fault_intersection_patch.size > 0:
        fault_intersection_patch = np.nan_to_num(fault_intersection_patch, nan=0.0, posinf=0.0, neginf=0.0)
        metadata["meta_fault_intersection_fraction"] = float(np.mean(fault_intersection_patch > 0.0))

    lith_patch = _safe_extract_patch_by_key(zvol, "faulted_lithology", origin, patch_size)
    if lith_patch is not None and lith_patch.size > 0:
        lith_patch = np.nan_to_num(lith_patch, nan=0.0, posinf=0.0, neginf=0.0)
        sand, shale, water = compute_lithology_fractions(lith_patch, sand_threshold)
        metadata["meta_sand_fraction"] = sand
        metadata["meta_shale_fraction"] = shale
        metadata["meta_water_fraction"] = water

    closure_patch = _safe_extract_patch_by_key(zvol, "closure_segments_id", origin, patch_size)
    if closure_patch is not None and closure_patch.size > 0:
        closure_patch = np.nan_to_num(closure_patch, nan=0.0, posinf=0.0, neginf=0.0)
        metadata["meta_closure_fraction"] = float(np.mean(closure_patch > 0.0))

    flat_spot_patch = _safe_extract_patch_by_key(zvol, "flat_spot", origin, patch_size)
    if flat_spot_patch is not None and flat_spot_patch.size > 0:
        flat_spot_patch = np.nan_to_num(flat_spot_patch, nan=0.0, posinf=0.0, neginf=0.0)
        metadata["meta_flat_spot_fraction"] = float(np.mean(flat_spot_patch > 0.0))

    onlap_patch = _safe_extract_patch_by_key(zvol, "onlap_segments", origin, patch_size)
    if onlap_patch is not None and onlap_patch.size > 0:
        onlap_patch = np.nan_to_num(onlap_patch, nan=0.0, posinf=0.0, neginf=0.0)
        metadata["meta_onlap_fraction"] = float(np.mean(onlap_patch > 0.0))
        metadata["meta_onlap_variability"] = float(np.std(onlap_patch))

    channel_patch = _safe_extract_patch_by_key(zvol, "faults/faulted_channel_labels", origin, patch_size)
    if channel_patch is not None and channel_patch.size > 0:
        channel_patch = np.nan_to_num(channel_patch, nan=0.0, posinf=0.0, neginf=0.0)
        metadata["meta_channel_fraction"] = float(np.mean(channel_patch > 0.0))
        metadata["meta_channel_core_fraction"] = float(np.mean(channel_patch >= 2.0))

    # Composite structural complexity: dip variability + azimuth dispersion + fault-intersection density.
    metadata["meta_structural_complexity"] = float(
        max(0.0,
            0.35 * (metadata["meta_dip_std_deg"] / 45.0)
            + 0.20 * metadata["meta_azimuth_circular_variance"]
            + 0.20 * metadata["meta_fault_intersection_fraction"]
            + 0.15 * metadata["meta_onlap_variability"]
            + 0.10 * metadata["meta_channel_fraction"]
        )
    )

    for key, val in metadata.items():
        if not np.isfinite(val):
            metadata[key] = 0.0
    return metadata


def sample_patches_from_model(
    zvol,
    seismic_key,
    geoscore_key,
    patch_size,
    n_patches_per_vol=100,
    allow_overlap=True,
    return_metadata=False,
    return_origin=False,
    label_z_offset=0,
    sand_threshold=DEFAULT_SAND_THRESHOLD,
):
    # zvol: root group for a model_data.zarr (zarr.core.Array or Group)
    # seismic_key: key in zvol pointing to seismic array
    # geoscore_key: key in zvol for geologic_score
    if seismic_key not in zvol or geoscore_key not in zvol:
        return []
    seismic = np.asarray(zvol[seismic_key])
    geoscore = np.asarray(zvol[geoscore_key])
    if seismic.ndim != 3:
        return []
    shape = seismic.shape

    if geoscore.ndim == 2:
        geoscore = geoscore[:, :, np.newaxis]
    if geoscore.ndim != 3:
        geoscore = np.zeros(shape, dtype='f4')
    # clip geoscore to non-negative
    geoscore = np.nan_to_num(geoscore, nan=0.0)
    sx, sy, sz = patch_size
    picks = pick_weighted_positions(
        geoscore,
        shape,
        patch_size,
        n_patches_per_vol,
        n_candidates=1000,
        allow_overlap=allow_overlap,
    )
    patches = []
    for (i,j,k) in picks:
        patch = seismic[i:i+sx, j:j+sy, k:k+sz]
        if patch.shape == (sx, sy, sz):
            if return_metadata:
                metadata = compute_patch_derived_metadata(
                    zvol,
                    (i, j, k),
                    patch_size,
                    geoscore_key=geoscore_key,
                    label_z_offset=label_z_offset,
                    sand_threshold=sand_threshold,
                )
                if return_origin:
                    patches.append((patch, metadata, (i, j, k)))
                else:
                    patches.append((patch, metadata))
            elif return_origin:
                patches.append((patch, (i, j, k)))
            else:
                patches.append(patch)
    return patches


def class_mask(name, values, onlap_threshold=DEFAULT_ONLAP_THRESHOLD, sand_threshold=DEFAULT_SAND_THRESHOLD):
    v = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if name == "onlap":
        return v >= onlap_threshold
    if name == "sand":
        return v >= sand_threshold
    return v > 0.0


def parse_class_quotas(items):
    quotas = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or name not in LABEL_CLASS_ORDER:
            raise ValueError(f"--class_quotas entries must be CLASS=FRACTION with CLASS in {LABEL_CLASS_ORDER}; got {item!r}")
        quotas[name] = float(value)
        if quotas[name] < 0:
            raise ValueError(f"Quota for {name} must be >= 0")
    return quotas


def normalize_shares(class_quotas, background_fraction):
    """Slot shares per class plus background, normalized to sum to 1."""
    shares = {c: float(class_quotas.get(c, 0.0)) for c in LABEL_CLASS_ORDER}
    shares[BACKGROUND] = float(background_fraction)
    total = sum(shares.values())
    if total <= 0:
        raise ValueError("class quotas plus background fraction must be > 0")
    return {k: v / total for k, v in shares.items()}


def allocate_slots(n, shares, rng):
    """Largest-remainder allocation of n slots to shares, returned in random order."""
    names = list(shares)
    exact = np.array([shares[k] * n for k in names])
    counts = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - counts), kind="stable")[: n - counts.sum()]:
        counts[i] += 1
    slots = [name for name, c in zip(names, counts) for _ in range(int(c))]
    return [slots[i] for i in rng.permutation(len(slots))]


def build_anchor_index(
    zvol,
    seismic_shape,
    label_z_offset=0,
    max_coords=200000,
    rng=None,
    onlap_threshold=DEFAULT_ONLAP_THRESHOLD,
    sand_threshold=DEFAULT_SAND_THRESHOLD,
    classes=LABEL_CLASS_ORDER,
):
    """Reservoir-sampled class voxel coordinates (in seismic index space), read chunk by chunk.

    Returns {class: {"coords": (n, 3) int32, "ids": (n,) int64, "count": total class voxels,
    "valid_voxels": voxels in the seismic-covered label region}}.
    """
    rng = rng or np.random.default_rng(0)
    sx, sy, sz = (int(v) for v in seismic_shape)
    off = int(label_z_offset)
    index = {}
    for name in classes:
        key = LABEL_CLASS_SOURCES[name]
        entry = {"coords": np.zeros((0, 3), np.int32), "ids": np.zeros(0, np.int64), "count": 0, "valid_voxels": 0}
        index[name] = entry
        if key not in zvol:
            continue
        arr = zvol[key]
        region = (min(sx, arr.shape[0]), min(sy, arr.shape[1]), (max(0, off), min(arr.shape[2], off + sz)))
        z_lo, z_hi = region[2]
        if z_hi <= z_lo:
            continue
        entry["valid_voxels"] = int(region[0] * region[1] * (z_hi - z_lo))
        chunks = getattr(arr, "chunks", None) or arr.shape
        keys = np.zeros(0)
        for slc in iter_chunk_slices(arr.shape, chunks):
            xs = slice(slc[0].start, min(slc[0].stop, region[0]))
            ys = slice(slc[1].start, min(slc[1].stop, region[1]))
            zs = slice(max(slc[2].start, z_lo), min(slc[2].stop, z_hi))
            if xs.stop <= xs.start or ys.stop <= ys.start or zs.stop <= zs.start:
                continue
            block = np.asarray(arr[xs, ys, zs])
            mask = class_mask(name, block, onlap_threshold, sand_threshold)
            nz = np.nonzero(mask)
            n_new = int(nz[0].size)
            if n_new == 0:
                continue
            entry["count"] += n_new
            coords = np.stack([nz[0] + xs.start, nz[1] + ys.start, nz[2] + zs.start - off], axis=1).astype(np.int32)
            ids = np.rint(np.nan_to_num(block[nz].astype(np.float64))).astype(np.int64)
            new_keys = rng.random(n_new)
            keys = np.concatenate([keys, new_keys])
            entry["coords"] = np.concatenate([entry["coords"], coords])
            entry["ids"] = np.concatenate([entry["ids"], ids])
            if keys.size > max_coords:
                keep = np.argpartition(keys, max_coords - 1)[:max_coords]
                keys, entry["coords"], entry["ids"] = keys[keep], entry["coords"][keep], entry["ids"][keep]
    return index


def _origin_from_anchor(anchor, seismic_shape, patch_size, anchor_jitter, rng):
    a = np.asarray(anchor, dtype=np.int64)
    p = np.asarray(patch_size, dtype=np.int64)
    if anchor_jitter == "center":
        o = a - p // 2
    else:
        o = a - rng.integers(0, p)
    return tuple(int(v) for v in np.clip(o, 0, np.asarray(seismic_shape) - p))


def _uniform_origin(seismic_shape, patch_size, rng):
    return tuple(int(rng.integers(0, s - p + 1)) for s, p in zip(seismic_shape, patch_size))


def sample_anchored_origins(
    index,
    seismic_shape,
    patch_size,
    n,
    shares,
    rng,
    anchor_jitter="uniform",
    max_patches_per_object=DEFAULT_MAX_PATCHES_PER_OBJECT,
    max_redraws=20,
):
    """Return (origins, anchor_class_indices, stats); class index -1 means background."""
    stats = {
        "anchored": {c: 0 for c in LABEL_CLASS_ORDER},
        "fallback": {c: 0 for c in LABEL_CLASS_ORDER},
        "object_cap_fallback": {c: 0 for c in LABEL_CLASS_ORDER},
    }
    object_counts = {}
    origins, anchor_classes = [], []
    for slot in allocate_slots(n, shares, rng):
        origin = None
        if slot != BACKGROUND:
            coords = index.get(slot, {}).get("coords", np.zeros((0, 3)))
            if len(coords) == 0:
                stats["fallback"][slot] += 1
            else:
                ids = index[slot]["ids"]
                for _ in range(max_redraws):
                    i = int(rng.integers(len(coords)))
                    if slot in OBJECT_ID_CLASSES:
                        obj = (slot, int(ids[i]))
                    else:
                        obj = (slot,) + tuple(int(c) // int(p) for c, p in zip(coords[i], patch_size))
                    if max_patches_per_object and object_counts.get(obj, 0) >= max_patches_per_object:
                        continue
                    object_counts[obj] = object_counts.get(obj, 0) + 1
                    origin = _origin_from_anchor(coords[i], seismic_shape, patch_size, anchor_jitter, rng)
                    break
                if origin is None:
                    stats["object_cap_fallback"][slot] += 1
        if origin is None:
            origins.append(_uniform_origin(seismic_shape, patch_size, rng))
            anchor_classes.append(-1)
        else:
            origins.append(origin)
            anchor_classes.append(LABEL_CLASS_ORDER.index(slot))
            stats["anchored"][slot] += 1
    return origins, anchor_classes, stats


def compute_label_presence(
    zvol,
    origin,
    patch_size,
    label_z_offset=0,
    presence_min_voxels=32,
    onlap_threshold=DEFAULT_ONLAP_THRESHOLD,
    sand_threshold=DEFAULT_SAND_THRESHOLD,
    return_masks=False,
):
    """Per-class voxel counts and presence flags for the label window matching a seismic patch."""
    i, j, k = origin
    px, py, pz = patch_size
    k += int(label_z_offset)
    counts = {}
    masks = np.zeros((len(LABEL_CLASS_ORDER), px, py, pz), dtype=np.uint8) if return_masks else None
    for ci, name in enumerate(LABEL_CLASS_ORDER):
        key = LABEL_CLASS_SOURCES[name]
        counts[name] = 0
        if key not in zvol:
            continue
        block = np.asarray(zvol[key][i:i + px, j:j + py, k:k + pz])
        if block.shape != (px, py, pz):
            continue
        mask = class_mask(name, block, onlap_threshold, sand_threshold)
        counts[name] = int(mask.sum())
        if masks is not None:
            masks[ci] = mask
    presence = {name: int(counts[name] >= presence_min_voxels) for name in LABEL_CLASS_ORDER}
    return counts, presence, masks


def compute_inclusion_weight(counts, realized_shares, densities, patch_voxels):
    """Importance weight p_uniform(origin) / q_mixture(origin) for a class-anchored patch.

    With uniform jitter, a class-c anchored draw lands on an origin with probability roughly
    proportional to the class-c voxel count in that patch, normalized by the class density.
    """
    denom = realized_shares.get(BACKGROUND, 0.0)
    for name in LABEL_CLASS_ORDER:
        s = realized_shares.get(name, 0.0)
        rho = densities.get(name, 0.0)
        if s > 0 and rho > 0:
            denom += s * counts.get(name, 0) / (rho * patch_voxels)
    return float(1.0 / denom) if denom > 0 else 1.0


def sample_labeled_patches(
    zvol,
    seismic_key,
    patch_size,
    n,
    rng,
    sampling_mode="class_anchored",
    shares=None,
    anchor_jitter="uniform",
    max_patches_per_object=DEFAULT_MAX_PATCHES_PER_OBJECT,
    anchor_index_max_coords=200000,
    presence_min_voxels=32,
    onlap_threshold=DEFAULT_ONLAP_THRESHOLD,
    sand_threshold=DEFAULT_SAND_THRESHOLD,
    label_z_offset=0,
    store_label_patches=False,
    geoscore_key="geologic_score",
):
    """Sample patches with 'class_anchored' or 'uniform' origins; returns (items, stats)."""
    if seismic_key not in zvol:
        return [], None
    seismic = np.asarray(zvol[seismic_key])
    if seismic.ndim != 3 or any(s < p for s, p in zip(seismic.shape, patch_size)):
        return [], None
    shape = seismic.shape
    stats = None
    densities = {}
    if sampling_mode == "uniform":
        origins = [_uniform_origin(shape, patch_size, rng) for _ in range(n)]
        anchors = [-1] * n
    elif sampling_mode == "class_anchored":
        shares = shares or normalize_shares(DEFAULT_CLASS_QUOTAS, DEFAULT_BACKGROUND_FRACTION)
        index = build_anchor_index(
            zvol, shape, label_z_offset, anchor_index_max_coords, rng, onlap_threshold, sand_threshold
        )
        densities = {c: (e["count"] / e["valid_voxels"] if e["valid_voxels"] else 0.0) for c, e in index.items()}
        origins, anchors, stats = sample_anchored_origins(
            index, shape, patch_size, n, shares, rng, anchor_jitter, max_patches_per_object
        )
    else:
        raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")

    realized = {BACKGROUND: anchors.count(-1) / max(n, 1)}
    for ci, name in enumerate(LABEL_CLASS_ORDER):
        realized[name] = anchors.count(ci) / max(n, 1)
    patch_voxels = int(np.prod(patch_size))
    sx, sy, sz = patch_size
    items = []
    for (i, j, k), anchor in zip(origins, anchors):
        counts, presence, masks = compute_label_presence(
            zvol, (i, j, k), patch_size, label_z_offset, presence_min_voxels,
            onlap_threshold, sand_threshold, return_masks=store_label_patches,
        )
        weight = (
            compute_inclusion_weight(counts, realized, densities, patch_voxels)
            if sampling_mode == "class_anchored" else 1.0
        )
        items.append({
            "patch": seismic[i:i + sx, j:j + sy, k:k + sz],
            "metadata": compute_patch_derived_metadata(
                zvol, (i, j, k), patch_size, geoscore_key,
                sand_threshold=sand_threshold, label_z_offset=label_z_offset,
            ),
            "origin": (i, j, k),
            "anchor_class": anchor,
            "inclusion_weight": weight,
            "counts": counts,
            "presence": presence,
            "label_patch": masks,
        })
    return items, stats


def iter_chunk_slices(shape, chunks):
    for i in range(0, shape[0], chunks[0]):
        i1 = min(i + chunks[0], shape[0])
        for j in range(0, shape[1], chunks[1]):
            j1 = min(j + chunks[1], shape[1])
            for k in range(0, shape[2], chunks[2]):
                k1 = min(k + chunks[2], shape[2])
                yield (slice(i, i1), slice(j, j1), slice(k, k1))


def compute_array_stats(seismic):
    shape = seismic.shape
    chunks = getattr(seismic, "chunks", None)
    if chunks is None:
        chunks = shape

    count = 0
    mean = 0.0
    m2 = 0.0
    vmin = float("inf")
    vmax = float("-inf")

    for slc in iter_chunk_slices(shape, chunks):
        block = np.asarray(seismic[slc], dtype=np.float64)
        if block.size == 0:
            continue

        n = int(block.size)
        bmean = float(block.mean())
        bm2 = float(np.square(block - bmean).sum())
        bmin = float(block.min())
        bmax = float(block.max())

        vmin = min(vmin, bmin)
        vmax = max(vmax, bmax)

        if count == 0:
            count = n
            mean = bmean
            m2 = bm2
            continue

        delta = bmean - mean
        new_count = count + n
        mean = mean + delta * (n / new_count)
        m2 = m2 + bm2 + (delta * delta) * (count * n / new_count)
        count = new_count

    if count == 0:
        raise RuntimeError("No seismic samples available to compute stats.")

    variance = m2 / count
    std = math.sqrt(max(variance, 0.0))
    return {
        "shape": shape,
        "count": count,
        "mean": mean,
        "std": std,
        "min": vmin,
        "max": vmax,
    }


def compute_dataset_stats(volumes, seismic_key):
    # Numerically stable aggregation of mean/std across all 3D volumes.
    count = 0
    mean = 0.0
    m2 = 0.0

    for vol in volumes:
        try:
            z = cast(Any, zarr.open(str(vol), mode="r"))
            if seismic_key not in z:
                continue
            seismic = cast(Any, z[seismic_key])
            if getattr(seismic, "ndim", None) != 3:
                continue
            stats = compute_array_stats(seismic)
            n = int(stats["count"])
            bmean = float(stats["mean"])
            bstd = float(stats["std"])
            bm2 = (bstd * bstd) * n

            if count == 0:
                count = n
                mean = bmean
                m2 = bm2
                continue

            delta = bmean - mean
            new_count = count + n
            mean = mean + delta * (n / new_count)
            m2 = m2 + bm2 + (delta * delta) * (count * n / new_count)
            count = new_count
        except Exception as e:
            print("Failed while computing stats for", vol, e)

    if count == 0:
        raise RuntimeError("Unable to compute dataset stats: no readable seismic data found.")
    variance = m2 / count
    std = math.sqrt(max(variance, 0.0))
    return mean, std


def apply_scaling(patch, scaling_mode, scaling_mean, scaling_std):
    if scaling_mode == "none":
        return patch
    eps = 1e-8
    std = float(max(abs(scaling_std), eps))
    if scaling_mode == "divide_by_std":
        return patch / std
    if scaling_mode == "zscore":
        return (patch - float(scaling_mean)) / std
    raise ValueError(f"Unsupported scaling mode: {scaling_mode}")


def has_temp_folder_sibling(volume_zarr_path):
    # Exclude seismic folders when a temp_folder variant exists next to them.
    volume_dir = volume_zarr_path.parent
    volume_name = volume_dir.name
    if not volume_name.startswith("seismic__"):
        return False
    temp_name = volume_name.replace("seismic__", "temp_folder__", 1)
    temp_dir = volume_dir.with_name(temp_name)
    return temp_dir.exists() and temp_dir.is_dir()


def list_source_volumes(source, exclude_dirs=()):
    """Sorted model_data.zarr stores under source, skipping any under a folder named in exclude_dirs."""
    src = Path(source)
    excluded = set(exclude_dirs or ())
    vols = []
    for vol in sorted(src.rglob("model_data.zarr")):
        if has_temp_folder_sibling(vol):
            continue
        if excluded and excluded.intersection(vol.relative_to(src).parts[:-1]):
            continue
        vols.append(vol)
    return vols


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="directory that contains model_data.zarr folders")
    p.add_argument("--out", required=True)
    p.add_argument("--patch_size", type=int, nargs='+', default=[32], help="Patch size: one value for cubic or three values X Y Z")
    p.add_argument("--n_patches", type=int, default=5000)
    p.add_argument("--n_per_volume", type=int, default=100)
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sampling seed. If omitted, generate a new seed from system entropy and record it in the output Zarr attrs.",
    )
    p.add_argument("--seismic_key", type=str, default=DEFAULT_SEISMIC_KEY)
    p.add_argument(
        "--exclude_dir",
        action="append",
        default=[],
        metavar="NAME",
        help="Skip source volumes under a folder with this name (repeatable), e.g. --exclude_dir validation.",
    )
    p.add_argument("--geoscore_key", type=str, default="geologic_score")
    p.add_argument(
        "--scaling",
        choices=["none", "divide_by_std", "zscore"],
        default="divide_by_std",
        help="Amplitude scaling applied to each sampled patch (default: divide_by_std).",
    )
    p.add_argument(
        "--derive_dataset_stats",
        action="store_true",
        help="Derive global dataset mean/std from all source seismic volumes before sampling (default: enabled).",
    )
    p.add_argument(
        "--no_derive_dataset_stats",
        dest="derive_dataset_stats",
        action="store_false",
        help="Disable dataset-wide stats derivation and use provided --dataset_mean/--dataset_std values.",
    )
    p.add_argument("--dataset_mean", type=float, default=None, help="Global mean used for z-score when --derive_dataset_stats is not set.")
    p.add_argument("--dataset_std", type=float, default=None, help="Global std used for divide-by-std or z-score when --derive_dataset_stats is not set.")
    p.set_defaults(allow_overlap=True)
    p.set_defaults(derive_dataset_stats=True)
    p.add_argument("--allow_overlap", dest="allow_overlap", action="store_true", help="Allow overlapping/duplicate patch centers (default).")
    p.add_argument("--no_overlap", dest="allow_overlap", action="store_false", help="Disallow overlapping by sampling unique candidate centers.")
    p.add_argument(
        "--sampling_mode",
        choices=["geoscore", "class_anchored", "uniform"],
        default="geoscore",
        help="geoscore: legacy geologic_score-weighted origins; class_anchored: origins anchored on label classes; uniform: natural prevalence.",
    )
    p.add_argument(
        "--class_quotas",
        nargs="+",
        default=None,
        metavar="CLASS=FRACTION",
        help=f"Share of anchors per class (classes: {', '.join(LABEL_CLASS_ORDER)}). Default: 0.125 for each class except sand.",
    )
    p.add_argument("--background_fraction", type=float, default=DEFAULT_BACKGROUND_FRACTION, help="Share of uniform (non-anchored) patches.")
    p.add_argument("--anchor_jitter", choices=["uniform", "center"], default="uniform", help="Where the anchor voxel lands in the patch.")
    p.add_argument("--max_patches_per_object", type=int, default=DEFAULT_MAX_PATCHES_PER_OBJECT, help="Cap per object (segment id, or coarse cell for classes without ids); 0 disables.")
    p.add_argument("--anchor_index_max_coords", type=int, default=200000, help="Reservoir cap on stored anchor coordinates per class per volume.")
    p.add_argument("--presence_min_voxels", type=int, default=32, help="Minimum class voxels for label_presence_<class> = 1.")
    p.add_argument("--onlap_threshold", type=float, default=DEFAULT_ONLAP_THRESHOLD)
    p.add_argument("--sand_threshold", type=float, default=DEFAULT_SAND_THRESHOLD)
    p.add_argument("--store_label_patches", action="store_true", help="Also store (N, 7, X, Y, Z) uint8 label patches (voxel mode).")
    p.add_argument(
        "--label_z_offset",
        type=int,
        default=0,
        help="Label depth offset: seismic[z] matches label[z + offset]. Synthoseis data: 1 (verified in WP0).",
    )
    p.add_argument(
        "--disjoint_from",
        action="append",
        default=[],
        metavar="ZARR",
        help="Fail if any source volume is listed in this existing output store's source_volumes (repeatable).",
    )
    p.add_argument(
        "--dip_sample_count",
        type=int,
        default=DEFAULT_DIP_SAMPLE_COUNT,
        help="Voxel dip samples stored per patch (dip_samples_deg, dip_samples_z) so training can re-derive dip "
             "labels after depth-warp augmentation; 0 disables.",
    )
    args = p.parse_args()
    patch_size = normalize_patch_size(args.patch_size)
    sampling_seed = int(args.seed) if args.seed is not None else secrets.randbits(32)
    random.seed(sampling_seed)
    np.random.seed(sampling_seed)
    label_rng = np.random.default_rng(sampling_seed)
    # Separate stream so dip sampling never changes patch origins for a given seed.
    dip_rng = np.random.default_rng([sampling_seed, 1])
    print(f"Sampling seed: {sampling_seed}")

    class_quotas = parse_class_quotas(args.class_quotas) if args.class_quotas else dict(DEFAULT_CLASS_QUOTAS)
    shares = normalize_shares(class_quotas, args.background_fraction)

    src = Path(args.source)
    out = Path(args.out)

    vols = list_source_volumes(src, args.exclude_dir)
    if not vols:
        print("No model_data.zarr volumes found under", src)
        return
    for other in args.disjoint_from:
        other_vols = set(cast(list, zarr.open_group(str(other), mode="r").attrs.get("source_volumes", [])))
        overlap = sorted(other_vols.intersection(str(v) for v in vols))
        if overlap:
            raise SystemExit(f"{len(overlap)} source volumes overlap with {other}, e.g. {overlap[0]}")

    out.parent.mkdir(parents=True, exist_ok=True)

    # create destination zarr
    dst = cast(Any, zarr.open(str(out), mode="w"))

    def create(name, shape, dtype, chunks):
        # zarr 2.x uses create_dataset on groups, zarr 3.x uses create_array
        if hasattr(dst, 'create_dataset'):
            return dst.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks)
        return dst.create_array(name, shape=shape, dtype=dtype, chunks=chunks)

    create("patches", (args.n_patches, patch_size[0], patch_size[1], patch_size[2]), "f4", (1, patch_size[0], patch_size[1], patch_size[2]))

    written = 0
    metadata_arrays = {}

    scaling_mean = 0.0 if args.dataset_mean is None else float(args.dataset_mean)
    scaling_std = 1.0 if args.dataset_std is None else float(args.dataset_std)
    if args.scaling != "none":
        if args.derive_dataset_stats:
            scaling_mean, scaling_std = compute_dataset_stats(vols, args.seismic_key)
            print(f"Derived dataset stats: mean={scaling_mean:.6f}, std={scaling_std:.6f}")
        elif args.dataset_std is None:
            raise ValueError("--dataset_std is required when --scaling is enabled and --derive_dataset_stats is not set.")

    dst.attrs["scaling_mode"] = args.scaling
    dst.attrs["scaling_mean"] = float(scaling_mean)
    dst.attrs["scaling_std"] = float(scaling_std)
    dst.attrs["sampling_seed"] = sampling_seed
    dst.attrs["source_volumes"] = [str(vol) for vol in vols]
    dst.attrs["exclude_dirs"] = list(args.exclude_dir)
    dst.attrs["sand_threshold"] = float(args.sand_threshold)
    dst.attrs["dip_mean_class_edges_deg"] = list(DIP_MEAN_CLASS_EDGES_DEG)
    dst.attrs["dip_range_class_edges_deg"] = list(DIP_RANGE_CLASS_EDGES_DEG)
    dst.attrs["sampling_mode"] = args.sampling_mode
    dst.attrs["label_z_offset"] = int(args.label_z_offset)
    dst.attrs["label_class_order"] = list(LABEL_CLASS_ORDER)
    dst.attrs["label_class_sources"] = dict(LABEL_CLASS_SOURCES)
    dst.attrs["presence_min_voxels"] = int(args.presence_min_voxels)
    dst.attrs["onlap_threshold"] = float(args.onlap_threshold)
    if args.sampling_mode == "class_anchored":
        dst.attrs["class_quotas"] = {c: shares[c] for c in LABEL_CLASS_ORDER}
        dst.attrs["background_fraction"] = shares[BACKGROUND]
        dst.attrs["anchor_jitter"] = args.anchor_jitter
        dst.attrs["max_patches_per_object"] = int(args.max_patches_per_object)
        dst.attrs["anchor_index_max_coords"] = int(args.anchor_index_max_coords)
    patches_dst = cast(Any, dst["patches"])
    vec_chunks = (min(args.n_patches, 2048),)
    provenance_arrays = {}
    for key in ("source_volume_index", "origin_x", "origin_y", "origin_z"):
        provenance_arrays[key] = create(key, (args.n_patches,), "i4", vec_chunks)
    presence_arrays = {c: create(f"label_presence_{c}", (args.n_patches,), "u1", vec_chunks) for c in LABEL_CLASS_ORDER}
    anchor_class_dst = create("anchor_class", (args.n_patches,), "i1", vec_chunks)
    inclusion_weight_dst = create("inclusion_weight", (args.n_patches,), "f4", vec_chunks)
    label_patches_dst = None
    if args.store_label_patches:
        lp_shape = (len(LABEL_CLASS_ORDER),) + tuple(patch_size)
        label_patches_dst = create("label_patches", (args.n_patches,) + lp_shape, "u1", (1,) + lp_shape)
    dip_samples_buf = dip_z_buf = None
    if args.dip_sample_count > 0:
        k = int(args.dip_sample_count)
        # Buffered in memory (N x K x 3 bytes) and written once; per-row writes would rewrite whole chunks.
        dip_samples_buf = np.zeros((args.n_patches, k), np.float16)
        dip_z_buf = np.zeros((args.n_patches, k), np.uint8 if patch_size[2] <= 256 else np.uint16)
    totals = {k: {c: 0 for c in LABEL_CLASS_ORDER} for k in ("anchored", "fallback", "object_cap_fallback")}
    skipped_missing_labels = []

    for volume_index, vol in enumerate(vols):
        print("Scanning", vol)
        try:
            z = cast(Any, zarr.open(str(vol), mode="r"))
            if args.seismic_key in z:
                seismic = cast(Any, z[args.seismic_key])
                if getattr(seismic, "ndim", None) == 3:
                    vol_stats = compute_array_stats(seismic)
                    print(
                        "Volume stats:",
                        f"shape={vol_stats['shape']}",
                        f"mean={vol_stats['mean']:.6f}",
                        f"std={vol_stats['std']:.6f}",
                        f"min={vol_stats['min']:.6f}",
                        f"max={vol_stats['max']:.6f}",
                    )
            if args.sampling_mode == "geoscore":
                patch_items = [
                    {"patch": pch, "metadata": metadata, "origin": origin, "anchor_class": -1,
                     "inclusion_weight": 1.0, "presence": None, "label_patch": None}
                    for pch, metadata, origin in sample_patches_from_model(
                        z,
                        args.seismic_key,
                        args.geoscore_key,
                        patch_size,
                        n_patches_per_vol=args.n_per_volume,
                        allow_overlap=args.allow_overlap,
                        return_metadata=True,
                        return_origin=True,
                        label_z_offset=args.label_z_offset,
                        sand_threshold=args.sand_threshold,
                    )
                ]
            else:
                missing_labels = [key for key in LABEL_CLASS_SOURCES.values() if key not in z]
                if missing_labels:
                    print(f"Skipping {vol}: missing label arrays {missing_labels}")
                    skipped_missing_labels.append(str(vol))
                    continue
                n_vol = min(args.n_per_volume, args.n_patches - written)
                patch_items, vol_counts = sample_labeled_patches(
                    z,
                    args.seismic_key,
                    patch_size,
                    n_vol,
                    label_rng,
                    sampling_mode=args.sampling_mode,
                    shares=shares,
                    anchor_jitter=args.anchor_jitter,
                    max_patches_per_object=args.max_patches_per_object,
                    anchor_index_max_coords=args.anchor_index_max_coords,
                    presence_min_voxels=args.presence_min_voxels,
                    onlap_threshold=args.onlap_threshold,
                    sand_threshold=args.sand_threshold,
                    label_z_offset=args.label_z_offset,
                    store_label_patches=args.store_label_patches,
                    geoscore_key=args.geoscore_key,
                )
                if vol_counts:
                    for kind, per_class in vol_counts.items():
                        for c, v in per_class.items():
                            totals[kind][c] += int(v)
            for item in patch_items:
                if written >= args.n_patches:
                    break
                pch, metadata, origin = item["patch"], item["metadata"], item["origin"]
                pch = apply_scaling(pch.astype("f4"), args.scaling, scaling_mean, scaling_std)
                patches_dst[written] = pch.astype("f4")
                if not metadata_arrays:
                    for key in DERIVED_METADATA_KEYS:
                        metadata_arrays[key] = create(key, (args.n_patches,), "f4", vec_chunks)
                    dst.attrs["derived_metadata_keys"] = list(DERIVED_METADATA_KEYS)
                for key in DERIVED_METADATA_KEYS:
                    metadata_arrays[key][written] = np.float32(metadata.get(key, 0.0))
                provenance_arrays["source_volume_index"][written] = np.int32(volume_index)
                provenance_arrays["origin_x"][written] = np.int32(origin[0])
                provenance_arrays["origin_y"][written] = np.int32(origin[1])
                provenance_arrays["origin_z"][written] = np.int32(origin[2])
                presence = item["presence"]
                label_patch = item["label_patch"]
                if presence is None:
                    _, presence, label_patch = compute_label_presence(
                        z, origin, patch_size, args.label_z_offset, args.presence_min_voxels,
                        args.onlap_threshold, args.sand_threshold, return_masks=args.store_label_patches,
                    )
                for c in LABEL_CLASS_ORDER:
                    presence_arrays[c][written] = np.uint8(presence[c])
                anchor_class_dst[written] = np.int8(item["anchor_class"])
                inclusion_weight_dst[written] = np.float32(item["inclusion_weight"])
                if label_patches_dst is not None:
                    label_patches_dst[written] = label_patch
                if dip_samples_buf is not None and dip_z_buf is not None:
                    dip_samples_buf[written], dip_z_buf[written] = compute_patch_dip_samples(
                        z, origin, patch_size, args.dip_sample_count, dip_rng, label_z_offset=args.label_z_offset
                    )
                written += 1
            if written >= args.n_patches:
                break
        except Exception as e:
            print("Failed to read", vol, e)
    if args.sampling_mode == "class_anchored":
        dst.attrs["anchored_counts"] = totals["anchored"]
        dst.attrs["fallback_counts"] = totals["fallback"]
        dst.attrs["object_cap_fallback_counts"] = totals["object_cap_fallback"]
        print("Anchored counts:", totals["anchored"])
        print("Fallback counts:", totals["fallback"], "object cap fallbacks:", totals["object_cap_fallback"])
    dst.attrs["n_written"] = int(written)
    if dip_samples_buf is not None and dip_z_buf is not None:
        write_dip_samples(dst, dip_samples_buf, dip_z_buf)
    if args.sampling_mode != "geoscore":
        dst.attrs["skipped_volumes_missing_labels"] = skipped_missing_labels
    print(f"Wrote {written} patches to {out}")


if __name__ == "__main__":
    main()
