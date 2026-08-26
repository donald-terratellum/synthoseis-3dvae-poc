#!/usr/bin/env python3

"""Sample 3D patches from existing model_data.zarr stores into a destination zarr store.

Usage:
    python scripts/sample_patches.py --source /path/to/fake_data --out data/train.zarr --patch_size 32 32 32 --n_patches 5000 \
        --seismic_key seismicCubes_cumsum__fullstack --geoscore_key geologic_score --n_per_volume 100
"""

from pathlib import Path
import sys

SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)

import argparse
import random
import secrets
import numpy as np
import zarr
import math
from typing import Any, cast


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
)


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
    eps = 1e-8
    gx, gy, gz = np.gradient(structural_patch.astype(np.float32), edge_order=1)
    horizontal_mag = np.sqrt(gx * gx + gy * gy)
    dip_rad = np.arctan2(horizontal_mag, np.abs(gz) + eps)
    dip_deg = np.degrees(dip_rad)

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
    return dip_mean_deg, dip_std_deg, azimuth_mean_deg, azimuth_circular_variance


def compute_patch_derived_metadata(zvol, origin, patch_size, geoscore_key, dip_source_key="geologic_age_faulted"):
    metadata = {k: 0.0 for k in DERIVED_METADATA_KEYS}

    geoscore_patch = _safe_extract_patch_by_key(zvol, geoscore_key, origin, patch_size)
    if geoscore_patch is not None and geoscore_patch.size > 0:
        geoscore_patch = np.nan_to_num(geoscore_patch, nan=0.0, posinf=0.0, neginf=0.0)
        metadata["meta_geologic_score_mean"] = float(np.mean(geoscore_patch))

    structural_patch = _safe_extract_patch_by_key(zvol, dip_source_key, origin, patch_size)
    if structural_patch is not None and structural_patch.size > 0:
        structural_patch = np.nan_to_num(structural_patch, nan=0.0, posinf=0.0, neginf=0.0)
        dip_mean_deg, dip_std_deg, azimuth_mean_deg, azimuth_circular_variance = _compute_dip_azimuth_features(structural_patch)
        metadata["meta_dip_mean_deg"] = dip_mean_deg
        metadata["meta_dip_std_deg"] = dip_std_deg
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

    # faulted_lithology ranges [-1, 1] in synthetic data: map to [0, 1] for sandness.
    lith_patch = _safe_extract_patch_by_key(zvol, "faulted_lithology", origin, patch_size)
    if lith_patch is not None and lith_patch.size > 0:
        lith_patch = np.nan_to_num(lith_patch, nan=0.0, posinf=0.0, neginf=0.0)
        sandness = np.clip((lith_patch + 1.0) * 0.5, 0.0, 1.0)
        metadata["meta_sand_fraction"] = float(np.mean(sandness))
        metadata["meta_shale_fraction"] = float(np.mean(1.0 - sandness))

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
    p.add_argument("--seismic_key", type=str, default="seismicCubes_cumsum__fullstack")
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
    args = p.parse_args()
    patch_size = normalize_patch_size(args.patch_size)
    sampling_seed = int(args.seed) if args.seed is not None else secrets.randbits(32)
    random.seed(sampling_seed)
    np.random.seed(sampling_seed)
    print(f"Sampling seed: {sampling_seed}")

    src = Path(args.source)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # create destination zarr
    dst = cast(Any, zarr.open(str(out), mode="w"))
    # zarr 2.x uses create_dataset on groups, zarr 3.x uses create_array
    if hasattr(dst, 'create_dataset'):
        dst.create_dataset("patches", shape=(args.n_patches, patch_size[0], patch_size[1], patch_size[2]), dtype="f4", chunks=(1, patch_size[0], patch_size[1], patch_size[2]))
    else:
        dst.create_array("patches", shape=(args.n_patches, patch_size[0], patch_size[1], patch_size[2]), dtype="f4", chunks=(1, patch_size[0], patch_size[1], patch_size[2]))

    written = 0
    metadata_arrays = {}
    vols = sorted(vol for vol in src.rglob("model_data.zarr") if not has_temp_folder_sibling(vol))
    if not vols:
        print("No model_data.zarr volumes found under", src)
        return

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
    patches_dst = cast(Any, dst["patches"])
    provenance_arrays = {}
    for key in ("source_volume_index", "origin_x", "origin_y", "origin_z"):
        if hasattr(dst, 'create_dataset'):
            provenance_arrays[key] = dst.create_dataset(key, shape=(args.n_patches,), dtype="i4", chunks=(min(args.n_patches, 2048),))
        else:
            provenance_arrays[key] = dst.create_array(key, shape=(args.n_patches,), dtype="i4", chunks=(min(args.n_patches, 2048),))

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
            patch_items = sample_patches_from_model(
                z,
                args.seismic_key,
                args.geoscore_key,
                patch_size,
                n_patches_per_vol=args.n_per_volume,
                allow_overlap=args.allow_overlap,
                return_metadata=True,
                return_origin=True,
            )
            for pch, metadata, origin in patch_items:
                if written >= args.n_patches:
                    break
                pch = apply_scaling(pch.astype("f4"), args.scaling, scaling_mean, scaling_std)
                patches_dst[written] = pch.astype("f4")
                if not metadata_arrays:
                    for key in DERIVED_METADATA_KEYS:
                        if hasattr(dst, 'create_dataset'):
                            metadata_arrays[key] = dst.create_dataset(key, shape=(args.n_patches,), dtype="f4", chunks=(min(args.n_patches, 2048),))
                        else:
                            metadata_arrays[key] = dst.create_array(key, shape=(args.n_patches,), dtype="f4", chunks=(min(args.n_patches, 2048),))
                    dst.attrs["derived_metadata_keys"] = list(DERIVED_METADATA_KEYS)
                for key in DERIVED_METADATA_KEYS:
                    metadata_arrays[key][written] = np.float32(metadata.get(key, 0.0))
                provenance_arrays["source_volume_index"][written] = np.int32(volume_index)
                provenance_arrays["origin_x"][written] = np.int32(origin[0])
                provenance_arrays["origin_y"][written] = np.int32(origin[1])
                provenance_arrays["origin_z"][written] = np.int32(origin[2])
                written += 1
            if written >= args.n_patches:
                break
        except Exception as e:
            print("Failed to read", vol, e)
    print(f"Wrote {written} patches to {out}")


if __name__ == "__main__":
    main()
