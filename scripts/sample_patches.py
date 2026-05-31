#!/usr/bin/env python3

"""Sample 3D patches from existing model_data.zarr stores into a destination zarr store.

Usage:
  python scripts/sample_patches.py --source /path/to/fake_data --out data/train.zarr --patch_size 32 --n_patches 5000 \
    --seismic_key seismicCubes_cumsum__fullstack --geoscore_key geologic_score --n_per_volume 100
"""

from pathlib import Path
import argparse
import random
import numpy as np
import zarr
import math
from typing import Any, cast


def candidate_positions(shape, patch_size, n_candidates=500):
    max_x = shape[0] - patch_size
    max_y = shape[1] - patch_size
    max_z = shape[2] - patch_size
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
    scores = []
    for (i,j,k) in candidates:
        # geoscore can have different extents than seismic; out-of-range slices get zero weight.
        patch_score = geoscore[i:i+patch_size, j:j+patch_size, k:k+patch_size]
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


def sample_patches_from_model(zvol, seismic_key, geoscore_key, patch_size, n_patches_per_vol=100, allow_overlap=True):
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
        patch = seismic[i:i+patch_size, j:j+patch_size, k:k+patch_size]
        if patch.shape == (patch_size, patch_size, patch_size):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="directory that contains model_data.zarr folders")
    p.add_argument("--out", required=True)
    p.add_argument("--patch_size", type=int, default=32)
    p.add_argument("--n_patches", type=int, default=5000)
    p.add_argument("--n_per_volume", type=int, default=100)
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

    src = Path(args.source)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # create destination zarr
    dst = cast(Any, zarr.open(str(out), mode="w"))
    # zarr 2.x uses create_dataset on groups, zarr 3.x uses create_array
    if hasattr(dst, 'create_dataset'):
        dst.create_dataset("patches", shape=(args.n_patches, args.patch_size, args.patch_size, args.patch_size), dtype="f4", chunks=(1, args.patch_size, args.patch_size, args.patch_size))
    else:
        dst.create_array("patches", shape=(args.n_patches, args.patch_size, args.patch_size, args.patch_size), dtype="f4", chunks=(1, args.patch_size, args.patch_size, args.patch_size))

    written = 0
    vols = list(src.rglob("model_data.zarr"))
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
    patches_dst = cast(Any, dst["patches"])

    for vol in vols:
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
            patches = sample_patches_from_model(
                z,
                args.seismic_key,
                args.geoscore_key,
                args.patch_size,
                n_patches_per_vol=args.n_per_volume,
                allow_overlap=args.allow_overlap,
            )
            for pch in patches:
                if written >= args.n_patches:
                    break
                pch = apply_scaling(pch.astype("f4"), args.scaling, scaling_mean, scaling_std)
                patches_dst[written] = pch.astype("f4")
                written += 1
            if written >= args.n_patches:
                break
        except Exception as e:
            print("Failed to read", vol, e)
    print(f"Wrote {written} patches to {out}")


if __name__ == "__main__":
    main()
