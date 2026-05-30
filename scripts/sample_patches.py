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


def pick_weighted_positions(geoscore, patch_size, n_picks, n_candidates=1000):
    # geoscore: numpy array shaped (X,Y,Z)
    shape = geoscore.shape
    candidates = candidate_positions(shape, patch_size, n_candidates=n_candidates)
    if not candidates:
        return []
    scores = []
    for (i,j,k) in candidates:
        score = float(geoscore[i:i+patch_size, j:j+patch_size, k:k+patch_size].sum())
        scores.append(score)
    scores = np.array(scores)
    if scores.sum() <= 0:
        # fallback: uniform random unique picks
        chosen = random.sample(candidates, min(n_picks, len(candidates)))
        return chosen
    probs = scores / scores.sum()
    idx = np.random.choice(len(candidates), size=min(n_picks, len(candidates)), replace=False, p=probs)
    return [candidates[i] for i in idx]


def sample_patches_from_model(zvol, seismic_key, geoscore_key, patch_size, n_patches_per_vol=100):
    # zvol: root group for a model_data.zarr (zarr.core.Array or Group)
    # seismic_key: key in zvol pointing to seismic array
    # geoscore_key: key in zvol for geologic_score
    if seismic_key not in zvol or geoscore_key not in zvol:
        return []
    seismic = np.asarray(zvol[seismic_key])
    geoscore = np.asarray(zvol[geoscore_key])
    shape = seismic.shape
    # clip geoscore to non-negative
    geoscore = np.nan_to_num(geoscore, nan=0.0)
    picks = pick_weighted_positions(geoscore, patch_size, n_patches_per_vol, n_candidates=1000)
    patches = []
    for (i,j,k) in picks:
        patch = seismic[i:i+patch_size, j:j+patch_size, k:k+patch_size]
        patches.append(patch)
    return patches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="directory that contains model_data.zarr folders")
    p.add_argument("--out", required=True)
    p.add_argument("--patch_size", type=int, default=32)
    p.add_argument("--n_patches", type=int, default=5000)
    p.add_argument("--n_per_volume", type=int, default=100)
    p.add_argument("--seismic_key", type=str, default="seismicCubes_cumsum__fullstack")
    p.add_argument("--geoscore_key", type=str, default="geologic_score")
    args = p.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # create destination zarr
    dst = zarr.open(str(out), mode="w")
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

    for vol in vols:
        print("Scanning", vol)
        try:
            z = zarr.open(str(vol), mode="r")
            patches = sample_patches_from_model(z, args.seismic_key, args.geoscore_key, args.patch_size, n_patches_per_vol=args.n_per_volume)
            for pch in patches:
                if written >= args.n_patches:
                    break
                dst["patches"][written] = pch.astype("f4")
                written += 1
            if written >= args.n_patches:
                break
        except Exception as e:
            print("Failed to read", vol, e)
    print(f"Wrote {written} patches to {out}")


if __name__ == "__main__":
    main()
