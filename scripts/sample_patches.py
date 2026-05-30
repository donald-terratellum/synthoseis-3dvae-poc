#!/usr/bin/env python3

"""Sample 3D patches from existing zarr volumes into a destination zarr store.

Usage:
  python scripts/sample_patches.py --source /path/to/synthoseis --out data/train.zarr --patch_size 32 --n_patches 5000
"""

from pathlib import Path
import argparse
import random
import numpy as np
import zarr
import math


def sample_patches_from_volume(z, patch_size, n_patches_per_vol=100):
    shape = z.shape  # (x,y,z) assumed
    max_x = shape[0] - patch_size
    max_y = shape[1] - patch_size
    max_z = shape[2] - patch_size
    if max_x < 0 or max_y < 0 or max_z < 0:
        return []
    patches = []
    for _ in range(n_patches_per_vol):
        i = random.randint(0, max_x)
        j = random.randint(0, max_y)
        k = random.randint(0, max_z)
        patch = z[i:i+patch_size, j:j+patch_size, k:k+patch_size]
        patches.append(patch)
    return patches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--patch_size", type=int, default=32)
    p.add_argument("--n_patches", type=int, default=5000)
    p.add_argument("--n_per_volume", type=int, default=100)
    args = p.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    dst = zarr.open(str(out), mode="w")
    dst.create_dataset("patches", shape=(args.n_patches, args.patch_size, args.patch_size, args.patch_size), dtype="f4", chunks=(1, args.patch_size, args.patch_size, args.patch_size))

    written = 0
    vols = list(src.rglob("*.zarr"))
    if not vols:
        print("No zarr volumes found under", src)
        return

    for vol in vols:
        print("Scanning", vol)
        try:
            z = zarr.open(str(vol), mode="r")
            # pick a reasonable number per volume
            take = min(args.n_per_volume, args.n_patches - written)
            patches = sample_patches_from_volume(z, args.patch_size, n_patches_per_vol=take)
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
