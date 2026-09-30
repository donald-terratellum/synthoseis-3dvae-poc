#!/usr/bin/env python3
"""Add per-patch dip samples to an existing sampled patch zarr (made before --dip_sample_count existed).

Usage:
    uv run python scripts/add_dip_samples.py --data data/synth_train_anchored_32-32-64.zarr
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
from typing import Any, cast

import numpy as np
import zarr

from scripts.sample_patches import (
    _compute_dip_azimuth_features,
    _safe_extract_patch_by_key,
    compute_patch_dip_samples,
    write_dip_samples,
)
from src.geology_label_augment import DEFAULT_DIP_SAMPLE_COUNT


def add_dip_samples(data_path, n_samples=DEFAULT_DIP_SAMPLE_COUNT, seed=None, verify=0, dip_source_key="geologic_age_faulted"):
    """Fill dip_samples_deg / dip_samples_z from the source volumes; returns the max |meta_dip_mean_deg| mismatch checked."""
    dst = cast(Any, zarr.open_group(str(data_path), mode="r+"))
    n = int(dst.attrs.get("n_written", dst["patches"].shape[0]))
    patch_size = tuple(int(v) for v in dst["patches"].shape[1:4])
    sources = list(dst.attrs["source_volumes"])
    offset = int(dst.attrs.get("label_z_offset", 0))
    vol_idx = np.asarray(dst["source_volume_index"][:n])
    origins = np.stack([np.asarray(dst[k][:n]) for k in ("origin_x", "origin_y", "origin_z")], axis=1)
    stored_mean = np.asarray(dst["meta_dip_mean_deg"][:n]) if "meta_dip_mean_deg" in dst else None
    seed = int(dst.attrs.get("sampling_seed", 0)) if seed is None else int(seed)
    rng = np.random.default_rng([seed, 1])

    dips = np.zeros((n, int(n_samples)), np.float16)
    dip_z = np.zeros((n, int(n_samples)), np.uint8 if patch_size[2] <= 256 else np.uint16)
    worst = 0.0
    checked = 0
    for v in np.unique(vol_idx):
        rows = np.flatnonzero(vol_idx == v)
        zvol = cast(Any, zarr.open(str(sources[int(v)]), mode="r"))
        print(f"Volume {int(v) + 1}/{len(sources)}: {len(rows)} patches from {sources[int(v)]}")
        for r in rows:
            origin = tuple(int(c) for c in origins[r])
            dips[r], dip_z[r] = compute_patch_dip_samples(
                zvol, origin, patch_size, n_samples, rng, dip_source_key=dip_source_key, label_z_offset=offset
            )
            if stored_mean is not None and checked < int(verify):
                patch = _safe_extract_patch_by_key(zvol, dip_source_key, (origin[0], origin[1], origin[2] + offset), patch_size)
                if patch is not None:
                    patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
                    worst = max(worst, abs(_compute_dip_azimuth_features(patch)[0] - float(stored_mean[r])))
                    checked += 1
    write_dip_samples(dst, dips, dip_z)
    dst.attrs["dip_sample_seed"] = seed
    if checked:
        print(f"Verified {checked} patches: max |meta_dip_mean_deg| mismatch {worst:.4f} deg")
    return worst


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True, help="Sampled patch zarr to update in place.")
    p.add_argument("--dip_sample_count", type=int, default=DEFAULT_DIP_SAMPLE_COUNT)
    p.add_argument("--seed", type=int, default=None, help="Default: the store's sampling_seed.")
    p.add_argument("--verify", type=int, default=200, help="Recompute meta_dip_mean_deg for this many patches as an origin/offset check.")
    p.add_argument("--max_mismatch_deg", type=float, default=0.01)
    args = p.parse_args()
    worst = add_dip_samples(args.data, args.dip_sample_count, args.seed, args.verify)
    if worst > args.max_mismatch_deg:
        raise SystemExit(f"Stored dip metadata does not match the source volumes (max mismatch {worst:.4f} deg).")


if __name__ == "__main__":
    main()
