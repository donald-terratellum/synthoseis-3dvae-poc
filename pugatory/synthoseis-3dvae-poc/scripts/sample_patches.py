#!/usr/bin/env python3
"""Sample 3D patches from model_data.zarr stores under a source directory into a destination zarr.

Designed to work with your synthoseis zarr layout. Uses 'geologic_score' to bias sampling.
"""
from pathlib import Path
import argparse
import zarr
import numpy as np
from tqdm import tqdm


def find_model_zarrs(source):
    p = Path(source)
    for d in p.glob('**/model_data.zarr'):
        yield d


def load_geoscore(zarr_group, geoscore_key='geologic_score'):
    g = zarr.open(str(zarr_group), mode='r')
    if geoscore_key in g:
        return np.asarray(g[geoscore_key])
    # fallback: try attributes
    return None


def sample_from_volume(zarr_group, seismic_key, geoscore_key, n_per_volume, patch_size):
    g = zarr.open(str(zarr_group), mode='r')
    seismic = np.asarray(g[seismic_key])
    # seismic shape assumed (H,W,D) or (D,H,W) — try to detect
    if seismic.shape[0] < seismic.shape[-1]:
        # assume (H,W,D)
        H,W,D = seismic.shape
    else:
        D,H,W = seismic.shape
        seismic = np.transpose(seismic, (1,2,0))
    # get geoscore
    if geoscore_key in g:
        geo = np.asarray(g[geoscore_key])
        if geo.shape != (H,W):
            geo = np.mean(geo, axis=0) if geo.ndim==3 else geo
        # normalize to probabilities
        prob = np.maximum(geo, 0).astype(np.float64)
        if prob.sum() == 0:
            prob = None
        else:
            prob = prob.ravel()
    else:
        prob = None
    coords = []
    for _ in range(n_per_volume):
        if prob is None:
            x = np.random.randint(0, H - patch_size + 1)
            y = np.random.randint(0, W - patch_size + 1)
        else:
            idx = np.random.choice(np.arange(prob.size), p=prob/prob.sum())
            xr = idx // W
            yr = idx % W
            x = np.clip(xr + np.random.randint(-patch_size//2, patch_size//2+1), 0, H-patch_size)
            y = np.clip(yr + np.random.randint(-patch_size//2, patch_size//2+1), 0, W-patch_size)
        z = np.random.randint(0, D - patch_size + 1)
        coords.append((x,y,z))
    patches = []
    for x,y,z in coords:
        patch = seismic[x:x+patch_size, y:y+patch_size, z:z+patch_size]
        if patch.shape == (patch_size, patch_size, patch_size):
            patches.append(patch.astype('f4'))
    return patches


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--patch_size', type=int, default=32)
    p.add_argument('--n_patches', type=int, default=4096)
    p.add_argument('--n_per_volume', type=int, default=256)
    p.add_argument('--seismic_key', default='seismicCubes_cumsum__fullstack')
    p.add_argument('--geoscore_key', default='geologic_score')
    args = p.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # open dst
    dst = zarr.open(str(out), mode='w')
    dst.create_array('patches', shape=(args.n_patches, args.patch_size, args.patch_size, args.patch_size), dtype='f4', chunks=(1,args.patch_size,args.patch_size,args.patch_size))

    i = 0
    for zpath in find_model_zarrs(src):
        patches = sample_from_volume(zpath, args.seismic_key, args.geoscore_key, args.n_per_volume, args.patch_size)
        for pch in patches:
            if i >= args.n_patches:
                break
            dst['patches'][i] = pch
            i += 1
        if i >= args.n_patches:
            break
    print('wrote', i, 'patches to', out)

if __name__ == '__main__':
    main()