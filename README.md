# synthoseis-3dvae-poc

Proof-of-concept 3D VAE training on synthetic seismic patches (32^3).

Overview

- Sample 32^3 patches from existing zarr corpus at /Users/donaldpg/synthoseis/synthoseis
- Train a small 3D convolutional VAE (PyTorch)
- Save model checkpoints, sample reconstructions, and training logs

Quickstart

```bash
# use Astral uv runtime
uv run python -m pip install --upgrade pip
uv sync

# sample patches (example)
uv run python scripts/sample_patches.py --source /Users/donaldpg/synthoseis/synthoseis --out data/train.zarr --patch_size 32 --n_patches 5000

# train (example)
uv run python train.py --data data/train.zarr --batch_size 8 --epochs 100 --device cuda
```
