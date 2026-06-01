# VAE Training Component

This component covers:
1. Seismic patch data preparation.
2. 3D VAE training and checkpointing.

## Scope
- Build training zarr patch datasets from seismic cubes.
- Train the 3D VAE model (with optional discriminator path).
- Save checkpoints and training metrics.

## Quick start
```bash
uv sync

uv run python scripts/sample_patches.py \
  --source /path/to/seismic_source \
  --out data/train.zarr \
  --patch_size 32 \
  --n_patches 5000

uv run python docs/train.py \
  --data data/train.zarr \
  --batch_size 8 \
  --epochs 100 \
  --device auto
```

## Main files
- docs/train.py
- scripts/sample_patches.py
- src/model.py
- src/augmentations.py

## Notes
- Patch size is 32x32x32.
- Checkpoints are written under configured output directories (for example checkpoints/).
