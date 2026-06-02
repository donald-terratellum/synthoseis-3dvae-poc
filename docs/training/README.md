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

# create training patches
uv run python scripts/sample_patches.py \
  --source /Users/donaldpg/synthoseis/fake_data \
  --patch_size 32 32 64 \
  --n_patches 35000 \
  --n_per_volume 2500 \
  --seismic_key seismicCubes_cumsum__17_degrees \
  --geoscore_key geologic_score \
  --out data/train_32-32-64.zarr

# create validation patches
uv run python scripts/sample_patches.py \
  --source /Users/donaldpg/synthoseis/fake_data/validation \
  --patch_size 32 32 64 \
  --n_patches 10000 \
  --n_per_volume 2500 \
  --seismic_key seismicCubes_cumsum__17_degrees \
  --geoscore_key geologic_score \
  --out data/val_32-32-64.zarr

# resumed training
uv run python scripts/train.py \
  --data data/train_32-32-64.zarr \
  --validation_data data/val_32-32-64.zarr \
  --batch_size 50 \
  --number_batches 100 \
  --epochs 150 \
  --augment \
  --vertical_warp_prob 0.5 \
  --mixup_augment_prob 0.5 \
  --learning_rate 1e-4 \
  --weight_decay 1e-4 \
  --kl_schedule warmup \
  --kl_start 0.0 \
  --kl_end 1e-3 \
  --kl_warmup_epochs 15 \
  --lr_scheduler plateau \
  --lr_scheduler_patience 3 \
  --lr_scheduler_factor 0.5 \
  --early_stopping_patience 8 \
  --best_checkpoint_name vae_best.pt \
  --use_discriminator \
  --gan_weight 0.1 \
  --out_dir checkpoints_gan_vwarp2 \
  --gan_balance_controller \
  --gan_balance_disc_lr_down_mult 0.5 \
  --discriminator_learning_rate 2.5e-6 \
  --resume /Users/donaldpg/synthoseis-3dvae-poc/checkpoints_gan_vwarp/vae_best.pt

Use `--patch_size X Y Z` for anisotropic examples. If you provide a single value, it is applied to all three axes.

```

## Main files

- scripts/train.py
- scripts/sample_patches.py
- src/model.py
- src/augmentations.py

## Notes

- Patch size can be specified as one value (broadcast to all three axes) or three values (X Y Z).
- Checkpoints are written under configured output directories (for example checkpoints/).
