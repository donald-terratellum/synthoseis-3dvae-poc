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

Successful training run (reported)

```bash
cd ~/synthoseis-3dvae-poc

rm -rf data/train.zarr

uv run python scripts/sample_patches.py \
	--source /Users/donaldpg/synthoseis/fake_data \
	--patch_size 32 \
	--n_patches 50000 \
	--n_per_volume 4200 \
	--seismic_key seismicCubes_cumsum__17_degrees \
	--geoscore_key geologic_score \
	--out data/train.zarr

uv run python train.py --data data/train.zarr --batch_size 100 --number_batches 100 --epochs 75
```

Training knobs (concise)

- `--augment`: enable on-the-fly data augmentation during training.
- Paired (input + label) augmentations: `--swap_xy_prob`, `--flip_x_prob`, `--flip_y_prob`, `--vertical_warp_prob`.
- Input-only augmentation: 3x3 trace-cluster zeroing (`--zero_cluster_min`, `--zero_cluster_max`) is applied to input only, never to label.
- `--validation_extrema_only` / `--no_validation_extrema_only`: toggle extrema-only validation input (default: on).
- `--resume`: resume model weights from a checkpoint.
- `--weight_decay`: AdamW regularization strength.
- `--kl_schedule`: `warmup` or `fixed` KL weighting.
- `--kl_start`, `--kl_end`, `--kl_warmup_epochs`: KL warmup controls.
- `--kl_fixed`: KL weight when `--kl_schedule fixed`.
- `--lr_scheduler`: `plateau` or `none`.
- `--lr_scheduler_patience`, `--lr_scheduler_factor`, `--lr_scheduler_min_lr`: scheduler controls.
- `--early_stopping_patience`, `--early_stopping_min_delta`: stop criteria on validation loss.
- `--best_checkpoint_name`: filename for best checkpoint in `--out_dir`.
- `--save_epoch_checkpoints` / `--no_save_epoch_checkpoints`: keep per-epoch checkpoints or best-only.

Discriminator (GAN) setup (concise)

- Enable with `--use_discriminator`.
- Generator objective becomes `VAE loss + gan_weight * g_gan_loss` (`--gan_weight`).
- Discriminator learns real-label vs reconstructed-fake classification each train step; validation path stays augmentation-free and does not use discriminator loss.
- Optional discriminator-specific optimizer knobs: `--discriminator_learning_rate`, `--discriminator_weight_decay`, `--discriminator_base_ch`.
- Per-epoch logs include `d_gan_acc` in stdout and `d_gan_acc_pct` in `training_metrics.csv`.
- `d_gan_acc` interpretation on balanced real/fake batches: ~50% means chance-level discrimination; higher values mean stronger separation.

Recommended stable config

```bash
uv run python train.py \
	--data data/train.zarr \
	--batch_size 100 \
	--number_batches 100 \
	--epochs 150 \
	--augment \
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
	--out_dir checkpoints
```

Successful discriminator config (reported)

```bash
uv run python train.py \
	--data data/train.zarr \
	--batch_size 100 \
	--number_batches 50 \
	--epochs 150 \
	--augment \
	--vertical_warp_prob 0.5 \
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
	--gan_weight 0.2 \
	--out_dir checkpoints_gan_vwarp
```
