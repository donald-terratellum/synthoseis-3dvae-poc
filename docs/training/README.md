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

# Build training patches from canonical synthetic volumes.
# Source pattern: /Volumes/Crucial X9/fake_data/seismic__2026.*__synthoseis_run_*/model_data.zarr
uv run python scripts/sample_patches.py \
  --source '/Volumes/Crucial X9/fake_data' \
  --patch_size 32 32 64 \
  --n_patches 120000 \
  --n_per_volume 600 \
  --seismic_key seismicCubes_cumsum_fullstack \
  --geoscore_key geologic_score \
  --out data/synth_train_32-32-64.zarr

# Build validation patches from the same synthetic source.
uv run python scripts/sample_patches.py \
  --source '/Volumes/Crucial X9/fake_data' \
  --patch_size 32 32 64 \
  --n_patches 24000 \
  --n_per_volume 200 \
  --seismic_key seismicCubes_cumsum_fullstack \
  --geoscore_key geologic_score \
  --out data/synth_val_32-32-64.zarr

# Geology-aware retraining pass (derived metadata vectors are read automatically).
uv run python scripts/train.py \
  --data data/synth_train_32.zarr \
  --validation_data data/synth_val_32.zarr \
  --patch_size 32 32 64 \
  --batch_size 12 \
  --number_batches 300 \
  --epochs 120 \
  --augment \
  --vertical_warp_prob 0.5 \
  --mixup_augment_prob 0.2 \
  --learning_rate 1e-4 \
  --weight_decay 1e-4 \
  --kl_schedule warmup \
  --kl_start 0.0 \
  --kl_end 1e-3 \
  --kl_warmup_epochs 20 \
  --lr_scheduler plateau \
  --lr_scheduler_patience 4 \
  --lr_scheduler_factor 0.5 \
  --early_stopping_patience 12 \
  --geology_loss_weight 0.10 \
  --geology_metadata_keys \
    meta_dip_mean_deg \
    meta_dip_std_deg \
    meta_azimuth_mean_deg \
    meta_azimuth_circular_variance \
    meta_fault_intersection_fraction \
    meta_geologic_score_mean \
    meta_sand_fraction \
    meta_shale_fraction \
    meta_flat_spot_fraction \
    meta_onlap_fraction \
    meta_onlap_variability \
    meta_channel_fraction \
    meta_channel_core_fraction \
    meta_structural_complexity \
  --best_checkpoint_name vae_best.pt \
  --out_dir checkpoints/synth_geoaware_v1

Use --patch_size X Y Z for anisotropic examples. If you provide a single value, it is applied to all three axes.

```

## Derived metadata pipeline

scripts/sample_patches.py now writes per-patch derived metadata arrays alongside patches. These keys are attached in zarr attrs under derived_metadata_keys and include:

- meta_dip_mean_deg
- meta_dip_std_deg
- meta_azimuth_mean_deg
- meta_azimuth_circular_variance
- meta_fault_intersection_fraction
- meta_geologic_score_mean
- meta_sand_fraction
- meta_shale_fraction
- meta_flat_spot_fraction
- meta_onlap_fraction
- meta_onlap_variability
- meta_channel_fraction
- meta_channel_core_fraction
- meta_structural_complexity

These vectors are computed from synthetic label volumes (for example geologic_age_faulted and fault_intersection_segments) and used by scripts/train.py when geology loss is enabled.

## New geology-aware train.py options

- --geology_loss_weight FLOAT
  - Default: 0.0
  - Set > 0 to activate metadata-to-latent similarity shaping.
- --geology_metadata_keys KEY [KEY ...]
  - Default keys are the derived metadata set above.
  - Use this if you want to drop or reorder features for ablations.

## Latent geology validation diagnostics

When geology metadata is enabled, validation compares pairwise cosine similarity of deterministic encoder `mu` vectors with pairwise cosine similarity of the selected geology metadata. Diagnostic cubes use the tokenizer's deterministic preprocessing (per-cube standard-deviation normalization followed by trace-extrema retention), so augmentation randomness does not contaminate epoch-to-epoch trends. This is the same preprocessing, latent representation, and similarity family used by the seismic tokenizer.

- `validation/geology_latent_pair_cosine_correlation`: Pearson correlation across all validation-pair cosine values. Higher is better; `1` means identical pair ordering.
- `validation/geology_latent_cosine_separation`: mean latent cosine for the most geologically similar 10% of pairs minus the mean for the least similar 10%. Larger positive values indicate better separation.
- `validation/geology_latent_neighbor_overlap`: fraction of each patch's metadata top-k neighbors also found among its latent top-k neighbors. Higher is better and most directly reflects tokenizer retrieval behavior.
- `validation/geology_latent_similar_cosine` and `validation/geology_latent_dissimilar_cosine`: the two components of the separation metric.

Configure diagnostic cost and retrieval neighborhood size with:

```bash
--geology_diagnostic_max_samples 512 \
--geology_diagnostic_neighbor_k 5
```

The values are written to TensorBoard each epoch. Correlation, separation, and neighbor overlap are also appended to `training_metrics.csv` and printed after the epoch summary. Monitor trends on validation data rather than absolute training-batch values.

## Recommended values for the first full run

- patch creation
  - --patch_size 32 32 64
  - --n_patches 120000 (train), 24000 (val)
  - --n_per_volume 600 (train), 200 (val) — must be high enough that n_patches ≤ n_volumes × n_per_volume; otherwise the preallocated zarr is filled with silence and PMSE explodes
  - --seismic_key seismicCubes_cumsum_fullstack
  - --geoscore_key geologic_score
- training core
  - --batch_size 12
  - --number_batches 300
  - --epochs 120
  - --learning_rate 1e-4
  - --weight_decay 1e-4
  - --kl_schedule warmup --kl_start 0.0 --kl_end 1e-3 --kl_warmup_epochs 20
  - --geology_loss_weight 0.10
- augmentation and stability
  - --augment
  - --vertical_warp_prob 0.5
  - --mixup_augment_prob 0.2
  - --lr_scheduler plateau --lr_scheduler_patience 4 --lr_scheduler_factor 0.5
  - --early_stopping_patience 12

If training is unstable, lower geology pressure first (for example --geology_loss_weight 0.05), then increase once reconstruction is stable.

## Minimal smoke command

For a quick integration check before long training:

```bash
uv run python scripts/train.py \
  --data data/synth_train_32-32-64.zarr \
  --validation_data data/synth_val_32-32-64.zarr \
  --batch_size 4 \
  --number_batches 2 \
  --epochs 1 \
  --learning_rate 1e-4 \
  --weight_decay 1e-4 \
  --kl_schedule fixed \
  --kl_fixed 1e-4 \
  --geology_loss_weight 0.10 \
  --out_dir checkpoints/smoke_geoaware \
  --best_checkpoint_name vae_best.pt \
  --no_save_epoch_checkpoints
```

## Troubleshooting: NaN or Inf validation loss

Symptoms: training loss in the hundreds of thousands, validation loss nan/inf from epoch 1.

Root cause: the sampled zarr file has far fewer real patches than the preallocated size. The zarr is created with shape (n_patches, ...) but if n_volumes × n_per_volume < n_patches, the remainder is zarr zero-fill. The PMSE reconstruction loss divides by label energy, which is 0 for zero patches, producing huge or infinite loss values. This also causes metadata vectors of all zeros which produce NaN in the geology loss on MPS.

Check:

```bash
uv run python - <<'PY'
import zarr, numpy as np
z = zarr.open('data/synth_train_32-32-64.zarr', mode='r')
patches = np.asarray(z['patches'])
zero_rows = int(np.all(patches.reshape(len(patches), -1) == 0.0, axis=1).sum())
print(f'zero patches: {zero_rows}/{len(patches)} ({100.0*zero_rows/len(patches):.1f}%)')
PY
```

Fix: ensure n_per_volume × n_volumes ≥ n_patches. With ~200 volumes, use n_per_volume=600 for n_patches=120000.

If training fails with an error like:

- KeyError: Required geology metadata key 'meta_sand_fraction' was not found in the dataset.

then your patch zarr was created before the latest metadata schema update. Regenerate both train and validation patch datasets with scripts/sample_patches.py, then rerun training.

Quick check:

```bash
uv run python - <<'PY'
import zarr
z = zarr.open('data/synth_train_32-32-64.zarr', mode='r')
print('derived_metadata_keys:', z.attrs.get('derived_metadata_keys'))
print('has meta_sand_fraction:', 'meta_sand_fraction' in z)
PY
```

If you must train with older patch files, pass only keys that exist in those files to --geology_metadata_keys.

## Better rerun strategy

For better stability, do not start with the full geology key set at full weight on the first pass. Use a staged run:

1. First pass: regenerate patches, then train with a smaller geology weight and only the structural keys.
   - --geology_loss_weight 0.02 to 0.05
   - --geology_metadata_keys meta_dip_mean_deg meta_dip_std_deg meta_azimuth_mean_deg meta_azimuth_circular_variance meta_structural_complexity
2. Second pass: once reconstruction is stable, resume from the best checkpoint and add the remaining interpretation keys.
   - --resume checkpoints/synth_geoaware_v1/vae_best.pt
   - add meta_sand_fraction, meta_shale_fraction, meta_flat_spot_fraction, meta_onlap_fraction, meta_onlap_variability, meta_channel_fraction, meta_channel_core_fraction

Recommended rerun sequence:

```bash
uv run python scripts/sample_patches.py \
  --source '/Volumes/Crucial X9/fake_data' \
  --patch_size 32 32 64 \
  --n_patches 120000 \
  --n_per_volume 600 \
  --seismic_key seismicCubes_cumsum_fullstack \
  --geoscore_key geologic_score \
  --out data/synth_train_32-32-64.zarr

uv run python scripts/sample_patches.py \
  --source '/Volumes/Crucial X9/fake_data' \
  --patch_size 32 32 64 \
  --n_patches 24000 \
  --n_per_volume 200 \
  --seismic_key seismicCubes_cumsum_fullstack \
  --geoscore_key geologic_score \
  --out data/synth_val_32-32-64.zarr
```

Then train with a smaller first-pass geology weight:

```bash
uv run python scripts/train.py \
  --data data/synth_train_32-32-64.zarr \
  --validation_data data/synth_val_32-32-64.zarr \
  --patch_size 32 32 64 \
  --batch_size 12 \
  --number_batches 300 \
  --epochs 120 \
  --augment \
  --vertical_warp_prob 0.5 \
  --mixup_augment_prob 0.2 \
  --learning_rate 1e-4 \
  --weight_decay 1e-4 \
  --kl_schedule warmup \
  --kl_start 0.0 \
  --kl_end 1e-3 \
  --kl_warmup_epochs 20 \
  --lr_scheduler plateau \
  --lr_scheduler_patience 4 \
  --lr_scheduler_factor 0.5 \
  --early_stopping_patience 12 \
  --geology_loss_weight 0.03 \
  --geology_metadata_keys meta_dip_mean_deg meta_dip_std_deg meta_azimuth_mean_deg meta_azimuth_circular_variance meta_structural_complexity \
  --best_checkpoint_name vae_best.pt \
  --out_dir checkpoints/synth_geoaware_v1
```

If that run is stable and validation keeps improving, resume from the best checkpoint and expand the metadata keys to include the channel, flat-spot, onlap, and lithology-derived features.

## Main files

- scripts/train.py
- scripts/sample_patches.py
- src/model.py
- src/augmentations.py

## Notes

- Patch size can be specified as one value (broadcast to all three axes) or three values (X Y Z).
- Checkpoints are written under configured output directories (for example checkpoints/).

## Advanced experiment guides

- Latent alignment and encoder/decoder balancing guide (Markdown): [latent_alignment_experiments.md](latent_alignment_experiments.md)
- Latent alignment and encoder/decoder balancing guide (HTML): [latent_alignment_experiments.html](latent_alignment_experiments.html)
