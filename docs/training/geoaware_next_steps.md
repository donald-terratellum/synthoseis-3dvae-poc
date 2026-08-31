# Geo-Aware Next Steps

This guide is the shortest path from the successful screening run to a final checkpoint and benchmark comparison.

Current dataset counts:

- source: 180 synthetic seismic datasets
- validation: 25 synthetic seismic datasets

Current split:

- about 87.8% train / 12.2% validation

## Step 1: regenerate the training data

Run:

```bash
rm -rf data/synth_train_32-32-64.zarr data/synth_val_32-32-64.zarr

uv run python scripts/sample_patches.py \
  --source '/Volumes/Crucial X9/fake_data' \
  --patch_size 32 32 64 \
  --n_patches 108000 \
  --n_per_volume 600 \
  --seismic_key seismicCubes_cumsum_fullstack \
  --geoscore_key geologic_score \
  --out data/synth_train_32-32-64.zarr

uv run python scripts/sample_patches.py \
  --source '/Volumes/Crucial X9/fake_data/validation' \
  --patch_size 32 32 64 \
  --n_patches 5000 \
  --n_per_volume 200 \
  --seismic_key seismicCubes_cumsum_fullstack \
  --geoscore_key geologic_score \
  --out data/synth_val_32-32-64.zarr
```

## Step 2: run the full training job

Run:

```bash
uv run python scripts/train.py \
  --data data/synth_train_32-32-64.zarr \
  --validation_data data/synth_val_32-32-64.zarr \
  --patch_size 32 32 64 \
  --batch_size 12 \
  --number_batches 450 \
  --epochs 240 \
  --augment \
    --vertical_warp_prob 0.5 \
    --mixup_augment_prob 0.2 \
    --input_decimate_trilinear_prob 0.05 \
  --input_scaling divide_by_std \
  --learning_rate 5e-4 \
  --weight_decay 1e-4 \
  --kl_schedule warmup \
  --kl_start 0.0 \
  --kl_end 1e-3 \
  --kl_warmup_epochs 20 \
  --reconstruction_loss mae \
  --loss_mse_weight 1.0 \
  --lpips_weight 0.1 \
  --lr_scheduler plateau \
  --lr_scheduler_patience 4 \
  --lr_scheduler_factor 0.5 \
  --early_stopping_patience 12 \
  --geology_loss_weight 0.03 \
  --geology_diagnostic_max_samples 4096 \
  --geology_diagnostic_neighbor_k 5 \
  --geology_diagnostic_topk 5 10 20 \
  --geology_batch_sampler \
    --geology_batch_background_fraction 0.20 \
    --geology_batch_hard_fraction 0.20 \
    --geology_batch_min_negative_strata 2 \
  --geology_metadata_keys \
    meta_fault_fraction \
    meta_fault_intersection_fraction \
    meta_channel_fraction \
    meta_channel_core_fraction \
    meta_flat_spot_fraction \
    meta_onlap_fraction \
    meta_onlap_variability \
  --representative_selection_epoch 5 \
  --representative_plot_interval 5 \
  --refresh_representative_examples \
  --adaptive_sampling_by_mse \
    --sampling_snapshot_interval 5 \
    --sampling_improvement_window 5 \
    --sampling_improvement_weight 0.35 \
    --sampling_snapshot_filename adaptive_sampling_snapshots.pt \
  --best_checkpoint_name vae_best.pt \
  --out_dir checkpoints/synth_geoaware_v2_scratch_180x25 \
  --resume /Users/donaldpg/synthoseis-3dvae-poc/checkpoints/synth_geoaware_v2_scratch_180x25/vae_epoch105.pt
```

## Step 3: run the frozen benchmark

Run:

```bash
uv run python scripts/evaluate_geology_benchmark.py \
  --data data/synth_val_32-32-64.zarr \
  --checkpoint checkpoints/synth_geoaware_v2_scratch_180x25/vae_best.pt \
  --manifest docs/benchmarks/frozen_validation_manifest.json \
  --benchmark_size 512 \
  --seed 20260826 \
  --batch_size 32 \
  --diagnostic_topk 5 10 20 \
  --metadata_keys \
    meta_fault_fraction \
    meta_fault_intersection_fraction \
    meta_channel_fraction \
    meta_channel_core_fraction \
    meta_flat_spot_fraction \
    meta_onlap_fraction \
    meta_onlap_variability \
  --background_keys \
    meta_fault_fraction \
    meta_fault_intersection_fraction \
    meta_channel_fraction \
    meta_channel_core_fraction \
    meta_flat_spot_fraction \
    meta_onlap_fraction \
  --out_json docs/benchmarks/final_geoaware_report.json
```

## Step 4: inspect the output

Run:

```bash
python -m json.tool docs/benchmarks/final_geoaware_report.json
python -m json.tool docs/benchmarks/screening_seed20260826_report.json
diff -u docs/benchmarks/screening_seed20260826_report.json docs/benchmarks/final_geoaware_report.json || true
```

## Optional helper

If you prefer a single executable file, run:

```bash
bash scripts/geoaware_next_steps.sh
```

## Step 5: geology-loss pressure sweep (Run A and Run B)

Goal:

- compare two geology-loss weights from the same starting checkpoint
- keep validation diagnostics stable and deterministic
- keep run outputs separate for direct comparison

### Run A (geology_loss_weight = 0.07)

```bash
uv run python scripts/train.py \
  --data data/synth_train_32-32-64.zarr \
  --validation_data data/synth_val_32-32-64.zarr \
  --patch_size 32 32 64 \
  --batch_size 12 \
  --number_batches 450 \
  --epochs 20 \
  --seed 20260826 \
  --augment \
    --vertical_warp_prob 0.5 \
    --mixup_augment_prob 0.2 \
    --input_decimate_trilinear_prob 0.05 \
  --input_scaling divide_by_std \
  --learning_rate 2.5e-4 \
  --weight_decay 1e-4 \
  --kl_schedule warmup \
  --kl_start 0.0 \
  --kl_end 1e-3 \
  --kl_warmup_epochs 20 \
  --reconstruction_loss mae \
  --loss_mse_weight 1.0 \
  --lpips_weight 0.1 \
  --lr_scheduler plateau \
  --lr_scheduler_patience 4 \
  --lr_scheduler_factor 0.5 \
  --early_stopping_patience 20 \
  --geology_loss_weight 0.07 \
  --geology_loss_type huber \
  --geology_huber_delta 0.1 \
  --geology_offdiag_only \
  --geology_diagnostic_max_samples 4096 \
  --geology_diagnostic_neighbor_k 5 \
  --geology_diagnostic_topk 5 10 20 \
  --geology_batch_sampler \
    --geology_batch_background_fraction 0.05 \
    --geology_batch_hard_fraction 0.15 \
    --geology_batch_min_negative_strata 2 \
  --geology_metadata_keys \
    meta_fault_fraction \
    meta_fault_intersection_fraction \
    meta_channel_fraction \
    meta_channel_core_fraction \
    meta_flat_spot_fraction \
    meta_onlap_fraction \
    meta_onlap_variability \
  --representative_selection_epoch 5 \
  --representative_plot_interval 5 \
  --refresh_representative_examples \
  --no_validation_extrema_only \
  --adaptive_sampling_by_mse \
    --sampling_snapshot_interval 5 \
    --sampling_improvement_window 5 \
    --sampling_improvement_weight 0.10 \
    --sampling_snapshot_filename adaptive_sampling_snapshots.pt \
  --best_checkpoint_name vae_best.pt \
  --out_dir checkpoints/synth_geoaware_v2_sweep_runA_glw007 \
  --resume /Users/donaldpg/synthoseis-3dvae-poc/checkpoints/synth_geoaware_v2_scratch_180x25/vae_best.pt \
  | tee checkpoints/synth_geoaware_v2_sweep_runA_glw007/train.log
```

### Run B (geology_loss_weight = 0.10)

```bash
uv run python scripts/train.py \
  --data data/synth_train_32-32-64.zarr \
  --validation_data data/synth_val_32-32-64.zarr \
  --patch_size 32 32 64 \
  --batch_size 12 \
  --number_batches 450 \
  --epochs 20 \
  --seed 20260826 \
  --augment \
    --vertical_warp_prob 0.5 \
    --mixup_augment_prob 0.2 \
    --input_decimate_trilinear_prob 0.05 \
  --input_scaling divide_by_std \
  --learning_rate 2.5e-4 \
  --weight_decay 1e-4 \
  --kl_schedule warmup \
  --kl_start 0.0 \
  --kl_end 1e-3 \
  --kl_warmup_epochs 20 \
  --reconstruction_loss mae \
  --loss_mse_weight 1.0 \
  --lpips_weight 0.1 \
  --lr_scheduler plateau \
  --lr_scheduler_patience 4 \
  --lr_scheduler_factor 0.5 \
  --early_stopping_patience 20 \
  --geology_loss_weight 0.10 \
  --geology_loss_type huber \
  --geology_huber_delta 0.1 \
  --geology_offdiag_only \
  --geology_diagnostic_max_samples 4096 \
  --geology_diagnostic_neighbor_k 5 \
  --geology_diagnostic_topk 5 10 20 \
  --geology_batch_sampler \
    --geology_batch_background_fraction 0.05 \
    --geology_batch_hard_fraction 0.15 \
    --geology_batch_min_negative_strata 2 \
  --geology_metadata_keys \
    meta_fault_fraction \
    meta_fault_intersection_fraction \
    meta_channel_fraction \
    meta_channel_core_fraction \
    meta_flat_spot_fraction \
    meta_onlap_fraction \
    meta_onlap_variability \
  --representative_selection_epoch 5 \
  --representative_plot_interval 5 \
  --refresh_representative_examples \
  --no_validation_extrema_only \
  --adaptive_sampling_by_mse \
    --sampling_snapshot_interval 5 \
    --sampling_improvement_window 5 \
    --sampling_improvement_weight 0.10 \
    --sampling_snapshot_filename adaptive_sampling_snapshots.pt \
  --best_checkpoint_name vae_best.pt \
  --out_dir checkpoints/synth_geoaware_v2_sweep_runB_glw010 \
  --resume /Users/donaldpg/synthoseis-3dvae-poc/checkpoints/synth_geoaware_v2_scratch_180x25/vae_best.pt \
  | tee checkpoints/synth_geoaware_v2_sweep_runB_glw010/train.log
```

### One-shot bash CLI for both runs

```bash
bash -lc '
set -euo pipefail

BASE_CKPT="/Users/donaldpg/synthoseis-3dvae-poc/checkpoints/synth_geoaware_v2_scratch_180x25/vae_best.pt"

run_sweep () {
  local RUN_NAME="$1"
  local GEO_W="$2"
  local OUT_DIR="checkpoints/${RUN_NAME}"

  mkdir -p "${OUT_DIR}"
  uv run python scripts/train.py \
    --data data/synth_train_32-32-64.zarr \
    --validation_data data/synth_val_32-32-64.zarr \
    --patch_size 32 32 64 \
    --batch_size 12 \
    --number_batches 450 \
    --epochs 20 \
    --seed 20260826 \
    --augment \
      --vertical_warp_prob 0.5 \
      --mixup_augment_prob 0.2 \
      --input_decimate_trilinear_prob 0.05 \
    --input_scaling divide_by_std \
    --learning_rate 2.5e-4 \
    --weight_decay 1e-4 \
    --kl_schedule warmup \
    --kl_start 0.0 \
    --kl_end 1e-3 \
    --kl_warmup_epochs 20 \
    --reconstruction_loss mae \
    --loss_mse_weight 1.0 \
    --lpips_weight 0.1 \
    --lr_scheduler plateau \
    --lr_scheduler_patience 4 \
    --lr_scheduler_factor 0.5 \
    --early_stopping_patience 20 \
    --geology_loss_weight "${GEO_W}" \
    --geology_loss_type huber \
    --geology_huber_delta 0.1 \
    --geology_offdiag_only \
    --geology_diagnostic_max_samples 4096 \
    --geology_diagnostic_neighbor_k 5 \
    --geology_diagnostic_topk 5 10 20 \
    --geology_batch_sampler \
      --geology_batch_background_fraction 0.05 \
      --geology_batch_hard_fraction 0.15 \
      --geology_batch_min_negative_strata 2 \
    --geology_metadata_keys \
      meta_fault_fraction \
      meta_fault_intersection_fraction \
      meta_channel_fraction \
      meta_channel_core_fraction \
      meta_flat_spot_fraction \
      meta_onlap_fraction \
      meta_onlap_variability \
    --representative_selection_epoch 5 \
    --representative_plot_interval 5 \
    --refresh_representative_examples \
    --no_validation_extrema_only \
    --adaptive_sampling_by_mse \
      --sampling_snapshot_interval 5 \
      --sampling_improvement_window 5 \
      --sampling_improvement_weight 0.10 \
      --sampling_snapshot_filename adaptive_sampling_snapshots.pt \
    --best_checkpoint_name vae_best.pt \
    --out_dir "${OUT_DIR}" \
    --resume "${BASE_CKPT}" \
    | tee "${OUT_DIR}/train.log"
}

run_sweep "synth_geoaware_v2_sweep_runA_glw007" "0.07"
run_sweep "synth_geoaware_v2_sweep_runB_glw010" "0.10"
'
```

## Step 6: run apples-to-apples benchmark with a fixed 512 manifest

Run:

```bash
rm -f docs/benchmarks/frozen_validation_manifest_512_v2.json

uv run python scripts/evaluate_geology_benchmark.py \
  --data data/synth_val_32-32-64.zarr \
  --checkpoint checkpoints/synth_geoaware_v2_scratch_180x25/vae_best.pt \
  --manifest docs/benchmarks/frozen_validation_manifest_512_v2.json \
  --benchmark_size 512 \
  --seed 20260826 \
  --batch_size 32 \
  --diagnostic_topk 5 10 20 \
  --metadata_keys \
    meta_fault_fraction \
    meta_fault_intersection_fraction \
    meta_channel_fraction \
    meta_channel_core_fraction \
    meta_flat_spot_fraction \
    meta_onlap_fraction \
    meta_onlap_variability \
  --background_keys \
    meta_fault_fraction \
    meta_fault_intersection_fraction \
    meta_channel_fraction \
    meta_channel_core_fraction \
    meta_flat_spot_fraction \
    meta_onlap_fraction \
  --out_json docs/benchmarks/final_geoaware_report_512_v2.json
```

## Step 7: inspect the new benchmark output

Run:

```bash
uv run python -m json.tool docs/benchmarks/final_geoaware_report_512_v2.json
uv run python -m json.tool docs/benchmarks/screening_seed20260826_report.json
diff -u docs/benchmarks/screening_seed20260826_report.json docs/benchmarks/final_geoaware_report_512_v2.json || true
```
