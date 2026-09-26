#!/usr/bin/env bash
# Geology classifier + class-anchored sampling suite (plan 2026-09-25, Section 8).
#
# Usage: scripts/geoaware_classifier_suite.sh [sample|train|benchmark|all]
# Environment overrides: SOURCE, OUT_DIR, WARM_START, EPOCHS, BENCH_EPOCHS, PYTHON, OVERWRITE=1,
# PRINT_COMMANDS=1 (print commands without running them or checking inputs).
set -euo pipefail

cd "$(dirname "$0")/.."

STAGE="${1:-all}"
PYTHON="${PYTHON:-uv run python}"
SOURCE="${SOURCE:-/Volumes/CrucialX9/fake_data}"
TRAIN_DATA="${TRAIN_DATA:-data/synth_train_anchored_32-32-64.zarr}"
VAL_UNIFORM="${VAL_UNIFORM:-data/synth_val_uniform_32-32-64.zarr}"
FROZEN_VAL="${FROZEN_VAL:-data/synth_val_32-32-64.zarr}"
MANIFEST="${MANIFEST:-docs/benchmarks/frozen_validation_manifest.json}"
WARM_START="${WARM_START:-checkpoints/geoaware_v3_phase2_20260831/vae_epoch20.pt}"
OUT_DIR="${OUT_DIR:-checkpoints/geoaware_v4_classifier_20260925}"
EPOCHS="${EPOCHS:-40}"
BENCH_EPOCHS="${BENCH_EPOCHS:-10 20 30 40}"
LABEL_Z_OFFSET=1  # verified in WP0: seismic[z] <-> label[z + 1]
METADATA_KEYS=(
  meta_fault_fraction meta_fault_intersection_fraction
  meta_channel_fraction meta_channel_core_fraction
  meta_flat_spot_fraction meta_onlap_fraction meta_onlap_variability
)
read -r -a PY <<< "$PYTHON"

run() {
  if [[ "${PRINT_COMMANDS:-0}" == "1" ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

require() {
  [[ "${PRINT_COMMANDS:-0}" == "1" ]] && return 0
  for path in "$@"; do
    [[ -e "$path" ]] || { echo "Missing required input: $path" >&2; exit 1; }
  done
}

prepare_output() {
  [[ "${PRINT_COMMANDS:-0}" == "1" ]] && return 0
  if [[ -e "$1" ]]; then
    if [[ "${OVERWRITE:-0}" == "1" ]]; then
      rm -rf "$1"
    else
      echo "Output exists: $1 (set OVERWRITE=1 to replace it)" >&2
      exit 1
    fi
  fi
}

stage_sample() {
  require "$SOURCE" "$SOURCE/validation"
  prepare_output "$TRAIN_DATA"
  prepare_output "$VAL_UNIFORM"

  # Train: class-anchored, validation volumes excluded.
  run "${PY[@]}" scripts/sample_patches.py \
    --source "$SOURCE" \
    --exclude_dir validation \
    --patch_size 32 32 64 \
    --n_patches 108000 \
    --n_per_volume 600 \
    --seismic_key seismicCubes_cumsum_fullstack \
    --geoscore_key geologic_score \
    --sampling_mode class_anchored \
    --class_quotas fault=0.10 fault_x=0.10 flat_spot=0.10 channel=0.10 closure=0.10 onlap=0.10 sand=0.05 \
    --background_fraction 0.25 \
    --max_patches_per_object 24 \
    --presence_min_voxels 32 \
    --label_z_offset "$LABEL_Z_OFFSET" \
    --seed 20260925 \
    --out "$TRAIN_DATA"

  # Validation: natural prevalence; 20 of 25 validation volumes carry labels (4,000 patches).
  run "${PY[@]}" scripts/sample_patches.py \
    --source "$SOURCE/validation" \
    --patch_size 32 32 64 \
    --n_patches 4000 \
    --n_per_volume 200 \
    --seismic_key seismicCubes_cumsum_fullstack \
    --geoscore_key geologic_score \
    --sampling_mode uniform \
    --presence_min_voxels 32 \
    --label_z_offset "$LABEL_Z_OFFSET" \
    --disjoint_from "$TRAIN_DATA" \
    --seed 20260926 \
    --out "$VAL_UNIFORM"
}

stage_train() {
  require "$TRAIN_DATA" "$VAL_UNIFORM" "$WARM_START"
  run "${PY[@]}" scripts/train.py \
    --data "$TRAIN_DATA" \
    --validation_data "$VAL_UNIFORM" \
    --patch_size 32 32 64 \
    --batch_size 12 \
    --number_batches 450 \
    --epochs "$EPOCHS" \
    --seed 20260925 \
    --resume "$WARM_START" --resume_epoch 0 \
    --augment --vertical_warp_prob 0.5 --mixup_augment_prob 0.0 \
    --input_scaling divide_by_std \
    --learning_rate 5e-4 --weight_decay 1e-4 --encoder_lr_mult 0.1 \
    --kl_schedule warmup --kl_start 1e-3 --kl_end 1e-3 --kl_warmup_epochs 1 \
    --reconstruction_loss mae --loss_mse_weight 1.0 --lpips_weight 0.1 \
    --lr_scheduler plateau --lr_scheduler_patience 6 --lr_scheduler_factor 0.5 --lr_scheduler_min_lr 1e-5 \
    --early_stopping_patience 999 --save_epoch_checkpoints \
    --geology_projection --geology_proj_hidden 128 --geology_proj_dim 64 \
    --geology_contrastive_weight 0.5 --geology_contrastive_temperature 0.2 \
    --geology_uniformity_weight 0 \
    --geology_classifier --geology_classifier_mode patch \
    --geology_classifier_weight 0.1 --geology_classifier_loss focal --geology_classifier_focal_gamma 2.0 \
    --geology_classifier_label_smoothing 0.05 \
    --geology_batch_sampler \
    --geology_strata_source presence_labels \
    --geology_batch_background_fraction 0.05 \
    --geology_batch_hard_fraction 0.30 \
    --geology_batch_min_negative_strata 2 \
    --geology_batch_class_quota fault_x=1 flat_spot=1 channel=1 \
    --geology_metadata_keys "${METADATA_KEYS[@]}" \
    --geology_diagnostic_max_samples 512 --geology_diagnostic_neighbor_k 5 --geology_diagnostic_topk 5 10 20 \
    --best_checkpoint_name vae_best.pt \
    --out_dir "$OUT_DIR"
}

stage_benchmark() {
  require "$FROZEN_VAL" "$MANIFEST" "$VAL_UNIFORM" "$TRAIN_DATA"
  local name
  name="$(basename "$OUT_DIR")"
  for ep in $BENCH_EPOCHS; do
    require "$OUT_DIR/vae_epoch${ep}.pt"
    run "${PY[@]}" scripts/evaluate_geology_benchmark.py \
      --data "$FROZEN_VAL" \
      --manifest "$MANIFEST" \
      --checkpoint "$OUT_DIR/vae_epoch${ep}.pt" \
      --use_geo_embedding \
      --metadata_keys "${METADATA_KEYS[@]}" \
      --classifier_data "$VAL_UNIFORM" \
      --classifier_threshold_data "$TRAIN_DATA" \
      --out_json "docs/benchmarks/${name}_ep${ep}_zgeo.json"
  done
}

case "$STAGE" in
  sample) stage_sample ;;
  train) stage_train ;;
  benchmark) stage_benchmark ;;
  all) stage_sample; stage_train; stage_benchmark ;;
  *) echo "Unknown stage '$STAGE' (expected sample|train|benchmark|all)" >&2; exit 2 ;;
esac
