#!/usr/bin/env bash
# P4: warm-start unchanged P0b and append K=2 real examples for reconstruction/KL only.
# Usage: scripts/p4_real_mix_suite.sh
# Env: PYTHON, SEED, EPOCHS, BATCHES, LOG_DIR, OUT_ROOT, PRINT_COMMANDS=1
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
SEED="${SEED:-20260925}"
EPOCHS="${EPOCHS:-40}"
BATCHES="${BATCHES:-450}"
LOG_DIR="${LOG_DIR:-logs}"
OUT_ROOT="${OUT_ROOT:-checkpoints/p4_real_mix}"
TRAIN_DATA="data/synth_train_anchored_32-32-64.zarr"
VAL_DATA="data/synth_val_v2_uniform_32-32-64.zarr"
BENCH_DATA="data/synth_val_v2_32-32-64.zarr"
REAL_TRAIN_DATA="data/real_train_32-32-64.zarr"
REAL_VAL_DATA="data/real_validation_32-32-64.zarr"
REAL_TEST_DATA="data/real_test_32-32-64.zarr"
MANIFEST="docs/benchmarks/frozen_validation_manifest_v2.json"
WARM_START="checkpoints/geoaware_p0b_r2_seed${SEED}_r2/vae_best.pt"
OUT="${OUT_ROOT}_seed${SEED}"
LOG="${LOG_DIR}/p4_real_mix_seed${SEED}.log"
METADATA_KEYS=(
  meta_fault_fraction meta_fault_intersection_fraction
  meta_channel_fraction meta_channel_core_fraction
  meta_flat_spot_fraction meta_onlap_fraction meta_onlap_variability
)

printing() { [[ "${PRINT_COMMANDS:-0}" == "1" ]]; }
run_logged() {
  local log="$1"; shift
  if printing; then printf '%q ' "$@"; printf '\n'; else "$@" 2>&1 | tee -a "$log"; fi
}

if ! printing; then
  mkdir -p "$LOG_DIR"
  for required in "$WARM_START" "$REAL_TRAIN_DATA" "$REAL_VAL_DATA" "$REAL_TEST_DATA"; do
    if [[ ! -e "$required" ]]; then
      echo "Missing P4 input: $required" >&2
      exit 1
    fi
  done
  : > "$LOG"
  printf 'P4 seed=%s K=2 phase=off stretch=off warm_start=%s\n' "$SEED" "$WARM_START" | tee -a "$LOG"
fi

run_logged "$LOG" "$PYTHON" scripts/train.py \
  --data "$TRAIN_DATA" --validation_data "$VAL_DATA" \
  --real_data "$REAL_TRAIN_DATA" --real_validation_data "$REAL_VAL_DATA" --real_test_data "$REAL_TEST_DATA" \
  --real_batch_count 2 --real_recon_weight 1.0 \
  --patch_size 32 32 64 --batch_size 12 --number_batches "$BATCHES" --epochs "$EPOCHS" \
  --seed "$SEED" --resume "$WARM_START" --resume_epoch 0 \
  --augment --vertical_warp_prob 0.5 --phase_rotation_prob 0.0 --stretch_prob 0.0 \
  --dip_label_policy adjust --mixup_augment_prob 0.0 \
  --input_scaling divide_by_std --learning_rate 5e-4 --weight_decay 1e-4 --encoder_lr_mult 0.1 \
  --kl_schedule warmup --kl_start 1e-3 --kl_end 1e-3 --kl_warmup_epochs 1 \
  --reconstruction_loss mae --lpips_weight 0.1 \
  --lr_scheduler plateau --lr_scheduler_patience 6 --lr_scheduler_factor 0.5 --lr_scheduler_min_lr 1e-5 \
  --early_stopping_patience 999 --save_epoch_checkpoints \
  --geology_projection --geology_proj_hidden 128 --geology_proj_dim 64 \
  --geology_contrastive_weight 0.5 --geology_contrastive_temperature 0.2 --geology_uniformity_weight 0 \
  --geology_classifier --geology_classifier_mode patch --geology_classifier_weight 0.1 \
  --geology_classifier_loss focal --geology_classifier_focal_gamma 2.0 --geology_classifier_label_smoothing 0.05 \
  --geology_batch_sampler --geology_batch_background_fraction 0.05 --geology_batch_hard_fraction 0.30 \
  --geology_batch_min_negative_strata 2 --geology_strata_source presence_labels \
  --geology_batch_class_quota fault_x=1 flat_spot=1 channel=1 \
  --geology_metadata_keys "${METADATA_KEYS[@]}" \
  --geology_diagnostic_max_samples 512 --geology_diagnostic_neighbor_k 5 --geology_diagnostic_topk 5 10 20 \
  --best_checkpoint_name vae_best.pt --out_dir "$OUT"

for ep in 10 20 30 40; do
  ckpt="${OUT}/vae_epoch${ep}.pt"
  report="docs/benchmarks/p4_real_mix_seed${SEED}_ep${ep}_zgeo.json"
  run_logged "$LOG" "$PYTHON" scripts/evaluate_geology_benchmark.py \
    --data "$BENCH_DATA" --manifest "$MANIFEST" --checkpoint "$ckpt" --use_geo_embedding \
    --metadata_keys "${METADATA_KEYS[@]}" --classifier_data "$VAL_DATA" \
    --classifier_threshold_data "$TRAIN_DATA" --out_json "$report"
done
