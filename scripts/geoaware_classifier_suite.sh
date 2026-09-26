#!/usr/bin/env bash
# Geology classifier + class-anchored sampling experiments (plan 2026-09-25, Sections 7-8).
#
# Usage: scripts/geoaware_classifier_suite.sh [sample|train|benchmark|summary|all]
#   Runs: r1ctrl (Phase 2 recipe, leak-free geoscore data), r1 (class-anchored data),
#         r2 (r1 + classifier), r3 (r2 + presence strata + batch class quotas).
# Re-running is safe: finished datasets, runs, and reports are skipped (OVERWRITE=1 redoes them).
# Environment overrides: RUNS, SOURCE, OUT_PREFIX, WARM_START, EPOCHS, BENCH_EPOCHS, PYTHON,
# LOG_DIR, OVERWRITE=1, PRINT_COMMANDS=1 (print commands without running them or checking inputs).
# Long runs on macOS: nohup caffeinate -i scripts/geoaware_classifier_suite.sh all > logs/suite.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/.."

STAGE="${1:-all}"
PYTHON="${PYTHON:-uv run python}"
SOURCE="${SOURCE:-/Volumes/CrucialX9/fake_data}"
TRAIN_DATA="${TRAIN_DATA:-data/synth_train_anchored_32-32-64.zarr}"
CTRL_DATA="${CTRL_DATA:-data/synth_train_geoscore_noval_32-32-64.zarr}"
VAL_UNIFORM="${VAL_UNIFORM:-data/synth_val_uniform_32-32-64.zarr}"
FROZEN_VAL="${FROZEN_VAL:-data/synth_val_32-32-64.zarr}"
MANIFEST="${MANIFEST:-docs/benchmarks/frozen_validation_manifest.json}"
WARM_START="${WARM_START:-checkpoints/geoaware_v3_phase2_20260831/vae_epoch20.pt}"
OUT_PREFIX="${OUT_PREFIX:-checkpoints/geoaware_v4}"
RUNS="${RUNS:-r1ctrl r1 r2 r3}"
EPOCHS="${EPOCHS:-40}"
BENCH_EPOCHS="${BENCH_EPOCHS:-10 20 30 40}"
LOG_DIR="${LOG_DIR:-logs}"
LABEL_Z_OFFSET=1  # verified in WP0: seismic[z] <-> label[z + 1]
METADATA_KEYS=(
  meta_fault_fraction meta_fault_intersection_fraction
  meta_channel_fraction meta_channel_core_fraction
  meta_flat_spot_fraction meta_onlap_fraction meta_onlap_variability
)
read -r -a PY <<< "$PYTHON"

printing() { [[ "${PRINT_COMMANDS:-0}" == "1" ]]; }

run() {
  if printing; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

# run_logged LOGFILE CMD...: run and tee output to LOGFILE.
run_logged() {
  local log="$1"
  shift
  if printing; then
    run "$@"
  else
    mkdir -p "$(dirname "$log")"
    "$@" 2>&1 | tee "$log"
  fi
}

require() {
  printing && return 0
  for path in "$@"; do
    [[ -e "$path" ]] || { echo "Missing required input: $path" >&2; exit 1; }
  done
}

# is_done TARGET: finished zarr stores carry the n_written attr (written last); other targets just exist.
is_done() {
  local target="$1"
  if [[ "$target" == *.zarr ]]; then
    grep -qs '"n_written"' "$target/zarr.json" "$target/.zattrs"
  else
    [[ -e "$target" ]]
  fi
}

# should_build TARGET: 0 to (re)build, 1 to skip because TARGET is already complete.
should_build() {
  printing && return 0
  if is_done "$1" && [[ "${OVERWRITE:-0}" != "1" ]]; then
    echo "Skipping (complete): $1" >&2
    return 1
  fi
  return 0
}

clear_output() {
  printing || rm -rf "$1"
}

stage_sample() {
  require "$SOURCE" "$SOURCE/validation"

  if should_build "$TRAIN_DATA"; then
    clear_output "$TRAIN_DATA"
    # Train: class-anchored, validation volumes excluded.
    run_logged "$LOG_DIR/sample_train_anchored.log" "${PY[@]}" scripts/sample_patches.py \
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
  fi

  if should_build "$CTRL_DATA"; then
    clear_output "$CTRL_DATA"
    # Control: legacy geoscore sampling on the same 180 training volumes (leak-free baseline).
    run_logged "$LOG_DIR/sample_train_ctrl.log" "${PY[@]}" scripts/sample_patches.py \
      --source "$SOURCE" \
      --exclude_dir validation \
      --patch_size 32 32 64 \
      --n_patches 108000 \
      --n_per_volume 600 \
      --seismic_key seismicCubes_cumsum_fullstack \
      --geoscore_key geologic_score \
      --sampling_mode geoscore \
      --label_z_offset "$LABEL_Z_OFFSET" \
      --seed 20260925 \
      --out "$CTRL_DATA"
  fi

  if should_build "$VAL_UNIFORM"; then
    clear_output "$VAL_UNIFORM"
    # Validation: natural prevalence; 20 of 25 validation volumes carry labels (4,000 patches).
    run_logged "$LOG_DIR/sample_val_uniform.log" "${PY[@]}" scripts/sample_patches.py \
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
      --disjoint_from "$CTRL_DATA" \
      --seed 20260926 \
      --out "$VAL_UNIFORM"
  fi
}

has_classifier() { [[ "$1" == "r2" || "$1" == "r3" ]]; }

run_args() {
  local run_name="$1"
  case "$run_name" in
    r1ctrl) RUN_DATA="$CTRL_DATA"; RUN_EXTRA=() ;;
    r1) RUN_DATA="$TRAIN_DATA"; RUN_EXTRA=() ;;
    r2) RUN_DATA="$TRAIN_DATA"; RUN_EXTRA=("${CLASSIFIER_ARGS[@]}") ;;
    r3) RUN_DATA="$TRAIN_DATA"; RUN_EXTRA=("${CLASSIFIER_ARGS[@]}" "${PRESENCE_STRATA_ARGS[@]}") ;;
    *) echo "Unknown run '$run_name' (expected r1ctrl|r1|r2|r3)" >&2; exit 2 ;;
  esac
}

CLASSIFIER_ARGS=(
  --geology_classifier --geology_classifier_mode patch
  --geology_classifier_weight 0.1 --geology_classifier_loss focal --geology_classifier_focal_gamma 2.0
  --geology_classifier_label_smoothing 0.05
)
PRESENCE_STRATA_ARGS=(
  --geology_strata_source presence_labels
  --geology_batch_class_quota fault_x=1 flat_spot=1 channel=1
)

stage_train() {
  require "$VAL_UNIFORM" "$WARM_START"
  local run_name out_dir
  for run_name in $RUNS; do
    run_args "$run_name"
    out_dir="${OUT_PREFIX}_${run_name}"
    should_build "$out_dir/vae_epoch${EPOCHS}.pt" || continue
    require "$RUN_DATA"
    clear_output "$out_dir"
    printing || mkdir -p "$out_dir"
    run_logged "$LOG_DIR/train_${run_name}.log" "${PY[@]}" scripts/train.py \
      --data "$RUN_DATA" \
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
      --geology_batch_sampler \
      --geology_batch_background_fraction 0.05 \
      --geology_batch_hard_fraction 0.30 \
      --geology_batch_min_negative_strata 2 \
      --geology_metadata_keys "${METADATA_KEYS[@]}" \
      --geology_diagnostic_max_samples 512 --geology_diagnostic_neighbor_k 5 --geology_diagnostic_topk 5 10 20 \
      --best_checkpoint_name vae_best.pt \
      ${RUN_EXTRA[@]+"${RUN_EXTRA[@]}"} \
      --out_dir "$out_dir"
  done
}

stage_benchmark() {
  require "$FROZEN_VAL" "$MANIFEST" "$VAL_UNIFORM" "$TRAIN_DATA"
  local run_name ep ckpt report
  local classifier_eval=()
  for run_name in $RUNS; do
    run_args "$run_name"
    classifier_eval=()
    if has_classifier "$run_name"; then
      classifier_eval=(--classifier_data "$VAL_UNIFORM" --classifier_threshold_data "$TRAIN_DATA")
    fi
    for ep in $BENCH_EPOCHS; do
      ckpt="${OUT_PREFIX}_${run_name}/vae_epoch${ep}.pt"
      report="docs/benchmarks/$(basename "$OUT_PREFIX")_${run_name}_ep${ep}_zgeo.json"
      should_build "$report" || continue
      require "$ckpt"
      run "${PY[@]}" scripts/evaluate_geology_benchmark.py \
        --data "$FROZEN_VAL" \
        --manifest "$MANIFEST" \
        --checkpoint "$ckpt" \
        --use_geo_embedding \
        --metadata_keys "${METADATA_KEYS[@]}" \
        ${classifier_eval[@]+"${classifier_eval[@]}"} \
        --out_json "$report"
    done
  done
}

stage_summary() {
  printing && return 0
  "${PY[@]}" - "$(basename "$OUT_PREFIX")" <<'EOF'
import glob, json, sys
prefix = sys.argv[1]
print(f"{'report':<40} {'n@5':>7} {'n@10':>7} {'macroAUROC':>11} {'gate':>6}")
for path in sorted(glob.glob(f"docs/benchmarks/{prefix}_*_zgeo.json")):
    report = json.load(open(path))
    d = report["diagnostics"]
    c = report.get("classifier_metrics")
    auroc = f"{c['macro_auroc']:.3f}" if c and c.get("macro_auroc") is not None else "-"
    gate = str(c["sanity_gate"]["passed"]) if c else "-"
    print(f"{path.split('/')[-1]:<40} {d['neighbor_overlap_at_5']:>7.4f} {d['neighbor_overlap_at_10']:>7.4f} {auroc:>11} {gate:>6}")
print("Compare r1/r2/r3 against r1ctrl (the 0.139 baseline saw the validation volumes during training).")
EOF
}

case "$STAGE" in
  sample) stage_sample ;;
  train) stage_train ;;
  benchmark) stage_benchmark ;;
  summary) stage_summary ;;
  all) stage_sample; stage_train; stage_benchmark; stage_summary ;;
  *) echo "Unknown stage '$STAGE' (expected sample|train|benchmark|summary|all)" >&2; exit 2 ;;
esac
