#!/usr/bin/env bash
# Sample geology-aware patches (class-anchored labels + dip samples) for --data or --validation_data,
# and verify them. Training and validation MUST be produced by separate invocations of this script
# (SPLIT=train / SPLIT=validation) so their source synthoseis volumes are guaranteed disjoint and
# scripts/train.py's startup consistency check (assert_train_validation_consistency) can verify it.
#
# Usage:
#   SPLIT=train      scripts/sample_geoaware_patches.sh   # excludes SOURCE_ROOT/validation
#   SPLIT=validation scripts/sample_geoaware_patches.sh   # samples only from SOURCE_ROOT/validation
#
# Env overrides: PYTHON, SOURCE_ROOT, OUT, PATCH_SIZE, N_PATCHES, N_PER_VOLUME, SEED,
#   SAMPLING_MODE, LABEL_Z_OFFSET, DIP_SAMPLE_COUNT, VERIFY_COUNT, TRAIN_REF, LOG_DIR,
#   PRINT_COMMANDS=1
set -euo pipefail

cd "$(dirname "$0")/.."

SPLIT="${SPLIT:-train}"
PYTHON="${PYTHON:-.venv/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-/Volumes/CrucialX9/fake_data}"
PATCH_SIZE="${PATCH_SIZE:-32 32 64}"
SEED="${SEED:-20260929}"
LABEL_Z_OFFSET="${LABEL_Z_OFFSET:-1}"  # verified in WP0: seismic[z] <-> label[z + 1]
DIP_SAMPLE_COUNT="${DIP_SAMPLE_COUNT:-512}"
VERIFY_COUNT="${VERIFY_COUNT:-300}"
LOG_DIR="${LOG_DIR:-logs}"
LOG="${LOG_DIR}/sample_geoaware_patches_${SPLIT}_$(date +%Y%m%d_%H%M%S).log"
SCALE_ARGS=()

case "$SPLIT" in
  train)
    SOURCE_ARGS=(--source "$SOURCE_ROOT" --exclude_dir validation)
    OUT="${OUT:-data/synth_train_anchored_32-32-64.zarr}"
    N_PATCHES="${N_PATCHES:-108000}"
    N_PER_VOLUME="${N_PER_VOLUME:-600}"
    SAMPLING_MODE="${SAMPLING_MODE:-class_anchored}"
    ;;
  validation)
    SOURCE_ARGS=(--source "${SOURCE_ROOT}/validation")
    OUT="${OUT:-data/synth_val_anchored_32-32-64.zarr}"
    N_PATCHES="${N_PATCHES:-12000}"
    N_PER_VOLUME="${N_PER_VOLUME:-600}"
    # Natural-prevalence sampling is the documented convention for validation/classifier eval
    # (docs/plans/2026-09-25__..._plan.md Section 6); override with SAMPLING_MODE=class_anchored
    # only if you specifically want the training distribution mirrored for debugging.
    SAMPLING_MODE="${SAMPLING_MODE:-uniform}"
    TRAIN_REF="${TRAIN_REF:-data/synth_train_anchored_32-32-64.zarr}"
    if [[ ! -e "$TRAIN_REF" ]]; then
      echo "TRAIN_REF=$TRAIN_REF not found; sample the train split first, or set TRAIN_REF to an existing training zarr." >&2
      exit 1
    fi
    ;;
  *)
    echo "SPLIT must be 'train' or 'validation', got '$SPLIT'" >&2
    exit 1
    ;;
esac

printing() { [[ "${PRINT_COMMANDS:-0}" == "1" ]]; }

run_logged() {
  if printing; then
    printf '%q ' "$@"; printf '\n'
  else
    "$@" 2>&1 | tee -a "$LOG"
  fi
}

if ! printing; then
  mkdir -p "$LOG_DIR"
  echo "Logging to $LOG"
fi

if [[ "$SPLIT" == "validation" ]]; then
  # Reuse the training set's baked-in amplitude scaling so train/validation normalization matches
  # exactly (scripts/train.py's consistency check enforces this at training startup).
  read -r SCALING_MEAN SCALING_STD <<<"$("$PYTHON" - "$TRAIN_REF" <<'PY'
import sys, zarr
z = zarr.open(sys.argv[1], mode="r")
print(z.attrs["scaling_mean"], z.attrs["scaling_std"])
PY
)"
  SCALE_ARGS=(--no_derive_dataset_stats --dataset_mean "$SCALING_MEAN" --dataset_std "$SCALING_STD" --disjoint_from "$TRAIN_REF")
fi

class_quota_args=()
if [[ "$SAMPLING_MODE" == "class_anchored" ]]; then
  class_quota_args=(
    --class_quotas fault=0.10 fault_x=0.10 channel=0.10 closure=0.10 onlap=0.10 sand=0.05 flat_spot=0.10
    --background_fraction 0.25
    --max_patches_per_object 24
  )
fi

# shellcheck disable=SC2086
# ${arr[@]+"${arr[@]}"} avoids bash 3.2's "unbound variable" on an empty array under set -u (macOS default bash).
run_logged "$PYTHON" scripts/sample_patches.py \
  "${SOURCE_ARGS[@]+"${SOURCE_ARGS[@]}"}" \
  --patch_size $PATCH_SIZE \
  --n_patches "$N_PATCHES" \
  --n_per_volume "$N_PER_VOLUME" \
  --sampling_mode "$SAMPLING_MODE" \
  "${class_quota_args[@]+"${class_quota_args[@]}"}" \
  --presence_min_voxels 32 \
  --label_z_offset "$LABEL_Z_OFFSET" \
  --dip_sample_count "$DIP_SAMPLE_COUNT" \
  --seed "$SEED" \
  "${SCALE_ARGS[@]+"${SCALE_ARGS[@]}"}" \
  --out "$OUT"

# Re-derives dip samples from the source volumes and fails if they disagree with what was just
# written; this is an end-to-end check of the sampler + dip-sample-writing path, not a backfill.
run_logged "$PYTHON" scripts/add_dip_samples.py \
  --data "$OUT" \
  --dip_sample_count "$DIP_SAMPLE_COUNT" \
  --verify "$VERIFY_COUNT"

printing && exit 0

"$PYTHON" - "$OUT" <<'PY' 2>&1 | tee -a "$LOG"
import sys
import zarr

z = zarr.open(sys.argv[1], mode="r")
print("patches", z["patches"].shape)
print("n_written", z.attrs.get("n_written"))
print("source_volumes", len(z.attrs.get("source_volumes", [])))
print("skipped_volumes_missing_labels", len(z.attrs.get("skipped_volumes_missing_labels", [])))
print("dip_sample_count", z.attrs.get("dip_sample_count"))
print("label_class_order", z.attrs.get("label_class_order"))
print("scaling_mean", z.attrs.get("scaling_mean"), "scaling_std", z.attrs.get("scaling_std"))
PY
