#!/usr/bin/env bash
# WP0 (docs/plans/2026-09-27__pretrain_v2_encoder_loss_augmentation_transfer_plan.md, Section 8):
# generate 25 new leak-free labeled synthoseis volumes (runs 5000-5024), sample the two v2
# validation patch sets, and freeze the v2 benchmark manifest (also produces the P0 reference
# number: the adopted checkpoint re-benchmarked on the v2 manifest).
#
# Usage: scripts/wp0_generate_v2_validation.sh
# Env overrides: PYTHON, SYNTHOSEIS_DIR, GEN_SCRIPT, STAGING_ROOT, V2_ROOT, START_INDEX,
#   N_VOLUMES, LABEL_Z_OFFSET, DIP_SAMPLE_COUNT, WARM_START, MANIFEST_SIZE, LOG_DIR,
#   PRINT_COMMANDS=1
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
SYNTHOSEIS_DIR="${SYNTHOSEIS_DIR:-$HOME/synthoseis/synthoseis}"
GEN_SCRIPT="${GEN_SCRIPT:-/Users/donaldpg/synthoseis-pretrain-v2/generate_datasets.sh}"
BASE_CONFIG="${BASE_CONFIG:-$SYNTHOSEIS_DIR/config/example_bigger_ex.json}"
# CrucialX9 is ExFAT, which has no os.link (hard link) support; zarr v3's atomic write needs it.
# Generate on local APFS storage (same convention as synthoseis-pretrain-v2's replay tool), then
# move each finished dataset onto CrucialX9 with a plain copy+delete (no hard link involved).
STAGING_ROOT="${STAGING_ROOT:-$HOME/synthoseis/fake_data_staging}"
V2_ROOT="${V2_ROOT:-/Volumes/CrucialX9/fake_data_validation_v2}"
START_INDEX="${START_INDEX:-5000}"
N_VOLUMES="${N_VOLUMES:-25}"
LABEL_Z_OFFSET="${LABEL_Z_OFFSET:-1}"  # verified in WP0: seismic[z] <-> label[z + 1]
DIP_SAMPLE_COUNT="${DIP_SAMPLE_COUNT:-512}"
WARM_START="${WARM_START:-checkpoints/geoaware_v3_phase2_20260831/vae_epoch20.pt}"
MANIFEST_SIZE="${MANIFEST_SIZE:-2048}"
LOG_DIR="${LOG_DIR:-logs}"
LOG="${LOG_DIR}/wp0_generate_v2_validation_$(date +%Y%m%d_%H%M%S).log"
METADATA_KEYS=(
  meta_fault_fraction meta_fault_intersection_fraction
  meta_channel_fraction meta_channel_core_fraction
  meta_flat_spot_fraction meta_onlap_fraction meta_onlap_variability
)
REQUIRED_LABEL_KEYS=(
  fault_segments_id fault_intersection_segments "faults/faulted_channel_labels"
  closure_segments_id onlap_segments faulted_lithology flat_spot geologic_age_faulted
)

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

printing || mkdir -p "$STAGING_ROOT" "$V2_ROOT"

# The synthoseis config hardcodes project_folder/work_folder (main.py has no CLI override for
# this). Write a copy pointed at STAGING_ROOT, matching synthoseis-pretrain-v2's
# studies/recreate_datasets_with_segmentation.py::_write_overridden_config.
OVERRIDDEN_CONFIG="$STAGING_ROOT/_generation_config_v2.json"
printing || "$PYTHON" - "$BASE_CONFIG" "$OVERRIDDEN_CONFIG" "$STAGING_ROOT" <<'PY'
import json, sys
src, dst, root = sys.argv[1], sys.argv[2], sys.argv[3]
payload = json.loads(open(src, encoding="utf-8").read())
payload["project_folder"] = root
payload["work_folder"] = root
open(dst, "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True))
print(f"Wrote {dst} with project_folder/work_folder={root}")
PY

# volume_exists ROOT INDEX: true if a run with this index has a model_data.zarr under ROOT.
volume_exists() {
  printing && return 1
  compgen -G "${1}/seismic__*__synthoseis_run_$(printf '%04d' "$2")/model_data.zarr" > /dev/null
}

# promote_volume INDEX: move a finished dataset from staging (APFS) to V2_ROOT (ExFAT, CrucialX9).
promote_volume() {
  local idx tag match
  idx="$1"
  tag="synthoseis_run_$(printf '%04d' "$idx")"
  printing && return 0
  match=$(compgen -G "${STAGING_ROOT}/seismic__*__${tag}" | head -n 1) || {
    echo "ERROR: no finished dataset found in $STAGING_ROOT for $tag" >&2
    exit 1
  }
  echo "Promoting $match -> $V2_ROOT/" | tee -a "$LOG"
  mv "$match" "$V2_ROOT/"
  rm -rf "${STAGING_ROOT:?}"/temp_folder__*"${tag}"  # partial-run staging litter, if any
}

# --- 1) Generate volumes one at a time on local storage, then promote each onto CrucialX9 ---
for ((idx = START_INDEX; idx < START_INDEX + N_VOLUMES; idx++)); do
  tag_padded=$(printf '%04d' "$idx")
  if volume_exists "$V2_ROOT" "$idx"; then
    echo "Skipping generation of run $tag_padded: already present under $V2_ROOT" | tee -a "$LOG"
    continue
  fi
  if ! volume_exists "$STAGING_ROOT" "$idx"; then
    run_logged "$GEN_SCRIPT" -n 1 \
      --synthoseis-dir "$SYNTHOSEIS_DIR" \
      --config "$OVERRIDDEN_CONFIG" \
      -d "$STAGING_ROOT" \
      --start-index "$idx"
  fi
  promote_volume "$idx"

  if [[ "$idx" -eq "$START_INDEX" ]]; then
    printing || "$PYTHON" - "$V2_ROOT" "$START_INDEX" "${REQUIRED_LABEL_KEYS[@]}" <<'PY' 2>&1 | tee -a "$LOG"
import sys, glob, zarr

root, start_index, required = sys.argv[1], sys.argv[2], sys.argv[3:]
pattern = f"{root}/seismic__*__synthoseis_run_{int(start_index):04d}/model_data.zarr"
matches = sorted(glob.glob(pattern))
if not matches:
    raise SystemExit(f"No volume found matching {pattern}; generation may have failed.")
z = zarr.open(matches[0], mode="r")
missing = [k for k in required if k not in z]
if missing:
    raise SystemExit(f"{matches[0]} is missing label arrays: {missing}")
print(f"OK: {matches[0]} has every required label array.")
PY
  fi
done

# --- 3) Verify label_z_offset on a sample of the new volumes ---
run_logged "$PYTHON" scripts/verify_label_alignment.py \
  --source "$V2_ROOT" \
  --n_volumes 5

# --- 4) Sample the two v2 validation patch sets ---
run_logged "$PYTHON" scripts/sample_patches.py \
  --source "$V2_ROOT" \
  --patch_size 32 32 64 \
  --n_patches 5000 --n_per_volume 200 \
  --sampling_mode uniform \
  --label_z_offset "$LABEL_Z_OFFSET" \
  --dip_sample_count "$DIP_SAMPLE_COUNT" \
  --seed 20260927 \
  --out data/synth_val_v2_32-32-64.zarr

run_logged "$PYTHON" scripts/sample_patches.py \
  --source "$V2_ROOT" \
  --patch_size 32 32 64 \
  --n_patches 5000 --n_per_volume 200 \
  --sampling_mode uniform \
  --label_z_offset "$LABEL_Z_OFFSET" \
  --dip_sample_count "$DIP_SAMPLE_COUNT" \
  --seed 20260928 \
  --out data/synth_val_v2_uniform_32-32-64.zarr

# --- 5) Freeze the v2 manifest; also produces the P0 reference (adopted checkpoint on v2) ---
run_logged "$PYTHON" scripts/evaluate_geology_benchmark.py \
  --data data/synth_val_v2_32-32-64.zarr \
  --manifest docs/benchmarks/frozen_validation_manifest_v2.json \
  --benchmark_size "$MANIFEST_SIZE" \
  --seed 20260927 \
  --checkpoint "$WARM_START" \
  --use_geo_embedding \
  --metadata_keys "${METADATA_KEYS[@]}" \
  --out_json docs/benchmarks/adopted_v2_reference.json
