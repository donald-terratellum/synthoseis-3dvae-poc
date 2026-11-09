#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python="${PYTHON:-.venv/bin/python}"
log="${LOG_PATH:-logs/pre_p7_readiness_audit_20261004.log}"
reports="${REPORT_DIR:-/Volumes/CrucialX9/3dvae-poc_logs/pre_p7_readiness_audit_20261004}"
device="${DEVICE:-mps}"
samples="${SAMPLES:-128}"
control="checkpoints/geoaware_p0b_r2_seed20260925_r2/vae_best.pt"
candidate="/Volumes/CrucialX9/3dvae-poc_checkpoints/p5a_reconstruction_150_seed20260925/vae_epoch150.pt"
if [[ ! -d /Volumes/CrucialX9 ]]; then printf 'CrucialX9 is not mounted\n' >&2; exit 1; fi
if [[ -e "$log" || -e "$reports" ]]; then printf 'Refusing existing audit log/report directory\n' >&2; exit 1; fi
mkdir -p "$(dirname "$log")" "$reports"
{
  printf 'Pre-P7 readiness audit started %s; evaluation only; no training\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  for input_mode in clean extrema; do
    for split in train validation; do
      data="data/synth_val_v2_uniform_32-32-64.zarr"
      if [[ "$split" == train ]]; then data="data/synth_train_anchored_32-32-64.zarr"; fi
      printf 'AUDIT input_mode=%s synthetic_split=%s samples=%s\n' "$input_mode" "$split" "$samples"
      "$python" -u -m scripts.diagnose_p5_p6_reconstruction --evaluation_only \
        --device "$device" --samples "$samples" --batch_size 2 --input_mode "$input_mode" \
        --synthetic_data "$data" --audit_training_data data/synth_train_anchored_32-32-64.zarr \
        --checkpoints "p0b=$control" "p5a150=$candidate" \
        --out_json "$reports/${split}_${input_mode}.json"
    done
  done
  printf 'COMPLETE: parity, train/validation diagnostics and theoretical exposure audit; no new weights trained.\n'
} 2>&1 | tee "$log"