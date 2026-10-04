#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
LOG_PATH="${LOG_PATH:-logs/p5_p6_reconstruction_diagnostics.log}"
mkdir -p "$(dirname "$LOG_PATH")"
{
  printf 'P5/P6 reconstruction diagnostic started: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  "${PYTHON:-.venv/bin/python}" -u -m scripts.diagnose_p5_p6_reconstruction \
    --device "${DEVICE:-cpu}" --samples "${SAMPLES:-32}" \
    --fit_steps "${FIT_STEPS:-40}" --batch_size "${BATCH_SIZE:-2}" \
    --seed "${SEED:-20261003}" \
    --out_json "${OUT_JSON:-logs/p5_p6_reconstruction_diagnostics.json}"
} 2>&1 | tee "$LOG_PATH"