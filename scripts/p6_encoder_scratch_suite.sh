#!/usr/bin/env bash
# P6: E2 encoder from scratch, matched to the P5a two-stage training recipe.
# Env: SEED, RECON_EPOCHS, GEOLOGY_EPOCHS, BATCHES, OUT_ROOT, PRINT_COMMANDS=1
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/p5_p6_encoder_common.sh

run_encoder_arm p6 "" 0.05
