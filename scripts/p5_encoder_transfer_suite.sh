#!/usr/bin/env bash
# P5a/P5b: E2 encoder initialized from the pretrain-v2 r006 EMA weights.
# Env: SEED, RECON_EPOCHS, GEOLOGY_EPOCHS, BATCHES, OUT_ROOT, PRINT_COMMANDS=1
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/p5_p6_encoder_common.sh

PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-/Volumes/CrucialX9/pretrain_v2_checkpoints/sweep_20260620_080306_r006_u3_h32-64-128_lp0p000_tv0p001/best_val_epoch.pt}"
run_encoder_arm p5a "$PRETRAIN_CHECKPOINT" 0.05
run_encoder_arm p5b "$PRETRAIN_CHECKPOINT" 0.0
