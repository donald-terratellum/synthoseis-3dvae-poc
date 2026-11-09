#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python="${PYTHON:-.venv/bin/python}"
out="${OUT_DIR:-/Volumes/CrucialX9/3dvae-poc_checkpoints/p5a_reconstruction_150_seed20260925}"
log="${LOG_PATH:-/Volumes/CrucialX9/3dvae-poc_logs/p5a_reconstruction_150_seed20260925.log}"
local_log="logs/p5a_reconstruction_150_seed20260925.log"
source_checkpoint="checkpoints/p5_p6_encoder/p5a_seed20260925/stage1_reconstruction/vae_epoch60.pt"
control="checkpoints/geoaware_p0b_r2_seed20260925_r2/vae_best.pt"
metadata_keys=(meta_fault_fraction meta_fault_intersection_fraction meta_channel_fraction
  meta_channel_core_fraction meta_flat_spot_fraction meta_onlap_fraction meta_onlap_variability)
args=(
  -u -m scripts.train --resume "$source_checkpoint" --resume_epoch 60 --epochs 90
  --data data/synth_train_anchored_32-32-64.zarr --validation_data data/synth_val_v2_uniform_32-32-64.zarr
  --real_validation_data data/real_validation_32-32-64.zarr --real_test_data data/real_test_32-32-64.zarr
  --device mps --patch_size 32 32 64 --batch_size 12 --number_batches 450 --seed 20260925
  --encoder_arch resnetv2 --encoder_hidden_dims 32 64 128 --encoder_depth_profile deeper
  --encoder_stage_blocks 3 5 8 --encoder_norm instance --encoder_stem pretrain_v2 --encoder_input_axes zxy
  --decoder_hidden_dims 256 128 64 16 --decoder_block res --freeze_encoder_epochs 0
  --augment --vertical_warp_prob 0.5 --phase_rotation_prob 0 --stretch_prob 0 --dip_label_policy adjust
  --mixup_augment_prob 0 --input_scaling divide_by_std --learning_rate 0.00025 --encoder_lr_mult 0.1
  --weight_decay 1e-4 --kl_schedule warmup --kl_start 1e-3 --kl_end 1e-3 --kl_warmup_epochs 1
  --reconstruction_loss mae --lpips_weight 0.1 --lr_scheduler plateau --lr_scheduler_patience 6
  --lr_scheduler_factor 0.5 --lr_scheduler_min_lr 1e-5 --early_stopping_patience 999 --save_epoch_checkpoints
  --geology_projection --geology_proj_hidden 128 --geology_proj_dim 64
  --geology_contrastive_weight 0 --geology_uniformity_weight 0
  --geology_classifier --geology_classifier_mode patch --geology_classifier_weight 0.05
  --geology_classifier_loss focal --geology_classifier_focal_gamma 2 --geology_classifier_label_smoothing 0.05
  --geology_metadata_keys "${metadata_keys[@]}" --best_checkpoint_name vae_best.pt --out_dir "$out"
  --reconstruction_diagnostic_interval 30 --reconstruction_diagnostic_samples 32
)
if [[ "${PRINT_COMMANDS:-0}" == 1 ]]; then
  printf '%q ' "$python" "${args[@]}"; printf '\n'
  exit 0
fi
if [[ ! -d /Volumes/CrucialX9 ]]; then
  printf 'External drive /Volumes/CrucialX9 is not mounted.\n' >&2
  exit 1
fi
mkdir -p "$(dirname "$out")" "$(dirname "$log")" logs
if [[ "$log" != /* ]]; then log="$PWD/$log"; fi
if [[ -e "$out" || -e "$log" || -e "$local_log" || -L "$local_log" ]]; then
  printf 'Refusing to overwrite existing output or log: %s %s\n' "$out" "$log" >&2
  exit 1
fi
ln -s "$log" "$local_log"
{
  printf 'P5a reconstruction extension started %s; epochs 61-150; no geology stage\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf 'Weight continuation: optimizer moments and scheduler history were not saved; both restart. Encoder LR=2.5e-5, decoder/head LR=2.5e-4.\n'
  "$python" - "$source_checkpoint" "$control" "$(dirname "$out")" <<'PY'
import csv, shutil, sys
from pathlib import Path
import torch, zarr
source, control, output_parent = map(Path, sys.argv[1:])
checkpoint = torch.load(source, map_location='cpu', weights_only=False)
assert checkpoint['epoch'] == 60, 'expected epoch-60 reconstruction checkpoint'
with (source.parent / 'training_metrics.csv').open() as stream:
    last = list(csv.DictReader(stream))[-1]
assert int(last['epoch']) == 60 and abs(float(last['learning_rate']) - 2.5e-5) < 1e-12
assert control.is_file(), 'missing P0b reconstruction control'
assert torch.backends.mps.is_available(), 'MPS required for this long run'
for path in ('data/synth_train_anchored_32-32-64.zarr', 'data/synth_val_v2_uniform_32-32-64.zarr',
             'data/real_validation_32-32-64.zarr', 'data/real_test_32-32-64.zarr'):
    root = zarr.open(path, mode='r')
    assert tuple(root['patches'].shape[1:]) == (32, 32, 64)
    assert root.attrs['scaling_mode'] == 'divide_by_std'
assert shutil.disk_usage(output_parent).free > 15 * 1024 ** 3, 'need at least 15 GiB free on checkpoint drive'
print('PREFLIGHT PASS: model epoch, final LR, control, data, MPS, disk checked', flush=True)
PY
  "$python" -u -m scripts.diagnose_p5_p6_reconstruction --evaluation_only --device mps --samples 32 \
    --checkpoints "p0b=$control" "p5a_epoch60=$source_checkpoint" \
    --out_json "${log%.log}_baseline.json"
  "$python" "${args[@]}"
  "$python" -u -m scripts.diagnose_p5_p6_reconstruction --evaluation_only --device mps --samples 5000 --batch_size 2 \
    --checkpoints "p0b=$control" "p5a_best=$out/vae_best.pt" "p5a_epoch150=$out/vae_epoch150.pt" \
    --out_json "${log%.log}_full_reconstruction_gate.json"
  printf 'COMPLETE: reconstruction extension and full-set gate evaluated; no geology stage launched.\n'
} 2>&1 | tee "$log"