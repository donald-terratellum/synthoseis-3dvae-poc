#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python="${PYTHON:-.venv/bin/python}"
seed=20260926
checkpoint_root="${CHECKPOINT_ROOT:-/Volumes/CrucialX9/3dvae-poc_checkpoints}"
p9_root="${P9_ROOT:-${checkpoint_root}/p9_mu512_seed${seed}}"
out="${OUT_ROOT:-${p9_root}/phase2_extension_110ep}"
source_checkpoint="${SOURCE_CHECKPOINT:-${p9_root}/e1_mu512/stage2_geology/vae_epoch40.pt}"
p0b_checkpoint="${P0B_CHECKPOINT:-checkpoints/geoaware_p0b_r2_seed${seed}_r2/vae_best.pt}"
p0b_benchmark="${P0B_BENCHMARK:-docs/benchmarks/geoaware_p0b_r2_seed${seed}_r2_ep30_zgeo.json}"
e1_reconstruction="${E1_RECONSTRUCTION:-${checkpoint_root}/p7_e1_seed${seed}_replication/e1/stage1_reconstruction/vae_best.pt}"
e1_geology="${E1_GEOLOGY:-${checkpoint_root}/p7_e1_seed${seed}_replication/e1/stage2_geology/vae_epoch30.pt}"
train_data="${TRAIN_DATA:-data/synth_train_anchored_32-32-64.zarr}"
validation_data="${VALIDATION_DATA:-data/synth_val_v2_uniform_32-32-64.zarr}"
benchmark_data="${BENCHMARK_DATA:-data/synth_val_v2_32-32-64.zarr}"
real_validation_data="${REAL_VALIDATION_DATA:-data/real_validation_32-32-64.zarr}"
real_test_data="${REAL_TEST_DATA:-data/real_test_32-32-64.zarr}"
manifest="${MANIFEST:-docs/benchmarks/frozen_validation_manifest_v2.json}"
device="${DEVICE:-auto}"

case "$device" in
  auto)
    device="$("$python" -c 'import torch; print("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))')"
    ;;
  cuda|mps|cpu) ;;
  *) printf 'DEVICE must be auto, cuda, mps, or cpu (got %s)\n' "$device" >&2; exit 2 ;;
esac
if [[ "$device" == cuda ]] && ! "$python" -c 'import torch; assert torch.cuda.is_available()'; then
  printf 'CUDA was requested but is unavailable in this Python environment.\n' >&2
  exit 1
fi
if [[ "$device" == mps ]] && ! "$python" -c 'import torch; assert torch.backends.mps.is_available()'; then
  printf 'MPS was requested but is unavailable in this Python environment.\n' >&2
  exit 1
fi

stage1_dir="${p9_root}/e1_mu512/stage1_reconstruction"
stage2_dir="${out}/e1_mu512/stage2_geology"
eval_epochs=(50 60 70 80 90 100 110 120 130 140 150)
metadata=(
  meta_fault_fraction meta_fault_intersection_fraction meta_channel_fraction
  meta_channel_core_fraction meta_flat_spot_fraction meta_onlap_fraction meta_onlap_variability
)

for path in "$source_checkpoint" "$stage1_dir/vae_best.pt" \
  "$p9_root/stage1_tokenizer_reconstruction.json" "$p9_root/latent_reconstruction_gate.json" \
  "$p0b_checkpoint" "$p0b_benchmark" "$e1_reconstruction" "$e1_geology" \
  "$train_data" "$validation_data" "$benchmark_data" "$real_validation_data" "$real_test_data" "$manifest"; do
  [[ -e "$path" ]] || { printf 'Missing required extension input: %s\n' "$path" >&2; exit 1; }
done
if [[ -e "$out" || -L "$out" ]]; then
  printf 'Refusing existing continuation output: %s\n' "$out" >&2
  exit 1
fi
if [[ -e "${LOG_PATH:-${out}/phase2_extension.log}" ]]; then
  printf 'Refusing existing log path: %s\n' "${LOG_PATH:-${out}/phase2_extension.log}" >&2
  exit 1
fi

"$python" - "$source_checkpoint" <<'PY'
import sys
import torch
from pathlib import Path

path = Path(sys.argv[1])
checkpoint = torch.load(path, map_location='cpu', weights_only=True)
assert int(checkpoint.get('latent_dim', -1)) == 512, 'source checkpoint must be mu512'
assert int(checkpoint.get('epoch', -1)) == 40, 'source checkpoint must be stage-2 epoch 40'
assert tuple(checkpoint.get('patch_shape', ())) == (32, 32, 64), 'unexpected patch shape'
print(f'Validated continuation source: {path} (mu512, stage-2 epoch 40)')
PY

common=(
  --data "$train_data" --validation_data "$validation_data"
  --real_validation_data "$real_validation_data" --real_test_data "$real_test_data"
  --device "$device" --patch_size 32 32 64 --latent_dim 512 --batch_size 12
  --number_batches 450 --seed "$seed"
  --encoder_arch residual --encoder_norm batch --encoder_input_axes xyz --decoder_block conv
  --freeze_encoder_epochs 0 --augment --vertical_warp_prob 0.5
  --phase_rotation_prob 0 --stretch_prob 0 --dip_label_policy adjust --mixup_augment_prob 0
  --input_scaling divide_by_std --input_extrema_prob 1 --input_sparse_keep_prob 0
  --input_decimate_trilinear_prob 0 --validation_extrema_only
  --learning_rate 5e-4 --weight_decay 1e-4 --encoder_lr_mult 0.1 --grad_clip 2
  --kl_schedule warmup --kl_start 1e-3 --kl_end 1e-3 --kl_warmup_epochs 1
  --reconstruction_loss mae --lpips_weight 0.1 --lr_scheduler plateau
  --lr_scheduler_patience 6 --lr_scheduler_factor 0.5 --lr_scheduler_min_lr 1e-5
  --early_stopping_patience 999 --save_epoch_checkpoints --best_checkpoint_name vae_best.pt
  --geology_projection --geology_proj_hidden 128 --geology_proj_dim 64
  --geology_uniformity_weight 0 --geology_classifier --geology_classifier_mode patch
  --geology_classifier_loss focal --geology_classifier_focal_gamma 2
  --geology_classifier_label_smoothing 0.05 --geology_metadata_keys "${metadata[@]}"
  --reconstruction_diagnostic_interval 30 --reconstruction_diagnostic_samples 32
  --reconstruction_diagnostic_modes clean extrema tokenizer --reconstruction_diagnostic_train
)
train_command=(
  "$python" -u -m scripts.train "${common[@]}"
  --resume "$source_checkpoint" --resume_epoch 40 --epochs 110
  --geology_classifier_weight 0.1 --geology_contrastive_weight 0.5
  --geology_contrastive_temperature 0.2 --geology_batch_sampler
  --geology_batch_background_fraction 0.05 --geology_batch_hard_fraction 0.30
  --geology_batch_min_negative_strata 2 --geology_strata_source presence_labels
  --geology_batch_class_quota fault_x=1 flat_spot=1 channel=1
  --out_dir "$stage2_dir"
)

if [[ "${PRINT_COMMANDS:-0}" == 1 ]]; then
  printf 'Continuation output: %s\n' "$out"
  printf 'Training device: %s; resume epoch: 40; additional epochs: 110; final epoch: 150\n' "$device"
  printf 'WARNING: model weights continue from epoch 40, but optimizer/scheduler state is reset.\n'
  printf '%q ' "${train_command[@]}"; printf '\n'
  for epoch in "${eval_epochs[@]}"; do
    printf 'Evaluate reconstruction and retrieval at epoch %s\n' "$epoch"
  done
  exit 0
fi

mkdir -p "$out"
log="${LOG_PATH:-${out}/phase2_extension.log}"
exec > >(tee -a "$log") 2>&1
printf 'P9 mu512 exploratory phase-2 extension; seed=%s; source=%s; device=%s\n' \
  "$seed" "$source_checkpoint" "$device"
printf 'Purpose: human curiosity/completeness, not a fair apples-to-apples experiment.\n'

"$python" -u -m scripts.p9_mu_scaling memory-probe --device "$device" --batch-size 12 \
  --latent-dims 512 --out-json "$out/memory_preflight.json"
"$python" - "$out/memory_preflight.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
if not report['results'][0]['fits_required_batch']:
    raise SystemExit('mu512 batch-12 memory preflight failed; continuation stopped.')
PY

mkdir -p "${out}/e1_mu512"
ln -s "$stage1_dir" "${out}/e1_mu512/stage1_reconstruction"
cp "$p9_root/stage1_tokenizer_reconstruction.json" "$out/stage1_tokenizer_reconstruction.json"
cp "$p9_root/latent_reconstruction_gate.json" "$out/latent_reconstruction_gate.json"

"${train_command[@]}"

for epoch in "${eval_epochs[@]}"; do
  checkpoint="${stage2_dir}/vae_epoch${epoch}.pt"
  [[ -f "$checkpoint" ]] || { printf 'Missing expected checkpoint: %s\n' "$checkpoint" >&2; exit 1; }
  "$python" -u -m scripts.evaluate_geology_benchmark \
    --data "$benchmark_data" --manifest "$manifest" --device "$device" \
    --checkpoint "$checkpoint" --use_geo_embedding --metadata_keys "${metadata[@]}" \
    --classifier_data "$validation_data" --classifier_threshold_data "$train_data" \
    --out_json "$out/mu512_ep${epoch}_zgeo.json"
  "$python" -u -m scripts.diagnose_p5_p6_reconstruction \
    --evaluation_only --contract_audit --allow_experimental_latent_dim \
    --input_mode tokenizer --device "$device" --samples 5000 --batch_size 2 \
    --seed 20261003 \
    --checkpoints "p0b=$p0b_checkpoint" "mu128_ep30=$e1_geology" \
      "mu512_ep${epoch}=$checkpoint" \
    --out_json "$out/mu512_ep${epoch}_tokenizer.json"
done

"$python" -u -m scripts.p9_mu_scaling finalize --root "$out" --seed "$seed" \
  --p0b-benchmark "$p0b_benchmark" --epochs "${eval_epochs[@]}"
"$python" - "$out/p9_mu512_decision.json" "$source_checkpoint" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
decision = json.loads(path.read_text())
decision['continuation_context'] = {
    'purpose': 'human curiosity and completeness after the original screening decision',
    'fair_apples_to_apples_experiment': False,
    'source_checkpoint': sys.argv[2],
    'additional_stage2_epochs': 110,
    'stage2_epoch_range': [41, 150],
    'optimizer_and_scheduler_state': 'reinitialized; not restored from the epoch-40 run',
    'interpretation': 'Exploratory continuation only; do not treat as a pre-registered matched screening result.',
}
path.write_text(json.dumps(decision, indent=2, allow_nan=False) + '\n')
print(json.dumps(decision, indent=2))
PY

printf 'Exploratory phase-2 extension complete. Results: %s\n' "$out"
printf 'TensorBoard: uv run tensorboard --logdir %q\n' "$out"
