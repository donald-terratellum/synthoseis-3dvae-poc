#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python="${PYTHON:-.venv/bin/python}"
seed="${SEED:-20260926}"
smoke="${SMOKE:-0}"
root="${OUT_ROOT:-/Volumes/CrucialX9/3dvae-poc_checkpoints/p9_mu512_seed${seed}}"
log="${LOG_PATH:-/Volumes/CrucialX9/3dvae-poc_logs/p9_mu512_seed${seed}.log}"
local_log="${LOCAL_LOG_PATH:-logs/p9_mu512_seed${seed}.log}"
source_initial="/Volumes/CrucialX9/3dvae-poc_checkpoints/p7_e1_seed${seed}_replication/initial/e1_initial.pt"
p0b_checkpoint="checkpoints/geoaware_p0b_r2_seed${seed}_r2/vae_best.pt"
p0b_benchmark="docs/benchmarks/geoaware_p0b_r2_seed${seed}_r2_ep30_zgeo.json"
e1_reconstruction="/Volumes/CrucialX9/3dvae-poc_checkpoints/p7_e1_seed${seed}_replication/e1/stage1_reconstruction/vae_best.pt"
e1_geology="/Volumes/CrucialX9/3dvae-poc_checkpoints/p7_e1_seed${seed}_replication/e1/stage2_geology/vae_epoch30.pt"
epochs=150
geology_epochs=40
batches=450
reconstruction_samples=5000
diagnostic_samples=32
diagnostic_interval=30
benchmark_milestones=(10 20 30 40)
metadata=(meta_fault_fraction meta_fault_intersection_fraction meta_channel_fraction
  meta_channel_core_fraction meta_flat_spot_fraction meta_onlap_fraction meta_onlap_variability)
if [[ "$smoke" == 1 ]]; then
  root="${OUT_ROOT:-/Volumes/CrucialX9/3dvae-poc_checkpoints/p9_mu512_seed${seed}_smoke}"
  log="${LOG_PATH:-/Volumes/CrucialX9/3dvae-poc_logs/p9_mu512_seed${seed}_smoke.log}"
  local_log="${LOCAL_LOG_PATH:-logs/p9_mu512_seed${seed}_smoke.log}"
  epochs=1; geology_epochs=1; batches=2; reconstruction_samples=64; diagnostic_samples=2; diagnostic_interval=1
  benchmark_milestones=(1)
fi

common=(
  --data data/synth_train_anchored_32-32-64.zarr
  --validation_data data/synth_val_v2_uniform_32-32-64.zarr
  --real_validation_data data/real_validation_32-32-64.zarr
  --real_test_data data/real_test_32-32-64.zarr
  --device mps --patch_size 32 32 64 --latent_dim 512 --batch_size 12
  --number_batches "$batches" --seed "$seed"
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
  --reconstruction_diagnostic_interval "$diagnostic_interval" --reconstruction_diagnostic_samples "$diagnostic_samples"
  --reconstruction_diagnostic_modes clean extrema tokenizer --reconstruction_diagnostic_train
)
stage1=(
  -u -m scripts.train "${common[@]}" --resume "$root/initial/e1_mu512_initial.pt"
  --resume_epoch 0 --epochs "$epochs" --geology_classifier_weight 0.05
  --geology_contrastive_weight 0 --paired_sampling --save_training_state
  --out_dir "$root/e1_mu512/stage1_reconstruction"
)
stage2=(
  -u -m scripts.train "${common[@]}" --resume "$root/e1_mu512/stage1_reconstruction/vae_best.pt"
  --resume_epoch 0 --epochs "$geology_epochs" --geology_classifier_weight 0.1
  --geology_contrastive_weight 0.5 --geology_contrastive_temperature 0.2
  --geology_batch_sampler --geology_batch_background_fraction 0.05
  --geology_batch_hard_fraction 0.30 --geology_batch_min_negative_strata 2
  --geology_strata_source presence_labels
  --geology_batch_class_quota fault_x=1 flat_spot=1 channel=1
  --out_dir "$root/e1_mu512/stage2_geology"
)

print_command() { printf '%q ' "$@"; printf '\n'; }
if [[ "${PRINT_COMMANDS:-0}" == 1 ]]; then
  printf 'P9 mu512 run; seed=%s; smoke=%s; output=%s\n' "$seed" "$smoke" "$root"
  print_command "$python" -u -m scripts.p9_mu_scaling memory-probe --device mps --batch-size 12 \
    --latent-dims 512 2048 --out-json "$root/memory_preflight.json"
  print_command "$python" -u -m scripts.p9_mu_scaling initialize --source-checkpoint "$source_initial" \
    --output "$root/initial/e1_mu512_initial.pt" --latent-dim 512 --seed "$seed"
  print_command "$python" "${stage1[@]}"
  print_command "$python" -u -m scripts.diagnose_p5_p6_reconstruction --evaluation_only \
    --contract_audit --allow_experimental_latent_dim --input_mode tokenizer --device mps \
    --samples "$reconstruction_samples" --batch_size 2 --seed 20261003 \
    --checkpoints "p0b=$p0b_checkpoint" "mu128=$e1_reconstruction" \
    "mu512=$root/e1_mu512/stage1_reconstruction/vae_best.pt" \
    --out_json "$root/stage1_tokenizer_reconstruction.json"
  print_command "$python" -u -m scripts.p7_control tokenizer-gate \
    --root "$root/stage1_tokenizer_reconstruction.json" --candidate mu512 --control p0b
  print_command "$python" -u -m scripts.p9_mu_scaling reconstruction-gate \
    --report "$root/stage1_tokenizer_reconstruction.json" --candidate mu512 --controls p0b mu128
  print_command "$python" "${stage2[@]}"
  for epoch in "${benchmark_milestones[@]}"; do
    print_command "$python" -u -m scripts.evaluate_geology_benchmark --data data/synth_val_v2_32-32-64.zarr \
      --manifest docs/benchmarks/frozen_validation_manifest_v2.json \
      --checkpoint "$root/e1_mu512/stage2_geology/vae_epoch${epoch}.pt" --use_geo_embedding \
      --metadata_keys "${metadata[@]}" --classifier_data data/synth_val_v2_uniform_32-32-64.zarr \
      --classifier_threshold_data data/synth_train_anchored_32-32-64.zarr \
      --out_json "$root/mu512_ep${epoch}_zgeo.json"
    if [[ "$smoke" != 1 ]]; then
      print_command "$python" -u -m scripts.diagnose_p5_p6_reconstruction --evaluation_only \
        --contract_audit --allow_experimental_latent_dim --input_mode tokenizer --device mps \
        --samples 5000 --batch_size 2 --seed 20261003 \
        --checkpoints "p0b=$p0b_checkpoint" "mu128_ep30=$e1_geology" \
          "mu512_ep${epoch}=$root/e1_mu512/stage2_geology/vae_epoch${epoch}.pt" \
        --out_json "$root/mu512_ep${epoch}_tokenizer.json"
    fi
  done
  if [[ "$smoke" != 1 ]]; then
    print_command "$python" -u -m scripts.p9_mu_scaling finalize --root "$root" --seed "$seed" \
      --p0b-benchmark "$p0b_benchmark"
  fi
  printf 'TensorBoard: uv run tensorboard --logdir %q\n' "$root"
  exit 0
fi

if [[ ! -d /Volumes/CrucialX9 ]]; then printf 'CrucialX9 is not mounted\n' >&2; exit 1; fi
if [[ -e "$root" || -e "$log" || -e "$local_log" || -L "$local_log" ]]; then
  printf 'Refusing existing experiment output/log: %s\n' "$root" >&2
  exit 1
fi
for path in "$source_initial" "$e1_reconstruction" "$e1_geology" "$p0b_checkpoint" "$p0b_benchmark" \
  data/synth_train_anchored_32-32-64.zarr data/synth_val_v2_uniform_32-32-64.zarr \
  data/real_validation_32-32-64.zarr data/real_test_32-32-64.zarr \
  docs/benchmarks/frozen_validation_manifest_v2.json; do
  [[ -e "$path" ]] || { printf 'Missing required experiment input: %s\n' "$path" >&2; exit 1; }
done
mkdir -p "$root" "$(dirname "$log")" "$(dirname "$local_log")"
ln -s "$log" "$local_log"
exec > >(tee -a "$log") 2>&1
printf 'P9 mu512 started %s; seed=%s; smoke=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$seed" "$smoke"

"$python" - "$source_initial" "$e1_reconstruction" "$e1_geology" "$p0b_checkpoint" "$p0b_benchmark" <<'PY'
import json, sys, torch, zarr
from pathlib import Path
source, e1_recon, e1_geo, p0b, benchmark = map(Path, sys.argv[1:])
assert torch.backends.mps.is_available(), 'MPS is required by the approved batch/device recipe'
assert source.is_file() and e1_recon.is_file() and e1_geo.is_file()
assert p0b.is_file() and benchmark.is_file()
assert json.loads(Path('docs/benchmarks/frozen_validation_manifest_v2.json').read_text())['indices']
for checkpoint_path, expected_epoch in ((source, 0), (e1_recon, 139), (e1_geo, 30), (p0b, None)):
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    assert payload['latent_dim'] == 128 and payload['patch_shape'] == [32, 32, 64]
    if expected_epoch is not None:
        assert payload['model_config']['encoder_arch'] == 'residual'
        assert payload['model_config']['decoder_hidden_dims'] is None
        assert payload['epoch'] == expected_epoch, (checkpoint_path, payload['epoch'])
    if checkpoint_path == source:
        assert payload['geology_projection'] and payload['geology_classifier']
for path in ('data/synth_train_anchored_32-32-64.zarr', 'data/synth_val_v2_uniform_32-32-64.zarr',
             'data/real_validation_32-32-64.zarr', 'data/real_test_32-32-64.zarr'):
    store = zarr.open(path, mode='r')
    assert tuple(store['patches'].shape[1:]) == (32, 32, 64)
    assert store.attrs['scaling_mode'] == 'divide_by_std'
print('P9 PREFLIGHT PASS: MPS, seed-2 E1/P0b checkpoints, manifest, and patch stores')
PY

"$python" -u -m scripts.p9_mu_scaling memory-probe --device mps --batch-size 12 \
  --latent-dims 512 2048 --out-json "$root/memory_preflight.json"
"$python" - "$root/memory_preflight.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
sizes = {item['latent_dim']: item for item in report['results']}
if not sizes[512]['fits_required_batch']:
    raise SystemExit('512-D did not pass the full-batch MPS memory preflight; training will not start.')
print('2048-D preliminary full-batch MPS result:', json.dumps(sizes[2048], sort_keys=True))
PY

"$python" -u -m scripts.p9_mu_scaling initialize \
  --source-checkpoint "$source_initial" --output "$root/initial/e1_mu512_initial.pt" \
  --latent-dim 512 --seed "$seed"
"$python" "${stage1[@]}"
"$python" -u -m scripts.diagnose_p5_p6_reconstruction --evaluation_only --contract_audit \
  --allow_experimental_latent_dim --input_mode tokenizer --device mps \
  --samples "$reconstruction_samples" --batch_size 2 --seed 20261003 \
  --checkpoints "p0b=$p0b_checkpoint" "mu128=$e1_reconstruction" \
    "mu512=$root/e1_mu512/stage1_reconstruction/vae_best.pt" \
  --out_json "$root/stage1_tokenizer_reconstruction.json"
if [[ "$smoke" != 1 ]]; then
  "$python" -u -m scripts.p7_control tokenizer-gate --root "$root/stage1_tokenizer_reconstruction.json" \
    --candidate mu512 --control p0b
  "$python" -u -m scripts.p9_mu_scaling reconstruction-gate \
    --report "$root/stage1_tokenizer_reconstruction.json" --candidate mu512 --controls p0b mu128
  "$python" - "$root/latent_reconstruction_gate.json" <<'PY'
import json, sys
if not json.load(open(sys.argv[1]))['passed']:
    raise SystemExit('512-D failed the full-tokenizer reconstruction gate; geology training is stopped.')
PY
else
  printf 'SMOKE: stage-1 gate is not adoption evidence; geology startup smoke will continue.\n'
fi
"$python" "${stage2[@]}"

if [[ "$smoke" == 1 ]]; then
  "$python" -u -m scripts.evaluate_geology_benchmark --data data/synth_val_v2_32-32-64.zarr \
    --manifest docs/benchmarks/frozen_validation_manifest_v2.json \
    --checkpoint "$root/e1_mu512/stage2_geology/vae_epoch1.pt" --use_geo_embedding \
    --metadata_keys "${metadata[@]}" --classifier_data data/synth_val_v2_uniform_32-32-64.zarr \
    --classifier_threshold_data data/synth_train_anchored_32-32-64.zarr \
    --out_json "$root/mu512_ep1_zgeo_smoke.json"
  printf 'P9 mu512 smoke complete; no full gate, adoption, or 2048 training was run.\n'
  exit 0
fi

for epoch in "${benchmark_milestones[@]}"; do
  "$python" -u -m scripts.evaluate_geology_benchmark --data data/synth_val_v2_32-32-64.zarr \
    --manifest docs/benchmarks/frozen_validation_manifest_v2.json \
    --checkpoint "$root/e1_mu512/stage2_geology/vae_epoch${epoch}.pt" --use_geo_embedding \
    --metadata_keys "${metadata[@]}" --classifier_data data/synth_val_v2_uniform_32-32-64.zarr \
    --classifier_threshold_data data/synth_train_anchored_32-32-64.zarr \
    --out_json "$root/mu512_ep${epoch}_zgeo.json"
  "$python" -u -m scripts.diagnose_p5_p6_reconstruction --evaluation_only --contract_audit \
    --allow_experimental_latent_dim --input_mode tokenizer --device mps \
    --samples 5000 --batch_size 2 --seed 20261003 \
    --checkpoints "p0b=$p0b_checkpoint" "mu128_ep30=$e1_geology" \
      "mu512_ep${epoch}=$root/e1_mu512/stage2_geology/vae_epoch${epoch}.pt" \
    --out_json "$root/mu512_ep${epoch}_tokenizer.json"
done
"$python" -u -m scripts.p9_mu_scaling finalize --root "$root" --seed "$seed" \
  --p0b-benchmark "$p0b_benchmark"
printf 'P9 mu512 screening complete. Compare TensorBoard runs with:\nuv run tensorboard --logdir %s\n' "$root"
printf 'No 2048-D training was launched; use memory_preflight.json and the 512-D decision to decide next.\n'
