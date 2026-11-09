#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python="${PYTHON:-.venv/bin/python}"
seed="${SEED:-20260925}"
smoke="${SMOKE:-0}"
recon_epochs=150
batches=450
root="${OUT_ROOT:-/Volumes/CrucialX9/3dvae-poc_checkpoints/p7_e0_e1_seed${seed}}"
log_root="${LOG_ROOT:-/Volumes/CrucialX9/3dvae-poc_logs}"
local_log="${LOCAL_LOG_PATH:-logs/p7_e0_e1_seed${seed}.log}"
if [[ "$smoke" == 1 ]]; then
  recon_epochs=1
  batches=2
  root="${OUT_ROOT:-/Volumes/CrucialX9/3dvae-poc_checkpoints/p7_smoke_seed${seed}}"
  local_log="${LOCAL_LOG_PATH:-logs/p7_smoke_seed${seed}.log}"
fi
suite_log="$log_root/$(basename "$local_log")"
printing="${PRINT_COMMANDS:-0}"
metadata=(meta_fault_fraction meta_fault_intersection_fraction meta_channel_fraction
  meta_channel_core_fraction meta_flat_spot_fraction meta_onlap_fraction meta_onlap_variability)
common=(
  --data data/synth_train_anchored_32-32-64.zarr --validation_data data/synth_val_v2_uniform_32-32-64.zarr
  --real_validation_data data/real_validation_32-32-64.zarr --real_test_data data/real_test_32-32-64.zarr
  --device mps --patch_size 32 32 64 --batch_size 12 --number_batches "$batches" --seed "$seed"
  --encoder_norm batch --encoder_input_axes xyz --decoder_block conv --freeze_encoder_epochs 0
  --augment --vertical_warp_prob 0.5 --phase_rotation_prob 0 --stretch_prob 0 --dip_label_policy adjust --mixup_augment_prob 0
  --input_scaling divide_by_std --input_extrema_prob 1 --input_sparse_keep_prob 0 --input_decimate_trilinear_prob 0 --validation_extrema_only
  --learning_rate 5e-4 --weight_decay 1e-4 --encoder_lr_mult 0.1
  --kl_schedule warmup --kl_start 1e-3 --kl_end 1e-3 --kl_warmup_epochs 1
  --reconstruction_loss mae --lpips_weight 0.1 --lr_scheduler plateau --lr_scheduler_patience 6 --lr_scheduler_factor 0.5 --lr_scheduler_min_lr 1e-5
  --early_stopping_patience 999 --save_epoch_checkpoints --geology_projection --geology_proj_hidden 128 --geology_proj_dim 64
  --geology_classifier --geology_classifier_mode patch --geology_classifier_loss focal --geology_classifier_focal_gamma 2 --geology_classifier_label_smoothing 0.05
  --geology_uniformity_weight 0 --geology_metadata_keys "${metadata[@]}" --best_checkpoint_name vae_best.pt
  --reconstruction_diagnostic_interval 30 --reconstruction_diagnostic_samples 32 --reconstruction_diagnostic_modes clean extrema --reconstruction_diagnostic_train
)
run_command() {
  local destination="$1"; shift
  if [[ "$printing" == 1 ]]; then
    printf '%q ' "$@"; printf '\n'
  elif [[ "$destination" == "$suite_log" ]]; then
    "$@"
  else
    "$@" 2>&1 | tee -a "$destination"
  fi
}
run_suite() {
  printf 'P7 seed=%s smoke=%s: matched fresh E0/E1, mu128, legacy decoder\n' "$seed" "$smoke"
  run_command "$suite_log" "$python" -u -m scripts.p7_control prepare --root "$root/initial" --seed "$seed"
  for arm in e0 e1; do
    arch=conv
    if [[ "$arm" == e1 ]]; then arch=residual; fi
    out="$root/$arm/stage1_reconstruction"
    arm_log="$log_root/p7_${arm}_seed${seed}_reconstruction.log"
    run_command "$arm_log" "$python" -u -m scripts.train "${common[@]}" --encoder_arch "$arch" \
      --resume "$root/initial/${arm}_initial.pt" --resume_epoch 0 --epochs "$recon_epochs" \
      --geology_classifier_weight 0.05 --geology_contrastive_weight 0 --paired_sampling --save_training_state --out_dir "$out"
  done
  if [[ "$smoke" == 1 ]]; then
    run_command "$suite_log" "$python" -u -m scripts.train "${common[@]}" --encoder_arch conv \
      --resume "$root/e0/stage1_reconstruction/vae_last.pt" --resume_training_state --epochs 1 \
      --geology_classifier_weight 0.05 --geology_contrastive_weight 0 --paired_sampling --save_training_state --out_dir "$root/e0/stage1_reconstruction"
    run_command "$suite_log" "$python" -u -m scripts.train "${common[@]}" --encoder_arch conv \
      --resume "$root/initial/e0_initial.pt" --resume_epoch 0 --epochs 2 \
      --geology_classifier_weight 0.05 --geology_contrastive_weight 0 --paired_sampling --save_training_state --out_dir "$root/uninterrupted"
    if [[ "$printing" != 1 ]]; then
      "$python" - "$root" <<'PY'
import json, sys, torch
from pathlib import Path
from src.tokenizer.core.model_adapter import VaeLatentAdapter
root = Path(sys.argv[1])
split = torch.load(root/'e0/stage1_reconstruction/vae_epoch2.pt', map_location='cpu', weights_only=True)
whole = torch.load(root/'uninterrupted/vae_epoch2.pt', map_location='cpu', weights_only=True)
for key, value in split['model_state_dict'].items():
  torch.testing.assert_close(value, whole['model_state_dict'][key], rtol=1e-4, atol=1e-3)
for key in ('torch_rng', 'mps_rng', 'loader_rng'):
  torch.testing.assert_close(split['training_state'][key], whole['training_state'][key], rtol=0, atol=0)
assert split['training_state']['optimizer']['param_groups'] == whole['training_state']['optimizer']['param_groups']
models = [VaeLatentAdapter(root/path, device='mps', load_classifier=True).model for path in
      ('e0/stage1_reconstruction/vae_epoch2.pt', 'uninterrupted/vae_epoch2.pt')]
inputs = torch.randn((2,1,32,32,64), generator=torch.Generator().manual_seed(100)).to('mps')
with torch.no_grad():
  latents = [model.encoder(inputs)[0] for model in models]
  predictions = [model.decoder(latent)[0] for model, latent in zip(models, latents)]
  torch.testing.assert_close(latents[0], latents[1], rtol=1e-4, atol=5e-5)
  torch.testing.assert_close(predictions[0], predictions[1], rtol=1e-4, atol=2e-4)
for epoch in (1, 2):
    assert json.loads((root/f'e0/stage1_reconstruction/sampling_epoch{epoch}.json').read_text())['consumed'] == 24
assert json.loads((root/'e0/stage1_reconstruction/sampling_epoch1.json').read_text())['indices'] == json.loads((root/'e1/stage1_reconstruction/sampling_epoch1.json').read_text())['indices']
print('P7 SMOKE PASS: paired indices, exact RNG/group restoration, full-state continuation within MPS output tolerance (not bit-identical GPU weights)')
PY
    fi
    return
  fi
  for mode in clean extrema; do
    run_command "$suite_log" "$python" -u -m scripts.diagnose_p5_p6_reconstruction --evaluation_only --device mps --samples 5000 --input_mode "$mode" \
      --checkpoints "p0b=checkpoints/geoaware_p0b_r2_seed${seed}_r2/vae_best.pt" \
      "e0=$root/e0/stage1_reconstruction/vae_best.pt" "e1=$root/e1/stage1_reconstruction/vae_best.pt" --out_json "$root/reconstruction_${mode}.json"
  done
  run_command "$suite_log" "$python" -u -m scripts.p7_control gate --root "$root"
  if [[ "$printing" != 1 ]]; then
    eligible=$("$python" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["geology_eligible"]))' "$root/reconstruction_gate.json")
    if [[ "$eligible" != 1 ]]; then printf 'P7 STOP: reconstruction gate failed; no geology or replication launched.\n'; return; fi
  fi
  for arm in e0 e1; do
    arch=conv
    if [[ "$arm" == e1 ]]; then arch=residual; fi
    out="$root/$arm/stage2_geology"
    run_command "$log_root/p7_${arm}_seed${seed}_geology.log" "$python" -u -m scripts.train "${common[@]}" --encoder_arch "$arch" \
      --resume "$root/$arm/stage1_reconstruction/vae_best.pt" --resume_epoch 0 --epochs 40 \
      --geology_classifier_weight 0.1 --geology_contrastive_weight 0.5 --geology_contrastive_temperature 0.2 \
      --geology_batch_sampler --geology_batch_background_fraction 0.05 --geology_batch_hard_fraction 0.30 \
      --geology_batch_min_negative_strata 2 --geology_strata_source presence_labels --geology_batch_class_quota fault_x=1 flat_spot=1 channel=1 --out_dir "$out"
    for epoch in 10 20 30 40; do
      run_command "$suite_log" "$python" -u -m scripts.evaluate_geology_benchmark \
        --data data/synth_val_v2_32-32-64.zarr --manifest docs/benchmarks/frozen_validation_manifest_v2.json \
        --checkpoint "$out/vae_epoch${epoch}.pt" --use_geo_embedding --metadata_keys "${metadata[@]}" \
        --classifier_data data/synth_val_v2_uniform_32-32-64.zarr --classifier_threshold_data data/synth_train_anchored_32-32-64.zarr \
        --out_json "$root/${arm}_ep${epoch}_zgeo.json"
    done
  done
  printf 'P7 COMPLETE: first-seed comparison; review retrieval before replication or adoption.\n'
}
if [[ "$printing" == 1 ]]; then run_suite; exit 0; fi
if [[ ! -d /Volumes/CrucialX9 ]]; then printf 'CrucialX9 is not mounted\n' >&2; exit 1; fi
if [[ -e "$root" || -e "$suite_log" || -e "$local_log" || -L "$local_log" ]]; then printf 'Refusing existing P7 output/log\n' >&2; exit 1; fi
mkdir -p "$(dirname "$root")" "$log_root" logs
"$python" - "$root" "$seed" <<'PY' | tee "$suite_log"
import shutil, sys, torch, zarr
from pathlib import Path
from scripts.train import assert_train_validation_consistency
root, seed = sys.argv[1:]
assert torch.backends.mps.is_available(), 'MPS required'
assert shutil.disk_usage(Path(root).parent).free > 15 * 1024**3, 'need 15 GiB free'
assert_train_validation_consistency('data/synth_train_anchored_32-32-64.zarr','data/synth_val_v2_uniform_32-32-64.zarr')
for path in ('data/synth_train_anchored_32-32-64.zarr','data/synth_val_v2_uniform_32-32-64.zarr','data/real_validation_32-32-64.zarr','data/real_test_32-32-64.zarr'):
    data=zarr.open(path,mode='r'); assert tuple(data['patches'].shape[1:])==(32,32,64)
    assert data.attrs['scaling_mode']=='divide_by_std'
assert Path(f'checkpoints/geoaware_p0b_r2_seed{seed}_r2/vae_best.pt').exists()
assert Path('docs/benchmarks/frozen_validation_manifest_v2.json').exists()
print('P7 PREFLIGHT PASS: data provenance/scaling, references, MPS and disk')
PY
ln -s "$suite_log" "$local_log"
run_suite 2>&1 | tee -a "$suite_log"