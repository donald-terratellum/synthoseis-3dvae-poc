#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python="${PYTHON:-.venv/bin/python}"
seed="${SEED:-20260926}"
smoke="${SMOKE:-0}"
root="${OUT_ROOT:-/Volumes/CrucialX9/3dvae-poc_checkpoints/p7_e1_seed${seed}_replication}"
log_root="${LOG_ROOT:-/Volumes/CrucialX9/3dvae-poc_logs}"
local_log="${LOCAL_LOG_PATH:-logs/p7_e1_seed${seed}_replication.log}"
suite_log="$log_root/$(basename "$local_log")"
recon_epochs=150
batches=450
diagnostic_samples=32
benchmark_samples=5000
if [[ "$smoke" == 1 ]]; then
  root="${OUT_ROOT:-/Volumes/CrucialX9/3dvae-poc_checkpoints/p7_e1_seed${seed}_replication_smoke}"
  log_root="${LOG_ROOT:-/Volumes/CrucialX9/3dvae-poc_logs}"
  local_log="${LOCAL_LOG_PATH:-logs/p7_e1_seed${seed}_replication_smoke.log}"
  suite_log="$log_root/$(basename "$local_log")"
  recon_epochs=1; batches=2; diagnostic_samples=2; benchmark_samples=64
fi
p0b_checkpoint="checkpoints/geoaware_p0b_r2_seed${seed}_r2/vae_best.pt"
metadata=(meta_fault_fraction meta_fault_intersection_fraction meta_channel_fraction
  meta_channel_core_fraction meta_flat_spot_fraction meta_onlap_fraction meta_onlap_variability)
common=(
  --data data/synth_train_anchored_32-32-64.zarr --validation_data data/synth_val_v2_uniform_32-32-64.zarr
  --real_validation_data data/real_validation_32-32-64.zarr --real_test_data data/real_test_32-32-64.zarr
  --device mps --patch_size 32 32 64 --batch_size 12 --number_batches "$batches" --seed "$seed"
  --encoder_norm batch --encoder_input_axes xyz --decoder_block conv --freeze_encoder_epochs 0
  --augment --vertical_warp_prob 0.5 --phase_rotation_prob 0 --stretch_prob 0 --dip_label_policy adjust --mixup_augment_prob 0
  --input_scaling divide_by_std --input_extrema_prob 1 --input_sparse_keep_prob 0 --input_decimate_trilinear_prob 0 --validation_extrema_only
  --learning_rate 5e-4 --weight_decay 1e-4 --encoder_lr_mult 0.1 --grad_clip 2
  --kl_schedule warmup --kl_start 1e-3 --kl_end 1e-3 --kl_warmup_epochs 1
  --reconstruction_loss mae --lpips_weight 0.1 --lr_scheduler plateau --lr_scheduler_patience 6 --lr_scheduler_factor 0.5 --lr_scheduler_min_lr 1e-5
  --early_stopping_patience 999 --save_epoch_checkpoints --best_checkpoint_name vae_best.pt
  --geology_projection --geology_proj_hidden 128 --geology_proj_dim 64 --geology_uniformity_weight 0
  --geology_classifier --geology_classifier_mode patch --geology_classifier_loss focal --geology_classifier_focal_gamma 2 --geology_classifier_label_smoothing 0.05
  --geology_metadata_keys "${metadata[@]}" --reconstruction_diagnostic_interval 30
  --reconstruction_diagnostic_samples "$diagnostic_samples" --reconstruction_diagnostic_modes clean extrema tokenizer
  --reconstruction_diagnostic_train
)
printing="${PRINT_COMMANDS:-0}"
run_command() {
  local destination="$1"; shift
  if [[ "$printing" == 1 ]]; then printf '%q ' "$@"; printf '\n';
  elif [[ "$destination" == "$suite_log" ]]; then "$@" 2>&1 | tee -a "$suite_log";
  else "$@" 2>&1 | tee -a "$destination" | tee -a "$suite_log";
  fi
}
status_line() {
  if [[ "$printing" == 1 ]]; then printf '%s\n' "$1";
  else printf '%s\n' "$1" | tee -a "$suite_log";
  fi
}
run_suite() {
  status_line "P7 E1 seed2 replication seed=$seed smoke=$smoke; fresh E0/E1 recon, E1 geology only after same-seed exact-tokenizer reconstruction gate"
  run_command "$suite_log" "$python" -u -m scripts.p7_control prepare --root "$root/initial" --seed "$seed"
  for arm in e0 e1; do
    arch=conv; [[ "$arm" == e1 ]] && arch=residual
    out="$root/$arm/stage1_reconstruction"
    arm_log="$log_root/p7_${arm}_seed${seed}_reconstruction.log"
    run_command "$arm_log" "$python" -u -m scripts.train "${common[@]}" --encoder_arch "$arch" \
      --resume "$root/initial/${arm}_initial.pt" --resume_epoch 0 --epochs "$recon_epochs" \
      --geology_classifier_weight 0.05 --geology_contrastive_weight 0 --paired_sampling --save_training_state --out_dir "$out"
  done
  local full_samples="$benchmark_samples"
  if [[ "$smoke" == 1 ]]; then full_samples=64; fi
  run_command "$suite_log" "$python" -u -m scripts.diagnose_p5_p6_reconstruction --evaluation_only --contract_audit \
    --device mps --samples "$full_samples" --batch_size 2 --seed 20261003 --input_mode tokenizer \
    --checkpoints "p0b=$p0b_checkpoint" "e0=$root/e0/stage1_reconstruction/vae_best.pt" \
      "e1=$root/e1/stage1_reconstruction/vae_best.pt" --out_json "$root/seed2_tokenizer_reconstruction.json"
  if [[ "$smoke" != 1 && "$printing" != 1 ]]; then
    run_command "$suite_log" "$python" -u -m scripts.p7_control tokenizer-gate --root "$root/seed2_tokenizer_reconstruction.json"
  elif [[ "$smoke" == 1 ]]; then
    status_line 'SMOKE: bypassing full-store reconstruction gate; geology command is startup-only and never an adoption result.'
  else
    status_line 'DRY RUN: tokenizer reconstruction gate must pass before the rendered geology command is run.'
  fi
  if [[ "$smoke" != 1 && "$printing" != 1 ]]; then
    local reconstruction_pass
    reconstruction_pass=$("$python" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["passed"]))' "$root/tokenizer_reconstruction_gate.json")
    if [[ "$reconstruction_pass" != 1 ]]; then status_line 'P7 SEED2 STOP: tokenizer reconstruction gate failed; no geology stage launched.'; return; fi
  fi
  local geology_epochs=40
  if [[ "$smoke" == 1 ]]; then geology_epochs=1; fi
  local geology_out="$root/e1/stage2_geology"
  run_command "$log_root/p7_e1_seed${seed}_geology.log" "$python" -u -m scripts.train "${common[@]}" --encoder_arch residual \
    --resume "$root/e1/stage1_reconstruction/vae_best.pt" --resume_epoch 0 --epochs "$geology_epochs" \
    --geology_classifier_weight 0.1 --geology_contrastive_weight 0.5 --geology_contrastive_temperature 0.2 \
    --geology_batch_sampler --geology_batch_background_fraction 0.05 --geology_batch_hard_fraction 0.30 \
    --geology_batch_min_negative_strata 2 --geology_strata_source presence_labels \
    --geology_batch_class_quota fault_x=1 flat_spot=1 channel=1 --out_dir "$geology_out"
  if [[ "$smoke" == 1 ]]; then status_line 'P7 SEED2 SMOKE COMPLETE: paired reconstruction and E1 geology startup verified; no adoption.'; return; fi
  for epoch in 10 20 30 40; do
    run_command "$suite_log" "$python" -u -m scripts.evaluate_geology_benchmark \
      --data data/synth_val_v2_32-32-64.zarr --manifest docs/benchmarks/frozen_validation_manifest_v2.json \
      --checkpoint "$geology_out/vae_epoch${epoch}.pt" --use_geo_embedding --metadata_keys "${metadata[@]}" \
      --classifier_data data/synth_val_v2_uniform_32-32-64.zarr --classifier_threshold_data data/synth_train_anchored_32-32-64.zarr \
      --out_json "$root/e1_ep${epoch}_zgeo.json"
  done
  if [[ "$printing" == 1 ]]; then status_line 'DRY RUN: after benchmarks, write seed2_replication_decision.json; no automatic adoption.'; return; fi
  "$python" - "$root" "docs/benchmarks/geoaware_p0b_r2_seed${seed}_r2_ep30_zgeo.json" <<'PY' | tee -a "$suite_log"
import json, sys
from pathlib import Path
root, baseline_path = map(Path, sys.argv[1:])
baseline = json.loads(baseline_path.read_text())
baseline_n5 = baseline['diagnostics']['neighbor_overlap_at_5']
baseline_n10 = baseline['diagnostics']['neighbor_overlap_at_10']
reconstruction = json.loads((root/'tokenizer_reconstruction_gate.json').read_text())
milestones = []
for epoch in (10, 20, 30, 40):
    report = json.loads((root/f'e1_ep{epoch}_zgeo.json').read_text())
    milestones.append({'epoch': epoch,
        'n5': report['diagnostics']['neighbor_overlap_at_5'],
        'n10': report['diagnostics']['neighbor_overlap_at_10'],
        'macro_auroc': report['classifier_metrics']['macro_auroc']})
eligible = [item for item in milestones if reconstruction['passed'] and item['n5'] > baseline_n5]
best = max(eligible, key=lambda item: (item['n5'], item['n10'])) if eligible else None
decision = {'seed': 20260926, 'baseline_n5': baseline_n5, 'baseline_n10': baseline_n10,
    'tokenizer_reconstruction_gate_passed': reconstruction['passed'], 'milestones': milestones,
    'eligible_replication': best is not None, 'best_eligible_milestone': best,
    'macro_auroc_sanity_target': 0.75,
    'macro_auroc_sanity_target_met': bool(best and best['macro_auroc'] >= 0.75),
    'adopted': False, 'automatic_adoption': False}
(root/'seed2_replication_decision.json').write_text(json.dumps(decision, indent=2, allow_nan=False)+'\n')
print('P7 SEED2 DECISION:', json.dumps(decision, indent=2), flush=True)
PY
  status_line 'P7 SEED2 COMPLETE: second-seed decision written; adoption remains a manual decision.'
}
if [[ "$printing" == 1 ]]; then run_suite; exit 0; fi
if [[ ! -d /Volumes/CrucialX9 ]]; then printf 'CrucialX9 is not mounted\n' >&2; exit 1; fi
if [[ -e "$root" || -e "$suite_log" || -e "$local_log" || -L "$local_log" ]]; then printf 'Refusing existing seed2 output or log\n' >&2; exit 1; fi
if pgrep -f '[p]ython.*-m scripts[.]train' >/dev/null; then printf 'An active scripts.train process exists; refusing concurrent training\n' >&2; exit 1; fi
mkdir -p "$(dirname "$root")" "$log_root" logs
mkdir "$root" || { printf 'Could not reserve unique seed2 output directory\n' >&2; exit 1; }
"$python" - "$root" "$seed" <<'PY' | tee "$suite_log"
import shutil, sys, torch, zarr
from pathlib import Path
from scripts.train import assert_train_validation_consistency
root, seed = sys.argv[1:]
assert seed == '20260926', 'this runner is specifically the seed-2 replication'
assert torch.backends.mps.is_available(), 'MPS required'
assert shutil.disk_usage(Path(root).parent).free > 15*1024**3, 'need 15 GiB free'
assert_train_validation_consistency('data/synth_train_anchored_32-32-64.zarr','data/synth_val_v2_uniform_32-32-64.zarr')
for path in ('data/synth_train_anchored_32-32-64.zarr','data/synth_val_v2_uniform_32-32-64.zarr','data/synth_val_v2_32-32-64.zarr','data/real_validation_32-32-64.zarr','data/real_test_32-32-64.zarr'):
    store=zarr.open(path,mode='r'); assert tuple(store['patches'].shape[1:])==(32,32,64)
    assert store.attrs['scaling_mode']=='divide_by_std'
train=zarr.open('data/synth_train_anchored_32-32-64.zarr',mode='r')
for key in ('source_volume_index','label_presence_fault_x','label_presence_flat_spot','label_presence_channel','meta_dip_mean_class','meta_dip_range_class','dip_samples_deg','dip_samples_z'):
    assert key in train, f'missing {key}'
manifest=__import__('json').loads(Path('docs/benchmarks/frozen_validation_manifest_v2.json').read_text())
assert manifest['dataset_size']==zarr.open('data/synth_val_v2_32-32-64.zarr',mode='r')['patches'].shape[0]
assert Path(f'checkpoints/geoaware_p0b_r2_seed{seed}_r2/vae_best.pt').is_file()
baseline=Path(f'docs/benchmarks/geoaware_p0b_r2_seed{seed}_r2_ep30_zgeo.json')
assert baseline.is_file(), f'missing same-seed frozen P0b benchmark: {baseline}'
print('P7 SEED2 PREFLIGHT PASS: same-seed P0b checkpoint/benchmark, data/provenance, MPS and disk',flush=True)
print('E1 will initialize fresh from E0 weights seeded 20260926; seed1 checkpoints are not loaded.',flush=True)
PY
ln -s "$suite_log" "$local_log"
run_suite >/dev/null