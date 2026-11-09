#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python="${PYTHON:-.venv/bin/python}"
smoke="${SMOKE:-0}"
out="${OUT_DIR:-/Volumes/CrucialX9/3dvae-poc_checkpoints/p7_e1_geology_pilot_seed20260925}"
log="${LOG_PATH:-/Volumes/CrucialX9/3dvae-poc_logs/p7_e1_geology_pilot_seed20260925.log}"
local_log="${LOCAL_LOG_PATH:-logs/p7_e1_geology_pilot_seed20260925.log}"
epochs=40
batches=450
interval=10
samples=32
if [[ "$smoke" == 1 ]]; then
  out="${OUT_DIR:-/Volumes/CrucialX9/3dvae-poc_checkpoints/p7_e1_geology_pilot_smoke_seed20260925}"
  log="${LOG_PATH:-/Volumes/CrucialX9/3dvae-poc_logs/p7_e1_geology_pilot_smoke_seed20260925.log}"
  local_log="${LOCAL_LOG_PATH:-logs/p7_e1_geology_pilot_smoke_seed20260925.log}"
  epochs=1; batches=2; interval=1; samples=2
fi
source_checkpoint="/Volumes/CrucialX9/3dvae-poc_checkpoints/p7_e0_e1_seed20260925/e1/stage1_reconstruction/vae_best.pt"
control="checkpoints/geoaware_p0b_r2_seed20260925_r2/vae_best.pt"
audit="/Volumes/CrucialX9/3dvae-poc_logs/p7_input_contract_audit_20261005"
reference_report="docs/benchmarks/geoaware_p0b_r2_seed20260925_r2_ep30_zgeo.json"
metadata=(meta_fault_fraction meta_fault_intersection_fraction meta_channel_fraction
  meta_channel_core_fraction meta_flat_spot_fraction meta_onlap_fraction meta_onlap_variability)
args=(
  -u -m scripts.train --resume "$source_checkpoint" --resume_epoch 0 --epochs "$epochs"
  --data data/synth_train_anchored_32-32-64.zarr --validation_data data/synth_val_v2_uniform_32-32-64.zarr
  --real_validation_data data/real_validation_32-32-64.zarr --real_test_data data/real_test_32-32-64.zarr
  --device mps --patch_size 32 32 64 --batch_size 12 --number_batches "$batches" --seed 20260925
  --encoder_arch residual --encoder_norm batch --encoder_input_axes xyz --decoder_block conv --freeze_encoder_epochs 0
  --augment --vertical_warp_prob 0.5 --phase_rotation_prob 0 --stretch_prob 0 --dip_label_policy adjust --mixup_augment_prob 0
  --input_scaling divide_by_std --input_extrema_prob 1 --input_sparse_keep_prob 0 --input_decimate_trilinear_prob 0 --validation_extrema_only
  --learning_rate 5e-4 --weight_decay 1e-4 --encoder_lr_mult 0.1 --grad_clip 2
  --kl_schedule warmup --kl_start 1e-3 --kl_end 1e-3 --kl_warmup_epochs 1
  --reconstruction_loss mae --lpips_weight 0.1 --lr_scheduler plateau --lr_scheduler_patience 6 --lr_scheduler_factor 0.5 --lr_scheduler_min_lr 1e-5
  --early_stopping_patience 999 --save_epoch_checkpoints --best_checkpoint_name vae_best.pt --out_dir "$out"
  --geology_projection --geology_proj_hidden 128 --geology_proj_dim 64 --geology_uniformity_weight 0
  --geology_classifier --geology_classifier_mode patch --geology_classifier_weight 0.1 --geology_classifier_loss focal
  --geology_classifier_focal_gamma 2 --geology_classifier_label_smoothing 0.05
  --geology_contrastive_weight 0.5 --geology_contrastive_temperature 0.2
  --geology_batch_sampler --geology_batch_background_fraction 0.05 --geology_batch_hard_fraction 0.30
  --geology_batch_min_negative_strata 2 --geology_strata_source presence_labels --geology_batch_class_quota fault_x=1 flat_spot=1 channel=1
  --geology_metadata_keys "${metadata[@]}" --geology_diagnostic_max_samples 512 --geology_diagnostic_neighbor_k 5 --geology_diagnostic_topk 5 10 20
  --reconstruction_diagnostic_interval "$interval" --reconstruction_diagnostic_samples "$samples"
  --reconstruction_diagnostic_modes clean extrema tokenizer
)
if [[ "${PRINT_COMMANDS:-0}" == 1 ]]; then printf '%q ' "$python" "${args[@]}"; printf '\n'; exit 0; fi
if [[ ! -d /Volumes/CrucialX9 ]]; then printf 'CrucialX9 is not mounted\n' >&2; exit 1; fi
if [[ -e "$out" || -e "$log" || -e "$local_log" || -L "$local_log" ]]; then printf 'Refusing existing pilot output/log\n' >&2; exit 1; fi
mkdir -p "$(dirname "$out")" "$(dirname "$log")" logs
ln -s "$log" "$local_log"
{
  printf 'E1 geology pilot started %s; mu128; prospective tokenizer contract; original P7 gate preserved\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  "$python" - "$source_checkpoint" "$audit" "$reference_report" "$(dirname "$out")" <<'PY'
import json, shutil, sys, torch, zarr
from pathlib import Path
from scripts.train import assert_train_validation_consistency
source, audit, reference, output_parent = map(Path, sys.argv[1:])
decision = json.loads((audit/'decision_summary.json').read_text())
assert decision['e1_contract_reconstruction_within_2pct'], 'contract audit failed'
assert not decision['original_clean_extrema_gate_changed'], 'original P7 gate must remain unchanged'
full = json.loads((audit/'full_tokenizer.json').read_text())
assert full['reconstruction_gate']['full_dataset_coverage']
assert Path(full['arms']['e1']['checkpoint']).resolve() == source.resolve()
checkpoint = torch.load(source, map_location='cpu', weights_only=True)
assert checkpoint['epoch'] == 149 and checkpoint['latent_dim'] == 128
assert checkpoint['patch_shape'] == [32,32,64] and checkpoint['model_config']['encoder_arch'] == 'residual'
assert checkpoint['model_config']['decoder_hidden_dims'] is None
assert reference.is_file(), 'missing exact P0b retrieval reference'
assert torch.backends.mps.is_available(), 'MPS required'
assert shutil.disk_usage(output_parent).free > 15*1024**3
assert_train_validation_consistency('data/synth_train_anchored_32-32-64.zarr','data/synth_val_v2_uniform_32-32-64.zarr')
train = zarr.open('data/synth_train_anchored_32-32-64.zarr', mode='r')
for key in ('label_presence_fault','label_presence_fault_x','label_presence_channel','label_presence_closure',
            'label_presence_onlap','label_presence_sand','label_presence_flat_spot','meta_dip_mean_class',
            'meta_dip_range_class','dip_samples_deg','dip_samples_z'):
    assert key in train, f'missing target/dip array {key}'
for path in ('data/synth_val_v2_uniform_32-32-64.zarr','data/synth_val_v2_32-32-64.zarr','data/real_validation_32-32-64.zarr','data/real_test_32-32-64.zarr'):
    store = zarr.open(path, mode='r'); assert tuple(store['patches'].shape[1:]) == (32,32,64)
    assert store.attrs['scaling_mode'] == 'divide_by_std'
manifest = json.loads(Path('docs/benchmarks/frozen_validation_manifest_v2.json').read_text())
assert manifest['dataset_size'] == zarr.open('data/synth_val_v2_32-32-64.zarr',mode='r')['patches'].shape[0]
print('PILOT PREFLIGHT PASS: contract, epoch149/mu128/legacy decoder, targets, provenance, manifest, references, MPS and disk',flush=True)
print('Warm start uses fresh AdamW/plateau state, not reconstruction optimizer continuation.',flush=True)
PY
  "$python" "${args[@]}"
  if [[ "$smoke" == 1 ]]; then printf 'E1 PILOT SMOKE COMPLETE: two-batch geology warm start and tokenizer diagnostics.\n'; exit 0; fi
  for epoch in 10 20 30 40; do
    "$python" -u -m scripts.evaluate_geology_benchmark \
      --data data/synth_val_v2_32-32-64.zarr --manifest docs/benchmarks/frozen_validation_manifest_v2.json \
      --checkpoint "$out/vae_epoch${epoch}.pt" --use_geo_embedding --metadata_keys "${metadata[@]}" \
      --classifier_data data/synth_val_v2_uniform_32-32-64.zarr --classifier_threshold_data data/synth_train_anchored_32-32-64.zarr \
      --out_json "$out/ep${epoch}_zgeo.json"
  done
  "$python" -u -m scripts.diagnose_p5_p6_reconstruction --evaluation_only --contract_audit --input_mode tokenizer \
    --device mps --samples 5000 --batch_size 2 --seed 20261003 \
    --checkpoints "p0b=$control" "ep10=$out/vae_epoch10.pt" "ep20=$out/vae_epoch20.pt" \
      "ep30=$out/vae_epoch30.pt" "ep40=$out/vae_epoch40.pt" --out_json "$out/tokenizer_reconstruction_guardrails.json"
  "$python" - "$out" "$reference_report" <<'PY'
import json, sys
from pathlib import Path
root, baseline_path = map(Path,sys.argv[1:])
baseline = json.loads(baseline_path.read_text())['diagnostics']['neighbor_overlap_at_5']
guardrails = json.loads((root/'tokenizer_reconstruction_guardrails.json').read_text())
assert guardrails['reconstruction_gate']['full_dataset_coverage']
reports = []
for epoch in (10,20,30,40):
    report = json.loads((root/f'ep{epoch}_zgeo.json').read_text())
    reports.append({'epoch':epoch,'n5':report['diagnostics']['neighbor_overlap_at_5'],
                    'n10':report['diagnostics']['neighbor_overlap_at_10'],
                    'macro_auroc':report['classifier_metrics']['macro_auroc'],
                    'reconstruction_within_2pct':guardrails['reconstruction_gate']['synthetic_recipe_and_real_mae_within_2pct'][f'ep{epoch}']})
eligible = [report for report in reports if report['n5'] > baseline and report['reconstruction_within_2pct']]
ranked = sorted(eligible,key=lambda report:(report['n5'],report['n10']),reverse=True)
decision = {'baseline_n5':baseline,'milestones':reports,'eligible_first_seed_candidate':ranked[0] if ranked else None,
            'second_seed_launched':False,'adopted':False,'original_p7_gate_modified':False}
(root/'pilot_decision.json').write_text(json.dumps(decision,indent=2,allow_nan=False)+'\n')
print('E1 PILOT DECISION:',json.dumps(decision,indent=2),flush=True)
PY
  printf 'COMPLETE: E1 geology pilot and milestone evaluation; review before any second seed or adoption.\n'
} 2>&1 | tee "$log"