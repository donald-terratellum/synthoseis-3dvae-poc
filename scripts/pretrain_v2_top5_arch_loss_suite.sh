#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python="${PYTHON:-.venv/bin/python}"
seed="${SEED:-20261008}"
smoke="${SMOKE:-0}"
printing="${PRINT_COMMANDS:-0}"
candidate_filter="${CANDIDATE:-all}"
root="${OUT_ROOT:-/Volumes/CrucialX9/3dvae-poc_checkpoints/pretrain_v2_top5_arch_loss_seed${seed}}"
log_root="${LOG_ROOT:-/Volumes/CrucialX9/3dvae-poc_logs/pretrain_v2_top5_arch_loss_seed${seed}}"
suite_log="$log_root/pretrain_v2_top5_arch_loss_seed${seed}.log"
local_log="${LOCAL_LOG_PATH:-logs/pretrain_v2_top5_arch_loss_seed${seed}.log}"
recon_epochs="${RECON_EPOCHS:-60}"
geology_epochs="${GEOLOGY_EPOCHS:-40}"
batches="${BATCHES:-450}"
samples=5000
candidates=(r006 r004 r007 r001 r011)

if [[ "$smoke" == 1 ]]; then
  root="${OUT_ROOT:-/Volumes/CrucialX9/3dvae-poc_checkpoints/pretrain_v2_top5_arch_loss_smoke_seed${seed}}"
  log_root="${LOG_ROOT:-/Volumes/CrucialX9/3dvae-poc_logs/pretrain_v2_top5_arch_loss_smoke_seed${seed}}"
  suite_log="$log_root/pretrain_v2_top5_arch_loss_smoke_seed${seed}.log"
  local_log="${LOCAL_LOG_PATH:-logs/pretrain_v2_top5_arch_loss_smoke_seed${seed}.log}"
  recon_epochs=1; geology_epochs=1; batches=2; samples=32
  candidates=("${SMOKE_CANDIDATE:-r006}")
fi

if [[ "$candidate_filter" != all ]]; then
  case "$candidate_filter" in r006|r004|r007|r001|r011) candidates=("$candidate_filter") ;; *) printf 'Unknown CANDIDATE=%s\n' "$candidate_filter" >&2; exit 2 ;; esac
fi

p0b_checkpoint="checkpoints/geoaware_p0b_r2_seed20260926_r2/vae_best.pt"
p0b_benchmark="docs/benchmarks/geoaware_p0b_r2_seed20260926_r2_ep30_zgeo.json"
train_data="data/synth_train_anchored_32-32-64.zarr"
validation_data="data/synth_val_v2_uniform_32-32-64.zarr"
real_train_data="data/real_train_32-32-64.zarr"
real_validation_data="data/real_validation_32-32-64.zarr"
real_test_data="data/real_test_32-32-64.zarr"
benchmark_data="data/synth_val_v2_32-32-64.zarr"
manifest="docs/benchmarks/frozen_validation_manifest_v2.json"
metadata=(meta_fault_fraction meta_fault_intersection_fraction meta_channel_fraction
  meta_channel_core_fraction meta_flat_spot_fraction meta_onlap_fraction meta_onlap_variability)

configure_candidate() {
  case "$1" in
    r006)
      source_checkpoint="/Volumes/CrucialX9/pretrain_v2_checkpoints/sweep_20260620_080306_r006_u3_h32-64-128_lp0p000_tv0p001/checkpoint_epoch_0050.pt"
      widths=(32 64 128); blocks=(3 5 8); mae=0.995; lpips=0.005; tv=0; gdl=0
      ;;
    r004)
      source_checkpoint="/Volumes/CrucialX9/pretrain_v2_checkpoints/sweep_20260620_035309_r004_u3_h40-80-160_lp0p000_tv0p001/checkpoint_epoch_0055.pt"
      widths=(40 80 160); blocks=(3 5 8); mae=0.995; lpips=0.005; tv=0.001; gdl=0
      ;;
    r007)
      source_checkpoint="/Volumes/CrucialX9/pretrain_v2_checkpoints/sweep_20260621_232252_r007_u4_h40-74-138-256_lp0p000_tv0p001/checkpoint_epoch_0085.pt"
      widths=(40 74 138 256); blocks=(3 5 8 5); mae=0.999; lpips=0; tv=0; gdl=0.001
      ;;
    r001)
      source_checkpoint="/Volumes/CrucialX9/pretrain_v2_checkpoints/sweep_20260621_104613_r001_u4_h40-74-138-256_lp0p000_tv0p010/checkpoint_epoch_0050.pt"
      widths=(40 74 138 256); blocks=(3 4 8 4); mae=0.995; lpips=0.005; tv=0; gdl=0
      ;;
    r011)
      source_checkpoint="/Volumes/CrucialX9/pretrain_v2_checkpoints/sweep_20260622_085428_r011_u4_h32-64-128-256_lp0p000_tv0p010/checkpoint_epoch_0095.pt"
      widths=(32 64 128 256); blocks=(3 4 6 3); mae=0.99; lpips=0; tv=0; gdl=0.01
      ;;
  esac
}

status_line() {
  if [[ "$printing" == 1 ]]; then printf '%s\n' "$1";
  else printf '%s\n' "$1" | tee -a "$suite_log";
  fi
}

run_command() {
  local stage_log="$1"; shift
  if [[ "$printing" == 1 ]]; then printf '%q ' "$@"; printf '\n';
  else "$@" 2>&1 | tee -a "$stage_log" "$suite_log";
  fi
}

make_stage1_args() {
  local candidate="$1" output="$2"
  stage1_args=(
    -u -m scripts.train --data "$train_data" --validation_data "$validation_data"
    --real_data "$real_train_data" --real_batch_count 2 --real_validation_data "$real_validation_data" --real_test_data "$real_test_data"
    --device mps --patch_size 32 32 64 --batch_size 12 --number_batches "$batches" --seed "$seed"
    --encoder_arch resnetv2 --encoder_hidden_dims "${widths[@]}" --encoder_stage_blocks "${blocks[@]}"
    --encoder_depth_profile baseline --encoder_norm instance --encoder_stem pretrain_v2 --encoder_input_axes zxy
    --decoder_block conv --init_encoder_from "$source_checkpoint" --freeze_encoder_epochs 0
    --augment --vertical_warp_prob 0.5 --phase_rotation_prob 0 --stretch_prob 0 --dip_label_policy adjust --mixup_augment_prob 0
    --input_scaling divide_by_std --input_extrema_prob 1 --input_sparse_keep_prob 0 --input_decimate_trilinear_prob 0 --validation_extrema_only
    --learning_rate 2.5e-4 --weight_decay 1e-4 --encoder_lr_mult 0.1 --grad_clip 2
    --kl_schedule warmup --kl_start 1e-3 --kl_end 1e-3 --kl_warmup_epochs 1
    --reconstruction_loss multi_component --recon_mae_weight "$mae" --recon_tv_weight "$tv" --recon_gdl_weight "$gdl" --lpips_weight "$lpips"
    --lr_scheduler plateau --lr_scheduler_patience 6 --lr_scheduler_factor 0.5 --lr_scheduler_min_lr 1e-5
    --early_stopping_patience 999 --best_checkpoint_name vae_best.pt --no_save_epoch_checkpoints
    --save_training_state --paired_sampling
    --geology_projection --geology_proj_hidden 128 --geology_proj_dim 64 --geology_uniformity_weight 0
    --geology_classifier --geology_classifier_mode patch --geology_classifier_weight 0
    --geology_classifier_loss focal --geology_classifier_focal_gamma 2 --geology_classifier_label_smoothing 0.05
    --geology_metadata_keys "${metadata[@]}" --out_dir "$output"
  )
  if [[ "$smoke" == 1 ]]; then stage1_args+=(--epochs "$recon_epochs"); else stage1_args+=(--epochs "$recon_epochs"); fi
}

make_stage2_args() {
  local source="$1" output="$2"
  stage2_args=(
    -u -m scripts.train --data "$train_data" --validation_data "$validation_data"
    --real_data "$real_train_data" --real_batch_count 2 --real_validation_data "$real_validation_data" --real_test_data "$real_test_data"
    --device mps --patch_size 32 32 64 --batch_size 12 --number_batches "$batches" --seed "$seed"
    --encoder_arch resnetv2 --encoder_hidden_dims "${widths[@]}" --encoder_stage_blocks "${blocks[@]}"
    --encoder_depth_profile baseline --encoder_norm instance --encoder_stem pretrain_v2 --encoder_input_axes zxy
    --decoder_block conv --resume "$source" --resume_epoch 0 --epochs "$geology_epochs"
    --augment --vertical_warp_prob 0.5 --phase_rotation_prob 0 --stretch_prob 0 --dip_label_policy adjust --mixup_augment_prob 0
    --input_scaling divide_by_std --input_extrema_prob 1 --input_sparse_keep_prob 0 --input_decimate_trilinear_prob 0 --validation_extrema_only
    --learning_rate 5e-4 --weight_decay 1e-4 --encoder_lr_mult 0.1 --grad_clip 2
    --kl_schedule warmup --kl_start 1e-3 --kl_end 1e-3 --kl_warmup_epochs 1
    --reconstruction_loss multi_component --recon_mae_weight "$mae" --recon_tv_weight "$tv" --recon_gdl_weight "$gdl" --lpips_weight "$lpips"
    --lr_scheduler plateau --lr_scheduler_patience 6 --lr_scheduler_factor 0.5 --lr_scheduler_min_lr 1e-5
    --early_stopping_patience 999 --best_checkpoint_name vae_best.pt --save_epoch_checkpoints
    --geology_projection --geology_proj_hidden 128 --geology_proj_dim 64 --geology_uniformity_weight 0
    --geology_classifier --geology_classifier_mode patch --geology_classifier_weight 0.1
    --geology_classifier_loss focal --geology_classifier_focal_gamma 2 --geology_classifier_label_smoothing 0.05
    --geology_contrastive_weight 0.5 --geology_contrastive_temperature 0.2
    --geology_batch_sampler --geology_batch_background_fraction 0.05 --geology_batch_hard_fraction 0.30
    --geology_batch_min_negative_strata 2 --geology_strata_source presence_labels
    --geology_batch_class_quota fault_x=1 flat_spot=1 channel=1
    --geology_metadata_keys "${metadata[@]}" --out_dir "$output"
  )
}

recon_base=(--evaluation_only --contract_audit --device mps --batch_size 2 --seed 20261003 --input_mode tokenizer)

run_candidate() {
  local candidate="$1"
  configure_candidate "$candidate"
  local candidate_root="$root/$candidate"
  local stage1_out="$candidate_root/stage1_reconstruction"
  local stage2_out="$candidate_root/stage2_geology"
  local candidate_log="$log_root/${candidate}_seed${seed}.log"
  if [[ "$printing" != 1 ]]; then mkdir -p "$candidate_root"; fi
  status_line "CANDIDATE $candidate: source=$source_checkpoint widths=${widths[*]} blocks=${blocks[*]} MAE=$mae LPIPS=$lpips TV=$tv GDL=$gdl"
  make_stage1_args "$candidate" "$stage1_out"
  run_command "$candidate_log" "$python" "${stage1_args[@]}"

  local stage1_report="$candidate_root/reconstruction_tokenizer_stage1.json"
  run_command "$candidate_log" "$python" -u -m scripts.diagnose_p5_p6_reconstruction "${recon_base[@]}" \
    --samples "$samples" --checkpoints "p0b=$p0b_checkpoint" "$candidate=$stage1_out/vae_best.pt" --out_json "$stage1_report"
  if [[ "$smoke" == 1 ]]; then
    status_line "SMOKE $candidate: skips full-coverage gate after 32-patch diagnostics; starts geology for one epoch only."
  elif [[ "$printing" == 1 ]]; then
    run_command "$candidate_log" "$python" -u -m scripts.p7_control tokenizer-gate --root "$stage1_report" --candidate "$candidate" --control p0b
    status_line "DRY RUN $candidate: geology is conditional on the full tokenizer gate passing."
  else
    run_command "$candidate_log" "$python" -u -m scripts.p7_control tokenizer-gate --root "$stage1_report" --candidate "$candidate" --control p0b
    local gate_pass
    gate_pass=$("$python" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["passed"]))' "$candidate_root/tokenizer_reconstruction_gate.json")
    if [[ "$gate_pass" != 1 ]]; then
      status_line "STOP $candidate: full tokenizer reconstruction gate failed; geology stage skipped."
      "$python" - "$candidate_root" "$candidate" <<'PY' | tee -a "$suite_log"
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); name=sys.argv[2]
(root/'candidate_decision.json').write_text(json.dumps({'candidate':name,'stage1_gate_passed':False,'geology_stage_ran':False,'eligible_for_replication':False,'adopted':False},indent=2)+'\n')
PY
      return 0
    fi
  fi

  make_stage2_args "$stage1_out/vae_best.pt" "$stage2_out"
  run_command "$candidate_log" "$python" "${stage2_args[@]}"
  if [[ "$smoke" == 1 ]]; then
    status_line "SMOKE COMPLETE $candidate: one reconstruction and one geology epoch; no adoption result."
    return 0
  fi
  if [[ "$printing" == 1 ]]; then
    for epoch in 10 20 30 40; do
      run_command "$candidate_log" "$python" -u -m scripts.evaluate_geology_benchmark \
        --data "$benchmark_data" --manifest "$manifest" --checkpoint "$stage2_out/vae_epoch${epoch}.pt" \
        --use_geo_embedding --metadata_keys "${metadata[@]}" --classifier_data "$validation_data" \
        --classifier_threshold_data "$train_data" --out_json "$candidate_root/${candidate}_ep${epoch}_zgeo.json"
    done
    run_command "$candidate_log" "$python" -u -m scripts.diagnose_p5_p6_reconstruction "${recon_base[@]}" \
      --samples 5000 --checkpoints "p0b=$p0b_checkpoint" "ep10=$stage2_out/vae_epoch10.pt" \
        "ep20=$stage2_out/vae_epoch20.pt" "ep30=$stage2_out/vae_epoch30.pt" "ep40=$stage2_out/vae_epoch40.pt" \
      --out_json "$candidate_root/reconstruction_tokenizer_geology_milestones.json"
    return 0
  fi
  for epoch in 10 20 30 40; do
    run_command "$candidate_log" "$python" -u -m scripts.evaluate_geology_benchmark \
      --data "$benchmark_data" --manifest "$manifest" --checkpoint "$stage2_out/vae_epoch${epoch}.pt" \
      --use_geo_embedding --metadata_keys "${metadata[@]}" --classifier_data "$validation_data" \
      --classifier_threshold_data "$train_data" --out_json "$candidate_root/${candidate}_ep${epoch}_zgeo.json"
  done
  local final_report="$candidate_root/reconstruction_tokenizer_geology_milestones.json"
  run_command "$candidate_log" "$python" -u -m scripts.diagnose_p5_p6_reconstruction "${recon_base[@]}" \
    --samples 5000 --checkpoints "p0b=$p0b_checkpoint" "ep10=$stage2_out/vae_epoch10.pt" \
      "ep20=$stage2_out/vae_epoch20.pt" "ep30=$stage2_out/vae_epoch30.pt" "ep40=$stage2_out/vae_epoch40.pt" \
    --out_json "$final_report"
  "$python" - "$candidate_root" "$final_report" "$p0b_benchmark" "$candidate" <<'PY' | tee -a "$suite_log"
import json,sys
from pathlib import Path
root,report_path,baseline_path,candidate=map(Path,sys.argv[1:])
report=json.loads(report_path.read_text()); baseline=json.loads(baseline_path.read_text())
baseline_n5=baseline['diagnostics']['neighbor_overlap_at_5']
guard=report['reconstruction_gate']['synthetic_recipe_and_real_mae_within_2pct']
milestones=[]
for epoch in (10,20,30,40):
    name=f'ep{epoch}'; benchmark=json.loads((root/f'{candidate}_ep{epoch}_zgeo.json').read_text())
    global_metrics=benchmark['global_metrics']; bootstrap=benchmark['bootstrap']
    milestones.append({'epoch':epoch,'n5':benchmark['diagnostics']['neighbor_overlap_at_5'],
        'n10':benchmark['diagnostics']['neighbor_overlap_at_10'],'eligible_queries':global_metrics['query_count'],
        'precision_at_5_ci95':[bootstrap['precision_at_5']['ci95_low'],bootstrap['precision_at_5']['ci95_high']],
        'ndcg_at_5_ci95':[bootstrap['ndcg_at_5']['ci95_low'],bootstrap['ndcg_at_5']['ci95_high']],
        'macro_auroc':benchmark['classifier_metrics']['macro_auroc'],
        'tokenizer_reconstruction_guardrail_passed':bool(guard[name])})
eligible=[item for item in milestones if item['tokenizer_reconstruction_guardrail_passed'] and item['n5']>baseline_n5]
best=max(eligible,key=lambda item:(item['n5'],item['n10'])) if eligible else None
decision={'candidate':str(candidate),'stage1_gate_passed':True,'stage2_completed':True,
    'baseline_n5':baseline_n5,'milestones':milestones,'eligible_for_independent_replication':best is not None,
    'best_eligible_milestone':best,'adopted':False,'automatic_adoption':False}
(root/'candidate_decision.json').write_text(json.dumps(decision,indent=2,allow_nan=False)+'\n')
print('CANDIDATE DECISION:',json.dumps(decision,indent=2),flush=True)
PY
}

if [[ "$printing" != 1 ]]; then
  if [[ ! -d /Volumes/CrucialX9 ]]; then printf 'CrucialX9 is not mounted\n' >&2; exit 1; fi
  if [[ -e "$root" || -e "$suite_log" || -e "$local_log" || -L "$local_log" ]]; then printf 'Refusing existing top-five output or logs\n' >&2; exit 1; fi
  if pgrep -f '[p]ython.*-m scripts[.]train' >/dev/null; then printf 'An active scripts.train process exists; refusing concurrent training\n' >&2; exit 1; fi
  mkdir -p "$(dirname "$root")" "$log_root" logs
  mkdir "$root" || { printf 'Could not reserve unique top-five output directory\n' >&2; exit 1; }
  : > "$suite_log"
  ln -s "$suite_log" "$local_log"
  "$python" - "$root" "$seed" <<'PY' | tee -a "$suite_log"
import shutil,sys,torch,zarr
from pathlib import Path
from scripts.train import assert_train_validation_consistency
root,seed=sys.argv[1:]
assert torch.backends.mps.is_available(),'MPS is required'
assert shutil.disk_usage(Path(root).parent).free > 150*1024**3,'need 150 GiB free for checkpoints and reports'
assert_train_validation_consistency('data/synth_train_anchored_32-32-64.zarr','data/synth_val_v2_uniform_32-32-64.zarr')
for path in ('data/synth_train_anchored_32-32-64.zarr','data/synth_val_v2_uniform_32-32-64.zarr',
             'data/synth_val_v2_32-32-64.zarr','data/real_train_32-32-64.zarr',
             'data/real_validation_32-32-64.zarr','data/real_test_32-32-64.zarr'):
    store=zarr.open(path,mode='r'); assert tuple(store['patches'].shape[1:])==(32,32,64),path
    assert store.attrs['scaling_mode']=='divide_by_std',path
for path in ('checkpoints/geoaware_p0b_r2_seed20260926_r2/vae_best.pt',
             'docs/benchmarks/geoaware_p0b_r2_seed20260926_r2_ep30_zgeo.json',
             'docs/benchmarks/frozen_validation_manifest_v2.json',
             '/Volumes/CrucialX9/pretrain_v2_checkpoints/sweep_20260620_080306_r006_u3_h32-64-128_lp0p000_tv0p001/checkpoint_epoch_0050.pt',
             '/Volumes/CrucialX9/pretrain_v2_checkpoints/sweep_20260620_035309_r004_u3_h40-80-160_lp0p000_tv0p001/checkpoint_epoch_0055.pt',
             '/Volumes/CrucialX9/pretrain_v2_checkpoints/sweep_20260621_232252_r007_u4_h40-74-138-256_lp0p000_tv0p001/checkpoint_epoch_0085.pt',
             '/Volumes/CrucialX9/pretrain_v2_checkpoints/sweep_20260621_104613_r001_u4_h40-74-138-256_lp0p000_tv0p010/checkpoint_epoch_0050.pt',
             '/Volumes/CrucialX9/pretrain_v2_checkpoints/sweep_20260622_085428_r011_u4_h32-64-128-256_lp0p000_tv0p010/checkpoint_epoch_0095.pt'):
    assert Path(path).is_file(),f'missing required experiment input: {path}'
print(f'TOP5 PREFLIGHT PASS: seed={seed}, five source encoders, P0b, datasets/manifest, MPS, scaling/provenance and disk',flush=True)
print('Each U-Net checkpoint initializes only the matching ResNetV2 encoder; the VAE decoder is skip-free and trained through mu/z_geo.',flush=True)
PY
fi

status_line "TOP-FIVE ARCHITECTURE/LOSS SCREEN seed=$seed candidates=${candidates[*]} smoke=$smoke"
for candidate in "${candidates[@]}"; do run_candidate "$candidate"; done
if [[ "$printing" == 1 || "$smoke" == 1 ]]; then exit 0; fi
"$python" - "$root" "${candidates[@]}" <<'PY' | tee -a "$suite_log"
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); names=sys.argv[2:]
decisions=[]
for name in names:
    path=root/name/'candidate_decision.json'
    decisions.append(json.loads(path.read_text()) if path.is_file() else {'candidate':name,'status':'incomplete'})
eligible=[item for item in decisions if item.get('eligible_for_independent_replication')]
summary={'seed':20261008,'candidate_decisions':decisions,
    'screen_winners_for_independent_replication':[item['candidate'] for item in eligible],
    'adopted':False,'automatic_adoption':False,
    'interpretation':'architecture/loss bundles only; matched-loss architecture controls are required before claiming an architecture-specific causal effect'}
(root/'top5_screen_decision.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print('TOP-FIVE SCREEN DECISION:',json.dumps(summary,indent=2),flush=True)
PY
status_line 'TOP-FIVE SCREEN COMPLETE: results recorded; no automatic adoption or second-seed launch.'