#!/usr/bin/env bash
set -euo pipefail

run_encoder_arm() {
  local arm="$1" init_encoder="$2" stage1_classifier_weight="$3"
  local python="${PYTHON:-.venv/bin/python}"
  local seed="${SEED:-20260925}"
  local recon_epochs="${RECON_EPOCHS:-60}"
  local geology_epochs="${GEOLOGY_EPOCHS:-40}"
  local batches="${BATCHES:-450}"
  local log_dir="${LOG_DIR:-logs}"
  local out_root="${OUT_ROOT:-checkpoints/p5_p6_encoder}"
  local train_data="data/synth_train_anchored_32-32-64.zarr"
  local val_data="data/synth_val_v2_uniform_32-32-64.zarr"
  local bench_data="data/synth_val_v2_32-32-64.zarr"
  local manifest="docs/benchmarks/frozen_validation_manifest_v2.json"
  local real_val="data/real_validation_32-32-64.zarr"
  local real_test="data/real_test_32-32-64.zarr"
  local arch_args=(
    --encoder_arch resnetv2 --encoder_hidden_dims 32 64 128
    --encoder_depth_profile deeper --encoder_stage_blocks 3 5 8
    --encoder_norm instance --encoder_stem pretrain_v2 --encoder_input_axes zxy
    --decoder_hidden_dims 256 128 64 16 --decoder_block res
  )
  local metadata_keys=(
    meta_fault_fraction meta_fault_intersection_fraction
    meta_channel_fraction meta_channel_core_fraction
    meta_flat_spot_fraction meta_onlap_fraction meta_onlap_variability
  )
  local out="${out_root}/${arm}_seed${seed}"
  local stage1_out="${out}/stage1_reconstruction"
  local stage2_out="${out}/stage2_geology"
  local log="${log_dir}/${arm}_seed${seed}.log"
  local stage1_args=(
    scripts/train.py
    --data "$train_data" --validation_data "$val_data"
    --real_validation_data "$real_val" --real_test_data "$real_test"
    --patch_size 32 32 64 --batch_size 12 --number_batches "$batches" --epochs "$recon_epochs"
    --seed "$seed" "${arch_args[@]}" --freeze_encoder_epochs 5
    --augment --vertical_warp_prob 0.5 --phase_rotation_prob 0.0 --stretch_prob 0.0
    --dip_label_policy adjust --mixup_augment_prob 0.0
    --input_scaling divide_by_std --learning_rate 5e-4 --weight_decay 1e-4 --encoder_lr_mult 0.1
    --kl_schedule warmup --kl_start 1e-3 --kl_end 1e-3 --kl_warmup_epochs 1
    --reconstruction_loss mae --lpips_weight 0.1
    --lr_scheduler plateau --lr_scheduler_patience 6 --lr_scheduler_factor 0.5 --lr_scheduler_min_lr 1e-5
    --early_stopping_patience 999 --save_epoch_checkpoints
    --geology_projection --geology_proj_hidden 128 --geology_proj_dim 64
    --geology_contrastive_weight 0 --geology_uniformity_weight 0
    --geology_classifier --geology_classifier_mode patch
    --geology_classifier_weight "$stage1_classifier_weight"
    --geology_classifier_loss focal --geology_classifier_focal_gamma 2.0
    --geology_classifier_label_smoothing 0.05
    --geology_metadata_keys "${metadata_keys[@]}"
    --best_checkpoint_name vae_best.pt --out_dir "$stage1_out"
  )
  if [[ -n "$init_encoder" ]]; then stage1_args+=(--init_encoder_from "$init_encoder"); fi

  local printing="${PRINT_COMMANDS:-0}"
  run_logged() {
    local destination="$1"; shift
    if [[ "$printing" == "1" ]]; then printf '%q ' "$@"; printf '\n'; else "$@" 2>&1 | tee -a "$destination"; fi
  }

  if [[ "$printing" != "1" ]]; then
    mkdir -p "$log_dir" "$out"
    for required in "$train_data" "$val_data" "$bench_data" "$manifest" "$real_val" "$real_test"; do
      if [[ ! -e "$required" ]]; then echo "Missing encoder experiment input: $required" >&2; return 1; fi
    done
    if [[ -n "$init_encoder" && ! -f "$init_encoder" ]]; then
      echo "Missing pretrained encoder checkpoint: $init_encoder" >&2
      return 1
    fi
    : > "$log"
    printf '%s seed=%s encoder=E2 phase=off stretch=off real_mixing=off init=%s\n' \
      "$arm" "$seed" "${init_encoder:-scratch}" | tee -a "$log"
  fi

  run_logged "$log" "$python" "${stage1_args[@]}"

  run_logged "$log" "$python" scripts/train.py \
    --data "$train_data" --validation_data "$val_data" \
    --real_validation_data "$real_val" --real_test_data "$real_test" \
    --patch_size 32 32 64 --batch_size 12 --number_batches "$batches" --epochs "$geology_epochs" \
    --seed "$seed" "${arch_args[@]}" --resume "${stage1_out}/vae_best.pt" --resume_epoch 0 \
    --augment --vertical_warp_prob 0.5 --phase_rotation_prob 0.0 --stretch_prob 0.0 \
    --dip_label_policy adjust --mixup_augment_prob 0.0 \
    --input_scaling divide_by_std --learning_rate 5e-4 --weight_decay 1e-4 --encoder_lr_mult 0.1 \
    --kl_schedule warmup --kl_start 1e-3 --kl_end 1e-3 --kl_warmup_epochs 1 \
    --reconstruction_loss mae --lpips_weight 0.1 \
    --lr_scheduler plateau --lr_scheduler_patience 6 --lr_scheduler_factor 0.5 --lr_scheduler_min_lr 1e-5 \
    --early_stopping_patience 999 --save_epoch_checkpoints \
    --geology_projection --geology_proj_hidden 128 --geology_proj_dim 64 \
    --geology_contrastive_weight 0.5 --geology_contrastive_temperature 0.2 --geology_uniformity_weight 0 \
    --geology_classifier --geology_classifier_mode patch --geology_classifier_weight 0.1 \
    --geology_classifier_loss focal --geology_classifier_focal_gamma 2.0 --geology_classifier_label_smoothing 0.05 \
    --geology_batch_sampler --geology_batch_background_fraction 0.05 --geology_batch_hard_fraction 0.30 \
    --geology_batch_min_negative_strata 2 --geology_strata_source presence_labels \
    --geology_batch_class_quota fault_x=1 flat_spot=1 channel=1 \
    --geology_metadata_keys "${metadata_keys[@]}" \
    --geology_diagnostic_max_samples 512 --geology_diagnostic_neighbor_k 5 --geology_diagnostic_topk 5 10 20 \
    --best_checkpoint_name vae_best.pt --out_dir "$stage2_out"

  for epoch in 10 20 30 40; do
    local checkpoint="${stage2_out}/vae_epoch${epoch}.pt"
    local report="docs/benchmarks/${arm}_seed${seed}_ep${epoch}_zgeo.json"
    run_logged "$log" "$python" scripts/evaluate_geology_benchmark.py \
      --data "$bench_data" --manifest "$manifest" --checkpoint "$checkpoint" --use_geo_embedding \
      --metadata_keys "${metadata_keys[@]}" --classifier_data "$val_data" \
      --classifier_threshold_data "$train_data" --out_json "$report"
  done
}
