cd /Users/donaldpg/synthoseis-3dvae-poc

rm -rf data/synth_train_32-32-64.zarr/ data/synth_val_32-32-64.zarr/

# Regenerate train patches — must fill all 120,000 slots
uv run python scripts/sample_patches.py \
  --source '/Volumes/Crucial X9/fake_data' \
  --patch_size 32 32 64 \
  --n_patches 120000 \
  --n_per_volume 600 \
  --seismic_key seismicCubes_cumsum_fullstack \
  --geoscore_key geologic_score \
  --out data/synth_train_32-32-64.zarr

# Regenerate val patches
uv run python scripts/sample_patches.py \
  --source '/Volumes/Crucial X9/fake_data' \
  --patch_size 32 32 64 \
  --n_patches 24000 \
  --n_per_volume 200 \
  --seismic_key seismicCubes_cumsum_fullstack \
  --geoscore_key geologic_score \
  --out data/synth_val_32-32-64.zarr

# Re-train the 3D VAE with the new data and the new geology loss
uv run python scripts/train.py \
  --data data/synth_train_32-32-64.zarr \
  --validation_data data/synth_val_32-32-64.zarr \
  --patch_size 32 32 64 \
  --batch_size 12 \
  --number_batches 300 \
  --epochs 200 \
  --augment \
    --vertical_warp_prob 0.5 \
    --mixup_augment_prob 0.2 \
    --input_decimate_trilinear_prob 0.05 \
  --input_scaling divide_by_std \
  --learning_rate 5e-4 \
  --weight_decay 1e-4 \
  --kl_schedule warmup \
  --kl_start 0.0 \
  --kl_end 1e-3 \
  --kl_warmup_epochs 20 \
  --reconstruction_loss mae \
  --loss_mse_weight 1.0 \
  --lpips_weight 0.1 \
  --lr_scheduler plateau \
  --lr_scheduler_patience 4 \
  --lr_scheduler_factor 0.5 \
  --early_stopping_patience 12 \
  --geology_loss_weight 0.03 \
  --geology_diagnostic_max_samples 512 \
  --geology_diagnostic_neighbor_k 5 \
  --geology_metadata_keys \
    meta_fault_fraction \
    meta_fault_intersection_fraction \
    meta_channel_fraction \
    meta_channel_core_fraction \
    meta_flat_spot_fraction \
    meta_onlap_fraction \
    meta_onlap_variability \
  --representative_selection_epoch 5 \
  --representative_plot_interval 5 \
  --refresh_representative_examples \
  --adaptive_sampling_by_mse \
    --sampling_snapshot_interval 5 \
    --sampling_improvement_window 5 \
    --sampling_improvement_weight .35 \
    --sampling_snapshot_filename SAMPLING_SNAPSHOT \
  --best_checkpoint_name vae_best.pt \
  --out_dir checkpoints/synth_geoaware_v1b \
  --resume /Users/donaldpg/synthoseis-3dvae-poc/checkpoints/synth_geoaware_v1b/vae_epoch657.pt
