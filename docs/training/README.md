# VAE Training Component

This component covers:

1. Seismic patch data preparation.
2. 3D VAE training and checkpointing.

## Scope

- Build training zarr patch datasets from seismic cubes.
- Train the 3D VAE model (with optional discriminator path).
- Save checkpoints and training metrics.

## Quick start

The recommended end-to-end geology experiment is the staged suite below. It keeps the
validation volumes out of training, uses the verified label-depth offset, and runs a
leak-free control before the anchored/classifier experiments. It is compute-intensive:
sampling takes hours, followed by four training runs. Configure `SOURCE` for your machine;
the default path in the script is specific to the original macOS workstation.

```bash
uv sync
mkdir -p logs
export SOURCE=/path/to/fake_data

# Preview commands before starting the long run.
PRINT_COMMANDS=1 scripts/geoaware_classifier_suite.sh all

# Linux: run sequentially in the background and keep stdout/stderr in the log.
nohup scripts/geoaware_classifier_suite.sh all > logs/suite.log 2>&1 &
tail -f logs/suite.log
```

The stages can be run separately: `sample`, `train`, `benchmark`, `summary`. Completed
stores/runs/reports are skipped on rerun; set `OVERWRITE=1` to regenerate them. See the
suite section below for run definitions and current experiment conclusions.

For a small reconstruction-only smoke test, first run the suite's `sample` stage, then:

```bash
uv run python scripts/train.py \
  --data data/synth_train_anchored_32-32-64.zarr \
  --validation_data data/synth_val_uniform_32-32-64.zarr \
  --patch_size 32 32 64 \
  --batch_size 4 --number_batches 2 --epochs 1 \
  --seed 20260925 \
  --learning_rate 1e-4 \
  --weight_decay 1e-4 \
  --kl_schedule fixed --kl_fixed 1e-4 \
  --out_dir checkpoints/smoke_reconstruction \
  --best_checkpoint_name vae_best.pt \
  --no_save_epoch_checkpoints
```

For one-dimensional sizes, `--patch_size N` broadcasts to all axes; anisotropic sizes use
`--patch_size X Y Z`. The production suite uses 32×32×64 patches.

## Derived metadata pipeline

scripts/sample_patches.py now writes per-patch derived metadata arrays alongside patches. These keys are attached in zarr attrs under derived_metadata_keys and include:

- meta_dip_mean_deg
- meta_dip_std_deg
- meta_dip_range_deg
- meta_dip_mean_class
- meta_dip_range_class
- meta_azimuth_mean_deg
- meta_azimuth_circular_variance
- meta_fault_intersection_fraction
- meta_geologic_score_mean
- meta_sand_fraction
- meta_shale_fraction
- meta_flat_spot_fraction
- meta_onlap_fraction
- meta_onlap_variability
- meta_channel_fraction
- meta_channel_core_fraction
- meta_structural_complexity
- meta_water_fraction
- meta_closure_fraction

Metadata arrays are computed from source label volumes (for example `geologic_age_faulted`
and `fault_intersection_segments`) and are stored in the patch zarr. Training reads the
selected arrays only when the corresponding metadata, contrastive, classifier, or sampler
objective needs them. The class-presence target arrays are documented in the classifier
section below.

## Metadata-to-latent regression loss (legacy objective)

- --geology_loss_weight FLOAT
  - Default: 0.0
  - Set > 0 to activate metadata-to-latent similarity shaping.
- --geology_metadata_keys KEY [KEY ...]
  - Defaults to `DEFAULT_DERIVED_METADATA_KEYS` in `scripts/train.py`; this is a subset of
    the arrays written by the sampler, not every array in the inventory above.
  - Explicitly pass this flag to choose or reorder metadata keys for ablations.

## Latent geology validation diagnostics

When geology metadata is enabled, validation compares pairwise cosine similarity of deterministic encoder `mu` vectors with pairwise cosine similarity of the selected geology metadata. Diagnostic cubes use the tokenizer's deterministic preprocessing (per-cube standard-deviation normalization followed by trace-extrema retention), so augmentation randomness does not contaminate epoch-to-epoch trends. This is the same preprocessing, latent representation, and similarity family used by the seismic tokenizer.

- `validation/geology_latent_pair_cosine_correlation`: Pearson correlation across all validation-pair cosine values. Higher is better; `1` means identical pair ordering.
- `validation/geology_latent_cosine_separation`: mean latent cosine for the most geologically similar 10% of pairs minus the mean for the least similar 10%. Larger positive values indicate better separation.
- `validation/geology_latent_neighbor_overlap`: fraction of each patch's metadata top-k neighbors also found among its latent top-k neighbors. Higher is better and most directly reflects tokenizer retrieval behavior.
- `validation/geology_latent_similar_cosine` and `validation/geology_latent_dissimilar_cosine`: the two components of the separation metric.

Configure diagnostic cost and retrieval neighborhood size with:

```bash
--geology_diagnostic_max_samples 512 \
--geology_diagnostic_neighbor_k 5
```

The values are written to TensorBoard each epoch. Correlation, separation, and neighbor overlap are also appended to `training_metrics.csv` and printed after the epoch summary. Monitor trends on validation data rather than absolute training-batch values.

## Current recommendations and experiment status

- For the evaluated geology-aware experiments, use
  `scripts/geoaware_classifier_suite.sh`; it owns the current patch sizes, splits, offsets,
  seeds, warm start, and flags. Avoid copying old hyperparameter snippets into a new run.
- The suite's training data contains 108,000 patches from 180 labeled training volumes;
  validation contains 4,000 uniform patches from 20 labeled validation volumes. Five of the
  25 validation volumes have no geology label arrays and are skipped.
- The old 0.139 checkpoint was trained on volumes overlapping the frozen benchmark source.
  Use the suite's leak-free `r1ctrl` run as the comparison control for new experiments.
- Current experiments show a modest R2 improvement over that control, but no repeatable
  result exceeds the previous 0.1385 score. The best classifier macro AUROC is below the
  0.75 sanity gate. Keep the Phase 2 checkpoint as the adopted model; voxel mode (WP6/R5)
  is not currently justified. See the [progress log](../sessions/2026-09-25-geology-classifier-progress.md)
  for measured results and the [plan](../plans/2026-09-25__geology_classifier_decoder_and_class_anchored_sampling_plan.md)
  for remaining gates.

## Minimal smoke command

After running the suite's `sample` stage, this checks the training CLI and data loader with
two batches. It does not test geology supervision:

```bash
uv run python scripts/train.py \
  --data data/synth_train_anchored_32-32-64.zarr \
  --validation_data data/synth_val_uniform_32-32-64.zarr \
  --patch_size 32 32 64 \
  --batch_size 4 \
  --number_batches 2 \
  --epochs 1 \
  --learning_rate 1e-4 \
  --weight_decay 1e-4 \
  --kl_schedule fixed \
  --kl_fixed 1e-4 \
  --out_dir checkpoints/smoke_reconstruction \
  --best_checkpoint_name vae_best.pt \
  --no_save_epoch_checkpoints
```

## Troubleshooting: NaN or Inf validation loss

Symptoms: training loss in the hundreds of thousands, validation loss nan/inf from epoch 1.

Root cause: the sampled zarr file has fewer written patches than its requested size. If the
sampler writes fewer than `n_patches`, the preallocated remainder is zero-filled. PMSE can
then become very large or non-finite on those empty rows. Check `n_written` before training;
the current suite uses MAE reconstruction loss.

Check:

```bash
uv run python - <<'PY'
import zarr, numpy as np
z = zarr.open('data/synth_train_anchored_32-32-64.zarr', mode='r')
patches = np.asarray(z['patches'])
zero_rows = int(np.all(patches.reshape(len(patches), -1) == 0.0, axis=1).sum())
print(f'zero patches: {zero_rows}/{len(patches)} ({100.0*zero_rows/len(patches):.1f}%)')
PY
```

Fix: check the sampler's final `n_written` attribute and ensure all requested patches were
written. The suite's standard training set uses 600 patches × 180 labeled training volumes
= 108,000 patches. Do not use the old preallocated zarr if it contains unwritten zero rows.

If training fails with an error like:

- KeyError: a required classifier target such as 'label_presence_fault' was not found in the dataset.

then the patch zarr predates the label-target schema or classifier flags were enabled on an
unlabeled dataset. Regenerate it with the suite's `sample` stage. For non-classifier legacy
training, leave `--geology_classifier` off and use only metadata keys present in the dataset.

Quick check:

```bash
uv run python - <<'PY'
import zarr
z = zarr.open('data/synth_train_anchored_32-32-64.zarr', mode='r')
print('derived_metadata_keys:', z.attrs.get('derived_metadata_keys'))
print('has label_presence_fault:', 'label_presence_fault' in z)
PY
```

`uniform` and `class_anchored` sampling skip source volumes without all label arrays; review
`skipped_volumes_missing_labels` in the output attrs before training/evaluation.

## Choosing and repeating experiments

Use the suite for the evaluated recipe instead of the older 120-epoch
`--geology_loss_weight` examples. Its named runs change one main factor at a time:

- `r1ctrl`: leak-free geoscore control;
- `r1`: class-anchored sampling;
- `r2`: R1 plus the patch classifier;
- `r3`: R2 plus presence-label strata and per-batch class quotas.

Set `RUNS="r2"` to rerun only R2, `EPOCHS=40` to choose the epoch count, and
`BENCH_EPOCHS="10 20 30 40"` to choose evaluation epochs. The suite does not expose the R4
classifier-weight sweep; use the experiment commands in the session summary for that
separate ablation.

**Cautions:**

- Compare new runs with `r1ctrl`, because the historical 0.139 checkpoint was trained on
  volumes overlapping the frozen validation source. The prior results show only a modest
  repeatable R2 gain over the leak-free control; no repeatable result exceeded 0.1385.
- The classifier sanity gate is macro AUROC ≥ 0.75. Current runs are below it; the highest
  observed value was 0.677. Do not start voxel mode unless retrieval improvement is
  repeatable and the WP6 gate is met.
- Compare reconstruction `val_loss` only on the same validation store and loss settings;
  regressions over 2% are not acceptable for adoption.
- Do not sample training from the parent directory without `--exclude_dir validation`.
  Keep `data/synth_val_32-32-64.zarr` and the frozen manifest unchanged for benchmark
  comparability.

## Geology classifier and class-anchored sampling

Plan: [2026-09-25__geology_classifier_decoder_and_class_anchored_sampling_plan.md](../plans/2026-09-25__geology_classifier_decoder_and_class_anchored_sampling_plan.md).
Progress and findings: [2026-09-25-geology-classifier-progress.md](../sessions/2026-09-25-geology-classifier-progress.md).

Set `SOURCE` to a directory containing the source volumes and its nested `validation/`
directory, then run the pipeline on Linux:

```bash
cd /path/to/synthoseis-3dvae-poc
uv sync
export SOURCE=/path/to/fake_data
mkdir -p logs

# Preview commands before starting the long run.
PRINT_COMMANDS=1 scripts/geoaware_classifier_suite.sh all

# Run sampling, training, benchmarks, and summary sequentially in the background.
nohup env SOURCE="$SOURCE" scripts/geoaware_classifier_suite.sh all > logs/suite.log 2>&1 &
echo $! > logs/suite.pid
tail -f logs/suite.log
```

Stages are `sample`, `train`, `benchmark`, and `summary`. For example,
`RUNS="r1ctrl r2" scripts/geoaware_classifier_suite.sh train` selects runs. Set
`OVERWRITE=1` only when you intend to delete and rebuild completed outputs. `nohup` is
optional if you want to keep the command attached to the current terminal; Linux does not
need macOS's `caffeinate`.

The runs, one primary variable each, all warm-started from Phase 2 epoch 20:

| Run | Training data | Adds |
| --- | --- | --- |
| `r1ctrl` | geoscore sampling, validation volumes excluded | nothing; this is the leak-free baseline |
| `r1` | class-anchored | anchored data |
| `r2` | class-anchored | + classifier decoder (focal, weight 0.1) |
| `r3` | class-anchored | + presence-label strata and quotas `fault_x=1 flat_spot=1 channel=1` |

Compare r1, r2, and r3 against `r1ctrl`, not 0.139. The old training set included the
validation volumes.

The script skips finished items: sampled stores that have the
`n_written` attr, runs whose last epoch checkpoint exists, and existing benchmark reports.
`OVERWRITE=1` rebuilds them. It stops at once on missing inputs. Per-step logs go to
`logs/`. It does not regenerate or overwrite `data/synth_val_32-32-64.zarr` or the frozen
manifest. The old frozen set remains for historical comparison; `r1ctrl` is the fair,
leak-free control.

Opt-in classifier, strata, quota, and anchored-sampling flags preserve their legacy defaults.
WP1 intentionally corrected the default seismic key and sand/shale metadata semantics.

### scripts/sample_patches.py

| Flag | Default | Meaning |
| --- | --- | --- |
| `--exclude_dir NAME` (repeatable) | none | Skip volumes under a folder with this name; use `validation` for training sets |
| `--sampling_mode {geoscore,class_anchored,uniform}` | `geoscore` | Legacy geoscore-weighted, class-anchored, or natural-prevalence origins |
| `--class_quotas CLASS=FRACTION ...` | 0.125 for each class except sand | Share of anchored patches per class (fault, fault_x, channel, closure, onlap, sand, flat_spot) |
| `--background_fraction` | 0.25 | Share of uniform (non-anchored) patches |
| `--anchor_jitter {uniform,center}` | `uniform` | Where the anchor voxel lands inside the patch |
| `--max_patches_per_object` | 24 | Upper limit per fault or closure segment id, or per coarse cell for other classes; 0 disables |
| `--anchor_index_max_coords` | 200000 | Reservoir cap on stored anchor coordinates per class per volume |
| `--presence_min_voxels` | 32 | Minimum class voxels for `label_presence_<class> = 1` |
| `--onlap_threshold` / `--sand_threshold` | 0.5 / 0.5 | Class rules for onlap and sand |
| `--label_z_offset` | 0 | Use **1** for synthoseis data: `seismic[z]` matches `label[z + 1]` |
| `--store_label_patches` | off | Also write `(N, 7, X, Y, Z)` uint8 label patches |
| `--disjoint_from ZARR` (repeatable) | none | Fail if any source volume also appears in that store's `source_volumes` |

New per-patch arrays:

- `label_presence_<class>`: 7 arrays, uint8;
- `anchor_class`: int8, −1 means background;
- `inclusion_weight`: undoes the oversampling of rare classes;
- `meta_water_fraction` and `meta_closure_fraction`;
- `meta_sand_fraction` and `meta_shale_fraction` are now computed over rock only.

The `uniform` and `class_anchored` modes skip volumes with no label arrays (5 of the 25
validation volumes) and list them in the `skipped_volumes_missing_labels` attr.

### scripts/train.py

| Flag | Default | Meaning |
| --- | --- | --- |
| `--geology_classifier` | off | Build the patch-level classifier head on `mu` |
| `--geology_classifier_mode {patch}` | `patch` | Voxel mode is planned (WP6) |
| `--geology_classifier_hidden` | 256 | Hidden width of the classifier |
| `--geology_classifier_weight` | 0.0 | Loss weight; > 0 requires `--geology_classifier` and `label_presence_*` arrays |
| `--geology_classifier_loss {bce,focal}` | `bce` | Presence loss, weighted by `pos_weight` = clip(n_neg / n_pos, 1, 50) |
| `--geology_classifier_focal_gamma` | 2.0 | Focal gamma |
| `--geology_classifier_label_smoothing` | 0.05 | For the dip-class cross-entropy terms |
| `--geology_classifier_classes ...` | all 9 targets | Targets included in the loss |
| `--geology_strata_source {metadata,presence_labels}` | `metadata` | Build sampler and SupCon strata from presence labels (keeps the rarest classes) |
| `--geology_strata_classes ...` | six classes, no sand | Classes used for presence strata |
| `--geology_batch_class_quota CLASS=COUNT ...` | none | Minimum patches per batch containing each listed class |
| `--no_geology_calibration_inclusion_weight` | weighting on | Turn off `inclusion_weight` weighting of the metadata calibration |

Checkpoints store `geology_classifier`, `geology_classifier_mode`, and
`geology_classifier_hidden`. Warm-starting from a checkpoint without the classifier works.
The tokenizer ignores the classifier weights.

### scripts/evaluate_geology_benchmark.py

| Flag | Default | Meaning |
| --- | --- | --- |
| `--classifier_data ZARR` | none | Natural-prevalence set (`--sampling_mode uniform`); adds `classifier_metrics` to the report |
| `--classifier_threshold_data ZARR` | none | Training split used to tune per-class F1 thresholds (default: fixed 0.5) |
| `--classifier_max_samples` | 5000 | Maximum patches read from each classifier dataset |
| `--classifier_preprocess {tokenizer,extrema}` | `tokenizer` | Retrieval-path or training-path input preprocessing |

`classifier_metrics` includes per-class AUROC, average precision, and F1. It also has macro
averages, dip-class accuracy against the majority-class rate, confusion matrices, and
`sanity_gate` (macro AUROC ≥ 0.75, and dip accuracy above the majority rate). n@5 / n@10
on the frozen manifest remain the primary metric.

### scripts/verify_label_alignment.py

This is a repeatable data-audit utility, not a test module. It measures the label/seismic
depth offset on real volumes; the verified convention is `seismic[z]` ↔ `label[z + 1]`.

```bash
uv run python scripts/verify_label_alignment.py \
  --source "$SOURCE" \
  --n_volumes 20 \
  --out_json data/wp0_label_alignment.json
```

## Further work and scope limits

- **Voxel classifier (WP6/R5):** not implemented. The current R2/R4 results do not show a
  repeatable n@5 improvement sufficient to justify the voxel-label storage and training
  cost; classifier macro AUROC also remains below 0.75. Keep `--geology_classifier_mode`
  at `patch`.
- **Per-batch segment-ID telemetry:** deferred. Segment IDs are not stored per sampled patch;
  current batch logs report per-class shares and quota fallbacks.
- **Tokenizer UI classifier filters:** not implemented; this is an optional follow-up after
  a classifier checkpoint is adopted.
- **Preprocessing parity:** classifier evaluation supports `tokenizer` and `extrema`
  preprocessing. Training uses dataset-scaled inputs plus augmentation choices, while the
  tokenizer normalizes each cube individually. Compare the modes when interpreting classifier
  scores; do not treat them as directly interchangeable.

## Main files

- scripts/train.py
- scripts/sample_patches.py
- scripts/evaluate_geology_benchmark.py
- scripts/geoaware_classifier_suite.sh
- src/model.py
- src/geology_classifier.py
- src/geology_sampler.py
- src/augmentations.py

## Notes

- Patch size can be specified as one value (broadcast to all three axes) or three values (X Y Z).
- Checkpoints are written under configured output directories (for example checkpoints/).

## Advanced experiment guides

- Latent alignment and encoder/decoder balancing guide (Markdown): [latent_alignment_experiments.md](latent_alignment_experiments.md)
- Latent alignment and encoder/decoder balancing guide (HTML): [latent_alignment_experiments.html](latent_alignment_experiments.html)
