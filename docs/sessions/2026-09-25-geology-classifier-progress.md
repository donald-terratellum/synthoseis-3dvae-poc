# 2026-09-25 — Geology classifier decoder and class-anchored sampling: progress log

Living document. Add a section under **Work package log** as each WP is finished, and keep
the **Status** table current.

- Branch: `geology-classifier-2026-09-25` (from `encoder-improvement-2026-08-21`)
- Plan: [2026-09-25__geology_classifier_decoder_and_class_anchored_sampling_plan.md](../plans/2026-09-25__geology_classifier_decoder_and_class_anchored_sampling_plan.md)
- Parent plan: [2026-09-01_geo-aware_improvement_plan.md](../training/2026-09-01_geo-aware_improvement_plan.md)

---

## Background

The 3D VAE encodes 32×32×64 seismic patches. A projection head maps the latent mean `mu` to a
unit-norm embedding, `z_geo`. The seismic tokenizer uses `z_geo` for similarity search:
patches with similar geology should rank near each other by cosine similarity.

- **Primary metric:** neighbor overlap at 5 (n@5) on the frozen 7-key validation manifest
  (`docs/benchmarks/frozen_validation_manifest.json`), tie-broken by n@10.
- **Current best:** `checkpoints/geoaware_v3_phase2_20260831/vae_epoch20.pt`, n@5 0.139 and
  n@10 0.227 (the `mu` baseline is 0.077). The recipe is supervised contrastive loss on
  `z_geo` with a geology-aware batch sampler.
- **Why a new plan:** the parent plan found that the current recipe has plateaued at about
  0.14. More epochs and a uniformity regularizer did not help, so gains need a structural
  change.
- **Target:** n@5 of 0.18–0.25 without a reconstruction `val_loss` regression greater than 2%.

The plan adds two structural levers:

1. **Classifier decoder on `mu`.** A multi-task head predicts presence of fault, fault
   intersection, channel, closure, onlap, sand, and flat spot, plus dip-mean and dip-range
   classes. This gives the encoder a direct supervised geology signal.
2. **Class-anchored patch sampling plus a label-driven batch sampler.** Rare classes (fault
   intersections are about 2% of uniform patches, flat spots about 5%) become common enough to
   learn, and each batch contains positives for them.

Source data: 205 synthoseis volumes at `/Volumes/CrucialX9/fake_data` (the 25 under
`validation/` are held out). Each volume has seismic `seismicCubes_cumsum_fullstack`
(300×300×1499) and label arrays (300×300×1510).

## Status

| WP | Scope | Status | Tests | Notes |
|---|---|---|---|---|
| prep | Source path fix (`CrucialX9`), dip-range / dip-class metadata in `sample_patches.py` | Done | Suite OK | Done before WP0 |
| WP0 | Label/seismic depth alignment | **Done** | 9 new; suite 112 OK | `label_z_offset = 1` |
| WP1 | Sand/shale fix, water/closure metadata, seismic key default, `--exclude_dir` | **Done** | 9 new; suite 121 OK | Train/val volume lists are disjoint (180 / 25) |
| WP2 | Class-anchored sampler | **Done** | 18 new; suite 139 OK | Weighted presence matches uniform on real data |
| WP3 | Classifier decoder (patch mode) | **Done** | 18 new; suite 157 OK | Warm-start smoke run trains; tokenizer ignores head |
| WP4 | Label-driven batch sampler | **Done** | 16 new; suite 173 OK | Quotas met in 100% of smoke batches |
| WP5 | Evaluation additions | **Done** | 13 new; suite 186 OK | R0 reproduces n@5 0.139; 5 of 25 validation volumes have no labels |
| WP6 | Voxel mode | Not started | — | Gated on WP3 improving n@5 (needs R2) |
| WP7 | Suite script and docs | **Done** | 4 new; suite 190 OK | `scripts/geoaware_classifier_suite.sh` |

## Benchmark results

| Run | Change vs current best | n@5 | n@10 | val_loss | Adopted |
|---|---|---:|---:|---:|---|
| baseline | Phase 2 epoch 20 (`z_geo`) | 0.139 | 0.227 | 0.128 | current |
| R0 | Re-benchmark of baseline with the WP5 script (2026-09-25) | 0.1385 | 0.2269 | — | sanity OK |

## Known defects (from the plan) and where they are fixed

| Defect | Fix in |
|---|---|
| Sand/shale fraction maps lithology with `(x+1)/2`, so shale counts as 0.5 sand and water counts as shale (lithology: −1 water, 0 shale, 1 sand) | WP1 (fixed) |
| `rglob` over `fake_data` also picks up `fake_data/validation`, leaking validation volumes into training | WP1 (fixed: `--exclude_dir`) |
| Default `--seismic_key` has a double underscore and matches no volume | WP1 (fixed) |
| Labels are read at the seismic origin, one sample off | WP2 (fixed when `--label_z_offset 1` is passed; the default of 0 keeps the old behavior) |

---

## Work package log

### WP0 — Label/seismic depth alignment (done)

**Question:** labels have depth 1510 and seismic has 1499. How do they line up?

**Method:**
- Traced the synthoseis code to see where the depth difference comes from.
- Confirmed empirically by correlating seismic reflection strength (envelope of the depth
  derivative) with lithology boundaries for offsets −15…+15.

**Result:** `seismic[z]` matches `label[z + 1]`.
- The labels carry 10 padding samples at the bottom. `Geomodels.py` allocates
  `cube_shape[2] + pad_samples`, and `Seismic.py` trims the elastic properties with
  `[:, :, :nz]`.
- The reflectivity (and therefore the seismic) has one sample fewer than the 1500-sample
  rock property model. `rfc[k]` is the interface between samples `k` and `k+1`, and the
  cumulative sum carries each step to the layer below. Synthoseis's own QC overlays use
  offset 0, so they are one sample off.
- On real data, the mean correlation peak is at +1 on both 8 and 20 volumes (sub-sample
  estimates +0.86 to +1.21). 17 of 20 volumes are within ±1. The outliers have two
  correlation peaks, most likely from regular layer spacing.

**Artifacts:**
- [scripts/verify_label_alignment.py](../../scripts/verify_label_alignment.py)
- [tests/test_label_alignment.py](../../tests/test_label_alignment.py)
- JSON reports in `data/` (gitignored)

**Decision:** proceed to WP1.

### WP1 — Data fixes and metadata (done)

**Goal:** fix the known metadata and data-split defects before building new samplers on
top of them.

**Changes** (in [scripts/sample_patches.py](../../scripts/sample_patches.py)):
- **Sand/shale fractions.** They are now computed over rock voxels only (lithology ≥ 0).
  A voxel counts as sand at lithology ≥ 0.5. Both fractions are 0 when a patch is all water.
- **New metadata.** `meta_water_fraction`, and `meta_closure_fraction`
  (`closure_segments_id > 0`).
- **Seismic key.** The default `--seismic_key` is now `seismicCubes_cumsum_fullstack`.
- **`--exclude_dir NAME`** (repeatable). It skips volumes under any folder with that name
  below `--source`.
- **Output attrs.** `exclude_dirs` and `sand_threshold` are recorded.
- **Shell scripts.** The training-set sampling steps in `scripts/train_vae3d.sh` and
  `scripts/geoaware_next_steps.sh` now pass `--exclude_dir validation`.

**Result:**
- Tests: a synthetic volume with a known water/shale/sand/closure layout gives exact
  fractions. Excluded volumes are never listed. An end-to-end `main()` run with the default
  seismic key writes patches only from non-excluded volumes.
- Real data:
  - Volume lists: 205 in total, 180 with `validation` excluded, and 25 in `validation/`.
    The two sets are disjoint, and together they cover all 205.
  - 200 patches from one volume: mean sand fraction 0.17 (sand present in 92% of patches),
    water in 2% of patches, closure in 23%. Sand + shale = 1 on every patch that contains
    rock.

**Compatibility:**
- The seven retrieval keys and the frozen manifest are unchanged, so the 0.139 baseline
  still applies.
- The training defaults in `scripts/train.py` are unchanged, so datasets built before WP1
  still load.
- Datasets built before WP1 keep the wrong sand/shale values. Regenerate them before using
  sand/shale as targets.

**Decision:** proceed to WP2.

### WP2 — Class-anchored sampler (done)

**Goal:** make rare geology classes common in training patches, and record per-patch class
labels for the classifier (WP3) and the batch sampler (WP4).

**Method** (in [scripts/sample_patches.py](../../scripts/sample_patches.py)):
- **`--sampling_mode {geoscore,class_anchored,uniform}`.** The default, `geoscore`, keeps the
  old behavior.
- **Anchor index.** For each volume, the sampler reads each label array chunk by chunk,
  finds the voxels of each class, and keeps a random sample of their coordinates (capped per
  class).
- **Placing patches.** Each patch slot is assigned to a class according to the quotas. The
  patch is then placed so that a random voxel of that class falls at a random position
  inside it. If a volume has no voxels of a class, that slot becomes a uniform (background)
  patch, and the fallback is counted.
- **`--max_patches_per_object`.** Limits how many patches can come from one object. Faults
  and closures are counted per segment id. Other classes have no ids, so they are counted
  per coarse, patch-sized region.
- **New per-patch arrays:**
  - `label_presence_<class>` for 7 classes: 1 if the patch has at least 32 voxels of the
    class;
  - `anchor_class`: the class the patch was anchored on;
  - `inclusion_weight`: a weight that undoes the oversampling of rare classes;
  - optional `label_patches` (`--store_label_patches`), for voxel mode.
- **Attrs.** All sampling parameters, the class order, and the anchored and fallback counts
  are recorded.
- **`--label_z_offset`.** Applies to label and metadata reads. The default is 0 for
  backward compatibility; pass 1 for synthoseis data.
- **`--disjoint_from OTHER.zarr`.** Fails if any source volume overlaps another output
  store (train/validation check).

**Result:**
- **Tests.** Covered by the unit tests:
  - `geoscore` output matches the pre-WP2 code exactly (golden hashes);
  - anchored patches always contain their class;
  - achieved shares are within ±3 percentage points of the quotas;
  - fallback counts are correct when a class is absent;
  - the per-object cap is honored;
  - weights are finite and positive;
  - label patch shapes and dtypes are correct;
  - the index reads one chunk at a time;
  - `--disjoint_from` rejects overlapping volumes.
- **Real volume (run_1204, 600 patches).** Class-anchored sampling takes 51 s and 1.8 GB
  peak memory; uniform sampling takes 46 s. That is about 2.5 h for 180 volumes.

  | Class | Uniform | Anchored | Anchored, weighted |
  |---|---:|---:|---:|
  | fault | 0.088 | 0.308 | 0.097 |
  | fault_x | 0.003 | 0.147 | 0.009 |
  | channel | 0.020 | 0.132 | 0.021 |
  | closure | 0.117 | 0.413 | 0.103 |
  | onlap | 0.107 | 0.235 | 0.109 |
  | flat_spot | 0.052 | 0.317 | 0.057 |
  | none of the six | 0.733 | 0.250 | — |

  - Every anchored patch contains its anchor class.
  - All 600 origins are unique.
  - The weighted presence rates recover the uniform rates, so the weights work.
- **Tuning note.** This volume has only 4 fault ids, so the original cap of 8 per object
  moved 48 of the 75 fault slots to background. The default is now **24**; it is an upper
  limit, so it should be generous. Raise it further or accept a lower fault share (faults
  are already 31% of patches through other anchors).

**Decision:** proceed to WP3.

### WP3 — Classifier decoder, patch mode (done)

**Goal:** give the encoder a direct supervised geology signal through `mu`, the same tensor
that `z_geo` is projected from.

**Method:**
- **Model** ([src/model.py](../../src/model.py)). A new `GeologyClassifierDecoder` sits on
  `mu`: a shared hidden layer (Linear → LayerNorm → GELU → Dropout), then three outputs:
  - presence logits for 7 classes;
  - 6 dip-mean classes;
  - 6 dip-range classes.

  `VAE3D` builds it only with `geology_classifier=True` and exposes it as
  `model.classify(mu)`. `forward()` is unchanged. `pos_weight` is stored with the model
  weights, so it is saved in every checkpoint.
- **Loss** ([src/geology_classifier.py](../../src/geology_classifier.py)). Presence uses BCE
  or focal BCE weighted by `pos_weight`, averaged over the selected classes. `pos_weight` is
  clip(n_neg / n_pos, 1, 50) per class, computed from the training split. Each selected dip
  target adds a cross-entropy term with label smoothing.
- **Training** ([scripts/train.py](../../scripts/train.py)):
  - New flags: `--geology_classifier`, `--geology_classifier_mode` (currently `patch`
    only), `--geology_classifier_hidden`, `--geology_classifier_weight`,
    `--geology_classifier_loss {bce,focal}`, `--geology_classifier_focal_gamma`,
    `--geology_classifier_label_smoothing`, `--geology_classifier_classes`.
  - The training dataset loads `label_presence_*` and the dip-class arrays only when the
    classifier weight is > 0, and fails with a clear message if they are missing.
  - A classifier weight > 0 requires `--geology_classifier`.
  - Checkpoints now store the classifier settings.
  - Warm-starting and resuming tolerate missing or unexpected classifier weights.
  - The epoch loss goes to TensorBoard as `train/geology_classifier_loss`.
- **Tokenizer** ([src/tokenizer/core/model_adapter.py](../../src/tokenizer/core/model_adapter.py)).
  The adapter ignores the classifier weights when loading a checkpoint.

**Result:**
- **Tests** ([tests/test_geology_classifier.py](../../tests/test_geology_classifier.py)):
  - output shapes are correct and all logits are finite;
  - `pos_weight` matches hand-computed values on known counts;
  - focal loss equals BCE at gamma 0;
  - the loss falls by more than half in 50 steps on a separable batch;
  - with the classifier off, one epoch reproduces the pre-WP3 loss and weights exactly;
  - a classifier weight of 0 is a no-op;
  - classifier gradients reach the encoder;
  - a warm start from a checkpoint without the classifier reports only classifier weights
    as missing, and a resume restores every weight exactly;
  - the tokenizer gives identical `z_geo` with and without the classifier weights;
  - missing presence arrays fail with a clear error.
- **Smoke run.** Two epochs of 10 batches, warm-started from the Phase 2 epoch-20
  checkpoint on 240 class-anchored patches, with focal loss at weight 0.1:
  - the classifier loss fell from 3.78 to 3.33;
  - `pos_weight` = [2.87, 1.0, 5.86, 1.0, 3.44, 1.0, 1.86];
  - the checkpoint loads in the tokenizer and returns unit-norm `z_geo`.
- **Bug found and fixed.** With only the classifier enabled, the dataset tried to read
  metadata keys it had never loaded. It now reads only the arrays it loaded, and a test
  covers this case.

**Notes:**
- `pos_weight` is 1 for a class with no positives. Fault intersections were absent from the
  smoke volume; the full 180-volume training split will have positives.
- The validation loss (`val_loss`) is still reconstruction only, so the 2% reconstruction
  guardrail still compares like with like. Classifier metrics on validation are WP5.

**Decision:** proceed to WP4.

### WP4 — Label-driven batch sampler (done)

**Goal:** build training batches and contrastive (SupCon) positive pairs directly from the
WP2 presence labels. Guarantee that rare classes appear in every batch, and fit the
metadata calibration at natural prevalence.

**Method:**
- **`--geology_strata_source presence_labels`.** A patch's stratum is the set of rare classes
  present in it. When more than `--geology_strata_max_active_keys` classes are present,
  only the rarest are kept, so positive pairs share rare classes. The same rule builds the
  sampler's strata and the per-batch SupCon labels. The default classes are the six rare
  ones (`--geology_strata_classes`); sand is excluded because it appears in about 90% of
  patches.
- **`--geology_batch_class_quota CLASS=COUNT`.** Sets a minimum number of patches containing
  each listed class in every batch. Quotas are filled before the positive-pair, background,
  hard-example, and negative steps. A quota that cannot be met is counted as a fallback.
- **Calibration weights.** The metadata calibration now uses the WP2 `inclusion_weight`
  when the dataset has it, via weighted quantiles. This is on by default and can be turned
  off with `--no_geology_calibration_inclusion_weight`. Unit weights take the original
  code path exactly.
- **Sampler stats.** New per-epoch stats, logged to TensorBoard and printed:
  - `class_share_<class>`: the share of each batch containing the class;
  - `class_quota_fallback_<class>`: quota shortfalls;
  - `class_quota_met_batch_rate`: the share of batches that met every quota.
- **Guards.** Presence strata and class quotas require `--geology_batch_sampler`. The sum
  of the quotas must not exceed the batch size.

**Result:**
- **Tests** ([tests/test_label_batch_sampler.py](../../tests/test_label_batch_sampler.py)):
  - presence strata match the expected signatures and keep the rarest classes;
  - with no quota, the sampler returns exactly the pre-WP4 batches (golden hash);
  - feasible quotas are met in every batch;
  - infeasible quotas are reported as fallbacks;
  - weighted calibration on a rebalanced dataset matches calibration on the
    natural-prevalence dataset within 2%, while the unweighted fit is clearly off;
  - SupCon produces a nonzero loss from presence strata.
- **Smoke run.** Two epochs of 10 batches on 300 class-anchored patches from run_1204,
  warm-started from Phase 2 epoch 20, with SupCon 0.5, classifier 0.1, and quotas
  `fault_x=1 flat_spot=1 channel=1`:
  - 100% of batches met the quotas, with no fallbacks;
  - batch shares: fault_x 21%, channel 17–18%, flat_spot 28–33%, closure 51–53%;
  - positive-pair rate 1.0, 7.4–8.0 unique strata per batch;
  - classifier loss 3.72 → 3.45.

**Notes:**
- Per-batch unique segment-id stats are not logged. WP2 does not store segment ids per
  patch; add that only if memorization becomes a concern.
- The background share achieved (16–21%) is above the 5% target. Patches with none of
  the six classes are also used when the sampler fills the rest of each batch.

**Decision:** proceed to WP5.

### WP5 — Evaluation additions (done)

**Goal:** report classifier quality on natural-prevalence validation data, while keeping
n@5 / n@10 on the frozen manifest as the primary metric.

**Method:**
- **Benchmark script** ([scripts/evaluate_geology_benchmark.py](../../scripts/evaluate_geology_benchmark.py)).
  New optional flags:
  - `--classifier_data`: a natural-prevalence zarr; class-anchored data is rejected;
  - `--classifier_threshold_data`: the training split, used to tune per-class F1
    thresholds (otherwise a fixed 0.5 is used);
  - `--classifier_max_samples`;
  - `--classifier_preprocess {tokenizer,extrema}`.

  A new `classifier_metrics` block reports:
  - per class: prevalence, AUROC, average precision (AP), threshold, and F1;
  - macro averages of AUROC, AP, and F1;
  - for dip-mean and dip-range: accuracy, majority-class rate, and a confusion matrix;
  - `sanity_gate`: macro AUROC ≥ 0.75, and dip accuracy above the majority-class rate.

  Without the new flags, the report is unchanged.
- **Metrics** ([src/geology_classifier.py](../../src/geology_classifier.py)): numpy-only
  AUROC, AP, best-F1 threshold, and confusion matrix.
- **Tokenizer adapter.** `VaeLatentAdapter(..., load_classifier=True)` loads the classifier
  head, and `classify_batch()` returns probabilities. By default the adapter still ignores
  the head.
- **Sampler fix** ([scripts/sample_patches.py](../../scripts/sample_patches.py)). The
  `uniform` and `class_anchored` modes now skip volumes with no label arrays and list them in
  `skipped_volumes_missing_labels`. The benchmark stops with a clear error if the evaluation
  set has no positive labels.

**Result:**
- **Tests** ([tests/test_classifier_eval.py](../../tests/test_classifier_eval.py) and 1 new
  sampler test):
  - metric values on hand-computed examples, including ties and absent classes;
  - perfect predictions pass the gate and uninformative ones fail it;
  - adapter probabilities match the model;
  - the existing report is byte-identical with or without classifier metrics;
  - class-anchored evaluation data and data with no positive labels are rejected;
  - volumes without labels are skipped.
- **R0 sanity.** The current best checkpoint on the frozen manifest gives n@5 0.1385 and
  n@10 0.2269, reproducing the baseline.
- **Pipeline check on real data.** A 75-step smoke classifier trained on one volume,
  evaluated on 600 uniform patches from the 20 labeled validation volumes:
  - macro AUROC 0.45 with tokenizer preprocessing and 0.53 with `extrema` preprocessing;
  - the gate correctly fails.

  This model is essentially untrained, so these numbers only show that the pipeline runs.

**Findings (important for R1–R3):**
- **5 of the 25 validation volumes (runs 0570–0574) contain no label arrays**, only
  seismic and `geologic_score`. All 180 training volumes have every label array.
  - Uniform validation sampling from `fake_data/validation` now uses 20 volumes. At
    `--n_per_volume 200` that gives 4,000 patches, not 5,000.
  - These 5 volumes give zero metadata in `geoscore` mode (not changed). They may also be
    in the frozen manifest's data.
- **The tokenizer's input preprocessing differs from the training inputs.**
  - Tokenizer: normalized by each patch's own standard deviation, then keep only trace
    extremes.
  - Training: dataset-wide scaling, then one of three transforms (keep extremes, sparse
    keep, or decimate).
  - On the smoke model's own training split, macro AUROC was 0.68 with training-style
    inputs and 0.57 with tokenizer inputs (sand: 0.70 vs 0.34).
  - This mismatch predates this work. It also affects `z_geo` retrieval, because n@5 is
    measured through the tokenizer path.
  - Report both variants in R2. Aligning the preprocessing is a separate follow-up and is
    not part of this plan.

**Decision:** proceed to WP7 (scripts and docs). WP6 (voxel mode) stays gated on WP3
improving n@5, which requires the R1/R2 training runs on the full sampled dataset.

### WP7 — Suite script and docs (done)

**Method:**
- [scripts/geoaware_classifier_suite.sh](../../scripts/geoaware_classifier_suite.sh)
  implements plan Section 8 in three stages (`sample | train | benchmark | all`), with
  `set -euo pipefail`.
  - **sample:** class-anchored training data (validation excluded, `--label_z_offset 1`,
    `--max_patches_per_object 24`), plus 4,000 uniform validation patches from the 20
    labeled validation volumes (`--disjoint_from` the training store).
  - **train:** the R3 settings, i.e. the Phase 2 recipe plus the classifier (focal loss,
    weight 0.1), presence strata, and quotas `fault_x=1 flat_spot=1 channel=1`.
  - **benchmark:** the frozen manifest plus classifier metrics, for each epoch in
    `BENCH_EPOCHS`.
  - Settings can be overridden with environment variables. `PRINT_COMMANDS=1` prints the
    commands without running them.
  - The script fails fast on missing inputs, and refuses to overwrite outputs unless
    `OVERWRITE=1` is set.
- [docs/training/README.md](../training/README.md) has a new section with tables of every
  new flag for `sample_patches.py`, `train.py`, and `evaluate_geology_benchmark.py`, plus
  the suite usage. The legacy quick-start now notes that its validation set overlaps the
  training volumes.

**Result:** [tests/test_classifier_suite_script.py](../../tests/test_classifier_suite_script.py) checks:
- the bash syntax;
- that every flag the suite passes exists in the target script's `--help`;
- that an unknown stage exits with code 2;
- that missing inputs fail fast.

**Decision:** implementation of WP0–WP5 and WP7 is complete. Next are the experiment runs.

---

## Next steps (experiments)

Everything below is automated by one script:

```bash
nohup caffeinate -i scripts/geoaware_classifier_suite.sh all > logs/suite.log 2>&1 &
```

1. **Sample** three stores:
   - class-anchored training data;
   - leak-free geoscore control data;
   - uniform validation data.

   Expect about 2.5 h each for the two training stores. Check the attrs
   (`anchored_counts`, `fallback_counts`, `skipped_volumes_missing_labels`) in
   `logs/sample_*.log` before trusting the training results.
2. **Train** `r1ctrl`, `r1`, `r2`, and `r3`, one primary variable each (plan Section 7).
   **Leakage found:** the old training set
   (`data/synth_train_32-32-64.zarr`) used all 205 volumes, including the 25 validation
   volumes that the frozen benchmark is drawn from. So the 0.139 baseline is optimistic.
   `r1ctrl` repeats the Phase 2 recipe on leak-free geoscore data and is the fair baseline.
3. **Benchmark** epochs 10/20/30/40 (`summary` prints the table), and add the rows to the
   results table. Adopt a run only if n@5 beats `r1ctrl` and `val_loss` regresses by no
   more than 2% relative to `r1ctrl` on the same validation set.
4. **Classifier checks.** Check `classifier_metrics.sanity_gate`, and compare
   `--classifier_preprocess tokenizer` with `extrema` (see the WP5 findings).
5. **WP6 (voxel mode)** only if R2 or R3 improves n@5.

<!-- Template for next WP:
### WPn — Title (status)

**Goal:**
**Method:**
**Result:**
**Artifacts:**
**Decision:**
-->

---

## Conventions

- Tests: `.venv/bin/python -m unittest discover -s tests` (stdlib `unittest`).
- Run benchmark scripts from the repo root. `scripts/tokenize.py` shadows the stdlib
  `tokenize` module.
- Do not commit benchmark manifests or milestone benchmark reports unless explicitly requested.
