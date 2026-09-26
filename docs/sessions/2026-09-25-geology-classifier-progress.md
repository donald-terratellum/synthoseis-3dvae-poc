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
| WP3 | Classifier decoder (patch mode) | Next | — | — |
| WP4 | Label-driven batch sampler | Not started | — | — |
| WP5 | Evaluation additions | Not started | — | — |
| WP6 | Voxel mode | Not started | — | Alignment no longer blocks it |
| WP7 | Suite script and docs | Not started | — | — |

## Benchmark results

| Run | Change vs current best | n@5 | n@10 | val_loss | Adopted |
|---|---|---:|---:|---:|---|
| baseline | Phase 2 epoch 20 (`z_geo`) | 0.139 | 0.227 | 0.128 | current |

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
