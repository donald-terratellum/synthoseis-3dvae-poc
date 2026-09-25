# 2026-09-25 — Geology classifier decoder and class-anchored sampling: progress log

Living document. Add a section under **Work package log** as each WP is finished, and keep
the **Status** table current.

- Branch: `geology-classifier-2026-09-25` (from `encoder-improvement-2026-08-21`)
- Plan (local, untracked because `docs/plans/` is in `.git/info/exclude`):
  `docs/plans/2026-09-25__geology_classifier_decoder_and_class_anchored_sampling_plan.md`
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
| WP1 | Sand/shale fix, water/closure metadata, seismic key default, `--exclude_dir` | Next | — | — |
| WP2 | Class-anchored sampler | Not started | — | Uses `--label_z_offset 1` |
| WP3 | Classifier decoder (patch mode) | Not started | — | — |
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
| Sand/shale fraction maps lithology with `(x+1)/2`, so shale counts as 0.5 sand and water counts as shale (lithology: −1 water, 0 shale, 1 sand) | WP1 |
| `rglob` over `fake_data` also picks up `fake_data/validation`, leaking validation volumes into training | WP1 (`--exclude_dir`) |
| Default `--seismic_key` has a double underscore and matches no volume | WP1 |
| Labels are read at the seismic origin, one sample off | WP2 (`--label_z_offset`) |

---

## Work package log

### WP0 — Label/seismic depth alignment (done)

**Question:** labels have depth 1510 and seismic has 1499. How do they line up?

**Method:**
- Traced the synthoseis code to see where the depth difference comes from.
- Confirmed empirically by correlating seismic reflection strength (envelope of the depth
  derivative) with lithology boundaries for offsets −15…+15.

**Result:** `seismic[z]` matches `label[z + 1]`.
- The labels carry 10 padding samples at the bottom.
- The reflectivity (and therefore the seismic) has one sample fewer than the 1500-sample
  rock property model.
- On real data, the mean correlation peak is at +1 on both 8 and 20 volumes (sub-sample
  estimates +0.86 to +1.21). 17 of 20 volumes are within ±1. The outliers have two
  correlation peaks, most likely from regular layer spacing.

**Artifacts:**
- [scripts/verify_label_alignment.py](../../scripts/verify_label_alignment.py)
- [tests/test_label_alignment.py](../../tests/test_label_alignment.py)
- Details: [2026-09-25-wp0-label-seismic-alignment.md](2026-09-25-wp0-label-seismic-alignment.md)
- JSON reports in `data/` (gitignored)

**Decision:** proceed to WP1.

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
