# 2026-09-25 — WP0: label/seismic depth alignment

Plan (local, untracked): `docs/plans/2026-09-25__geology_classifier_decoder_and_class_anchored_sampling_plan.md`.
Progress log: [2026-09-25-geology-classifier-progress.md](2026-09-25-geology-classifier-progress.md)

## Result

**`--label_z_offset 1`**. Convention: `seismic[..., z]` matches `label[..., z + 1]`.
The 10 padded label samples are at the **bottom**. Only `label[..., 1:1500]` has
matching seismic samples.

## Code evidence (`/Users/donaldpg/synthoseis`)

| Step | Location | Depth |
|---|---|---|
| Geomodel/label arrays are allocated with `cube_shape[2] + pad_samples` | `datagenerator/Geomodels.py` (~L73), `Parameters.py` (~L750) | 1500 + 10 = 1510 |
| Elastic properties are trimmed with `[:, :, :nz]`, which drops the bottom padding | `datagenerator/Seismic.py` ~L2139–2147 | 1500 |
| RFC uses Z−1 interfaces: `rfc[k]` is the interface between elastic `k` and `k+1` | `Seismic.py` ~L98; `rockphysics/RockPropertyModels.py` ~L614 | 1499 |
| Cumsum fullstack: zero-phase bandpass, `cumsum`, then bandpass again (shape unchanged) | `Seismic.py` `build_cumsum_fullstack_noise_free` | 1499 |

`cumsum` makes `seismic[k]` include `rfc[k]`, so the step at `k` shows the layer below the
interface, which is `label[k+1]`. The expected offset is therefore **+1**. Synthoseis's own QC
overlays use `label[:, :, :nz]` (offset 0), which puts them off by one sample.

## Empirical verification

Script: [scripts/verify_label_alignment.py](../../scripts/verify_label_alignment.py). It
correlates the envelope of d(seismic)/dz with |d(faulted_lithology)/dz| for offsets −15…+15.
As a second estimator, it computes |corr| of the signed derivatives. Each volume uses
3 chunk blocks at XY stride 5 (300 traces).

| Run | Volumes | Envelope, mean-curve peak | Refined (envelope / signed) | Volumes within ±1 of +1 |
|---|---:|---:|---:|---:|
| seed 20260925 | 8 | +1 | +1.21 / +1.05 | 7/8 |
| seed 1 | 20 | +1 | +1.07 / +0.86 | 17/20 (19/20 within ±2) |

- The outliers (run_1142 at +11, run_1120 at +14) have bimodal curves: one local peak at +1
  and a slightly higher one about 10 samples deeper. This looks like periodic layer
  thickness, not a real shift.
- Peak correlations are low (0.08–0.22) because lithology boundaries explain only part of the
  impedance contrasts. The peak position is still consistent across volumes.
- Reports: `data/wp0_label_alignment.json` and `data/wp0_label_alignment_20vol.json`. Both are
  gitignored and not committed.

## Tests

- New: [tests/test_label_alignment.py](../../tests/test_label_alignment.py) (9 tests). Covers
  the envelope, the peak refinement, recovery of known offsets (+1 for the synthoseis layout,
  +7, −3), an end-to-end zarr volume with summary/verification, and the not-verified case
  with fewer than 5 volumes.
- Full suite: `.venv/bin/python -m unittest discover -s tests`: 112 tests OK. The baseline
  before WP0 was 103 OK.

## Implications for later WPs

- WP2: pass `--label_z_offset 1`. The label window for a seismic patch at origin `z0` is
  `[z0+1, z0+1+pz)`, and it always fits because 1499 + 1 ≤ 1510.
- Patch-level presence (Stage 1) is insensitive to a 1-sample offset. Voxel mode (WP6) is
  no longer blocked by alignment.
- The existing `compute_patch_derived_metadata` reads labels at the seismic origin
  (offset 0). That is a 1-sample bias, which does not matter for patch fractions. Apply the
  offset when WP2 adds `--label_z_offset`.
