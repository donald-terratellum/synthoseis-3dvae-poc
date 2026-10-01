# WP0–P3 Transfer Experiments — Verification and Next-Steps Handoff

Started: 2026-09-29
Updated: 2026-09-30
Plan this belongs to: [docs/plans/2026-09-27__pretrain_v2_encoder_loss_augmentation_transfer_plan.md](../plans/2026-09-27__pretrain_v2_encoder_loss_augmentation_transfer_plan.md), Sections 8–9.

## Current status (2026-09-30)

The older verification instructions below are retained as provenance; all of their open items
are complete. The frozen v2 manifest matches the 5,000-patch dataset (`dataset_size=5000`,
2,048 benchmark indices), and P0, P0a, P0b, P1, P2, and P3 have finished.

- P0b remains the control: best n@5 was 0.0120 for seed `20260925` and 0.0109 for seed
  `20260926` (best-per-run mean 0.01146).
- P1 loss recipes L1–L4 were rejected. The strongest provisional result did not reproduce in
  the second seed.
- WP3 phase rotation is implemented with vectorized FFT rotation, triangular angle sampling,
  CLI controls, and unit tests. The full suite passes: 218 tests.
- P2 used `--phase_rotation_prob 1.0 --phase_range -60 0 40`, with each seed warm-started from
  its matching P0b best checkpoint. Best n@5 was 0.0091 and 0.0122 (mean 0.01065), below P0b's
  mean; phase rotation is rejected and must not be carried forward.
- P2 checkpoints are under `checkpoints/p2_phase_seed20260925` and
  `checkpoints/p2_phase_seed20260926`. Reports are
  `docs/benchmarks/p2_phase_seed<seed>_ep<10|20|30|40>_zgeo.json`; the combined log is
  `logs/p2_phase_suite.log`.
- WP3 zoom-in stretch is implemented with label-consistent dip adjustment and no-padding
  enforcement. A real training-Zarr smoke check and all 222 tests pass.
- P3 seed `20260925` improved to n@5 0.0130, but seed `20260926` regressed to 0.0086. The
  best-per-run mean was 0.01081 versus P0b 0.01146, so zoom-in stretch is rejected.
- P3 checkpoints are under `checkpoints/p3_zoom_stretch_seed20260925` and
  `checkpoints/p3_zoom_stretch_seed20260926`; reports follow
  `docs/benchmarks/p3_zoom_stretch_seed<seed>_ep<10|20|30|40>_zgeo.json`.

### Next action

Commit and push the P3 source, tests, runner, plan, and this summary. Exclude logs, checkpoints,
and generated benchmark reports. Then implement WP4 real-seismic reconstruction mixing and run
P4 from unchanged P0b with `K=2`, phase rotation off, and zoom-in stretch off. Start one seed and
replicate only if real MAE improves without worsening v2 n@5. Do not start encoder experiments
before the P4 decision.

This file is written so a lower-cost agentic model can execute it with minimal judgment calls.
Every step has an exact command and an unambiguous pass/fail check. Do the steps in order; do
not skip the verification step even though it looks like the pipeline already finished.

Run everything from the repo root with the venv active:

```bash
cd /Users/donaldpg/synthoseis-3dvae-poc
source .venv/bin/activate
```

---

## 1. Already verified (2026-09-29) — no action needed

These were checked and passed. Re-run only if you suspect something changed:

1. All 25 volumes exist:
   ```bash
   ls /Volumes/CrucialX9/fake_data_validation_v2 | grep -c '^seismic__'
   # Expect: 25
   ls /Volumes/CrucialX9/fake_data_validation_v2 | grep '^seismic__' | sed 's/.*run_//' | sort -n | sed -n '1p;$p'
   # Expect: 5000
   #         5024
   ```
2. No partial/leftover generation folders:
   ```bash
   ls /Volumes/CrucialX9/fake_data_validation_v2 | grep temp_folder
   # Expect: no output
   ```
3. Every volume has all 8 required label arrays (checked with a Python loop over
   `zarr.open(...).keys()` against `fault_segments_id`, `fault_intersection_segments`,
   `faults/faulted_channel_labels`, `closure_segments_id`, `onlap_segments`,
   `faulted_lithology`, `flat_spot`, `geologic_age_faulted`) — zero volumes missing anything.
4. Disk: `/Volumes/CrucialX9/fake_data_validation_v2` is 78G; CrucialX9 has 886Gi free of
   1.8Ti (53% used). No disk pressure.
5. `logs/wp0_generate_v2_validation_3.log` shows the pipeline reached the end: all 25 volumes
   scanned, `Wrote 5000 patches to data/synth_val_v2_32-32-64.zarr`,
   `Wrote 5000 patches to data/synth_val_v2_uniform_32-32-64.zarr`,
   `Wrote benchmark report: docs/benchmarks/adopted_v2_reference.json`.

---

## 2. OPEN ITEM — verify the manifest is not stale (do this first, do not skip)

**Why this matters:** before the real 25-volume run, a small throwaway test was run with only
1 volume present (`run_5000`), to sanity-check the generation script. That test sampled only
200 patches and ran `scripts/evaluate_geology_benchmark.py`, which — if
`docs/benchmarks/frozen_validation_manifest_v2.json` did not already exist — would have created
it with `dataset_size: 200`. After the test, the placeholder data files and benchmark JSON were
deleted, but the manifest file itself was not deleted in that cleanup.

`scripts/evaluate_geology_benchmark.py`'s `_load_or_create_manifest()` (around line 105) does
this:

```python
if manifest_path.exists():
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "indices" not in payload:
        raise ValueError(...)
    return payload   # returns AS-IS — no check that dataset_size matches the current dataset
```

So if the stale 200-row manifest was still on disk when the real 25-volume run reached its
benchmark step, `docs/benchmarks/adopted_v2_reference.json` would have been computed against a
200-example index space, not the real 5000-patch v2 validation set — silently, with no error.
A first look at the current manifest showed `dataset_size: 5000` (which would mean it's fine),
but this was not conclusively confirmed before this handoff, so **do not trust it — verify it
directly with the commands below.**

### 2.1 Run this exact check

```bash
.venv/bin/python -c "
import json, zarr
m = json.load(open('docs/benchmarks/frozen_validation_manifest_v2.json'))
z = zarr.open('data/synth_val_v2_32-32-64.zarr', mode='r')
real_size = int(z['patches'].shape[0])
print('manifest dataset_size:', m['dataset_size'])
print('real dataset size:', real_size)
print('manifest benchmark_size:', m['benchmark_size'], 'n_indices:', len(m['indices']))
print('max index in manifest:', max(m['indices']))
print('RESULT:', 'MATCH' if m['dataset_size'] == real_size else 'MISMATCH_STALE_MANIFEST')
"
```

### 2.2 Decision

- **Prints `RESULT: MATCH`** → the manifest is correct and reflects the real 5000-patch v2
  dataset. Do nothing further here. Go to Section 3.
- **Prints `RESULT: MISMATCH_STALE_MANIFEST`** → the manifest is stale. Fix it:

  ```bash
  rm -f docs/benchmarks/frozen_validation_manifest_v2.json docs/benchmarks/adopted_v2_reference.json

  .venv/bin/python scripts/evaluate_geology_benchmark.py \
    --data data/synth_val_v2_32-32-64.zarr \
    --manifest docs/benchmarks/frozen_validation_manifest_v2.json \
    --benchmark_size 2048 \
    --seed 20260927 \
    --checkpoint checkpoints/geoaware_v3_phase2_20260831/vae_epoch20.pt \
    --use_geo_embedding \
    --metadata_keys meta_fault_fraction meta_fault_intersection_fraction \
      meta_channel_fraction meta_channel_core_fraction \
      meta_flat_spot_fraction meta_onlap_fraction meta_onlap_variability \
    --out_json docs/benchmarks/adopted_v2_reference.json
  ```

  Then re-run the check in 2.1 and confirm it now prints `RESULT: MATCH` before continuing.

### 2.3 Report the eligible-query count

The plan (Section 8) wants this reported: "2048 examples should give about 4x the 52 eligible
queries of the old manifest."

```bash
.venv/bin/python -c "
import json
d = json.load(open('docs/benchmarks/adopted_v2_reference.json'))
print(json.dumps(d.get('diagnostics', {}), indent=2))
"
```

Record the printed `neighbor_overlap_at_5` / `neighbor_overlap_at_10` values — these are the
**P0 reference numbers** (adopted checkpoint, re-benchmarked on the v2 manifest).

---

## 3. Next steps per the plan (Section 9)

Only proceed here after Section 2 confirms `RESULT: MATCH` (or has been fixed to match).

### 3.1 Finish P0: also benchmark the R2 seed-1 checkpoint on the v2 manifest

The adopted-checkpoint half of P0 is done (Section 2.3 above). The plan also wants the R2
seed-1 experimental checkpoint re-benchmarked on the same v2 manifest, for the full P0 baseline
pair:

```bash
.venv/bin/python scripts/evaluate_geology_benchmark.py \
  --data data/synth_val_v2_32-32-64.zarr \
  --manifest docs/benchmarks/frozen_validation_manifest_v2.json \
  --checkpoint checkpoints/geoaware_p0a_dipfix_r2/vae_epoch30.pt \
  --use_geo_embedding \
  --metadata_keys meta_fault_fraction meta_fault_intersection_fraction \
    meta_channel_fraction meta_channel_core_fraction \
    meta_flat_spot_fraction meta_onlap_fraction meta_onlap_variability \
  --out_json docs/benchmarks/r2_seed1_ep30_v2_reference.json
```

(`checkpoints/geoaware_p0a_dipfix_r2/vae_epoch30.pt` is the P0a run from 2026-09-28 — old
manifest n@5 was 0.1346 there. This step re-measures the same checkpoint on the new v2
manifest so both P0 numbers are on a comparable basis.)

### 3.2 P0b: rerun the 180-volume training baseline on the v2 validation set

The data split is now fixed: training uses only the 180 datasets under
`/Volumes/CrucialX9/fake_data` with `--exclude_dir validation`; validation uses the 25 new
datasets under `/Volumes/CrucialX9/fake_data_validation_v2`. P0b therefore reruns the R2 recipe on the
existing 180-volume training patches, with the new v2 validation set and two seeds.

Use `data/synth_val_v2_uniform_32-32-64.zarr` as the validation input. It contains the v2
source-volume provenance, so it is the correct zarr path for `--disjoint_from` if a new training
set is sampled. The existing `data/synth_train_anchored_32-32-64.zarr` is already the correct
108,000-patch, label-complete training set and does not need to be recreated.

If regeneration is genuinely needed, use the corrected command:

```bash
.venv/bin/python scripts/sample_patches.py \
  --source /Volumes/CrucialX9/fake_data \
  --exclude_dir validation \
  --disjoint_from data/synth_val_v2_uniform_32-32-64.zarr \
  --patch_size 32 32 64 --n_patches 108000 --n_per_volume 600 \
  --seismic_key seismicCubes_cumsum_fullstack --geoscore_key geologic_score \
  --sampling_mode class_anchored \
  --class_quotas fault=0.10 fault_x=0.10 flat_spot=0.10 channel=0.10 closure=0.10 onlap=0.10 sand=0.05 \
  --background_fraction 0.25 --max_patches_per_object 24 --presence_min_voxels 32 \
  --label_z_offset 1 --dip_sample_count 512 --seed 20260929 \
  --out data/synth_train_anchored_32-32-64.zarr
```

Do not run that sampling command unless the existing 180-volume training set is missing or
needs regeneration. Then train the R2 recipe on
`data/synth_train_anchored_32-32-64.zarr`, 2 seeds, validating against
`data/synth_val_v2_uniform_32-32-64.zarr`, and benchmark both against
`docs/benchmarks/frozen_validation_manifest_v2.json`. Compare n@5 against the P0/P0a numbers
from Sections 2.3 and 3.1 — this establishes the clean v2 holdout baseline the plan calls P0b.

### 3.3 After P0b: proceed to P1

Per the plan, P1 is a loss-recipe sweep (L1–L4) warm-started from P0b's best recipe. Only start
this after P0b has run and been compared; do not skip ahead.

---

## 4. Optional: record the outcome in repo memory

If you have access to the memory tool for this repo, append a short note to
`/memories/repo/geoaware_model.md` under the "Pretrain-v2 transfer plan" section once Sections
2 and 3.1 are done, e.g.: "v2 manifest confirmed correct (dataset_size=5000); P0 adopted n@5=X,
R2 seed1 ep30 n@5=Y (v2 manifest)." This is optional and does not block continuing to P0b.

---

## Not something an agent can do: switching the chat model

The request to "switch to auto model selection for the 10% discount" is a client-side setting
in the chat UI's model picker (the dropdown next to the chat input), not something any tool
available to an agent can change. This must be done directly in the editor by the user.
