# Latent Shaping Execution Backlog (Issue-Ready)

## Purpose
Turn the latent-shaping plan into a step-by-step, issue-ready execution backlog across:
- synthoseis
- synthoseis-pretrain-v2
- synthoseis-3dvae-poc

This backlog treats the Updated: Geologic-Feature-Aware Latent Shaping Plan as canonical.

## Global Constraints and Decisions

### Canonical constraints
- Do not store full voxel-level feature volumes for every generated dataset due to disk limits on the Mac mini.
- Compute patch-level feature vectors from subset volumes while synthoseis runs execute.
- Tight integration is required between synthoseis generation and VAE patch export flow.
- Existing hook to leverage: synthoseis-pretrain-v2/generate_datasets.sh with --generate-vae-zarr.
- Critical: all geological label arrays are deleted inside synthoseis `main.py` by the `--zarr-out essential` prune step, BEFORE `vae_export` runs. Feature extraction must happen inside the synthoseis run, before that prune, not in the pretrain-v2 export step.
- Persist provenance metadata by copying per-dataset `model_config_*.json` and `model_parameters_*.txt` into the VAE output area, because source synthoseis dataset folders are intentionally rotated/deleted.

### Proposed feature list (user-provided)
- background
- fault
- fault intersection
- large fault throw > 15
- small fault throw < 3
- onlap surface
- salt boundary
- channel
- closure
- strat closure
- faulted closure
- hydrocarbon closure
- oil closure
- gas closure
- brine closure
- channel closure
- reservoir
- low dip < 2 degrees
- steep dip > 10 degrees
- dip standard deviation
- noise level (db)

### Important caveat
Many features are not independent and some overlap on the same voxels. This backlog explicitly includes:
- hierarchical feature grouping
- overlap-aware accounting
- documented normalization policy

### Normalization policy (initial)
- Generate 15-20 throw-away calibration datasets.
- For each feature, compute patch-level voxel-count distribution.
- Normalize each feature separately by either:
  - feature-specific max, or
  - feature-specific p95 (recommended default for robustness).
- Persist normalization stats as versioned metadata and reuse for subsequent training datasets.

### VAE provenance contract (v1)

This section defines exact attrs and on-disk layout so pretrain-v2 implementation is one-to-one with this backlog.

#### Applicability
- Applies to every exported VAE patch dataset `train_XXXX.zarr`.
- Pretrain-v2 note: augmented examples are generated on the fly and are not persisted per-example; this contract stores dataset-level provenance only.
- 3dvae-poc note: persisted patch datasets should preserve the same provenance attrs and metadata file layout.

#### Required zarr attrs (exact keys)
- `provenance.contract_version` (string): `vae_patch_provenance.v1`
- `provenance.source_dataset_dir` (string): absolute path to synthoseis dataset folder (for example: `.../fake_data/seismic__..._synthoseis_run_0580`)
- `provenance.source_model_data_zarr` (string): absolute path to source `model_data.zarr`
- `provenance.source_dataset_id` (string): dataset id used for export (for example `0580`)
- `provenance.metadata_dir_rel` (string): relative path from zarr parent dir to metadata folder (for example `train_0580.metadata`)
- `provenance.model_config_relpath` (string): relative path from zarr parent dir to copied config json
- `provenance.model_parameters_relpath` (string): relative path from zarr parent dir to copied parameters txt
- `provenance.metadata_copy_status` (string): `copied` or `missing`
- `provenance.metadata_copied_at_utc` (string): ISO-8601 UTC timestamp

#### Optional zarr attrs
- `provenance.source_run_tag` (string): synthoseis run tag if available
- `provenance.source_dataset_timestamp` (string): timestamp token parsed from synthoseis folder name
- `provenance.copy_tool` (string): exporter identifier (for example `synthoseis-pretrain-v2/vae_export`)
- `pretrain_v2.augmented_examples_persisted` (bool): must be `false` for pretrain-v2 on-the-fly augmentation workflow
- `pretrain_v2.augmentation_mode` (string): recommended `on_the_fly`

#### File layout convention (exact)
Given:
- `VAE_OUTPUT_ROOT=/.../data/train`
- output zarr `train_0580.zarr`

Use this structure:

```text
VAE_OUTPUT_ROOT/
  train_0580.zarr/
    ... zarr arrays/attrs ...
  train_0580.metadata/
    model_config_2026.41242533.json
    model_parameters_2026.41242533.txt
```

Rules:
- Metadata folder name is `<train_basename>.metadata` where `<train_basename>` is zarr name without `.zarr`.
- Preserve original metadata filenames when copying.
- Paths written to attrs must be relative (portable across machines) using POSIX separators.
- One config json + one parameters txt per exported dataset.

#### Validation rules
- Fail export if either metadata file is missing when `--vae-copy-metadata-files=true`.
- Fail export if any required `provenance.*` attr is missing.
- Log one-line provenance summary per dataset:
  - source dataset dir
  - output zarr path
  - copied metadata paths

---

## Delivery Phases

1. Phase A: Data contract and calibration pipeline
2. Phase B: Online patch-vector computation during generation
3. Phase C: VAE sampling/training integration
4. Phase D: Evaluation and production hardening

## Cost Estimate

Calibration basis: your actual Copilot billing Jun 1–9, 2026 averaged about $4/day (billed range $0.53–$7.25, median $4.05). The per-issue cost is derived directly from the task's likely-days estimate multiplied by that observed daily rate. The low/high column uses $2.50 and $6.50 per day respectively to match the observed spread.

| Issue | Likely days | Low cost | Likely cost | High cost |
|---|---:|---:|---:|---:|
| S1-1 | 0.5 d | $1.25 | $2.02 | $3.25 |
| S1-2 | 1 d | $2.50 | $4.05 | $6.50 |
| S1-3a | 0.5 d | $1.25 | $2.02 | $3.25 |
| S1-3b | 1 d | $2.50 | $4.05 | $6.50 |
| S1-3c | 1 d | $2.50 | $4.05 | $6.50 |
| S1-4 | 0.5 d | $1.25 | $2.02 | $3.25 |
| S1-5 | 1 d | $2.50 | $4.05 | $6.50 |
| S1-6 | 0.5 d | $1.25 | $2.02 | $3.25 |
| S1-7 | 1 d | $2.50 | $4.05 | $6.50 |
| P1-1 | 0.5 d | $1.25 | $2.02 | $3.25 |
| P1-2 | 0.5 d | $1.25 | $2.02 | $3.25 |
| P1-3 | 0.5 d | $1.25 | $2.02 | $3.25 |
| V1-1 | 1 d | $2.50 | $4.05 | $6.50 |
| V1-2 | 0.5 d | $1.25 | $2.02 | $3.25 |
| V1-3 | 1.5 d | $3.75 | $6.07 | $9.75 |
| V1-4 | 0.5 d | $1.25 | $2.02 | $3.25 |
| V1-5 | 0.5 d | $1.25 | $2.02 | $3.25 |
| V1-6 | 1 d | $2.50 | $4.05 | $6.50 |
| X1 | 0.5 d | $1.25 | $2.02 | $3.25 |
| X2 | 0.5 d | $1.25 | $2.02 | $3.25 |
| X3 | 0.25 d | $0.62 | $1.01 | $1.62 |
| **Total** | **14.75 d** | **$36** | **$60** | **$95** |

Interpretation:
- These costs are anchored to your observed spend of roughly $4/day of active Copilot agentic work, not a theoretical token-price model.
- The range ($36–$95) reflects the variance already present in your daily billing — light planning sessions sit at $0.53, heavy implementation days at $7.25.
- V1-3 (contrastive losses) is the single highest-risk issue on both time and cost due to model iteration.
- Days where you run multiple short issues in sequence will likely come in at the low end per issue; cross-repo issues requiring multiple context reads will tend toward the high end.

---

## Repository Backlog: synthoseis

## Epic S1: Patch Feature Telemetry and Export During Generation

### Issue S1-1: Define canonical patch-feature schema and overlap semantics
Repository: synthoseis
Type: design/spec

Problem
No canonical schema currently defines overlapping feature semantics, units, and normalization references.

Scope
- Define geologic_feature_schema.json with:
  - feature_names (ordered)
  - feature_descriptions
  - feature_dependencies (for overlapping or derived features)
  - measurement type per feature (voxel_count, ratio, scalar)
  - normalization policy fields (max/p95, calibration dataset id, version)
- Add independence notes and derived-feature logic declarations.

Acceptance criteria
- Schema committed and versioned.
- Includes all features in preliminary list, with explicit status:
  - ready now
  - requires implementation
  - derived from existing outputs
- One markdown doc explains overlap handling policy.

Labels
- planning
- data-contract
- geologic-features

Confidence scale
- best case: 0.25 day
- likely: 0.5 day
- worst case: 1 day

Dependencies
- none

### Issue S1-2: Implement feature counters on streaming subset windows (online mode)
Repository: synthoseis
Type: feature

Problem
Need patch-level vectors without writing full voxel label volumes for all features.

Critical timing constraint
All geological label arrays (fault planes, onlap, throw, salt, closures, etc.) are deleted by `main.py` when `--zarr-out essential` runs, before `vae_export` fires. Feature extraction MUST be invoked inside the synthoseis run, before the prune step at `main.py` line ~146, while those arrays are still present in `model_data.zarr`.

Scope
- Add online feature counter module invoked during run execution, before the `--zarr-out essential` prune step.
- For each selected subset patch window, compute feature counts/scalars and emit:
  - feature_count_raw[K]
  - feature_value_raw[K] (for non-count features like dip std, noise db)
  - patch coordinates and patch shape
- Persist per-run patch feature records in compact format:
  - zarr arrays or parquet/jsonl sidecar under dataset folder.
- The hook must be called in `build_model()` after all label arrays are written but before the prune block.

Acceptance criteria
- No full-volume per-feature arrays required for enabled online mode.
- Per-run output includes at least N patch records expected by downstream export.
- Feature sidecar is written and flushed before `remove_dataset()` is called.
- Runtime overhead remains acceptable (target under 15 percent walltime increase).

Labels
- feature
- performance
- storage

Confidence scale
- best case: 0.5 day
- likely: 1 day
- worst case: 2 days

Dependencies
- S1-1

### Issue S1-3a: Expose already-materialized source volumes through the patch-feature contract
Repository: synthoseis
Type: feature

Problem
Several requested features already exist in synthoseis as full-volume labels or arrays, but they are not yet exposed through a stable patch-feature contract.

Critical timing constraint
These arrays exist in `model_data.zarr` only during the synthoseis run. They are deleted by the `--zarr-out essential` prune step before `vae_export` fires in `generate_datasets.sh`. All patch-window extraction from these volumes must occur within the synthoseis run, before pruning.

Scope
Expose and document these already-materialized sources:
- fault_intersection_segments / fault_intersections
- onlap_segments / faulted_onlap_segments
- fault_segments / fault_segments_id
- fault_plane_throw
- max_fault_throw
- fault_plane_azimuth
- geologic_age / faulted_depth_maps_gaps (for downstream horizon-derived measures)

Acceptance criteria
- Extraction functions are callable from inside `build_model()`, before the prune step.
- Unit tests for each new feature extractor.

Labels
- feature
- geology
- technical-debt

Confidence scale
- best case: 0.25 day
- likely: 0.5 day
- worst case: 1 day

Dependencies
- S1-1

### Issue S1-3b: Implement derived feature helpers for throw buckets, salt boundaries, and dip classes
Repository: synthoseis
Type: feature

Problem
Some features can be derived from existing volumes, but the transformations are not yet packaged as reusable helpers.

Scope
- Derive throw-bucket features from `fault_plane_throw` and/or `max_fault_throw`.
- Derive salt boundary features from existing salt segment volumes and boundary extraction.
- Add a helper that converts gapped depth maps into a dip-derived representation usable by `sample_patches.py` at patch creation time.
- Use that helper to compute dip-class features and dip standard deviation.
- Declare overlap-heavy outputs as dependent in the schema rather than independent.

Acceptance criteria
- Derived helpers are available as reusable functions.
- Unit tests cover throw-bucket thresholds, salt boundary extraction, and dip conversion.

Labels
- feature
- geology
- derived

Confidence scale
- best case: 0.5 day
- likely: 1 day
- worst case: 2 days

Dependencies
- S1-1, S1-3a

### Issue S1-3c: Compute augmentation-sensitive metrics after patch extraction
Repository: synthoseis
Type: feature

Problem
Some metrics are affected by later subcube creation and augmentation, so they should be computed after patch extraction rather than stored as full-volume artifacts.

Scope
- Compute seismic loop thickness mean at patch time.
- Compute seismic loop thickness std at patch time.
- Compute patch-level noise level in dB after the subcube has been extracted and augmented.
- Ensure these are emitted into the patch-feature vector contract without requiring full-volume storage.

Acceptance criteria
- Metrics are produced from patch/subcube context, not from full-volume exports.
- Unit tests confirm the metrics change when augmentation changes the patch.

Labels
- feature
- augmentation
- patch-time

Confidence scale
- best case: 0.5 day
- likely: 1 day
- worst case: 2 days

Dependencies
- S1-1, S1-2

### Issue S1-7: Evaluate and optionally implement --make-vae-patches hook in synthoseis
Repository: synthoseis
Type: architecture / feature

Problem
The current architecture runs patch selection in `synthoseis-pretrain-v2/vae_export.py` after the synthoseis prune step. This works for seismic-only patches but is incompatible with feature extraction, since all label arrays are already deleted at that point.

Two integration strategies exist:
- Strategy A (feature sidecar): S1-2 writes a compact patch-feature sidecar during the synthoseis run; `vae_export.py` later joins sidecar to seismic patches by patch coordinates. Requires stable coordinate alignment between the two steps.
- Strategy B (--make-vae-patches): synthoseis itself runs patch selection + feature extraction in a single pass before pruning, writing a `vae_patches.zarr` directly. The `generate_datasets.sh` step then copies or indexes this output rather than re-reading deleted arrays.

Scope
- Prototype both strategies against a single run to compare output consistency and implementation complexity.
- Decide and document the chosen strategy as an architectural decision record.
- If Strategy B is chosen, implement a `--make-vae-patches` CLI flag in `synthoseis/main.py` that:
  - accepts subset size, n_subsets, and radius parameters (same contract as `vae_export.py`)
  - runs geologic-score-guided patch selection and feature extraction before the prune step
  - writes `vae_patches.zarr` alongside `model_data.zarr` in the run output folder
  - leaves the existing `--zarr-out essential` prune behaviour unchanged

Trade-offs to document
- Strategy A keeps concerns separated but requires coordinate-stable sidecar alignment.
- Strategy B eliminates the alignment problem but couples patch-sampling hyperparameters into synthoseis config and increases per-run wall time.
- Either strategy must avoid storing full-volume label arrays after the run (disk constraint).

Acceptance criteria
- Architectural decision record committed.
- If Strategy B: `--make-vae-patches` flag produces a `vae_patches.zarr` with patches and feature arrays from a single run without disk bloat.
- If Strategy A: coordinate alignment contract documented and covered by a round-trip test.

Labels
- architecture
- feature
- integration

Confidence scale
- best case: 0.5 day
- likely: 1 day
- worst case: 2 days

Dependencies
- S1-2, S1-3a

---

### Issue S1-4: Calibration run workflow for normalization stats (15-20 throw-away datasets)
Repository: synthoseis
Type: pipeline

Problem
Need stable feature scaling from sampled distributions.

Scope
- Add calibration mode CLI/config:
  - runs 15-20 datasets
  - aggregates per-feature distributions from online patch records
- Compute and store:
  - max
  - p95
  - optional p99
- Output calibration artifact:
  - feature_normalization_v1.json

Acceptance criteria
- One command produces normalization file from calibration runs.
- File records dataset ids, sample counts, and timestamp/version.
- Re-runnable with deterministic seed mode.

Labels
- pipeline
- calibration
- metrics

Confidence scale
- best case: 0.25 day
- likely: 0.5 day
- worst case: 1 day

Dependencies
- S1-2

### Issue S1-5: Generation controls to influence feature frequency
Repository: synthoseis
Type: feature

Problem
Need future ability to bias generation toward underrepresented features.

Scope
- Add optional generator weighting knobs (config or CLI) for target feature frequency.
- Introduce telemetry report comparing observed vs requested frequencies.

Acceptance criteria
- Feature-frequency controls are optional and backward compatible.
- Report per-run divergence from requested targets.

Labels
- generation
- balancing
- future-facing

Confidence scale
- best case: 0.5 day
- likely: 1 day
- worst case: 2 days

Dependencies
- S1-2, S1-4

### Issue S1-6: Storage guardrails and fail-fast checks
Repository: synthoseis
Type: reliability

Problem
Disk pressure is a hard constraint.

Scope
- Add storage budget checks for online feature export artifacts.
- Hard-fail if estimated artifact size exceeds threshold.
- Add cleanup policy for throw-away calibration datasets while retaining normalization artifact.

Acceptance criteria
- Pipeline aborts early with actionable message under low-space scenarios.
- Calibration mode can auto-prune run folders after stats are consolidated.

Labels
- reliability
- storage

Confidence scale
- best case: 0.25 day
- likely: 0.5 day
- worst case: 1 day

Dependencies
- S1-2, S1-4

---

## Repository Backlog: synthoseis-pretrain-v2

## Epic P1: Orchestrated Integration from Generation to VAE Export

### Issue P1-1: Extend generate_datasets.sh --generate-vae-zarr path to pass feature-artifact inputs
Repository: synthoseis-pretrain-v2
Type: integration

Problem
The script triggers VAE zarr export but does not yet pass canonical patch-feature metadata contract.

Scope
- Extend --generate-vae-zarr flow to pass:
  - feature schema path
  - normalization artifact path
  - per-run patch-feature sidecar path
- Copy provenance metadata files from each synthoseis dataset folder into the corresponding VAE output folder:
  - `model_config_*.json`
  - `model_parameters_*.txt`
- Add CLI flags:
  - --vae-feature-schema
  - --vae-feature-normalization
  - --vae-feature-sidecar-name
  - --vae-copy-metadata-files (default true)
  - --vae-metadata-output-dir (optional override; default next to `train_XXXX.zarr`)

Acceptance criteria
- End-to-end command produces VAE zarr with feature vectors when artifacts exist.
- If artifacts are missing, script fails with clear diagnostics (or controlled fallback mode).
- For each exported VAE dataset, exactly one copied `model_config_*.json` and one copied `model_parameters_*.txt` are present in the VAE output location.
- Export writes provenance attrs into the VAE zarr (source synthoseis dataset path and copied metadata file paths).

Labels
- integration
- cli

Confidence scale
- best case: 0.25 day
- likely: 0.5 day
- worst case: 1 day

Dependencies
- S1-1, S1-2, S1-4

### Issue P1-2: Add artifact discovery and validation before export
Repository: synthoseis-pretrain-v2
Type: reliability

Problem
Need robust checks that expected artifacts exist and are consistent.

Scope
- Validate feature artifact compatibility:
  - feature vector length
  - schema version
  - normalization version
  - patch shape compatibility
- Add preflight summary in logs.
- Validate provenance metadata availability (`model_config_*.json`, `model_parameters_*.txt`) before source dataset cleanup.
- Validate copied metadata paths are written into destination zarr attrs.

Acceptance criteria
- Export aborts before work if contract mismatch is detected.
- Logs clearly identify mismatch fields.
- Export aborts with actionable message if required metadata files are missing.

Labels
- reliability
- validation

Confidence scale
- best case: 0.25 day
- likely: 0.5 day
- worst case: 1 day

Dependencies
- P1-1

### Issue P1-3: Throughput benchmarking for integrated generation/export loop
Repository: synthoseis-pretrain-v2
Type: perf

Problem
Need to confirm online feature extraction + export does not break throughput.

Scope
- Benchmark baseline vs integrated run over 3-5 runs.
- Report:
  - runtime delta
  - disk footprint delta
  - export success rate

Acceptance criteria
- Benchmark doc committed with recommendation thresholds.

Labels
- perf
- benchmarking

Confidence scale
- best case: 0.25 day
- likely: 0.5 day
- worst case: 1 day

Dependencies
- P1-1, P1-2

---

## Repository Backlog: synthoseis-3dvae-poc

## Epic V1: Feature-Aware Patch Dataset and Training Controls

### Issue V1-1: Extend scripts/sample_patches.py output schema for feature vectors
Repository: synthoseis-3dvae-poc
Type: feature

Problem
Current patch zarr stores patches only, without feature vector arrays.

Scope
- Add output arrays:
  - feature_vector_raw [N, K]
  - feature_vector_norm [N, K]
  - feature_primary_label [N]
  - feature_quality_mask [N] (optional)
- Persist attrs:
  - feature_names
  - normalization source/version
  - schema version
  - source synthoseis dataset path
  - copied metadata file paths (`model_config_*.json`, `model_parameters_*.txt`)

Acceptance criteria
- New zarr contains backward-compatible patches array plus feature arrays.
- Existing training path still works when feature arrays absent.
- Training/evaluation tooling can read provenance attrs without opening source synthoseis folders.

Labels
- feature
- data

Confidence scale
- best case: 0.5 day
- likely: 1 day
- worst case: 2 days

Dependencies
- P1-1, P1-2

### Issue V1-2: Training dataset loader support for feature vectors with fallback
Repository: synthoseis-3dvae-poc
Type: feature

Problem
Training loader currently returns only x/y tensors.

Scope
- Extend dataset class to optionally return feature tensors and labels.
- Add fallback mode to reconstruction-only training if feature arrays are unavailable.

Acceptance criteria
- Dataloader supports both modes under flags.
- Unit tests for shape, dtype, and fallback behavior.

Labels
- training
- dataloader

Confidence scale
- best case: 0.25 day
- likely: 0.5 day
- worst case: 1 day

Dependencies
- V1-1

### Issue V1-3: Add feature-head and geologic-contrastive losses behind flags
Repository: synthoseis-3dvae-poc
Type: feature

Problem
Need issue-ready implementation of updated objective.

Scope
- Add model head for K-dimensional feature prediction.
- Add loss terms:
  - feature BCE or focal-style objective
  - similarity-weighted geologic contrastive loss
- Add CLI controls:
  - --enable-feature-head
  - --enable-geologic-contrastive
  - --lambda-feat
  - --lambda-geo
  - --geo-tau

Acceptance criteria
- Losses compute and backprop under feature-enabled mode.
- Baseline behavior unchanged when flags are off.

Labels
- model
- loss

Confidence scale
- best case: 0.5 day
- likely: 1.5 days
- worst case: 3 days

Dependencies
- V1-2

### Issue V1-4: Balanced sampler for feature representation
Repository: synthoseis-3dvae-poc
Type: feature

Problem
Need batch composition with sufficient positives per feature profile.

Scope
- Add weighted/bucketed sampling strategies based on primary labels and/or feature-vector rarity.
- Add sampler diagnostics per epoch.

Acceptance criteria
- Configurable sampler improves per-batch feature coverage.
- Diagnostics emitted to logs and tensorboard.

Labels
- sampler
- imbalance

Confidence scale
- best case: 0.25 day
- likely: 0.5 day
- worst case: 1 day

Dependencies
- V1-1

### Issue V1-5: Curriculum config profiles and safety guardrails
Repository: synthoseis-3dvae-poc
Type: pipeline

Problem
Need repeatable phase-based training with regression guardrails.

Scope
- Add preset configs for:
  - Phase 1 (feature head only)
  - Phase 2 (enable geologic contrastive)
  - Phase 3 (optional GAN ramp)
- Add auto-stop/rollback triggers when reconstruction degrades beyond threshold.

Acceptance criteria
- One-command launch for each phase.
- Guardrails documented and active.

Labels
- training
- curriculum
- reliability

Confidence scale
- best case: 0.25 day
- likely: 0.5 day
- worst case: 1 day

Dependencies
- V1-3

### Issue V1-6: Latent evaluation package (probe + clustering + drift)
Repository: synthoseis-3dvae-poc
Type: evaluation

Problem
Need objective evidence that latent space improves geologic organization.

Scope
- Add evaluation scripts/notebook for:
  - linear probe
  - UMAP/t-SNE plots
  - reconstruction drift dashboard
- Include comparison against baseline checkpoint.

Acceptance criteria
- Generates report artifacts for go/no-go decisions.

Labels
- evaluation
- metrics

Confidence scale
- best case: 0.5 day
- likely: 1 day
- worst case: 2 days

Dependencies
- V1-3

---

## Cross-Repository Program Tickets

### Issue X1: Feature registry and dependency graph
Repositories: all
Type: governance

Scope
- Create single source-of-truth feature registry shared across repos.
- Include dependency graph showing non-independent features.

Acceptance criteria
- Same feature ordering and names used in all repos.
- CI check prevents drift.

### Issue X2: Contract tests in CI
Repositories: synthoseis-pretrain-v2, synthoseis-3dvae-poc
Type: test

Scope
- Add fixture dataset with mini feature sidecar and schema.
- Validate loader, exporter, and trainer compatibility.

Acceptance criteria
- CI fails on schema mismatch or shape mismatch.

### Issue X3: Versioning and migration policy
Repositories: all
Type: governance

Scope
- Define semantic versioning for feature schema and normalization artifacts.
- Add migration notes for breaking changes.

Acceptance criteria
- Docs published and referenced by all scripts.

---

## Step-by-Step Execution Order (Recommended)

1. S1-1 schema first.
2. S1-2 online counters.
3. S1-3a source-volume exposure.
4. S1-3b derived feature helpers.
5. S1-3c patch-time augmentation-sensitive metrics.
6. S1-7 architecture decision: Strategy A (sidecar) vs Strategy B (--make-vae-patches).
7. S1-4 calibration (15-20 throw-away runs) + normalization artifact.
8. P1-1 and P1-2 integration in generate_datasets.sh export path (shape depends on S1-7 outcome).
9. V1-1 and V1-2 data-path support in VAE repo.
10. V1-3 losses and flags.
11. V1-4 sampler.
12. V1-5 curriculum profiles.
13. V1-6 evaluation package.
14. S1-5 generation-frequency controls.
15. X1/X2/X3 governance and CI hardening.

---

## Feature Grouping Recommendation (to handle non-independence)

To reduce leakage and collinearity issues, tag each feature as one of:
- base_binary_voxel_feature
- derived_structural_feature
- closure_subtype
- scalar_patch_stat

Suggested examples:
- base_binary_voxel_feature: fault, channel, salt boundary, onlap surface, reservoir
- derived_structural_feature: fault intersection, large throw, small throw, strat closure, faulted closure
- closure_subtype: hydrocarbon, oil, gas, brine, channel closure
- scalar_patch_stat: low dip, steep dip, dip std, noise db

Use this grouping in modeling decisions:
- train with all features available
- optionally apply group-aware weighting to avoid over-counting related closure features

---

## Issue Template (copy/paste)

Title
- [repo] <short imperative task>

Background
- Why this is needed now.

Scope
- In-scope items.
- Out-of-scope items.

Implementation notes
- Key files/modules.
- CLI/config changes.

Acceptance criteria
- Bullet list of measurable pass conditions.

Validation
- Unit tests.
- Smoke test command.
- Expected outputs/artifacts.

Dependencies
- Blocking issue ids.

Risk and rollback
- Main risks.
- Rollback or feature-flag strategy.

---

## Immediate Next 5 Tickets to Open

1. S1-1 Define canonical patch-feature schema and overlap semantics.
2. S1-2 Implement online subset-window feature counters in synthoseis.
3. S1-3a expose existing source volumes.
4. S1-3b implement derived helpers.
5. S1-3c implement patch-time augmentation-sensitive metrics.
6. S1-4 Build calibration workflow and normalization artifact from 15-20 throw-away runs.
7. P1-1 Extend --generate-vae-zarr integration to pass feature artifacts.
8. V1-1 Extend sample_patches output schema with feature arrays and metadata.
