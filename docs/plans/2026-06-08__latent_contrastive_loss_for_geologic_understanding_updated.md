# Updated Plan Review and Execution Blueprint

## Scope
This document reviews the existing latent-shaping plan and defines an updated implementation blueprint across the VAE repo plus related upstream data repositories.

## Direct Assessment

### 1) Is the intent clear?
Yes, mostly clear.

- Intent clarity score: 8.5 out of 10.
- Strengths:
  - Clear target: preserve reconstruction quality while improving geologic separability in latent space.
  - Clear mechanism: feature prediction objective plus contrastive latent objective.
  - Clear staged training curriculum.
- Remaining ambiguity:
  - Which geologic labels are mandatory in phase 1 versus optional.
  - Exact data contract for label volumes and patch-level vectors across repositories.
  - Success thresholds are descriptive but not yet strict pass/fail gates.

### 2) Probability of success at achieving target
Estimated probability: 0.64.

Rationale:
- Positive factors:
  - Existing modular loss stack and training controls already exist.
  - Existing geologic-score-guided sampling exists in related pipeline code.
  - Staged curriculum lowers regression risk.
- Risk factors:
  - Mixed-label patches can dilute class-separation signal if similarity weighting and sampling are not tuned.
  - Label imbalance and label-noise transfer from upstream generation can reduce contrastive effectiveness.
  - Reconstruction-latent tradeoff can regress if lambda schedules are too aggressive.

### 3) Probability of implementing successfully
Estimated probability: 0.82.

Rationale:
- Implementation complexity is moderate, not extreme.
- Most work is additive and can be feature-flagged.
- Existing repositories already contain neighboring capabilities: zarr-backed storage, geologic-score arrays, channel label generation, and weighted sampling utilities.

### 4) Is supersession from preliminary to updated plan clear?
Partially clear, but not explicit enough for execution governance.

Current state:
- The document visibly contains a preliminary section followed by a section titled Updated: Geologic-Feature-Aware Latent Shaping Plan.
- The updated section clearly shifts from class-style SupCon toward similarity-weighted geologic contrastive alignment over feature-fraction vectors.

Gap:
- There is no explicit supersession declaration that states which preliminary decisions are replaced.

Required fix:
- Add a short governance block at the top:
  - Status: Updated section is canonical.
  - Superseded items: primary-label-only SupCon framing, binary threshold-only patch labels, optional N-pair branch.
  - Retained items: staged curriculum, reconstruction-preservation objective, label-aware sampling requirement.

### 5) What changes are needed in related repositories for patch-level geologic feature vectors?
Summary: both upstream repositories need a stable data contract so this VAE repo can consume feature vectors without custom per-run logic.

---

## Updated Cross-Repo Implementation Plan

## A. Canonical Decision and Data Contract

### A1. Canonical objective
Use similarity-weighted contrastive alignment based on patch-level geologic fraction vectors.

### A2. Canonical patch feature vector
For each sampled patch produce:
- feature_fraction: [fault, channel, unconformity, strat_boundary, background], float32, values in [0,1], sum to 1.
- feature_presence: same order, binary thresholded helper vector.
- primary_label: argmax over feature_fraction (int32).
- feature_entropy: optional scalar for curriculum diagnostics.

### A3. Canonical metadata
Store label semantics once per dataset:
- feature_names
- label_ids used upstream
- threshold policy
- normalization policy
- source dataset identifier and generator version

---

## B. Repository Workstreams

## B1. synthoseis repository (upstream generation)
Primary role: produce authoritative voxel-level geologic label volumes and metadata in model outputs.

Required additions:
1. Ensure model_data.zarr retains or writes these arrays in a stable schema:
- fault_label volume (binary or multi-fault id map collapsed to fault presence)
- channel label volume
- unconformity label volume or proxy layer
- strat-boundary label volume or derivable boundary proxy
- optional background derivation rule

2. Add feature manifest artifact in each generated dataset folder:
- geologic_feature_schema.json with label-id mapping and provenance.

3. Add generator-level config switch:
- export_geologic_feature_labels true/false
- strict mode fails run if required label arrays are missing.

4. Add tests:
- schema presence test for generated datasets
- dimensional consistency test between seismic and each label volume
- non-empty label sanity checks (for expected classes)

Notes from current state:
- Current pipeline already stores geologic_score and channel label products, which is a strong base.
- Essential-output mode currently keeps very few datasets; if training on feature vectors, essential-output should include required label arrays or a derived patch-vector artifact.

## B2. synthoseis-pretrain-v2 repository (intermediate data loading and pretraining)
Primary role: add reusable patch-vector extraction and optional balanced sampling utilities that can be shared by downstream consumers.

Required additions:
1. Dataloader extension:
- Add optional return fields feature_fraction and primary_label.
- Compute vectors from voxel labels at crop time when label arrays are available.
- Keep backward compatibility with current tuple output under feature flag.

2. Sampling extension:
- Add balanced or quota-based sampling by primary_label.
- Preserve current geologic-score sampling path as fallback.

3. Persisted sidecar option:
- Optional geologic_feature_selected_points.json or equivalent, including selected centers and feature statistics.

4. CLI/config additions:
- enable_geologic_feature_vectors
- geologic_feature_keys mapping
- geologic_feature_thresholds
- balanced_feature_sampling and per-class quotas

5. Tests:
- dataloader output contract test (old mode and new mode)
- vector normalization test (sum to 1)
- sampler class-balance smoke test

## B3. synthoseis-3dvae-poc repository (this repo)
Primary role: consume patch vectors and train with feature head plus geologic contrastive objective.

Required additions:
1. Sampling script changes:
- Extend patch output zarr schema to include feature_fraction, feature_presence, primary_label.
- Preserve existing patches array for backward compatibility.

2. Training dataset changes:
- Load new arrays when present.
- Fall back gracefully to reconstruction-only mode when absent.

3. Loss stack changes:
- Add feature prediction loss and geologic contrastive loss behind flags.
- Add lambda schedules and warmup ramps.

4. CLI changes:
- enable_feature_head
- enable_geologic_contrastive
- lambda_feat
- lambda_geo
- feature thresholds and similarity temperature

5. Evaluation additions:
- linear probe metric on frozen latent
- cluster-separation metrics by primary_label
- reconstruction regression guardrails

---

## C. Supersession Clarification to Add in Plan Header

Add this exact governance section at top of future plan versions:

- Plan status: Updated plan is canonical.
- Supersedes: preliminary sections that assume single-primary-label SupCon as default objective.
- Retains: staged curriculum, reconstruction-first training, balanced sampling requirement.
- Effective date: 2026-06-09.

---

## D. Probability-Based Milestones and Go-No-Go Gates

## D1. Milestone 1: Data contract readiness
Exit criteria:
- Upstream datasets contain required label arrays and schema metadata.
- Patch-level vectors reproducibly computed in intermediate loader.

Probability after M1: target success rises from 0.64 to 0.72.

## D2. Milestone 2: Feature-head only training
Exit criteria:
- Reconstruction metrics within 3 to 5 percent of baseline.
- Feature-head predictive metrics materially above random baseline.

Probability after M2: target success rises to 0.76.

## D3. Milestone 3: Geologic contrastive enabled
Exit criteria:
- Latent probe and clustering improve versus M2.
- Reconstruction remains within agreed drift band.

Probability after M3: target success rises to 0.81.

## D4. Go-no-go guardrails
Pause or rollback if any occurs:
- Sustained reconstruction regression beyond guardrail for 3 checkpoints.
- Label-vector quality checks fail (distribution collapse or mapping drift).
- Class-balance sampler cannot maintain minimum per-batch positive counts.

---

## E. Execution Order

1. Finalize schema and metadata contract in synthoseis.
2. Implement vector extraction and balanced sampling in synthoseis-pretrain-v2.
3. Extend patch sampling and training in synthoseis-3dvae-poc.
4. Run staged curriculum and evaluate against gates.
5. Promote canonical experiment profile only after M3 criteria pass.

---

## F. Final Assessment Snapshot

- Intent clarity: high, with minor contract ambiguities.
- Probability of achieving target: 0.64 currently, expected to exceed 0.80 if milestones pass.
- Probability of implementing successfully: 0.82 currently.
- Supersession visibility: partial, needs explicit governance block.
- Related-repo changes: required and well-scoped, mainly data-contract and loader evolution.
