# Tokenizer Geologic Similarity Training Plan

## Status and scope

- Status: implementation-ready; similarity semantics, promotion gates, compute constraints, and training-origin protocol are confirmed.
- Repository: `synthoseis-3dvae-poc`.
- Primary downstream workflow: compare pairs of seismic patches using cosine similarity between deterministic encoder `mu` vectors in the seismic tokenizer.
- Primary objective: make tokenizer-near patches geologically similar and tokenizer-distant patches geologically dissimilar.
- Constraint: reconstruction is secondary, but it must remain within an agreed regression guardrail.
- Supersedes for this repository: primary-label-only contrastive learning and reconstruction-only adaptive sampling as the intended final design.
- Retains from earlier plans: continuous patch metadata, staged training, balanced geology-aware batches, reconstruction preservation, fixed tokenizer API, and validation against a frozen baseline.

## Current system and motivation

The VAE encoder emits `mu` and `logvar`. During VAE training, the decoder reconstructs from a stochastic sample:

```text
z = mu + epsilon * exp(0.5 * logvar)
```

The tokenizer does not use sampled `z`. It preprocesses each cube deterministically, encodes it, and compares deterministic `mu` vectors with cosine similarity. Geology-oriented objectives and retrieval validation must therefore operate on normalized `mu`, while reconstruction continues to use sampled `z`.

The current geology loss is directionally correct. For the selected metadata vector, it matches the mini-batch pairwise cosine-similarity matrix in metadata space to the pairwise cosine-similarity matrix in `mu` space. This directly trains the representation and similarity family used by the tokenizer.

Current limitations:

1. Batch size 12 supplies only 66 unique off-diagonal pairs and does not ensure useful positive, negative, or hard pairs.
2. Random batches can contain too few geology-rich or geologically comparable patches.
3. Raw metadata fractions are normalized per patch but not calibrated per feature, so feature prevalence and scale can distort metadata cosine.
4. An all-zero metadata vector has zero similarity even to another all-zero vector; background semantics are undefined.
5. The current loss includes the similarity-matrix diagonal, which is constant and provides no separation signal.
6. Mean-squared similarity matching treats all pairs equally rather than emphasizing retrieval mistakes near the top of the ranking.
7. The current geology weight, `0.03`, has not been calibrated against gradient magnitudes from MAE, LPIPS, and KL.
8. Adaptive sampling is driven only by per-patch reconstruction loss and recent reconstruction improvement. It does not use geology rarity, pair availability, latent-neighbor errors, or hard positive/negative relationships.

The epoch-657 checkpoint has a preliminary 128-patch tokenizer-aligned baseline:

| Metric | Value |
|---|---:|
| Pair cosine correlation | 0.025940 |
| Similar-pair latent cosine | 0.224700 |
| Dissimilar-pair latent cosine | 0.212624 |
| Cosine separation | 0.012076 |
| Neighbor overlap at 5 | 0.064063 |
| Random neighbor-overlap reference | 0.039370 |

These values indicate weak, slightly above-random geological organization. They are provisional because the final baseline must use the frozen validation set, 512 samples, repeated short-run seeds where sampling is involved, and the exact metrics defined below.

## Desired outcome

The trained encoder should satisfy all of the following:

1. `mu` cosine ranking reflects the selected geology semantics for faults, channels, flat spots, onlaps, closures, anomalous amplitudes, and their meaningful combinations.
2. Similar patches move closer and dissimilar patches move farther apart in normalized `mu` space.
3. Retrieval quality improves at the top of the ranking, not only in global pairwise correlation.
4. Results are deterministic for a fixed checkpoint and patch because evaluation uses tokenizer preprocessing and `mu`.
5. The tokenizer API, latent dimension, checkpoint contract, and cosine-similarity implementation remain unchanged.
6. Reconstruction MAE and LPIPS remain inside agreed regression limits relative to the frozen baseline.
7. Adaptive sampling continues to expose difficult reconstruction examples while materially increasing useful geology pairs and latent-neighbor corrections.

## Canonical representations

### Tokenizer latent

- Use encoder `mu`, never sampled `z`, for geology losses, pair mining, adaptive geology scores, and retrieval metrics.
- L2-normalize `mu` immediately before cosine-oriented objectives.
- Continue using sampled `z` for VAE decoding and reconstruction training.

### Geology representation

The supervision target must combine feature quantity with orientation- and translation-invariant geometry and topology. Matching fractions alone are insufficient: two patches with different morphology or topology should be less similar, while rotating or translating the same geological structure within a patch should not reduce similarity.

The five equally important primary feature families are:

1. fault
2. channel
3. flat spot
4. onlap
5. closure

Initial quantity features:

1. fault fraction
2. fault-intersection fraction as a fault modifier
3. channel fraction
4. channel-core fraction as a channel modifier
5. flat-spot fraction
6. onlap fraction
7. onlap variability
8. closure fraction
9. closure subtype fractions where available, including simple, faulted, stratigraphic, oil, gas, brine, and other retained closure labels
10. anomalous-amplitude quantity and strength derived from seismic amplitudes using a training-fitted, robust threshold policy

Closure source arrays must be audited in Phase 0. Expected synthetic Zarr candidates include `simple_closures`, `faulted_closures`, `strat_closures`, `oil_closures`, `gas_closures`, `brine_closures`, `closure_segments_id`, and any canonical aggregate closure array present in the generated schema. The implementation must record which arrays are authoritative and how overlapping closure subtypes are combined.

For each primary family, add descriptors that preserve morphology and topology without encoding absolute orientation or location. Candidate descriptors include:

- connected-component count and component-size distribution;
- Euler characteristic or equivalent hole/connectivity measure;
- skeleton length, endpoint count, branch-point count, and loop count where meaningful;
- surface-to-volume or boundary-to-area ratio;
- compactness and elongation eigenvalue ratios, excluding eigenvector direction;
- distance-distribution summaries relative to each feature's own centroid, excluding absolute centroid coordinates;
- translation-invariant relative-distance summaries between co-occurring feature families.

Do not include absolute centroid coordinates, absolute orientation angles, or rotation-sensitive directional components in the similarity target. Unit tests must prove that translating or rotating a synthetic label structure leaves its target similarity effectively unchanged, while changing its connectivity or shape lowers similarity.

Represent the target as grouped sub-vectors rather than one flat unweighted list:

```text
geology_target = {
        primary_presence_and_fraction,
        family_modifiers,
        invariant_morphology,
        invariant_topology,
        cross_family_relations,
        amplitude_anomaly,
}
```

Give the five primary families equal top-level weight. Normalize descriptors within each family before combining family similarities so families with more descriptors do not dominate. Fault intersection modifies the fault-family similarity rather than acting as a sixth equally weighted family. Channel core follows the same parent-modifier pattern for channels.

Mixed-feature ordering must be explicit. A fault-plus-channel patch is closest to another fault-plus-channel patch, partially similar to fault-only and channel-only patches, and farthest from a patch containing neither. Continuous multi-label family overlap supplies this ordering; interaction and cross-family geometry descriptors distinguish different fault-plus-channel arrangements.

Fit feature calibration statistics on the training split only. Persist them in the sampled Zarr attributes and checkpoints. Apply the same transform to training, validation, diagnostics, and post-hoc evaluation.

Recommended initial transform:

1. replace non-finite values with zero;
2. apply `log1p(value / scale)` to sparse non-negative fractions, with a scale derived from the nonzero training distribution;
3. robustly scale each feature using training median and interquartile range or a documented percentile scale;
4. apply explicit feature weights only after an unweighted baseline;
5. retain an explicit `has_selected_geology` mask for loss masking and cohort reporting, but do not turn background into a positive geology class;
6. L2-normalize the final vector for cosine targets.

Background patches are neutral for geology similarity. They continue to train reconstruction, LPIPS, and KL, but pairs involving a patch with no selected geology are excluded from geology geometry/ranking losses and from positive/negative retrieval denominators. They are reported as a separate benchmark cohort to detect accidental latent collapse or false high-similarity retrieval.

The calibration artifact must include feature order, family hierarchy, modifier relationships, descriptor definitions, transform, fitted statistics, zero/background policy, closure source mapping, anomalous-amplitude policy, schema version, and source training dataset identity.

## Training objective

Use a staged multi-objective loss:

```text
L_total = L_reconstruction
        + lambda_lpips * L_lpips
        + lambda_kl * L_kl
        + lambda_geometry * L_geometry
        + lambda_rank * L_rank
        + optional lambda_invariance * L_invariance
```

### Geometry preservation loss

Retain the current continuous metadata-to-latent geometry concept, with these corrections:

- operate on normalized `mu`;
- use calibrated grouped geology targets;
- exclude diagonal pairs;
- exclude neutral background pairs;
- combine equal-weight family similarity with invariant geometry/topology similarity;
- optionally downweight ambiguous middle-similarity pairs;
- report the loss independently from total loss;
- test zero-vector and duplicate-vector behavior explicitly.

Start with off-diagonal weighted regression between latent cosine and metadata cosine. Huber loss is the recommended first candidate because it is less sensitive than MSE to noisy metadata targets.

### Retrieval-ranking loss

Add an objective that emphasizes tokenizer ranking behavior. The recommended first implementation is continuous supervised contrastive ranking:

- positives: metadata cosine above a configurable positive threshold or within the metadata top-k neighborhood;
- negatives: metadata cosine below a configurable negative threshold;
- hard positives: geologically similar pairs with low latent cosine;
- hard negatives: geologically dissimilar pairs with high latent cosine;
- ambiguous pairs between thresholds: ignored by the ranking term but still available to geometry regression.

Implement a bounded memory queue of detached `mu` and metadata vectors so pair mining is not limited to 12 examples. Queue entries must come from recent batches only, carry dataset indices, and never create a self-pair. Gradients flow through current-batch `mu`; queued vectors are comparison anchors without gradients.

Compare at least two candidates under identical seeds:

1. margin-based pair or triplet ranking on cosine distance;
2. continuous supervised contrastive loss with metadata-similarity weighting.

Adopt the simpler candidate unless the more complex objective provides a repeatable retrieval gain.

### Augmentation invariance

The tokenizer uses deterministic extrema preprocessing, while training includes warping, mixup, and occasional decimation. Add an optional low-weight consistency term between normalized `mu` for two valid views of the same source patch. Do not treat mixup views as exact positives unless their metadata target is mixed by the same coefficient.

This term is accepted only if it improves tokenizer-preprocessed validation retrieval without violating reconstruction guardrails.

### Loss scheduling and gradient monitoring

1. Resume with reconstruction, LPIPS, and KL behavior unchanged.
2. Warm up calibrated geometry loss from zero.
3. Add ranking loss only after geometry metrics are finite and stable.
4. Increase geology weights gradually; do not select weights solely by total-loss scale.
5. Log each weighted and unweighted component plus encoder gradient norms attributable to reconstruction and geology objectives on periodic probe batches.
6. Stop increasing geology weight when reconstruction reaches its guardrail or geology gradients consistently dominate the encoder.

## Geology-aware batch construction

Independent weighted sampling is insufficient for a pair objective because it cannot guarantee batch composition. Replace the training sampler with a geology-aware batch sampler while retaining a reconstruction-difficulty component.

Each batch should target:

- at least one positive pair for each represented geology stratum when available;
- multiple negative strata;
- a configurable fraction of background patches;
- a configurable fraction of reconstruction-hard patches;
- source-volume diversity to reduce memorization of one synthetic realization;
- no duplicate dataset index within a batch unless explicitly testing augmentation consistency.

Create coarse strata from calibrated metadata using presence thresholds and/or clustering. Multi-label patches remain multi-label; a primary stratum is only a sampling aid and must not replace the continuous target representation. Include single-family, mixed-family, closure-subtype, anomalous-amplitude, and neutral-background cohorts. Pair construction must enforce the confirmed ordering for mixed patches.

## Adaptive sampling redesign

Adaptive sampling should remain useful for reconstruction while becoming explicitly useful for geological retrieval.

### Per-example signals

Maintain normalized moving estimates for:

1. `recon_difficulty`: current per-example configured reconstruction loss;
2. `recon_learning`: recent reconstruction improvement, preserving current behavior;
3. `geology_rarity`: inverse frequency of the example's geology stratum or local metadata neighborhood;
4. `latent_neighbor_error`: disagreement between metadata neighbors and latent cosine neighbors;
5. `hard_positive_error`: low latent cosine for metadata-similar pairs;
6. `hard_negative_error`: high latent cosine for metadata-dissimilar pairs;
7. `staleness`: time since the example was last evaluated or sampled.

### Composite priority

Use a bounded, normalized score such as:

```text
priority = w_recon * recon_difficulty
         + w_learning * positive_recon_learning
         + w_rarity * geology_rarity
         + w_neighbor * latent_neighbor_error
         + w_pair * max(hard_positive_error, hard_negative_error)
         + w_staleness * staleness
```

Recommended initial weight budget:

- reconstruction difficulty and learning: 40 percent total;
- geology rarity and batch composition: 25 percent total;
- latent-neighbor and hard-pair error: 30 percent total;
- staleness/exploration: 5 percent total.

These are starting values, not acceptance criteria. Apply probability floors and caps so rare/noisy examples cannot monopolize training and every example retains a nonzero chance of selection.

### Two-stage sampling

Use two stages rather than drawing examples independently:

1. choose geology strata and required positive/negative relationships for the next batch;
2. choose examples within those constraints using the composite adaptive priority.

Recompute expensive latent-neighbor scores at the existing snapshot interval on deterministic, unaugmented inputs. Update cheap reconstruction signals continuously or at the current full-dataset snapshot interval. Persist score components separately so sampling behavior can be audited and resumed.

### Ablation requirement

Compare:

1. uniform shuffle;
2. current reconstruction-adaptive sampler;
3. geology-balanced sampler without adaptive scores;
4. full geology-aware adaptive sampler.

The full sampler is accepted only if it improves retrieval metrics beyond the current sampler while preserving reconstruction within the guardrail.

## Evaluation protocol

### Frozen benchmark

Before changing training behavior:

1. freeze train and validation Zarr identities, sampling seeds, patch origins, metadata schema, and calibration artifact;
2. save baseline checkpoint identity and command;
3. evaluate the same validation examples in the same order with tokenizer preprocessing;
4. use repeated short seeded screens for candidate comparisons, then run the selected final configuration once from the confirmed clean/staged origin;
5. prohibit validation examples from the pair-mining queue and adaptive training scores.

Build balanced synthetic query cohorts for each primary family, feature combinations, invariant-geometry variants, closure subtypes, anomalous amplitudes, and neutral background. Synthetic labels are the current release evidence because no expert-reviewed real-seismic query set is available. Record expert-reviewed real-seismic evaluation as future work, not a current promotion gate.

### Primary tokenizer metrics

- metadata-neighbor overlap at `k = 5, 10, 20`;
- recall at k for metadata-defined positives;
- precision at k and normalized discounted cumulative gain at k;
- hard-negative rate in top-k results;
- cosine separation between top and bottom metadata-similarity tails;
- pairwise cosine correlation as a secondary global metric.

Report bootstrap confidence intervals over query patches. Slice every metric by geology feature, feature prevalence, background, source volume, and mixed-feature complexity.

### Reconstruction guardrails

- validation MAE no more than 3 percent worse than baseline;
- validation LPIPS no more than 5 percent worse than baseline;
- no feature slice more than 10 percent worse in MAE;
- no visual collapse, amplitude instability, or decoder artifacts in fixed representative patches.

These reconstruction limits are confirmed promotion gates.

Checkpoint selection must be constrained multi-objective selection. A checkpoint is eligible only if it passes reconstruction guardrails; among eligible checkpoints, rank primarily by validation neighbor overlap and nDCG.

Promotion additionally requires at least 25 percent relative improvement over baseline in neighbor overlap and nDCG at 5, improvement in at least three of the five primary feature families, and no statistically credible regression in any primary family. These criteria are confirmed.

Early stopping and the learning-rate scheduler must not continue to use total reconstruction-oriented validation loss as the sole decision signal. Add a configurable constrained retrieval score for checkpoint promotion while retaining reconstruction as a rejection gate.

## Implementation phases

### Phase 0: Lock semantics and baseline

Files:

- `docs/plans/2026-08-26__tokenizer_geologic_similarity_training_plan.md`
- `scripts/train.py`
- a new reusable evaluation script under `scripts/`

Tasks:

1. Measure Mac mini throughput and record the screening and final-run wall-clock budgets.
2. Freeze and record the benchmark dataset and checkpoint.
3. Move the post-hoc latent diagnostic command into a reusable CLI.
4. Capture the 512-example baseline and per-feature retrieval slices.
5. Add reconstruction guardrail metrics and constrained checkpoint-selection specification.

Exit gate: deterministic baseline report can be reproduced from one command.

Runtime gate: benchmark dataset generation, training throughput, validation, adaptive snapshots, and tokenizer diagnostics separately on the Mac mini. Use the measured sustained rate to project the final run before launching it. The projected final training run must complete in no more than 14 wall-clock days with at least a 20 percent contingency margin; therefore the nominal projection should be at most 11.2 days. Reduce epochs, batches per epoch, diagnostic frequency, or snapshot frequency before reducing benchmark correctness or violating reconstruction/retrieval gates.

### Phase 1: Metadata calibration and metric correctness

Files:

- `scripts/sample_patches.py`
- `scripts/train.py`
- new focused metadata-calibration module under `src/`
- focused tests under `tests/`

Tasks:

1. Fit and persist training-only feature calibration.
2. Define background and all-zero semantics.
3. Exclude diagonals from geometry loss.
4. Add Huber and weighted-pair options behind CLI flags.
5. Add top-k retrieval metrics and feature-sliced reporting.

Exit gate: synthetic tests prove identical metadata ranks nearest, opposite metadata ranks farther away, zero/background handling is defined, and validation is deterministic.

### Phase 2: Geology-aware batches

Files:

- `scripts/train.py`
- new sampler module under `src/`
- sampler tests under `tests/`

Tasks:

1. Build reproducible multi-label geology strata.
2. Implement a batch sampler with positive, negative, background, and source-volume quotas.
3. Log achieved batch composition and fallback frequency.
4. Preserve an explicit uniform-sampling mode for ablation.

Exit gate: every feasible test batch satisfies configured pair constraints and fixed seeds reproduce index sequences.

### Phase 3: Ranking objective and queue

Files:

- new geology-loss module under `src/`
- `scripts/train.py`
- focused loss and queue tests under `tests/`

Tasks:

1. Implement cosine margin/triplet baseline.
2. Implement continuous supervised contrastive candidate.
3. Add bounded detached memory queue and hard-pair mining.
4. Add schedules, component logging, and gradient diagnostics.
5. Run short controlled experiments before full training.

Exit gate: synthetic embeddings optimize in the expected direction, no self-pairs or validation leakage occur, and one candidate beats geometry-only training on frozen retrieval metrics.

### Phase 4: Geology-aware adaptive sampling

Files:

- sampler module under `src/`
- `scripts/train.py`
- adaptive snapshot schema and tests

Tasks:

1. Add rarity, neighbor-error, hard-pair, and staleness signals.
2. Normalize each signal robustly and persist it separately.
3. Implement two-stage constrained adaptive batch construction.
4. Add floors, caps, effective-sample-size diagnostics, and resume support.
5. Run the four required sampler ablations.

Exit gate: full adaptive sampling improves retrieval over reconstruction-only adaptive sampling without violating reconstruction guardrails.

### Phase 5: Augmentation invariance and final tuning

Tasks:

1. Test same-patch view consistency at low weight.
2. Verify metadata handling for mixup or exclude mixup from paired consistency.
3. Tune geology weights and thresholds on training folds only.
4. Run multiple short seeded screens that are long enough to compare trend direction and stability.
5. Promote one configuration to the clean or clearly staged final run.

Exit gate: short seeded screens show consistent gains, bootstrap confidence intervals on the frozen benchmark support promotion, and the selected full run remains within the 14-day wall-clock limit.

### Phase 6: Tokenizer acceptance test

Files:

- tokenizer integration tests
- final report under `docs/`

Tasks:

1. Load the candidate checkpoint through `VaeLatentAdapter`.
2. Run real tokenizer queries against fixed synthetic reference patches.
3. Save top-k results with geology metadata and cosine scores.
4. Compare baseline and candidate blind to checkpoint identity where practical.
5. Document exact commands, checkpoints, calibration artifact, and results.

Exit gate: candidate improves retrieval metrics, passes reconstruction gates, preserves tokenizer compatibility, and produces qualitatively credible top-k matches.

## Test requirements

- metadata transform round-trip and schema-version tests;
- no train/validation calibration leakage;
- geometry loss excludes diagonal and remains finite for zero vectors;
- positive and negative pair direction tests;
- queue capacity, eviction, self-pair, and detach tests;
- batch-composition and deterministic-seed tests;
- adaptive-score normalization, floor, cap, and resume tests;
- tokenizer adapter uses deterministic `mu` and unchanged output shape;
- retrieval metric tests with known rankings;
- reconstruction guardrail and constrained checkpoint-selection tests;
- existing reconstruction, LPIPS, deep-supervision, sampling, and tokenizer regression suites.

## Observability and artifacts

Log to TensorBoard and CSV:

- unweighted and weighted loss components;
- neighbor overlap, recall, precision, nDCG, hard-negative rate, cosine gap, and pair correlation;
- metrics by geology feature and background status;
- positive/negative pairs per batch and fallback counts;
- queue occupancy and hard-pair counts;
- each adaptive score component, probability entropy, effective sample size, min/max probability, and stratum exposure;
- validation MAE and LPIPS deltas from baseline;
- checkpoint eligibility and rejection reason.

Persist:

- metadata calibration artifact;
- frozen benchmark manifest;
- adaptive sampling state;
- experiment configuration and seeds;
- baseline and candidate retrieval reports;
- final checkpoint-selection decision.

## Confirmed semantic decisions

1. Similarity combines feature fractions with similar geometry and topology.
2. Absolute orientation and absolute location within a patch must not reduce similarity.
3. Background patches are neutral to geology objectives.
4. Mixed-feature patches are closest to the same combination, partially similar to matching single-feature patches, and farthest from patches containing neither feature.
5. Fault, channel, flat spot, onlap, and closure are equally important primary families.
6. Fault intersection modifies fault similarity rather than acting as an equal primary family.
7. Queries are varied and include combinations, anomalous amplitudes, and hydrocarbon-trapping closure geometries.
8. Synthetic labeled data is the current quantitative acceptance source; no expert-reviewed real-seismic set is currently available.
9. Reconstruction promotion limits are 3 percent validation MAE, 5 percent LPIPS, and 10 percent per-slice MAE regression.
10. Retrieval promotion requires 25 percent relative improvement in neighbor overlap and nDCG at 5, improvement in at least three of five primary families, and no statistically credible primary-family regression.

## Confirmed execution constraints

1. Training and synthetic seismic generation run on one Mac mini.
2. Monetary experiment cost is not a limiting constraint; machine wall-clock time is the limiting resource.
3. The selected final VAE training run must finish in less than 14 wall-clock days.
4. Phase 0 must profile sustained throughput and enforce a nominal projection of at most 11.2 days, preserving a 20 percent contingency margin.
5. Synthetic dataset generation and final training should not compete concurrently for CPU, memory, storage bandwidth, or accelerator resources. Generate, validate, and freeze the required datasets before the final run.
6. Use checkpointing and complete resume state for the optimizer, scheduler, loss schedules, adaptive sampler, memory queue policy, random generators, and elapsed-training counters. An interruption must not require restarting the final run.
7. Use epoch 657 for fast feasibility experiments and objective/sampler screening.
8. Use a clean or clearly staged baseline-to-geology run for final promotion evidence so prior optimizer history does not confound the result.
9. Use repeated short seeded screens rather than three sequential full runs. Quantify final retrieval uncertainty with frozen-query bootstrap confidence intervals and feature-cohort slices.
10. If the projected final run exceeds the runtime gate, first reduce redundant epochs or batches and expensive diagnostic cadence. Do not weaken the frozen benchmark, reconstruction guardrails, or promotion criteria to meet runtime.

## Agent handoff checklist

An agentic implementation should not begin model changes until it can state:

1. the confirmed semantic decisions above;
2. the exact frozen train and validation dataset identities and seeds;
3. the metadata schema and calibration policy;
4. positive, negative, and background semantics;
5. reconstruction and retrieval promotion thresholds;
6. the measured Mac mini throughput, short-screen budget, and projected final runtime;
7. that synthetic labels are the current acceptance source and real-seismic expert review is deferred;
8. the first focused test that will falsify each implementation hypothesis.

Once those are recorded, implementation should proceed phase by phase, with one focused executable validation immediately after each initial edit and no promotion past a phase gate without saved evidence.