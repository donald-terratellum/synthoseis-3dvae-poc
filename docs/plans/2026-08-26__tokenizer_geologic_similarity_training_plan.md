# Tokenizer Geologic Similarity Training Plan

## Status and scope

- Status: proposed implementation plan; decisions in "Clarifications required" must be confirmed before implementation.
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

These values indicate weak, slightly above-random geological organization. They are provisional because the final baseline must use the frozen validation set, 512 samples, multiple seeds where sampling is involved, and the exact metrics defined below.

## Desired outcome

The trained encoder should satisfy all of the following:

1. `mu` cosine ranking reflects the selected geology semantics for faults, fault intersections, channels, channel cores, flat spots, onlaps, and onlap variability.
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

### Geology vector

Initial ordered features:

1. fault fraction
2. fault-intersection fraction
3. channel fraction
4. channel-core fraction
5. flat-spot fraction
6. onlap fraction
7. onlap variability

Fit feature calibration statistics on the training split only. Persist them in the sampled Zarr attributes and checkpoints. Apply the same transform to training, validation, diagnostics, and post-hoc evaluation.

Recommended initial transform:

1. replace non-finite values with zero;
2. apply `log1p(value / scale)` to sparse non-negative fractions, with a scale derived from the nonzero training distribution;
3. robustly scale each feature using training median and interquartile range or a documented percentile scale;
4. apply explicit feature weights only after an unweighted baseline;
5. append a background/no-selected-geology indicator if background similarity is part of the desired semantics;
6. L2-normalize the final vector for cosine targets.

The calibration artifact must include feature order, transform, fitted statistics, zero/background policy, schema version, and source training dataset identity.

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
- use calibrated metadata vectors;
- exclude diagonal pairs;
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

Create coarse strata from calibrated metadata using presence thresholds and/or clustering. Multi-label patches remain multi-label; a primary stratum is only a sampling aid and must not replace the continuous target vector.

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
4. evaluate at least three training seeds for final candidate comparisons;
5. prohibit validation examples from the pair-mining queue and adaptive training scores.

### Primary tokenizer metrics

- metadata-neighbor overlap at `k = 5, 10, 20`;
- recall at k for metadata-defined positives;
- precision at k and normalized discounted cumulative gain at k;
- hard-negative rate in top-k results;
- cosine separation between top and bottom metadata-similarity tails;
- pairwise cosine correlation as a secondary global metric.

Report bootstrap confidence intervals over query patches. Slice every metric by geology feature, feature prevalence, background, source volume, and mixed-feature complexity.

### Reconstruction guardrails

Recommended defaults pending user confirmation:

- validation MAE no more than 3 percent worse than baseline;
- validation LPIPS no more than 5 percent worse than baseline;
- no feature slice more than 10 percent worse in MAE;
- no visual collapse, amplitude instability, or decoder artifacts in fixed representative patches.

Checkpoint selection must be constrained multi-objective selection. A checkpoint is eligible only if it passes reconstruction guardrails; among eligible checkpoints, rank primarily by validation neighbor overlap and nDCG.

Early stopping and the learning-rate scheduler must not continue to use total reconstruction-oriented validation loss as the sole decision signal. Add a configurable constrained retrieval score for checkpoint promotion while retaining reconstruction as a rejection gate.

## Implementation phases

### Phase 0: Lock semantics and baseline

Files:

- `docs/plans/2026-08-26__tokenizer_geologic_similarity_training_plan.md`
- `scripts/train.py`
- a new reusable evaluation script under `scripts/`

Tasks:

1. Resolve all decisions under "Clarifications required."
2. Freeze and record the benchmark dataset and checkpoint.
3. Move the post-hoc latent diagnostic command into a reusable CLI.
4. Capture the 512-example baseline and per-feature retrieval slices.
5. Add reconstruction guardrail metrics and constrained checkpoint-selection specification.

Exit gate: deterministic baseline report can be reproduced from one command.

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
4. Run at least three final seeds.

Exit gate: gains are repeatable and confidence intervals do not indicate a one-seed result.

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

## Clarifications required

The following decisions materially change implementation. Recommended defaults are provided so an agent can proceed after confirmation.

### 1. Meaning of geologic similarity

Question: Should similarity mean proportional agreement across all seven features, shared presence of a dominant feature, or a hierarchy where some features matter more?

Recommended default: continuous calibrated seven-feature similarity, with equal initial feature weights and feature-sliced evaluation. Do not impose domain weights until baseline distributions and retrieval errors are reviewed.

### 2. Background patches

Question: Should two patches containing none of the selected features be considered similar, neutral, or not useful for tokenizer matching?

Recommended default: add an explicit background indicator, treat background-background as weak positives, and cap their share of every batch and benchmark.

### 3. Mixed geology

Question: Should a fault-plus-channel patch be close to fault-only, channel-only, both, or only other mixed patches?

Recommended default: use continuous overlap so it can be partially similar to both; reserve hard positives for sufficiently high metadata cosine.

### 4. Feature priority

Question: Are faults, channels, flat spots, and onlaps equally important to the user workflow? Is fault intersection a distinct target or primarily a modifier of fault similarity?

Recommended default: keep all seven dimensions, report each separately, and treat intersection/core features as independent dimensions until retrieval review supports hierarchical weighting.

### 5. Spatial and orientation invariance

Question: Should two patches with the same feature fraction but different geometry, orientation, topology, or spatial arrangement be considered similar?

This is the largest semantic uncertainty. Fractions alone cannot distinguish a straight channel from a branching channel or one fault from a fault network.

Recommended default: regard fractions as weak supervision, not complete ground truth. Plan a later metadata extension for morphology, orientation, topology, and relative spatial arrangement after the first retrieval baseline.

### 6. Positive and negative thresholds

Question: What metadata cosine values define an acceptable positive and definite negative?

Recommended default: derive thresholds from training-distribution quantiles, initially top 10 percent as positives and bottom 10 percent as negatives, then inspect examples before freezing them.

### 7. Reconstruction regression budget

Question: What degradation is acceptable in MAE, LPIPS, and feature-specific reconstruction?

Recommended default: at most 3 percent validation MAE regression, 5 percent LPIPS regression, and 10 percent MAE regression in any geology slice.

### 8. Retrieval success threshold

Question: What minimum improvement justifies promoting a model?

Recommended default: at least 25 percent relative improvement over baseline neighbor overlap and nDCG at 5, improvement in at least three of four primary feature families, and no statistically credible regression in any primary family.

### 9. Query population

Question: Will production queries mostly contain one clear feature, mixed geology, background, or subtle analogs with low feature fractions?

Recommended default: construct and report separate benchmark cohorts rather than allowing the most common patch type to dominate the aggregate.

### 10. Synthetic-to-real expectations

Question: Is synthetic retrieval performance the release criterion, or must a small expert-reviewed real-seismic query set also pass?

Recommended default: use synthetic labels for quantitative development and require a fixed expert-reviewed real-seismic smoke benchmark before declaring production readiness.

### 11. Compute and experiment budget

Question: How many full training runs and seeds are affordable?

Recommended minimum: short screening runs for objective choices, followed by three full seeds for the selected configuration and baseline-equivalent control.

### 12. Resume versus clean retraining

Question: Should geology shaping continue from epoch 657 or begin from a reconstruction baseline before geology-aware scheduling?

Recommended default: use epoch 657 for fast feasibility experiments, but confirm the final result with a clean or clearly staged baseline-to-geology run so optimizer history and prior training do not confound conclusions.

## Agent handoff checklist

An agentic implementation should not begin model changes until it can state:

1. the confirmed answers to all high-impact clarifications above;
2. the exact frozen train and validation dataset identities and seeds;
3. the metadata schema and calibration policy;
4. positive, negative, and background semantics;
5. reconstruction and retrieval promotion thresholds;
6. experiment and compute budget;
7. whether the final evidence requires real-seismic expert review;
8. the first focused test that will falsify each implementation hypothesis.

Once those are recorded, implementation should proceed phase by phase, with one focused executable validation immediately after each initial edit and no promotion past a phase gate without saved evidence.