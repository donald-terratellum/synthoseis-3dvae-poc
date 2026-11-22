# Streamed overlap and fluid-aware semantic retrieval: execution plan

Updated: 2026-11-22, with the refreshed source inventory and the owner's revised, smoothed observable-event target.

## 1. Authority, scope, and approval gates

This is the governing implementation contract, replacing the provisional options in
the November 8 version. Implement opt-in streamed training and evaluation, not a change
to production tokenizer defaults. Do not launch training from this planning request.
The [GitHub handoff](2026-11-11__streamed_overlap_copilot_implementation.md) contains
the paste-ready prompt, cloud/local division, and ordered implementation tasks.

### Current bounded local milestone

The owner authorized local implementation using the Mac's `/Volumes` data and this
repository's `.venv`; commits and a push are required on the existing task branch.
Datasets, private audit outputs, secrets and unrelated work must remain uncommitted.
Production training is still separately gated; the target and preflight work below
must not be mistaken for permission to launch the training suite.

**2026-11-22 refreshed local audit:** Run
`streamed-overlap-target-audit-20261122-03` inventories all recursive Zarr array
keys/shapes/chunks/dtypes and source/config metadata times for 180 training and 25
validation sources. All 205 have the required semantic arrays and no inventory issues;
all retain `faulted_net_to_gross` and `faults/faulted_channel_labels`, but none retain
`faults/faulted_channel_ng`. Inspection of the current neighboring generator and 13
passing synthetic channel/closure tests confirms its current code inserts channel N/G
into `faulted_net_to_gross` before closure and elastic-property construction. The saved
stores do not retain the per-channel N/G array, and their exact generation-time dirty
worktree/command is unknown; therefore this verifies the implementation path, not every
historical source's provenance. The owner accepted oil/gas closure fractions as a coarse
operational proxy, with that provenance limitation recorded. Full results are in the
[target-contract proposal](2026-11-11__streamed_overlap_target_contract_proposal.md).
No final-test crops were read.

The owner reports that all datasets under `/Volumes/CrucialX9/fake_data` were
recreated. SHA values and timestamps are inconclusive indicators of which generation
produced a source, so do not use them alone to reject or classify sources. The owner is
regenerating `/Volumes/CrucialX9/fake_data_validation_v2`; keep it separate from the
current validation cohort and do not read, audit, or mix it until regeneration is
complete. Audit it separately before considering it for any use. The current
`/Volumes/CrucialX9/fake_data` train and validation sources remain the primary cohort.

The owner has also approved a five-sample, nearest-edge depth smoother blended with
the original seismic before strict adjacent-sample Z-axis peak/trough detection,
same-index non-padded label support, per-source filter metadata rules, closure-proxy
policy, and the proposed evaluation success criteria. A bounded diagnostic of 45,000
full-depth traces from 12 training and 6 validation sources using the previous raw
detector found no zero-event trace; its counts are historical and must not be used to
characterize the smoothed detector. The updated smoothed-detector diagnostic scanned
the same 45,000 traces and found no zero-event trace; p10/median/p90 event counts were
197/215/243 for training and 184/217/239 for validation. The streamed pair-target
implementation, bounded target-yield preflight and frozen primary-validation manifest
have since been implemented and checked; see the current local evidence and its
limitations in the [handoff](2026-11-11__streamed_overlap_copilot_implementation.md).
These bounded checks do not certify every source or open the separate training-launch
gate. The remaining runner and evaluation integration must not be bypassed.

Proceed autonomously on the approved decisions below. Do not repeatedly ask about
settled choices. Only stop for an unresolved scientific contract, failed safety or
resource prerequisite, conflicting user changes, or an explicit approval gate:

1. **Local target-contract completion:** implement the approved labels and detector,
   report source/target validity and yield, and freeze the approved evaluation manifest
   and cohorts. Array names are not sufficient evidence. Do not replace missing labels
   or enable semantic training until the implementation and preflight checks pass.
2. **Local training-launch approval:** after implementation, tests, and local smoke pass,
   present the actual commands, epochs, batches per epoch, resource measurements, and
   resolved configuration once. Obtain approval before the sequential training suite.
   The user deliberately left epochs and batches per epoch unspecified.

A GitHub PR can implement and test the infrastructure without local volumes. It cannot
certify source semantics, MPS behavior, or scientific success from synthetic fixtures.
If cloud work precedes gate 1, explicitly report `LOCAL_TARGET_VALIDATION_PENDING`;
provide the audit tools and complete the approved target implementation after the
local review. Never call that intermediate state production-training-ready.

## 2. Evidence from previous experiments

See the [mu512 screening and extension assessment](../sessions/2026-11-10-p9-mu512-screening-outcome-and-next-steps.md).

| Same-seed reference, seed 20260926 | n@5 | n@10 | Macro AUROC |
|---|---:|---:|---:|
| P0b epoch 30 | 0.0109375 | 0.016667 | 0.6307 |
| mu128 E1 geology epoch 30 | 0.0109375 | 0.020182 | 0.7126 |
| Original mu512 geology epoch 40 | 0.010417 | 0.019792 | 0.6976 |
| Exploratory mu512 extension epoch 50 | 0.011979 | 0.021094 | 0.7419 |
| Exploratory mu512 extension epoch 100 | 0.012500 | 0.021354 | 0.7125 |
| Exploratory mu512 extension epoch 150 | 0.009115 | 0.022005 | 0.7214 |

The original mu512 screen failed. Extension epochs 50, 90, and 100 numerically beat
P0b with reconstruction guardrails satisfied; eight other evaluated extension milestones
did not. Epoch 100's n@5 gain was 14.3% relative but only 0.0015625 absolute. Its
full-store tokenizer synthetic recipe total and real MAE were 3.32% and 4.99% lower
than P0b's reconstruction control. No independent-seed confirmation exists.

The extension reset optimizer/scheduler state at epoch 40; it was not an uninterrupted
or matched 150-epoch screen. `vae_best.pt` selects synthetic-validation-loss epoch 138,
not retrieval epoch 100. Available precision/NDCG bootstrap intervals are not n@5
uncertainty. All extension classifier AUROCs missed 0.75. Earlier phase/stretch and
mu128 E1 numerical gains did not replicate.

Consequences for this experiment:

- Start fresh E1 mu128 controls, not mu512 epoch-100 warm starts. Preserve mu512 as
  an evaluation reference, not an adopted winner or evidence for increasing width.
- Do not add P0b duration experiments, 2048-D training, old classifier/SupCon objectives,
  or a latent-width sweep to this implementation.
- Evaluate retrieval during training; neither lower reconstruction loss nor a good
  pair predictor demonstrates better cosine neighbors.
- Keep independent checkpoint selections for reconstruction and retrieval, with
  explicit identities and metrics. Do not ambiguously name one file "best."
- Save actual optimizer/scheduler/RNG/sampling state for exact epoch-boundary resume.
  A weights-only restart is a different attempt, not exact continuation.
- Historical n@5 remains useful continuity evidence, not the primary measure of the
  newly requested directional, fluid-aware semantic task.

## 3. Approved scientific contract

### Two outputs

1. Physical overlap: the fraction of the canonical 3D source windows shared, not
   an overlap map.
2. Directional, fluid-adjusted **observable seismic-event coverage**: the proportion
   of A's peaks and troughs in the shared region, discounted for differences in
   fluid proportions. A is the reference. This replaces the earlier equal-weight
   distinct-layer and cross-volume analog definitions.

An event is not necessarily a geological layer: thin layers may be unresolvable,
and channels can replace host layers. This target is an observable-event proxy,
not an exact count of matching layers or a direct waveform-similarity measurement.
Because all disjoint and different-source pairs score zero, this objective does not
reward geological similarity between non-overlapping windows or independent volumes.
The signal defines the events; integer age values do not define event identities.
Source coordinates and geology labels generate targets; they are not predictor inputs.

### Event coverage and fluid adjustment

Use the original, unaugmented dense seismic and canonical source coordinates:

- For the same source, intersect A and B in XYZ. If there is no 3D intersection,
  the directional target is zero. Different-source pairs also have target zero;
  numeric age values across independent volumes do not identify common horizons.
- In the shared XY footprint, stream full-depth original A traces in XY chunks and
  first apply `scipy.ndimage.uniform_filter1d` with `size=5`, `axis=2`, and
  `mode="nearest"`. Blend the original and filtered values as
  `0.2 * original + 0.8 * filtered`. Then detect strict local maxima and minima
  along Z: a peak must be greater than both adjacent blended samples, and a trough
  must be less than both. Compare `volume[:,:,1:-1]` with `volume[:,:,:-2]` and
  `volume[:,:,2:]`, then add one to each returned Z index. Use no prominence threshold;
  the first and last Z samples are excluded naturally.
  Sum events in A's shared XYZ region for the numerator, and sum events over the full
  A depth for those same traces for the denominator. A zero denominator for an
  overlapping pair is unavailable, with an explicit reason. The overlap boundary
  must not create artificial extrema.
- In the shared XYZ region, average `geologic_age_faulted` at its shallowest and
  deepest Z indices, across the shared XY footprint. Use these two values as the
  age-range bounds. For each *full* A and B window, select voxels within this range
  using native same-index support wherever both arrays are available. Exclude known
  padded label indices 1500:1510. Source volumes need not have identical Z lengths:
  use only fully supported same-index windows and skip unsupported windows with an
  explicit reason. Do not silently clip, pad, or shift arrays to force equal shapes,
  and do not apply a global correlation-derived +1/+2 Z shift.
- For each window, compute oil and gas fractions from `oil_closures > 0.5` and
  `gas_closures > 0.5`, divided by **all** selected voxels, not just sand. At voxels
  where both masks are positive, count one half toward each. Let `other = 1 - oil - gas`.
  This mathematical remainder includes shale and water as well as brine-bearing sand;
  it does not classify all those voxels as brine.
- Multiply the event ratio by
  `1 - 0.5 * (abs(oil_A-oil_B) + abs(gas_A-gas_B) + abs(other_A-other_B))`.
  Thus full oil versus full gas has factor 0, and 10% oil versus no oil or gas
  has factor 0.9. This is a deliberately coarse, window-level proxy: it can miss
  differences when fluid distributions swap between layers, and is not per-layer
  correspondence or a measured amplitude difference.
- The closure fractions are an owner-approved, coarse operational proxy, not verified
  per-voxel physical-fluid labels. Do not infer brine from absence of an oil/gas
  closure; `other` remains only the formula's mathematical remainder. Record the
  unknown per-source generation-time worktree/command as a limitation, not as a reason
  to block this proxy. Missing, non-finite or otherwise unusable labels, or an empty
  selected age-range support, must yield an explicit unavailable reason, not a zero
  target.

Detect events and calculate both labels before rotation, flips, scaling or added
noise; augment A and B independently afterward. Audit how canonical support and
transformed views relate, including sufficient real context. Do not leak source IDs,
coordinates or labels to the predictors.

Owner decisions now fix the event detector, native label support, closure-mask proxy,
noise metadata sources and evaluation success criteria. Gate 1 remains closed only
until these rules are implemented and tested, source/target validity and yield are
measured, and the deterministic evaluation manifest/cohorts are frozen. The dirty
generation-time provenance remains a documented limitation, not a proxy veto. Do not
reinterpret earlier sampled sources as validation of every stored source. Version the
complete target contract and cover its rules with synthetic cases before authorizing
training.

## 4. Model and objectives

### Initial implementation

- Fresh E1 residual encoder, batch normalization, original skip-free decoder, single
  fullstack channel, XYZ axes, patch shape `(32, 32, 64)`, latent dimension 128.
- Reconstruct A through sampled VAE `z_A`. Use deterministic `mu_A` and `mu_B` for
  pair prediction. No B decoding or B KL term in this first experiment.
- Shared trainable encoder. Concatenate A and B for the encoder pass so BatchNorm sees
  the same view mixture in all matched arms.
- Physical overlap predictor: calibrated cosine on mu,
  `sigmoid(a * cosine(mu_A, mu_B) + b)`, with positive `a`.
- Directional event-coverage predictor: a small ordered pair head on mu_A and mu_B. Preserve
  argument order; a construction using only absolute difference/product is symmetric
  and cannot implement the requested directional target. Reuse existing head patterns,
  persist the head configuration, and test A/B order sensitivity with an asymmetric
  synthetic target. Do not feed coordinates, source IDs, age, or fluid masks to the head.
- Evaluate actual mu-cosine retrieval separately from the semantic pair-head score.
  A semantic head's success is not a cosine-retrieval success.
- No production search/reranking/UI changes. A later order-preserving spatial alignment
  model and a later z_geo objective ablation require a separately approved comparison.

Cosine embeddings can learn tolerance through their loss, but a single symmetric scalar
cannot represent both directional percentages or arbitrary independent semantic and
overlap rankings. A learned pair head is a baseline, not a mathematical distance metric.
If mu128 loses ordered/spatial information, measure that limitation before expanding
the initial implementation into spatial alignment or a larger bottleneck.

```text
A,B -> shared encoder -> mu_A,logvar_A,mu_B,logvar_B
mu_A,logvar_A -> sampled z_A -> decoder -> reconstructed dense A
mu_A,mu_B -> calibrated cosine -> physical overlap
ordered(mu_A,mu_B) -> semantic head -> directional fluid-adjusted event coverage

L = MAE_A + 0.1 * LPIPS_A + 0.001 * KL_A
    + 0.1 * MSE_overlap
    + 0.1 * MSE_semantic_on_valid_targets
```

Train each arm end-to-end with one combined loss, one backward pass, and one optimizer
step per batch. Do not detach or cache `mu_A`/`mu_B`, freeze the shared encoder, or
precompute latents for downstream-head training. In S3, reconstruction and KL gradients
flow through the VAE path, while overlap and semantic gradients flow through their
heads and back through `mu_A`/`mu_B` into the same encoder; update the enabled heads
and encoder together. The encoder's lower learning-rate multiplier is a slower update,
not a freeze. This joint objective is the initial design; do not insert separate
encoder/head training phases unless measured instability or validation evidence
justifies proposing a revised schedule for approval.
For each arm, compute fresh `mu_A` and `mu_B` from the current batch in the shared
autograd graph. Add a synthetic gradient-flow test for every enabled objective,
verifying nonzero gradients reach the shared encoder as well as that objective's head.

Use current repository reconstruction/LPIPS/KL arithmetic, including voxel normalization
of KL. Define valid semantic-loss reduction explicitly; log counts for each semantic
task slice and invalid reason. Do not allow one validity category silently to dominate.
The target approval must settle any task balancing needed beyond valid-target averaging.

AdamW: base LR `5e-4`, encoder multiplier `0.1`, weight decay `1e-4`, gradient clip `2`.
Fixed KL weight `0.001`; old geology classifier, SupCon, uniformity, GAN, real mixing,
deep supervision, phase and polarity objectives are disabled. Log all parameter-group
LRs, not just the first group.

Use the existing reconstruction-validation ReduceLROnPlateau rule in every arm:
patience 6, factor 0.5, minimum LR `1e-5`. Match the rule, not assumed resulting LR
trajectories. Do not schedule from incomparable train/validation objectives.

## 5. Streaming, pair sampling, and reproducibility

Training root: `/Volumes/CrucialX9/fake_data`. Volume-balanced streaming from original
source stores is mandatory. Never materialize a training-patch corpus or preload whole
source cubes. Slow streaming is acceptable; do not trade correctness or disk space for
throughput by silently caching an unbounded crop collection.

Seeds reproduce a sampling sequence; they do not restrict training to a finite stored
patch pool. Draw fresh origins and augmentation parameters across batches and epochs.
Checkpoint the sequence position. Epoch and batch counts only define execution budget.

Per batch of 12 pairs:

| Category | Pair count |
|---|---:|
| Same-source canonical overlapping windows | 6 |
| Same-volume disjoint windows | 3 |
| Different-volume windows | 3 |

Two of the six same-source pairs have zero canonical shift. For the other four,
integer offsets are drawn within +/-25% of each patch dimension: X/Y +/-8 and Z +/-16.
This intentionally does not train 50%-Z shifts; the 32-sample depth-offset case remains
a required formula test and an out-of-training-range validation slice.

For equal canonical supports L and origin difference d:

```text
overlap = 0                                  if source IDs differ
overlap = product_i max(0, 1 - abs(d_i)/L_i)  otherwise
```

This is intersection volume divided by one window's volume, not IoU. Use half-open
index intervals. For `(32,32,64)`, offsets `(0,0,0)`, `(8,8,16)`, `(32,0,0)`,
`(-8,0,0)`, `(0,0,32)` yield `1`, `0.421875`, `0`, `0.75`, `0.5`.
Physical labels refer to canonical supports before nuisance transforms.

Use disjoint whole-volume splits. Exclude `validation`, `test`, and `test_synthoseis`
from recursive training discovery, including resolved-path/symlink aliases. Assert
exclusions in every training path. Group related realizations when provenance requires
it. Do not infer independence from different folder names alone.

Reuse [source sampler helpers](../../scripts/sample_patches.py) where their contracts
apply, but do not inherit silent failed-read/zero-label fallbacks into the new loader.
Open source handles lazily per worker; bound open handles, decoded chunks and prefetch.
Validate shapes, axes, context, keys, finite values and read errors explicitly. Choose
valid origins instead of clipping offsets but leaving labels unchanged. Record source,
origins and transform seeds in streamed provenance, not in an in-memory growing list.

## 6. Augmentation, targets, and per-volume bandlimited noise

Apply independently to each view, with identical distributions across pair categories:

| Transform | Approved contract |
|---|---|
| Horizontal rotation about Z | Every view; uniform 0-360 degrees |
| X flip, Y flip, X/Y swap | Each with probability 0.5 |
| XYZ scaling | Every view; each axis independently uniform 0.9-1.1 |
| Noise | Depth-only, probability 0.5; added-noise SNR uniform 12-60 dB |
| Z flip, phase rotation, polarity reversal | Disabled |

Crop sufficient real context for geometric transforms. Derive and test context bounds
for rotations, scale and filter transients; do not invent padding signatures that reveal
the pair category. Preserve 3D semantics under transforms using the approved contract.
Physical overlap stays canonical; label support and semantic geometry must not silently
disagree with what the transformed seismic view shows.

Input/target order:

1. Read bounded context and apply nuisance transforms in dense amplitude space.
2. Add independently generated, correctly filtered noise when selected.
3. Normalize each transformed cube with the tokenizer std rule, without mean subtraction.
4. Keep trace extrema for encoder input.
5. Reconstruct the corresponding normalized, augmented dense A, including its noise.

This is not denoising to a clean target. Log retained-extrema fraction and noise statistics.
Use [the shared tokenizer preprocessor](../../src/tokenizer/core/preprocess.py); do not
introduce a competing normalization path.

For each training/validation source, use `bandpass_bandlimits` in its associated
`model_parameters_*.txt` as the actual sampled low/high cutoffs. The matched
`model_config_*.json` `bandwidth_low` and `bandwidth_high` are allowed ranges, not
the sampled cutoffs; `bandwidth_ord` supplies the order. The inspected 205 sources
use 4 ms sampling and order 4; all actual text cutoffs fell inside their JSON ranges
(low 3.03-6.00 Hz, high 22.69-44.92 Hz). Reproduce the neighboring generator's
depth-axis Butterworth/Gustafsson filtering and pole-decay context from its code,
but do not make the cloud trainer depend on importing that checkout. Persist exact
per-source settings and reject missing, contradictory or Nyquist-invalid metadata;
never substitute a global band or silently disable noise.

The same `.txt` files' `sn_db` describes noise already present in the stored source,
not a filter parameter or the extra-noise augmentation SNR. It falls outside its JSON
`signal_to_noise_ratio_db` range in 44/205 inspected sources. Record that mismatch
as provenance metadata/warning; do not reuse `sn_db` for augmentation or reject
otherwise valid filter cutoffs solely for that warning. Do not read final-test seismic
for training or selection.

Added-noise RMS relative to pre-noise signal RMS is `10 ** (-SNR_dB/20)`:
approximately 0.251 at 12 dB and 0.001 at 60 dB. Normalize filtered noise to the
requested RMS. A non-finite or zero
pre-noise RMS has undefined SNR and must produce an explicit invalid reason rather
than a silent fallback. Measure actual spectral/RMS behavior after filtering.

## 7. Source audit and semantic-contract review

The November 8 read-only audit found 180 immediate training candidates, 25 validation
stores (20 with the six candidate semantic arrays, five without), and four real-test
stores exposing only `volume`. Revalidate locally; these are observations, not promises
that every source or historical checkpoint meets the new contract.

Candidate arrays:

- `faulted_lithology`, `geologic_age_faulted`
- `gas_closures`, `oil_closures`, `brine_closures`
- `faults/faulted_channel_labels`

The refreshed audit observed seismic arrays `(300,300,1499)` and label arrays
`(300,300,1510)`. The approved convention is native same-index support, excluding
known padded label indices `1500:1510`; do not apply the historical `z+1` hypothesis.
Different deepest Z extents are not by themselves evidence of a misalignment. Do not
require equal Z shapes or pad/truncate arrays to match; use only fully supported
same-index windows, and investigate only if there is strong evidence of an interior
alignment problem.

Observed float seismic chunks were `(50,50,1499)`, about 15 MB uncompressed each.
Small streamed crops can decode multiple full-depth chunks. Bound decoded caches and
measure amplification instead of assuming crop bytes equal I/O or peak memory.

Produce a bounded local audit report and proposed target contract:

1. Source IDs/splits, seismic keys, axes, shapes, dtype/chunks, parameter-file mapping,
   generator provenance, age/fluid encodings and label alignment.
2. Fluid coverage, mask overlaps, missing labels and generator evidence for brine
   outside hydrocarbon closures; explicit invalid reasons for unverified sources.
3. Visual/numeric training and validation examples of peaks/troughs, event boundaries,
   thickness/NTG variation, fluid contacts, faults, channels and no-reservoir cases.
4. Proposed event detection and age-range validity rules, relevance and eligibility
   definitions, and the yield and limitations of the approved window-level correction.
5. Noise metadata/filter contract and actual bounded-crop memory/throughput measurements.
6. Exact validation selection/adoption thresholds and cohort safeguards.

Request one consolidated approval of those unresolved choices. If evidence cannot resolve
them, explain the blocked cases. Neither the label audit nor the GitHub author may
silently convert the two-output request into overlap-only training.

## 8. Matched arms and launch budget

Run sequentially:

| Arm | Training objectives |
|---|---|
| S0 | Evaluation only: historical checkpoint references |
| S1 | Streamed reconstruction + KL control |
| S2 | S1 + physical-overlap loss |
| S3 | S2 + approved directional semantic loss |

S3 is not the old plan's z_geo arm; z_geo and spatial alignment are deferred experiments.
S2-S1 isolates overlap; S3-S2 isolates added semantic supervision. There is no initial
semantic-only arm, so do not claim to isolate semantic-only training.
The arms are separate controlled comparisons, not phases within a run: do not train
the encoder first and then freeze it for the pair heads.

For S1-S3, use the same initialized VAE tensors, source stream, A/B transforms, encoder
view mixture, reconstruction targets, optimizer rule and total A exposure for a given
seed. Isolate model initialization, data, augmentation, VAE sampling and evaluation RNGs
so adding a head does not shift the training crop stream. Save initialization and config
identities. Unused auxiliary loss is off in control arms; do not change sampler content.
Twelve pairs means 24 encoder views and 12 decoded A views per optimizer step.

Epochs and batches per epoch are **required launch arguments with no guessed defaults**.
The launch review must state total pairs, A reconstructions, resource footprint, measured
time estimates and evaluation cost. A slower loader is acceptable; changed batch size,
crop shape or objectives require approval rather than automatic resource adaptation.
Record seed explicitly for reproduction, not as a crop-count cap. Repeat a promising
configuration with an independent seed before adoption; do not automatically launch
an unapproved replication or extend a completed run.

## 9. Evaluation and selection

### Cadence

- Each epoch: fixed small held-out pair validation, tokenizer reconstruction and
  tokenizer-aligned mu-cosine retrieval on a deterministic validation manifest.
- Every five epochs and at the selected checkpoint: exact tokenizer CLI searches on
  bounded held-out subvolumes, plus directional pair-head evaluation.
- End of the approved pilot: historical frozen-manifest ranking for continuity.
- Final selected configuration only: approved frozen new synthetic test evaluation.

Freeze small evaluation counts, subvolume bounds, stride, queries, candidates and
runtime/memory limits in the launch configuration after local smoke measurements.
Do not silently shrink evaluation on a slow run. Restore model training mode and RNG
state after evaluation. Identify every scored checkpoint by arm, attempt, step and hash.

Use [tokenizer CLI](../../scripts/tokenize.py) in explicit VAE mode with explicit mu
embedding and scoring for new arms. Historical references retain both their identity
and embedding configuration: report P0b/mu512 deployed z_geo results separately from
mu results, never label different embedding modes as a matched representation test.
Do not evaluate an untrained projection head as a learned retrieval representation.

Primary selection: directional pair-head NDCG@10, using the approved fluid-adjusted
event target as relevance; report mu-cosine ranking separately. Different-volume
pairs are zero-target negatives, not a cross-volume analog task. Use the fixed
eligibility, tie-breaking and success rules below; freeze their manifests and
cohorts before model results are examined.

The owner approved linear DCG gain, only known-label queries with at least one valid
positive, and a frozen candidate manifest with eligibility independent of the model.
Require at least 200 eligible positive queries overall and at least 20 queries from
five source volumes in each predeclared positive-query cohort. The primary pair-head
must improve NDCG@10 by at least 0.01 absolute over a repeated uniform-random ranking
baseline, with a positive lower bound of the paired source-cluster bootstrap 95%
interval. No mandatory cohort may regress by more than 0.02 absolute NDCG@10, and
the direction of gain must repeat with an independent seed. If the eligibility floor
is not met, stop and report it; do not lower the threshold after seeing results.
Compare mu-cosine separately against the matched S1 reconstruction-only mu baseline.
Never score an untrained pair head.

Report physical-overlap MAE/RMSE/bias by overlap bin and pair category; compare constant
and waveform baselines. Report event-coverage error, NTG differences, fluid mismatch,
thin layers and no-reservoir cases. Compute actual mu-cosine ranking NDCG separately
from the primary pair-head ranking NDCG; do not describe head success as cosine success.
Compare ranking baselines only with explicitly defined, valid scorers. S1, S2 and
historical checkpoints do not have a trained semantic head; do not score an untrained
head as their pair-head baseline or claim their cosine score is the same kind of scorer.

Historical n@5/n@10, supported precision/NDCG, hard-negative rate and reconstruction
remain secondary continuity evidence. n@5 is exact metadata-neighbor overlap, not
precision@5; bootstrap intervals for other metrics are not its confidence interval.
Estimate paired differences with source-volume-aware uncertainty; correlated crops
are not independent geological realizations.

Retain the +2% matched synthetic tokenizer recipe-total and real tokenizer-MAE guardrail.
Evaluate candidates and S1/P0b on identical inputs, targets and latent-sampling protocol,
with full coverage of the approved reconstruction manifest. Report which reconstruction
reference checkpoint is used; P0b reconstruction-best and retrieval epoch 30 are different.
Historical real-test data is guardrail-only, not epoch selection. Use real validation
for routine progress; do not repeatedly score the new final test.

Save separate `best_retrieval.pt`, `best_reconstruction.pt` and `last.pt` identities.
Validate eligibility before selecting retrieval; never silently use reconstruction loss,
overlap loss, classifier AUROC or the last epoch as retrieval success.

### Bounded search versus full-volume streaming search

Training always streams. Initial search evaluation may load an explicitly bounded
held-out subvolume: a `(128,128,256)` float32 region is about 16 MiB and contains
343 fully contained `(32,32,64)` windows at stride `(16,16,32)`. This is an example,
not an approved fixed evaluation manifest.

A `(300,300,1499)` float32 cube alone is about 515 MiB; the current search also allocates
score-map buffers. Whole-volume bounded-memory I/O and disk-backed score accumulation
are deferred, not prerequisites for streamed training. The pilot must not load a whole
cube under the guise of bounded subvolume evaluation. Preserve current padding/Hann
semantics within the bounded region and report best-window ranking/coordinate error
separately from the Hann-blended voxel-map peak.

### Final holdouts

Thirty newly generated source volumes exist under
`/Volumes/CrucialX9/fake_data/test_synthoseis`. Freeze an audited manifest only after
checking category coverage and independent provenance. The earlier recommendation
of about 20 queries per volume (600 total) is a starting point for the gate-1 contract,
not a power calculation. Do not calibrate labels, select weights or train on this set.
Metadata-only inventory is permitted; final-test seismic predictions are not routine
epoch diagnostics.

The four real-test volumes have no established stored layer/fluid labels. They support
historical reconstruction and physical/localization checks, not quantitative fluid-aware
ground truth without independent annotations. Do not claim the new synthetic holdout or
old real guardrail is an untouched real-domain semantic test.

## 10. Persistent logging, completed-batch CSV, and resume

The current [trainer](../../scripts/train.py) writes epoch metrics and TensorBoard
events; it does not already provide the required durable completed-batch journal.
Its full-state resume also has paired-reconstruction restrictions. Extend this opt-in
path explicitly; do not advertise existing flags as sufficient for streamed pairs.

Required output layout:

```text
logs/<run-id>/
  training.log                 combined runner/training stdout and stderr
  tensorboard.log              TensorBoard stdout and stderr
  tensorboard.pid              server created by this run
  run_manifest.json            resolved config and artifact locations

/Volumes/CrucialX9/3dvae-poc_checkpoints/streamed_overlap/<run-id>/
  source_manifest.json
  target_contract.json         approved version/hash
  resolved_config.json
  <arm>/
    batch_metrics.csv
    training_metrics.csv       existing epoch semantics preserved
    tensorboard/
    checkpoints/
    evaluation/
```

Provide a per-arm monotonic `batch_metrics.csv` journal. After each successful optimizer
step, write one row, flush **and fsync every row**. Minimum fields:

- Run/arm/seed, attempt ID, UTC timestamp, epoch, batch index, logical global step,
  unique event ID, replay-of-event identity when applicable.
- Completed pairs, reconstructed A examples and cumulative committed exposure.
- Total loss, reconstruction MAE, LPIPS, normalized/unweighted KL, weighted components,
  overlap and semantic losses, valid semantic counts and per-task slice counts.
- All group LRs, pre-clip gradient norm, batch/data-read elapsed time and measured memory.
- Reference to deterministic source/transform provenance; do not put growing raw arrays
  or entire crops in a CSV cell. Use a streamed sidecar if needed.

Log uncomputed metrics as empty/not-applicable with explicit schema semantics, not
success-shaped zero defaults. Non-finite loss/gradients or failed optimizer/log writes
must stop with a traceback/nonzero status; do not call failed/skipped steps completed.

Epoch CSV and TensorBoard remain; add batch-step component tags and epoch retrieval,
reconstruction, pair-error, coverage, noise and throughput tags. CSV is the durable
journal; TensorBoard is visualization, not the only record.

At every 100 completed optimizer steps and at each epoch boundary (if not already
logged), run no-gradient inference with the latest weights on fixed deterministic
training and validation monitor batches. Emit the reconstruction, overlap and
semantic loss components to stdout and TensorBoard. Use inference mode, restore the
prior model mode and RNG states, and record monitor time as overhead. These fixed
monitor batches are for progress visibility, not checkpoint selection; formal
validation and retrieval evaluation retain the cadence above.

Atomically save full epoch-boundary training state: model/head and optimizer/scheduler,
all RNGs and deterministic loader stream positions, sampling/config/manifest hashes,
selection/early-stopping state and committed CSV event watermark. Verify before appending
on resume; incompatible settings, lost state or changed sources require a new explicit
run, never an optimizer reset disguised as resume.

Rows after the last checkpoint survive a crash but their model updates may not. Preserve
those attempt rows, mark them uncommitted relative to recovery via an append-only recovery
record, and record replay identity on new rows. Do not truncate their history or count
replayed exposure twice in the recovered model trajectory. Test interrupted-write/torn-tail
handling and step/checkpoint reconciliation without silent record loss. Epoch-boundary
resume can repeat computation; exact mid-epoch resume is not required.

## 11. Shell runner and TensorBoard lifecycle

Implement `scripts/streamed_overlap_experiment.sh`, with documented modes:

```text
--help
--mode audit
--mode smoke
--mode train
--mode resume
--mode evaluate
--mode tensorboard
--mode tensorboard-status
--mode tensorboard-stop
--print-commands
```

Audit, no-optimizer smoke, preparation, evaluation-only and TensorBoard lifecycle
modes are wired. Train/resume commands can be previewed, but execution remains
blocked pending integrated real-data smoke, resource review and separate launch
approval. Acknowledgement flags are not evidence of these prerequisites.
Evaluation requires a completed suite and explicit resource review. See the
[handoff's runnable commands](2026-11-11__streamed_overlap_copilot_implementation.md).
Reuse repository helpers rather than introduce a second trainer.

Runner requirements:

- `set -euo pipefail`, unbuffered Python, combined stdout/stderr capture from preflight
  onward to `logs/<run-id>/training.log`, and correct nonzero status through `tee`.
  Log-file write failures must not produce successful completion. Do not truncate a
  prior log; resume creates a new identified attempt and preserves history.
- Create-exclusive fresh output paths. Record source/target/config hashes, git revision,
  runtime/library versions, device, exact commands and attempt IDs.
- Audit mode may inspect metadata and bounded training/validation labels; it never
  trains. Print-only mode must have no output-directory, service or training side effects.
- Smoke uses tiny synthetic fixtures in cloud and a few real streamed training/validation
  pairs locally. The no-optimizer smoke checks each downstream loss separately for a
  finite nonzero encoder gradient; this is a gradient-path check, not a result about
  improved representations. Smoke must not bypass a production gate.
- Production train/resume require approved contract and launch config. Check mounted
  volume, readable sources/checkpoints, bounded caches, disk headroom, real batch-12-pair
  forward/backward memory and source exclusions before starting.
- Run arms sequentially. Never silently lower batch size, preload whole cubes, fall back
  to CPU, drop semantic labels, disable noise or skip required evaluation.
- Start TensorBoard for the suite root on `127.0.0.1:6006`, capture its output, record
  its specific PID and verify HTTP readiness. An occupied port is an explicit error;
  do not kill or appropriate another process. A verified run-owned server may be reused
  for resume. Service failure must be surfaced, not silently ignored.
- Leave this run's TensorBoard running after training, as requested. Separate its
  process lifetime from trainer cleanup; print URL, PID, log path and a targeted stop
  command. On local Copilot launch use the environment's supported detached-service
  mechanism and verify liveness; do not claim persistence from an ephemeral attached
  command. If the environment cannot keep it running, stop and explain the limitation.
- Handle interruption/exit visibly; preserve journals and last complete checkpoint,
  identify any incomplete evaluation, and never print "complete" for a partial run.

After code and local validation are ready, the launch approval must provide actual
values. Then the user can run, from repository root:

```sh
# These scripts are deliverables of the implementation, not present at planning time.
bash scripts/streamed_overlap_experiment.sh --mode audit \
  --config <local-audit-config.json> --run-id <new-audit-id>

# After target-contract approval:
bash scripts/streamed_overlap_experiment.sh --mode smoke \
  --config <approved-local-config.json> --run-id <new-smoke-id>

# Proposed commands; execution remains blocked until the launch gate is removed:
bash scripts/streamed_overlap_experiment.sh --mode train \
  --config <prepared-suite-inputs.json> --run-dir <new-suite-directory> \
  --run-id <pinned-training-id> --config-sha256 <prepared-config-sha256> \
  --epochs <approved-epochs> --batches-per-epoch <approved-batches> \
  --mps-memory-fraction <limit> --min-suite-free-bytes <bytes> \
  --min-log-free-bytes <bytes> --acknowledge-training-approval \
  --acknowledge-training-resources

# Resume must use recorded state/commit watermark, not weights only:
bash scripts/streamed_overlap_experiment.sh --mode resume \
  --config <prepared-suite-inputs.json> --run-dir <existing-suite-directory> \
  --run-id <new-resume-attempt-id> --tensorboard-run-id <pinned-training-id> \
  --config-sha256 <prepared-config-sha256> --epochs <approved-epochs> \
  --batches-per-epoch <approved-batches> --mps-memory-fraction <limit> \
  --min-suite-free-bytes <bytes> --min-log-free-bytes <bytes> \
  --acknowledge-training-approval --acknowledge-training-resources
```

Replace placeholders only with approved values. Print exact, copyable commands in the
local readiness report; production training must not start with unresolved placeholders.
Document standalone TensorBoard restart and PID-specific shutdown for later inspection.

The equivalent standalone visualization command, only when no server owns the port, is:

```sh
uv run tensorboard --logdir <existing-suite-directory> --host 127.0.0.1 --port 6006
```

For the normal runner path, print `http://127.0.0.1:6006` and the actual owned PID.
Shutdown instructions must identify that recorded process and verify its ownership
before `kill <recorded-tensorboard-pid>`; never use name-based process killing.

## 12. Implementation sequence and acceptance

1. Preserve existing work and establish an implementation branch from the saved mu512
   work, not old `main`; see the GitHub handoff. Do not overwrite historical artifacts.
2. Inspect prior art in model/trainer, sampler, preprocessing, adapter/search, diagnostics,
   checkpoint/state helpers and tests before adding code.
3. Implement manifests, bounded streaming, deterministic paired transforms/provenance
   and read-only audit tooling. Keep unresolved targets explicitly blocked.
4. Obtain gate-1 local approval; implement/version the exact target generator and
   approved evaluation rules, with synthetic fixtures covering the reviewed cases.
5. Integrate the shared-encoder pair heads/objectives and durable journals/full-state
   epoch recovery. Preserve old checkpoint loading and old default training behavior.
6. Add deterministic epoch evaluation and bounded exact CLI searches using shared
   preprocessing, no whole-store benchmark materialization.
7. Implement shell runner, all modes, TensorBoard lifecycle and clear readiness report.
8. Run targeted tests, full existing suite where dependencies permit, and actual local
   streamed-crop/MPS smoke. Report cloud and local checks separately.
9. Obtain launch approval, then run only the approved sequential budget and report
   results, uncertainty, cohort regressions and checkpoint selection. No automatic adoption.

Required focused tests:

- Canonical overlap numeric examples, symmetry, touching/disjoint supports and boundaries.
- Source exclusion through symlinks, axis/offset alignment, schema/parameter errors,
  unknown masks and bounded reads; no full-volume array conversion.
- Directional event counts on multiple traces, overlap boundaries, zero event count,
  disjoint and cross-volume zero targets, age-range selection, distinct fluid fractions,
  sampled mask overlap, proxy limitations and unavailable-source reasons.
- Every approved nuisance transform, context sufficiency, shared label/view geometry,
  noise band/sample spacing/Nyquist, RMS and negligible-energy handling.
- Fresh independent epoch sampling, worker uniqueness, bounded caches, arm-matched
  provenance and resume reproducibility independent of head initialization/evaluation.
- Finite pair-head gradients into encoder, correct latent/reconstruction shapes,
  directional head order sensitivity and encoder/adapter parity.
- Per-step fsync journaling, failures, torn-tail detection, epoch-state crash recovery,
  replay identities, committed exposure and no silent optimizer/scheduler restart.
- Validation-only selection, separate cosine/head rankings, exact metric definitions,
  cohort failures, missing evaluation and distinct best-checkpoint identities.
- Runner no-op print mode, fresh-output refusal, nonzero child/log-write failures,
  TensorBoard occupied-port/readiness/PID ownership, and survival after training exits.
- Legacy mu128/mu512 checkpoint and tokenizer/default-path compatibility.

Use existing tools and tests; no new dependencies unless necessary and justified.
Cloud fixtures must be tiny fabricated arrays/metadata, not uploads of source data.

## 13. Explicitly deferred work

No P0b extension, additional mu512/2048 training, EMA encoder, full-volume streaming
search, learned scorer production reranking, spatial alignment model, z_geo objective,
multi-angle inputs or optional voxel decoders in the initial scope. These may be proposed
later with a specific failure diagnosis and matched controls, not silently bundled in.

The unavoidable local target and launch approvals are not failures of autonomy. They
are the only planned scientific decision points; the rest of the implementation should
proceed without repeated chat confirmations.
