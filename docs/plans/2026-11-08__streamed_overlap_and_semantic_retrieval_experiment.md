# Streamed overlap supervision and semantic seismic retrieval experiment

Status: investigation and implementation plan; no training implementation or run authorized by this document.

## 1. Goal and interpretation

Train a full 3D VAE for reconstruction while a second view passes through the encoder.
Compare the deterministic latent means (`mu`) to predict the fraction of spatial
overlap between the two source windows. Measure whether this improves the tokenizer's
actual purpose: retrieving geologically similar locations, not merely matching the
same waveform or finding the original location.

Confirmed requirements:

- Save this plan as Markdown.
- Preserve current work on the current branch before creating an experiment branch.
- Perform subsequent implementation and experiments on that new branch.
- Stream crops directly from synthoseis source datasets; do not require a newly
  materialized training-patch corpus or load whole source cubes into memory.
- Source root: `/Volumes/CrucialX9/fake_data`. Its `validation`, `test`, and
  `test_synthoseis` subfolders are holdouts, not training sources. Exclude all three from recursive
  training discovery, including any optional real-data reconstruction path.
- Maintain full-VAE reconstruction training alongside the paired-view objective.
- Approximately 50% of pairs should be two views of the same source window, possibly
  translated; the other 50% should be different windows or volumes.
- Same-source views should tolerate horizontal rotations, horizontal flips, X/Y
  swapping, modest XYZ resizing, bandlimited noise, and translation.
- Predict overlap in [0, 1], including the stated 32-sample depth-offset example.
- Evaluate tokenizer behavior during training, not just the auxiliary loss.
- Predict a separate layering-match value that includes reservoir fluid fill. Gas sand
  must not count as matching brine sand merely because they belong to the same layer.

The clarification form from the initial discussion was interrupted. The recommendations
below are provisional, not accepted changes to the requested experiment.

## 2. Assessment and corrections to the initial critique

### What is promising

Augmented views can teach acquisition/orientation robustness without depending on the
quality of synthetic geology labels. Streaming gives access to real neighboring source
samples, making translation augmentation physically meaningful. Evaluating the deployed
retrieval path can expose improvements or regressions hidden by reconstruction loss.

### Principal limitation: overlap is not semantic relevance

Two spatially disjoint windows can show the same fault, channel, onlap, or reflector
geometry. Conversely, a large overlap can miss a small but important geological feature.
Assigning all disjoint pairs target 0 teaches instance discrimination, not necessarily
geological equivalence. Cross-volume negatives can also be false semantic negatives.

Physical overlap cannot always be inferred from content: identical repeating strata can
occur in different places. An image-only predictor cannot resolve all such cases, even
with perfect optimization. Coordinate provenance provides the training target, but must
not be supplied as a feature to the predictor.

Overlap learning might improve retrieval through useful structure recognition. It might
instead learn waveform identity or volume-specific artifacts. Either outcome needs
measurement; neither improvement nor failure should be assumed in advance.

### Representation and scoring

- Use a shared, trainable encoder as the simplest starting point.
- An independently initialized encoder copy does not guarantee a common embedding
  coordinate system. However, an independent copy can still train the primary encoder
  if the joint loss backpropagates into it; the initial statement that it could not was
  too strong. Weight sharing avoids that extra problem.
- EMA/teacher encoders are a different experiment, not a necessary ingredient.
- Reconstruction needs information about orientation and precise spatial arrangement;
  forcing `mu` to discard these can conflict with reconstruction.
- The existing `z_geo` projection can provide a retrieval representation with different
  priorities. It still backpropagates into the shared encoder: it reduces direct
  bottleneck conflict but does not guarantee reconstruction preservation.
- Keep direct-`mu` overlap training as the literal proposed experiment. Test `z_geo`
  overlap training as a separate ablation, not a silent replacement.
- A pair MLP can predict overlap without making cosine neighbors better. A calibrated
  cosine predictor is more aligned with the current search metric, but restrictive:
  overlap targets need not be representable by a single global cosine geometry.

### Repository evidence

- [Training](../../scripts/train.py) already has reconstruction, KL, optional geology
  objectives, checkpointing, and epoch diagnostics.
- [Model](../../src/model.py) exposes vector `mu`, a geology projection head, and
  optional classifier heads.
- [Tokenizer CLI](../../scripts/tokenize.py) supports `mu` and `z_geo`, with cosine
  or dot-product search. `z_geo` is the CLI's default VAE embedding; UI latent mode
  defaults to `pooled`, so evaluations must explicitly select VAE mode.
- [Tokenizer preprocessing](../../src/tokenizer/core/preprocess.py) divides by each
  cube's standard deviation, then retains trace extrema.
- [Search engine](../../src/tokenizer/core/search_engine.py) scores sliding windows
  and blends their scores into a voxel map with a Hann taper. The voxel-map peak is
  not equivalent to the highest-scoring individual window.
- [Geology benchmark](../../scripts/evaluate_geology_benchmark.py) provides
  tokenizer-aligned patch ranking, not a full `search-volume` localization test.
  It currently materializes the patch array before selecting the manifest subset.
- Correction: the current [patch sampler](../../scripts/sample_patches.py) DOES write
  `source_volume_index` and `origin_x/y/z`. The initial claim that stored patches have
  no origins was incorrect. Audit older stores before assuming they have this schema.
- [Training documentation](../training/README.md) records modest classifier/retrieval
  gains and failed adoption gates. [October screen](../sessions/2026-10-07-pretrain-v2-top5-architecture-loss-screen.md)
  reports reconstruction regressions, not retrieval results for the skipped geology
  stages. These findings do not establish the outcome of this new objective.

## 3. Safe Git workflow: preserve first, branch second

At planning time the current branch is `geology-classifier-2026-09-25` and it contains
modified and untracked work. Do not revert it, overwrite it, or stash it as a substitute
for the requested persistent save. This plan does not itself execute Git commits.

Before implementation:

1. Review all modified and untracked files, including this plan. Confirm intended
   source/documentation files and check for secrets, large data, checkpoints, and logs.
2. Save all intended current work in a normal commit on the current branch.
   Stage reviewed paths explicitly; do not indiscriminately stage datasets or secrets.
   If required work is ignored or large, arrange appropriate persistent storage and
   record its location before proceeding.
3. Verify the save commit and a clean worktree. Resolve any remaining work deliberately;
   do not silently discard it. Pushing the saved branch is an optional remote backup,
   subject to the user's normal workflow.
4. Create a fresh branch from that saved commit.
5. Perform all experiment implementation, configuration, tests, and result-documentation
   changes on the new branch. Do not modify historical benchmark manifests.

Illustrative commands (run only after review; replace the path placeholders):

```bash
git branch --show-current
git status --short
git diff --stat
git diff

git add <reviewed-paths>
git diff --cached --stat
git diff --cached
git commit -m "Save current work before streamed overlap experiment" \
  -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"

git status --short
git log -1 --oneline
# Proceed only after confirming all intended work is saved and the worktree is clean.
git switch -c experiment/streamed-overlap-retrieval
git branch --show-current
```

If the new branch already exists, stop and choose a new name; do not reset or reuse it
without checking its contents. Any later commits should include the same co-author trailer.

## 4. Pair definition and overlap contract

Let A and B be equal-sized, axis-aligned source windows, with shape L = (Lx, Ly, Lz),
integer source origins a and b, and offsets d = b - a.

```text
overlap(A, B) =
  0                                             if source volume IDs differ
  product_i max(0, 1 - abs(d_i) / L_i)           otherwise
```

Use half-open source index intervals. The product is intersection volume divided by
one window's volume, NOT intersection-over-union. For unequal source supports this
formula is insufficient; do not introduce unequal-support labels without a new contract.

Required examples for shape (32, 32, 64):

| Offset | Expected overlap |
| --- | --- |
| (0, 0, 0) | 1.0 |
| (8, 8, 16) | 0.421875 |
| (32, 0, 0) | 0.0 |
| (-8, 0, 0) | 0.75 |
| Any offset, different volume IDs | 0.0 |

Recommended first definition: calculate overlap from the canonical equal-sized source
windows BEFORE nuisance transforms. Rotations, flips, resampling, and noise do not
alter this label. Thus the target is canonical overlap, not the geometric overlap of
the final transformed voxel supports. This distinction needs user confirmation.

For each batch, use half same-source pairs and half negative pairs; use an even batch
size for an exact ratio. Same-source pairs include an explicit zero-shift subset.
At a maximum shift of 25% along each axis, their overlap is at least 0.421875.
Expanding the training range requires an explicit decision.

Negative policy must distinguish:

- disjoint windows from the same volume;
- windows from different synthoseis volumes;
- semantically similar negatives, where geology labels or expert review identify them.

Start with the requested negative policy as an ablation. For a semantic-oriented
variant, mask known geologically similar negatives or use geology-positive supervision.
Masking disjoint same-volume pairs alone does not eliminate cross-volume false negatives.
Do not expand to all in-batch pairs without applying provenance and false-negative rules.

## 5. Streamed input design

Implement an opt-in source-volume paired dataset integrated with the current trainer.
Reuse the sampler's source-discovery, key resolution, coordinate, and label helpers
where suitable rather than duplicating those contracts.

### Source manifest and splits

- List source dataset IDs, paths, seismic keys, stored axes, XYZ shape, dtype, chunk
  shape, and train/validation/test assignment.
- Exclude `validation`, `test`, and `test_synthoseis` from training discovery. Assert
  exclusions for paired sampling and any real-data reconstruction loader.
- Split by entire source volume before sampling. Hold out related generator realizations
  together where provenance identifies them. No translated crop may cross a split.
- Preserve historical frozen benchmarks. Create a separate immutable streamed-evaluation
  manifest with query/candidate origins and seeds.
- Validate label/seismic alignment when geology labels are used; preserve the verified
  synthoseis label-depth convention rather than applying it to unrelated sources.
- Sampling policy is a controlled variable: prefer a documented volume-balanced start,
  and report actual volume/feature prevalence rather than accidentally sampling large
  volumes or chunk-friendly locations more often.

### Bounded I/O

- Read only Zarr slices needed for each pair and its augmentation context.
- Open source handles lazily within DataLoader workers; do not inherit unsafe open
  handles or CUDA state across workers.
- Bound the worker-local source-handle/chunk cache and prefetch queue.
- Use a fixed number of samples/steps per epoch, with seeds derived from base seed,
  epoch, and global sample index. Worker assignment must not duplicate sample streams.
- Record actual origins and transforms. On resume, restore the sampling position and
  RNG state sufficiently to reproduce the documented epoch/step continuation.
- Validate source keys, axis order, finite values, available context, and read errors
  explicitly. Do not replace failed crops with zeros or quietly skip failed examples.
- Estimate memory from workers, prefetch, pair context sizes, batch tensors, decoder
  activations, and the optimizer. Benchmark throughput and peak memory before long runs.
- Boundary handling must not silently clip offsets while retaining the old overlap label.
  Sample valid origins or explicitly record resampling and its distributional effects.

The CLI currently loads whole volumes and the search engine allocates full-volume score
buffers. Training streaming alone will not make full-volume evaluation memory-bounded.
For the pilot, run exact CLI searches on explicitly bounded held-out subvolumes.
If original full cubes must also be streamed during evaluation, add a chunked search
input and disk-backed accumulation path preserving padding, stride, and Hann blending.
Test it against the existing implementation; treat this as a separate scoped work item.

## 6. Augmentation and preprocessing contract

Read sufficient real source context before rotating, resizing, or translating. Derive
context size from rotation angle, scale, and translation bounds; do not hardcode an
unverified larger-crop size. Avoid padding/corner artifacts that identify positive pairs.

Apply independent nuisance transforms to both views, with the same transform distribution
for positive and negative pairs. Otherwise augmentation artifacts become a label shortcut.
Do not flip Z unless separately approved.

| Transform | Proposed range and cautions |
| --- | --- |
| Horizontal rotation | Continuous azimuth about Z |
| X/Y flips and swap | Reflections and horizontal axis exchange |
| XYZ resize | 0.9-1.1, resampled to the fixed model input shape; Z changes wavelet/bed thickness and dip |
| Bandlimited noise | -30 to -12 dB; additive noise provides slight post-normalization variation; define filter axes/band and reference RMS before implementation |
| Translation | Integer source-origin shift up to 25% of each dimension in the literal proposal |
| Phase/polarity | Existing phase augmentation is available; inclusion is optional, not yet approved |

For noise amplitude in dB relative to signal RMS, use
`noise_rms / signal_rms = 10 ** (dB / 20)`, approximately 0.063-0.251 here.
Normalize the filtered noise to its intended RMS before adding it. Define what happens
for negligible-energy crops explicitly. Frequency units, passband, and anisotropic
sampling must be settled; isotropic XYZ filtering is not automatically physically valid.

Recommended branch input contract:

1. Crop and apply paired nuisance transforms in dense amplitude space.
2. Normalize each resulting cube using the tokenizer's std rule.
3. Retain trace extrema for the encoder input.
4. Reconstruct the corresponding transformed, normalized dense A view.

This is reconstruction of the augmented view, not denoising back to the original.
Noise-free targets would define a distinct denoising experiment. Record and approve that
choice before implementation. Audit the current trainer's scale/target behavior and give
the streamed reconstruction-only control exactly the same contract as the overlap runs.
Noise before extrema retention may create false extrema; measure the retained-voxel
fraction and robustness after this exact pipeline.

## 7. Model, loss, and training integration

Literal experiment:

```text
A -> shared encoder -> mu_A, logvar_A -> sample z_A -> decoder -> reconstruction_A
B -> shared encoder -> mu_B, logvar_B
mu_A, mu_B -> symmetric overlap predictor -> estimated canonical overlap
```

Use deterministic means for pair prediction and sampled latent values for the normal
VAE reconstruction path. Both encoder passes receive overlap gradients. B need not be
decoded in the first pilot. If using BatchNorm, concatenate A/B encoder inputs when
practical so the branches do not update statistics under different batch distributions.

```text
loss = reconstruction_loss
     + beta * KL_A
     + lambda_overlap * overlap_regression_loss
```

Retain the existing reconstruction recipe and KL schedule for a matched comparison.
Start with all unrelated auxiliary objectives disabled; add them only as separate
experiments. If both views are reconstructed later, define and average both KL and
reconstruction terms consistently.

Provisional overlap regression: MSE with a [0, 1] output, plus overlap-bin MAE and bias
for reporting. This is continuous regression, not binary classification.

Scoring variants:

- Calibrated cosine: `sigmoid(a * cosine(mu_A, mu_B) + b)`, with `a > 0`.
  Preserves ranking alignment with tokenizer cosine; finite sigmoid outputs approach,
  rather than exactly equal, the 0 and 1 target endpoints.
- Symmetric pair MLP: features such as absolute difference and elementwise product.
  If selected, report raw-cosine retrieval separately. Deploying the MLP as search
  scoring requires explicit adapter, CLI, search, UI, and checkpoint support.
- Projection variant: apply the chosen predictor to `z_geo` rather than `mu`.
  Record projection configuration and explicitly select this embedding in evaluation.

Log individual losses, overlap-bin counts, positive/negative proportions, embedding
norms/variance, pair-score distributions, augmentation parameters, and sample provenance.
Use a small predeclared overlap-weight sweep and watch reconstruction gradients/loss,
rather than copying a weight from a differently scaled auxiliary objective.

Save model/head config, optimizer/scheduler/RNG state, source-manifest identity, sampling
position, preprocessing contract, evaluation manifest identity, and experiment seed.
New behavior must be opt-in; old checkpoints and training paths must remain compatible.

## 8. Measure tokenizer performance during training

Evaluation runs in eval/inference mode on held-out volumes and deterministic manifests.
Restore training mode afterwards. Use the epoch checkpoint and record its identity.
Do not allow validation crops to enter training or repeatedly tune against the final test.

### Three distinct questions

1. **Did it learn the auxiliary task?**
   Held-out overlap MAE/RMSE, bias and prediction histograms by overlap bin, same-volume
   disjoint versus cross-volume pairs, and each augmentation. Compare with a constant
   predictor and simple waveform/pixel-similarity baseline.
2. **Is local matching robust?**
   Use `scripts/tokenize.py search-volume` in explicit VAE mode with fixed embedding,
   patch shape (32, 32, 64), stride, preprocessing, and checkpoint. Search a held-out
   subvolume for clean/augmented query views. Report best-window rank and coordinate
   error as well as voxel-map behavior; the Hann-blended map is not an overlap estimate.
   Arbitrarily augmented query injection may require a CLI query-input option reusing
   the existing token/search helpers; do not create a divergent preprocessing path.
3. **Did semantic retrieval improve?**
   Measure n@5/n@10 and supported ranking metrics on the unchanged historical manifest
   for continuity, plus a new held-out volume-aware benchmark that excludes overlapping
   query neighbors. Include separate cross-volume retrieval and per-geology cohorts.
   Metadata similarity is a proxy: supplement it with a small fixed expert-reviewed
   set of semantic matches where feasible.

Localization of the original patch, low overlap error, and high cosine for augmented
copies are NOT sufficient evidence for semantic retrieval.

### Cadence and bounded cost

- Every epoch: fixed small pair validation, reconstruction checks, and tokenizer-aligned
  patch retrieval using existing core preprocessing/adapter helpers.
- At a predeclared epoch interval and at the selected checkpoint: exact tokenizer CLI
  searches on bounded held-out subvolumes and the larger frozen retrieval evaluation.
- User must choose whether exact CLI searches are required every epoch.
- Stream selected benchmark rows rather than materializing the entire patch store.
  Use training-derived metadata calibration; avoid fitting it on validation/test data.
- Write explicit JSON/CSV/TensorBoard results with checkpoint, query provenance, split,
  embedding/scoring mode, runtime, memory, metrics, and failure status.
  The existing CLI benchmark JSON records throughput and score statistics, not semantic
  ground truth; add localization/ranking evaluation rather than treating it as accuracy.

For learned overlap-head inference, test adapter/head predictions against the training
model. For cosine inference, test encoder embeddings and rankings for parity instead.

## 9. Experiment ladder and decision gates

First run a tiny streaming smoke test, not a production job. Then compare:

| Run | Purpose |
| --- | --- |
| S0 | Existing adopted/checkpoint reference, evaluated under the fixed new protocol |
| S1 | Streamed reconstruction-only control with the same input/target/sampling contract |
| S2 | Literal proposal: shared encoder, overlap on `mu`, requested augmentation/negative policy |
| S3 | Same as S2, but overlap on `z_geo` |
| S4 | Semantic-oriented augmentation consistency with false-negative handling, after its loss is defined |

S1 versus S2 isolates the paired objective; S0 versus S1 reveals effects of streaming
and changed preprocessing. Historical `r1ctrl` is useful context but is not a substitute
for a matched streaming control. Do not assume inherited checkpoints are leak-free:
audit their training-source provenance against the new holdout.

Keep optimizer, reconstruction recipe, source split, sample exposure, and training
budget fixed. Report added compute from the B pass. Select on validation retrieval,
subject to reconstruction tolerance, not on overlap loss alone.

Predeclare before long runs:

- Semantic retrieval improvement required for adoption, including per-volume/cohort
  regressions and uncertainty. Repeat promising runs with at least two independent seeds.
- Reconstruction tolerance: initially retain the repository's +2% regression limit on
  matched synthetic AND real tokenizer-preprocessed evaluation, unless explicitly changed.
  Do not infer reconstruction quality from differently scaled losses.
- Acceptable overlap MAE and augmentation-stability thresholds; no thresholds approved yet.
- An untouched final held-out test evaluated only after selection.

If overlap/localization improves but disjoint semantic retrieval worsens, reject it for
the tokenizer purpose or retain it only as a separately labeled localization feature.
Classification metrics are secondary unless a classifier is deliberately retained.

## 10. Implementation work and acceptance tests

Suggested implementation order, all on the new branch:

1. Save existing work and establish the experiment branch.
2. Resolve the open contracts below; freeze source splits and evaluation manifests.
3. Add streamed pairs and provenance using shared source helpers.
4. Validate context crops, transforms, overlap labels, and preprocessing independently.
5. Add the opt-in overlap head/loss, trainer integration, metrics, and checkpoint support.
6. Add deterministic tokenizer-aligned epoch evaluation and bounded exact CLI searches.
7. Run the smoke and matched-control pilot; only then schedule the experiment ladder.
8. Document reproducible commands, resource budget, results, and adoption decision.

Tests must cover:

- All overlap examples above, signed offsets, touching/disjoint windows, and symmetry.
- Correct labels after source-boundary decisions and canonical nuisance transforms.
- Known-pattern XYZ reads, source axis permutations, keys, provenance, and split leakage.
- Streaming reads bounded by required context; no full-array conversion in training.
- Worker seed uniqueness, deterministic manifests, cache limits, and resumed sampling.
- Horizontal transforms without Z reversal, noise RMS/band, and no label-dependent artifacts.
- Dense target versus extrema input normalization and invalid/empty crops.
- Finite forward/backward losses; overlap gradients reach the primary encoder.
- Pair symmetry, prediction bounds, embedding collapse diagnostics, and head/adapter parity.
- Old checkpoint loading, resumed new checkpoint equivalence, and old CLI/default behavior.
- Existing versus chunked search parity if full-volume streaming evaluation is implemented.
- CLI output shape, fixed query provenance, independent-window scores, and Hann-map metrics.

Use existing focused tests/build checks. Documentation-only planning requires no training
or test run; implementation validation must use real streamed source crops as well as
synthetic unit fixtures.

## 11. Decisions still requiring clarification

1. Does "different dataset" mean a different synthoseis volume, a disjoint window in
   the same volume, or both? What fraction should each negative category receive?
2. Should invariance ignore absolute strike/azimuth, dip sign, thickness, phase, and
   polarity? These are not universally interchangeable geological attributes.
3. Keep the requested `mu` objective as the primary experiment, or make `z_geo` primary?
   Should overlap be calibrated cosine or a learned symmetric pair scorer?
4. Keep the 25% translation limit, or broaden it to include the 50%-Z example in training?
5. Accept canonical pre-transform overlap, or require true transformed-support overlap?
6. Reconstruct the augmented dense A view, or denoise to a clean target?
7. What seismic key, source root(s), axis order, chunk shapes, and compute/memory budget
   should the streaming pilot use? Define the noise frequency band and filter axes.
8. Does "using tokenize.py during training" require exact CLI searches every epoch,
   or lightweight per-epoch core evaluation plus periodic CLI searches?
9. Must full original search volumes be streamed too, or are bounded CLI subvolume
   searches sufficient for the first experiment?

Until these are answered, implement neither an assumed replacement objective nor a
long-running training job. The plan is saved; implementation is a separate next step.

## 12. Layering and fluid-aware semantic matching: clarified direction

The user clarified that the second predicted value should describe how much of A's
layering is duplicated in B, including fluid fill. Gas sand versus the same horizon
in a brine-filled aquifer is not a match. The same is true for gas vs oil, and oil vs brine. Thus rotation/shape robustness must not erase
fluid-sensitive seismic information.

Further confirmed requirements:

- Both identifying the same actual layers and finding analogous layering in unrelated
  locations are useful. Evaluate these separately; neither equals physical overlap.
- Reservoir fluid categories include gas, oil, and brine.
- Reservoir bodies may be channels or sand layers.
- The vertical ordering/pattern of layers must contribute to semantic matching.

Accordingly, the semantic target must preserve ordered vertical lithology/fluid patterns
AND lateral 3D geometry. An unordered material/fluid histogram or one trace at the
window center is insufficient. Separate same-horizon evaluation requires reliable
within-source stratigraphic provenance; local layer IDs alone cannot establish physical
identity across independently generated volumes. A separate same-horizon prediction
output is a possible extension, not yet an approved addition to the two requested outputs.

Recommended approach for discussion, not an approved implementation:

- Keep physical overlap and fluid-aware geological matching as distinct supervised
  targets. Zero physical overlap does not imply zero geological similarity.
- Audit the actual synthoseis source arrays and their units/encodings for layer/horizon
  identity, lithology, reservoir masks, and gas/oil/brine occupancy or saturation.
  The current sampler exposes sand/shale and geologic-age information, but the examined
  training-label path does not establish gas/brine targets. Above-seabed water is NOT
  a brine-reservoir label.
- Define ground-truth layer correspondences using stratigraphic order, material, fluid,
  and spatial structure. Layer IDs within one realization are not automatically
  comparable across independently generated volumes.
- Use these correspondences to supervise a fluid-aware retrieval embedding. Include
  hard negatives with similar layering but incompatible fluid labels, including the
  same horizon on opposite sides of a fluid contact.
- If patch embeddings cannot preserve the required distinctions, test auxiliary
  per-voxel lithology/fluid and boundary prediction, or spatial layer features, as
  separately justified additions. Do not assume whole-window class fractions encode
  the order or shape of layers.
- An ordered layer representation may help; a single vertical trace sequence is not
  sufficient for faulted, dipping, or laterally changing 3D geology.
- Decide whether the match value is a directional fraction of A, a symmetric score,
  an equal-per-layer fraction, or a volume-weighted fraction. Missing/incompatible
  reservoir intervals must not be averaged away by abundant matching background shale.
- Continue deploying a trained single-window embedding for efficient tokenizer search
  where possible. A richer pair scorer could rerank a shortlist, but that requires
  explicit evaluation and integration; it is not automatically equivalent to cosine.

Limitations and tests:

- Labels describe the target; seismic alone may not uniquely determine fluid fill.
  Assess held-out fluid distinguishability before treating predictions as definitive.
- Validate same-layer gas/brine contrast pairs and similar fluid-matched layering at
  distant locations. Report fluid-conditioned retrieval and errors near contacts.
- Review phase/polarity, amplitude normalization, and noise augmentation against the
  fluid task. Acquisition invariance must not remove the evidence needed for fluid
  discrimination. Do not assume gain/polarity transformations are harmless here.
- Clarify whether "duplicated layers" means the exact same physical horizons or
  comparable stratigraphic patterns from different horizons/volumes.

## 13. Read-only source schema audit

Inspected on 2026-11-08 without changing source stores or reading full seismic cubes.

| Location | Observed contents | Implication |
| --- | --- | --- |
| Source root, excluding holdout folders | 180 immediate runs with `model_data.zarr`; one training schema inspected | Training candidates; full schema/provenance audit still required |
| `validation` | 25 model stores; 20 have all six candidate semantic arrays, 5 do not | Labeled semantic validation is possible on the 20; report the missing-label subset separately |
| `test` | Four `.real.zarr` stores, each exposing only a `volume` array | Real-domain holdout, but no stored layer/fluid labels for quantitative semantic ground truth |

The sampled training store and a labeled validation store expose:

- `faulted_lithology`
- `geologic_age_faulted`
- `gas_closures`, `oil_closures`, `brine_closures`
- `faults/faulted_channel_labels`

This establishes candidate label availability, not their complete semantics. Verify
mask encodings, mutual exclusivity, fluid-contact handling, and whether closure masks
cover fluid-filled channels and all relevant reservoir sand. Do not assume unlabeled
sand is brine or above-seabed water identifies reservoir fluid.

The training schema has fullstack and 5/17/29-degree seismic arrays. Angle-dependent
amplitudes may provide useful fluid evidence. Comparing fullstack-only with multi-angle
inputs is a potential later ablation, not an approved input-channel change.

Seismic shape is `(300, 300, 1499)` and label shape is `(300, 300, 1510)` in the
inspected training store. Confirm axis and depth alignment with existing verified
helpers; do not infer a depth offset solely from the 11-sample length difference.
Float seismic chunks span `(50, 50, 1499)`, approximately 15 MB uncompressed each:
streamed small crops can still decode full-depth chunks. Include this amplification
in the bounded-cache and worker memory/throughput pilot.

The real test arrays have no array attributes establishing axis semantics. Verify
their existing conversion/storage contract before selecting XYZ crops.

Open evaluation decision: obtain an independently labeled synthetic test set, or
reserve an approved subset of the current training candidates before any new training.
The user can generate new volumes or use existing volumes with fair controls; no choice
has been finalized. New labeled synthetic volumes are recommended because reused
checkpoints cannot have trained on them. Existing-volume holdouts require checking
starting-checkpoint exposure or training matched controls without that exposure.

Retain the four real test volumes outside new training. They support reconstruction,
coordinate-derived physical-overlap evaluation, and controlled augmentation robustness.
Without labels or expert annotations, they cannot establish gas/oil/brine semantic
retrieval accuracy. Earlier repository experiments already evaluated these real volumes:
they are reusable historical benchmarks, not a completely untouched final test set.
Use labeled synthetic tests and real-domain tests as complementary evidence.

## 14. New labeled synthetic test collection

Thirty additional synthoseis test datasets now exist in
`/Volumes/CrucialX9/fake_data/test_synthoseis`.

The collection has the recommended initial size of **30 independently generated
volumes**. This is a practical evaluation starting point, not a power calculation or a
guarantee of narrow confidence intervals. Volume-level diversity and paired,
source-volume-aware uncertainty matter more than large numbers of correlated crops.

Generation checklist:

- New seeds and independent realizations, with no copied training/validation geometry.
- Same general generator distribution as training, with varied stratigraphy and structure.
- Collection-wide coverage of gas, oil, brine, reservoir sand layers, and channels.
  These categories may coexist within a volume; do not force one fluid per volume.
- Save seismic, lithology, geological age/horizon information, channel labels, and all
  fluid masks. Preserve generator settings, seeds, and any needed alignment metadata.
- Audit category coverage, fluid-mask semantics, and label alignment before finalizing
  the evaluation manifest. Missing cases cannot be supplied by repeatedly cropping
  volumes that do not contain them.

Start with approximately 20 fixed query windows per volume (600 queries total), selected
to cover layering/reservoir/fluid cases and background. Define candidate windows and
positive/negative correspondence rules separately; 600 queries is not 600 independent
geological realizations. Report category coverage and per-volume metrics, and estimate
uncertainty at the source-volume level.

Use existing validation volumes for epoch evaluation and model/settings selection.
Freeze this new test manifest and evaluate it after selection; do not use these new
volumes for training, calibration, auxiliary-loss tuning, or routine per-epoch evaluation.
Treat same-horizon recognition, disjoint analog retrieval, and gas/oil/brine mismatch
errors as separate test slices. Retain the existing real test volumes for complementary
real-domain evaluation.
