# 2026-08-21 — VAE retraining and latent-separation plan report

## Context and goals clarity

The context and goals are clear enough to proceed without additional questions. The core objective is well aligned with the repository’s direction:

- Preserve the working reconstruction and tokenizer workflow.
- Improve the latent geometry so geologically similar 3D windows cluster together in latent space.
- Use cosine similarity against a user-selected 3D window to help the user find matching geology elsewhere in the input volume.
- Use synthetic-data labels to push latent similarity up for matching geology and down for mismatched geology.

This report consolidates the plan documents found under the plans folder and compares them to the actual codebase implementation as of the current repository state.

## Plans reviewed

I reviewed the Markdown plans in the plans folder and found these documents:

- 2026-06-08__latent_contrastive_loss_execution_backlog_updated.md
- 2026-06-08__latent_contrastive_loss_for_geologic_understanding.md
- 2026-06-08__latent_contrastive_loss_for_geologic_understanding_updated.md
- deep_supervision_feature_ala_monai.md
- input-augmentations-sparse-keep-and-decimate-trilinear-2026-06-02.md
- pixelwise-lpips-gan-loss-implementation-plan-2026-06-06.md
- seismic_tokenizer_app.md

There were no subfolders under the plans folder, so the scope was limited to those seven Markdown plans.

## Implementation status report

### Summary table

| Area | Status | Evidence | Notes |
|---|---|---|---|
| Synthetic patch creation and 3D dataset assembly | Implemented | `scripts/sample_patches.py` creates 3D patch zarr datasets from `model_data.zarr`, sampling by seismic amplitude and geologic score. | This is the primary input-data pipeline currently in use. |
| Input scaling and deterministic augmentation pipeline | Implemented | `scripts/train.py` and `src/augmentations.py` contain scaling, pair augmentation, trace dropout, vertical warp, sparse keep, decimate-trilinear, and mixup support. | A mature augmentation layer already exists. |
| Baseline 3D VAE architecture | Implemented | `src/model.py` includes encoder, decoder, KL regularization, latent vector, and optional deep supervision. | The architecture is already production-grade for reconstruction. |
| Deep supervision heads | Implemented | `src/model.py` contains `deep_supervision` support, while `scripts/train.py` uses optional multi-scale deep-supervision loss paths. | This is already in the repo and not purely planned. |
| LPIPS + GAN loss path | Implemented | `scripts/train.py` includes `lpips_weight`, discriminator training, and GAN losses. | Reconstruction realism improves but latent separation is not yet deliberately geologic-aware. |
| Geologic feature vectors and feature-aware training labels | Not implemented | No plan item is yet wired into the actual patch dataset schema or training loader. | This remains the key missing piece for geology-aware latent geometry. |
| Contrastive or similarity-weighted latent shaping based on geology labels | Not implemented | The repository has latent alignment knobs, but no geology-conditioned contrastive objective is live in the training loop. | This is the primary gap behind the weak cosine-similarity separation. |
| Balanced geology-aware sampling | Ready to implement | The plan documents explicitly propose patch-level label vectors and balanced sampling. | The codebase has only geologic-score weighting, not label-aware balancing. |
| Evaluation of latent separation and feature probes | Ready to implement | The plans call for linear probes and clustering metrics, but there is no dedicated latent-evaluation package in the active pipeline. | This is needed before/after retraining to validate geology separation. |
| Tokenizer workflow for latency search by cosine similarity | Implemented | `src/tokenizer/*`, `scripts/tokenize.py`, and `src/tokenizer/core/similarity.py` already compute cosine/dot similarity on latent tensors against a target window. | The workflow exists; the issue is the latent signal quality, not the search mechanism itself. |

## A) Input data

### Current state

The current dataset pipeline does a solid job of producing 3D seismic patches with a geologic-score bias, but it does not yet produce a rich set of geology-aware labels for every patch.

Evidence in the current code:

- `scripts/sample_patches.py` samples patches from `model_data.zarr` using `seismic_key` and `geoscore_key`.
- The patch selection logic is weighted by geologic score, which is useful but not equivalent to class-aware geology labels.
- `scripts/train.py` loads `patches` only and does not currently load per-patch geology feature vectors, binary feature flags, or primary labels.
- The plan documents repeatedly call for patch-level feature vectors such as fault, channel, unconformity, strat boundary, and background proportions, but those are not in the live dataset contract.

### Comparison to the plan

The planning documents are internally consistent in identifying this as the main upstream data gap:

- The latent-shaping backlog calls for patch feature vectors, feature schema, normalization metadata, and provenance.
- The updated plan explicitly recommends a canonical patch feature vector:
  - `feature_fraction`
  - `feature_presence`
  - `primary_label`
  - optional entropy/quality indicators
- This has not been implemented in the repository yet.

### Assessment

- Implemented: raw patch sampling and geologic-score weighting.
- Ready to implement: patch-level geology metadata with feature fractions and labels.
- Not implemented: stable multi-label schema, feature normalizers, and label-aware dataset loaders.

### Key implication

Without label-aware input data, the VAE can only learn to reconstruct and align with scalar geologic score bias. It cannot learn a latent space where cosine similarity reflects geologic class identity or feature similarity in a user-driven search problem.

## B) Deep learning VAE architecture

### Current state

The architecture is already substantially mature and supports a good reconstruction baseline:

- `src/model.py` defines a 3D convolutional encoder and decoder.
- The encoder outputs `mu` and `logvar`, then reparameterizes to `z`.
- The decoder reconstructs a full 3D volume from the latent vector.
- The model supports optional deep supervision through auxiliary heads at decoder stages.
- `scripts/train.py` already includes a modular loss stack with pixel reconstruction, KL regularization, optional LPIPS, and optional GAN loss.

This is a solid backbone for a geology-aware latent model, but it is reconstruction-first rather than separation-first.

### Comparison to the plan

The plans are aligned with this architecture and propose extending it rather than replacing it:

- Add a feature head on top of the latent vector or mid-level decoder features.
- Add supervised contrastive or similarity-weighted contrastive loss on normalized latent embeddings.
- Keep the reconstruction stack intact while adding label-aware shaping on top.

### Assessment

- Implemented: layered encoder-decoder VAE, optional deep supervision, latent extraction, optional GAN/LPIPS.
- Ready to implement: a geologic feature head, latent contrastive branch, and paired objective plumbing.
- Not implemented: any training objective explicitly coupled to geology labels beyond generic reconstruction and latent alignment.

### Key implication

The latent space is currently optimized primarily for pixel reconstruction, not for geologic similarity geometry. That is why the search workflow can work technically while still being visually unconvincing to a user when cosine similarity values are interpreted.

## C) Training methodology: augmentation, losses, and loop design

### Current methodology

The training loop in `scripts/train.py` is already rich and flexible:

- Pair augmentations on input/target volumes
- Trace dropout
- Vertical warp
- Sparse input keep
- Decimate-and-interpolate transform
- Mixup augmentation
- KL warmup schedule
- Adaptive weighting and optional GAN balancing
- Resume logic and checkpoint saving
- Deep-supervision loss support

The default objective design is already a multi-component reconstruction objective with a variational prior term. The current reconstruction loss can be summarized as:

- reconstruction loss + KL penalty + optional LPIPS + optional GAN

This is appropriate for visual fidelity, but not sufficient for the downstream requirement of geology-specific latent clustering.

### Current loss design in the repo

From `scripts/train.py`, the loss package already includes:

- pixelwise reconstruction objective
- LPIPS loss
- GAN discriminator/generator loss
- deep-supervision aggregated reconstruction signal
- optional latent alignment losses already discussed in a training guide

This means the project has already been moving toward “structured latent” training, but not specifically toward geology-conditioned similarity learning.

### What is still missing for your target workflow

The missing ingredient is a label-aware similarity objective that explicitly enforces:

- similar geology => higher cosine similarity in latent space
- different geology => lower cosine similarity

The planning docs repeatedly recommend this approach, but it has not yet been implemented in the active training path.

### Assessment

- Implemented: augmentation variety, scale-robust training loop, optional deep supervision, LPIPS, GAN, resume, and checkpoint management.
- Ready to implement: geology-aware contrastive objective, balanced sampling, curriculum stages, and diagnostics.
- Not implemented: the actual training objective that uses synthetic geology labels to shape the latent geometry for 3D window matching.

## Recommendations for the next retraining plan

### Recommendation 1: Make geology-aware labels the first-class training signal

Use the synthetic training data to produce per-patch geology feature vectors and primary labels. For each 3D patch, compute:

- `feature_fraction`: per-feature proportions for the patch
- `feature_presence`: binary flag for each feature
- `primary_label`: geologic class that dominates the patch

This should be stored along with each patch in the output dataset, and it should follow a stable schema with feature names and normalization metadata.

Recommended feature set, consistent with the planning docs:

- fault
- channel
- unconformity
- strat boundary
- background
- optional derived categories such as closures, throws, and dip-based classes

### Recommendation 2: Add a geology contrastive training objective

The simplest and most effective next objective is a normalized latent similarity term that directly pushes geologically similar patches together and dissimilar ones apart.

A practical design is:

- encode the patch to latent vector `z`
- normalize `z` with L2 normalization
- compute cosine similarity between latent vectors in a batch
- treat patches with matching or similar `feature_fraction` patterns as positive pairs
- treat patches with different feature mixes as negatives

Recommended training objective:

- reconstruction loss remains as the backbone
- add a supervised contrastive loss or similarity-weighted pairwise loss
- optionally include a feature-prediction BCE head as a secondary signal

A clean starting objective is:

- `L_total = L_recon + λ_kl * L_kl + λ_geo * L_geo + λ_feat * L_feat`

where:

- `L_geo` is a geology-conditioned contrastive objective
- `L_feat` is a patch-level feature prediction objective if needed

### Recommendation 3: Use feature-similarity weighting instead of binary class-only labels

The best approach for your geology search problem is not only “same/different class” but also “how similar are the geology patterns.”

Use the feature fractions to build a soft similarity target:

- e.g., if two patches share a high fault fraction and similar channel fraction, their positive target similarity is high
- if one patch is channel-heavy and the other is fault-heavy, similarity should be low

This is much better aligned with the user experience requirement than a hard class label alone.

### Recommendation 4: Retain the current reconstruction stack while adding a label-aware branch

The current repo already has enough reconstruction capability; do not replace it. Instead, extend it with a parallel supervision path.

Recommended training phase ordering:

1. Baseline reconstruction checkpoint
2. Add feature head and feature prediction loss only
3. Add geology contrastive loss at a low weight
4. Increase `λ_geo` modestly after a stable checkpoint is reached
5. Keep LPIPS/GAN only if reconstruction remains stable

This preserves training stability while improving geology separation.

### Recommendation 5: Train for latent geometry, not just reconstruction quality

Before/after each retraining run, evaluate:

- cosine similarity distribution for same-feature pairs vs different-feature pairs
- latent probe accuracy for geology labels
- clustering/visualization of latent embeddings for selected patch classes
- rate of user-visible “reasonable match” decisions when comparing latent similarity across windows

This is essential: the output of cosine similarity must be meaningful to a human geoscientist.

## Concrete retraining workflow for the next phase

### Step 1: Extend the dataset schema

Update the patch dataset to include:

- per-patch feature fractions
- feature presence flags
- primary geology label
- provenance metadata

This should be stored in the zarr output alongside the patch arrays and be versioned.

### Step 2: Update the training loader

The dataset loader should optionally return:

- `x` (input patch)
- `y` (target patch)
- `feature_fraction`
- `feature_presence`
- `primary_label`

If those arrays are absent, training should still fall back to the current reconstruction-only mode.

### Step 3: Add geology-aware contrastive loss

Implement a loss that compares latent vectors using cosine similarity and enforces a margin or cross-entropy-style target based on geology similarity.

A practical first version:

- normalize latent embeddings
- for each anchor, create positive pairs from same or similar geology label
- create negative pairs from different geologic profiles
- use cross-entropy over cosine-similarity logits with temperature `τ`
- optionally add a small margin penalty for mismatched pairs

This directly addresses the user requirement to improve the ranking of similar geology in the 3D volume.

### Step 4: Use balanced sampling by geology label

Batch composition should not be allowed to drift toward background-heavy patches only.

Recommended sampler policy:

- keep a minimum fraction of fault, channel, unconformity, and other geology-rich patches in each batch
- oversample rare geology classes when needed
- avoid letting a few dominant features drown out the signal

### Step 5: Re-evaluate the tokenizer workflow against the improved latent space

After retraining, the current search workflow can remain unchanged. The real change is in the latent encoder output:

- user-selected patch latent tensor => better geologic descriptor
- volume-wide windows => better match ranking by cosine similarity
- result -> improved user confidence in “similar vs not similar” classification

## Additional required architecture-search component: U-Net-style encoder updates

### External repo access status

The remote-mounted user home does contain the upstream pretrain repo, and the session documents are under the remote user directory rather than the /Volumes mount. The repo was verified in the current VS Code remote environment under the user home and its session docs were reviewed successfully.

Those session summaries show a consistent pattern of architecture tuning on the encoder/decoder stack rather than isolated one-off changes:

- encoder depth profile selection with explicit baseline/deeper/deepest presets
- model depth configuration through CLI and validation checks
- resize-convolution decoder refactor to replace transpose-convolution upsampling
- multi-component loss additions that improve robustness under harder training conditions

This means the correct upstream precedent is not a generic “semi-random search” note, but a concrete search-and-refine workflow over encoder depth and decoder upsampling choices, validated by targeted tests.

### Why this matters

The repository already contains strong augmentation and loss infrastructure, but the next step to improve robustness under heavy augmentation is to also improve the encoder architecture itself. In the same spirit as the other repo’s semi-random search over U-Net-style architecture changes, the 3D VAE encoder should be tuned to preserve geologic structure under augmentation while staying compatible with the existing latent-search workflow.

This is especially relevant because your evaluation target is not just reconstruction quality in the abstract. It is reconstruction and latent geometry under heavy augmentation, with a final requirement that cosine similarity in latent space corresponds to geologic similarity in the volume.

### Architecture-search strategy to adopt

The upstream session summaries show a strong pattern that architecture improvements were made in a disciplined experimental loop, not by random code churn. The relevant examples from the pretrain repo are:

- encoder depth profile selection with presets (`baseline`, `deeper`, `deepest`) and explicit per-stage overrides
- multi-component loss work that adds PMSE and optional LPIPS to improve reconstruction under more difficult optimization setups
- resize-convolution decoder refactor validated by tests proving gradient flow and shape consistency

These are directly relevant to the 3D VAE retraining work here. The architecture-improvement step should therefore be structured as a constrained search over encoder depth and block-level changes, not a free-form random walk.

Recommended search dimensions:

- encoder depth and residual block count
- channel growth rate per stage
- normalization strategy (BatchNorm vs GroupNorm vs LayerNorm)
- activation choice and activation placement
- residual vs plain convolutional blocks
- use of pre-activation normalization and GELU/SiLU choice
- downsampling placement and block ordering
- skip/feature-routing choices that preserve fine-scale structure under augmentation
- optional lightweight attention or bottleneck enhancement only on encoder side

This search should not change the external latent interface used by the tokenizer. In other words, the latent tensor shape, semantic meaning, and downstream cosine-search contract must remain stable.

### Semi-random architecture search plan

1. Establish the baseline
   - train the current 3D VAE architecture under the same heavy augmentation regime
   - record validation loss, reconstruction metrics, and latent-separation diagnostics
   - freeze this as the baseline for comparison

2. Define a constrained search space
   - keep the decoder fixed as much as possible
   - vary only encoder-side configurations that are expected to improve invariance and robustness
   - enforce a fixed latent size and patch shape compatibility

3. Use a semi-random search loop
   - sample architecture variants from a bounded parameter space using a fixed random seed
   - evaluate each candidate on the same short training budget
   - rank candidates by a composite objective combining reconstruction quality and latent discrimination quality
   - mirror the upstream repo’s pattern of making architecture choices explicit and comparable through presets and validation

4. Run a two-stage selection process
   - Stage A: quick screening on a small synthetic subset or reduced training schedule
   - Stage B: full training for the top few candidates under the target regime

5. Require a win condition before promoting a candidate
   - candidate must improve on the baseline on the same validation dataset with heavy augmentation
   - candidate must not regress latent-space search behavior or tokenizer compatibility

### Verified upstream findings that should guide this step

From the remote pretrain repo session summaries already verified in the environment:

- `encoder-depth-profile-cli-selection-2026-06-13.md` showed that profile-based depth selection (`baseline`, `deeper`, `deepest`) is a practical and repeatable architecture-tuning mechanism.
- `multi-component-loss-unet-levels-task-a-f-2026-06-10.md` showed that configuration-level experiments around U-Net depth and multi-component loss are expected to be run in explicit stages with tests and documentation.
- `resize-convolution-decoder-phase1-2-implementation-2026-06-10.md` showed that resize-convolution replacements were validated with tests for shape consistency and gradient flow, rather than by visual inspection alone.

These are the exact types of evidence-based architecture experiments we should replicate here before deciding that a deeper or modified encoder is truly better.

### Composite evaluation metric for architecture selection

The architecture search should not be decided by reconstruction loss alone. A better objective is:

- reconstruction fidelity: MSE, MAE, PSNR, SSIM where available
- structure preservation under heavy augmentation: metrics on augmented validation set
- latent discriminability: probe accuracy for geology labels, same-vs-different patch similarity separation
- user-facing cosine-search quality: AUC or ranking metrics for matching geology windows

Recommended composite score:

- 0.4 * reconstruction quality
- 0.3 * augmentation robustness
- 0.3 * latent separation score

This helps avoid accepting architectures that only improve MSE while making latent similarity harder to interpret.

### Encoder-side architecture changes to prioritize

Because the other repo’s gains came from U-Net-like encoder-side changes, the 3D VAE should test modifications such as:

- residual encoder blocks with deeper early layers
- wider early encoders for preserving fine-scale seismic structure
- better feature-routing from shallow layers into the latent bottleneck
- stronger repeated downsampling with residual connections before the bottleneck
- normalization placements that are stable with heavy augmentation
- controlled dropout or stochastic depth only in the encoder, not the latent head

Important guardrail: these modifications should be added in a way that preserves the latent vector output contract used by cosine similarity matching.

### Rigorous test plan for any modified 3D VAE

Every candidate architecture must pass the following checks before it is accepted as an improvement.

#### 1. Reproduction baseline test

- train the current architecture and a modified candidate under identical seeds and augmentation settings
- compare the same validation split and batch distribution
- require that the modified model improves at least one primary metric without dropping other core metrics below a guardrail threshold

#### 2. Heavy-augmentation robustness test

- use the same augmentation recipe used in the target workflow: sparse keep, decimate/trilinear, vertical warp, trace dropout, mixup, and heavy perturbation intensity
- evaluate reconstruction metrics on heavily augmented validation patches
- require the modified encoder to outperform the baseline on these samples

#### 3. Latent separability test

- compute latent embeddings for a labeled validation set of geology classes
- evaluate:
  - linear probe accuracy on geology labels
  - cosine similarity distributions for same-feature vs different-feature pairs
  - inter-class and intra-class distance metrics
- require improved separation relative to the current architecture

#### 4. Search-quality test for the downstream workflow

- on a held-out volume, select several reference windows and measure retrieval quality with cosine similarity
- quantify whether the candidate encoder retrieves windows with matching geology more accurately than the baseline
- use a small manually labeled benchmark or synthetic labeled reference set as the ground truth

#### 5. Regresion and compatibility tests

- ensure checkpoint loading for tokenizer inference still works
- verify latent shape is unchanged
- verify encoder output still loads into the existing adapter code
- test inference under CPU, MPS, or CUDA as available

#### 6. Training stability tests

- run at least three seeded training runs for each candidate
- confirm the improved architecture is consistently better, not just a lucky one-off run
- watch for divergence, collapse, or unstable KL dynamics

### Suggested acceptance thresholds

A modified architecture should only be considered a success if it satisfies all of the following:

- reconstruction error on the heavy-augmentation validation set improves by a meaningful margin
- latent probe accuracy improves relative to baseline
- cosine similarity separation between related and unrelated geology improves relative to baseline
- no tokenizer compatibility regression is introduced
- no instability or collapse appears in repeated runs

This is stringent but necessary because the end product is not simply a visually nicer autoencoder; it is a latent descriptor that must help a user find geologically similar 3D windows.

### Recommended execution order

1. Review the upstream repo once available and extract exact U-Net encoder modifications and search procedure.
2. Reproduce the baseline current 3D VAE under heavy augmentation.
3. Run the architecture-search loop over encoder-side changes only.
4. Select the top 2–3 failing candidates and train them longer.
5. Compare them against the baseline using the above latency and latent-separation tests.
6. Only then promote the winner to the full retraining pipeline with geology-aware contrastive losses.

### Integration with the geology-aware contrastive plan

This architecture-search step should not replace the geology-aware objective. It is an additional improvement layer:

- first improve the encoder robustness under heavy augmentation
- then improve the latent geometry using geology-conditioned similarity targets
- then verify the final latent similarity aligns with geologic interpretation in the volume search workflow

This sequence reduces risk and keeps the work scientifically grounded: architecture changes improve feature extraction, and label-aware contrastive learning makes the latent space geologically meaningful.

## Bottom line

The repo is already capable of:

- reconstructing 3D seismic patches well,
- running a latent search workflow,
- computing cosine similarity between latent tensors,
- and using rich augmentation and multi-objective training.

What it still lacks is the combination of: (1) robust encoder-side architecture improvements under heavy augmentation, and (2) geology-conditioned latent shaping that turns raw latent vectors into geologic similarity descriptors. The next retraining phase should therefore include both an architecture search on the 3D VAE encoder and a geology-aware contrastive objective, with strict validation against the current architecture before any promotion to production.

The most important next development is therefore not a UI change but a data + objective + architecture change: encode geology label information into the patch dataset, use architecture search to make the encoder more robust under augmentation, and then train the VAE with a geology-aware similarity target that explicitly separates similar and dissimilar geologic features in latent space.
