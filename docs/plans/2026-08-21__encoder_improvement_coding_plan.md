# 2026-08-21 — Encoder improvement coding plan for geology-aware latent retraining

## Context

We are working on a 3D seismic VAE that already has a functioning reconstruction pipeline and a latent-based similarity workflow. The current system can encode a user-selected 3D window, compute latent vectors throughout a target volume, and compare them using cosine similarity. The workflow works technically, but the results are not compelling enough for a user to confidently say that two windows share the same geology when their cosine similarity is high or low.

The repository already includes a robust reconstruction-focused VAE, dataset generation and augmentation layers, deep supervision support, and a latent-search/search-engine pipeline. The main remaining weakness is latent geometry: the current latent space is not sufficiently structured around geology-specific similarity under heavy augmentation.

The upstream pretrain repo in the remote environment showed a useful precedent: architecture changes were implemented in a disciplined, staged way, including encoder-depth scheduling, multi-component loss updates, and resize-convolution decoder validation. Those are the kinds of changes we should reproduce here in a controlled manner.

## Goal

Improve the 3D VAE so that:

1. it is more robust to heavy augmentation,
2. its encoder is better at preserving geologic structure under input perturbation,
3. its latent representations separate geology more clearly,
4. cosine similarity becomes more aligned with geologic similarity in the user workflow,
5. the updated model remains compatible with the existing tokenizer/search workflow.

The final objective is to retrain a better latent encoder that can be used to search a 3D volume for geologically similar windows using user-selected target windows and cosine-similarity scoring.

## Human-readable summary

The current model is already expressive enough to reconstruct seismic patches, but it is not yet optimized to encode geology as a meaningful similarity metric. The problem is not the search engine itself; it is the latent representation it depends on. To improve this, we need two things in sequence:

- first, improve the encoder architecture under heavy augmentation so that the model is more robust and preserves structure,
- second, train with geology-aware objectives so similar geologic patterns become closer in latent space and dissimilar patterns become farther apart.

The repo already includes the right ingredients: training pipeline, augmentation hooks, optional deep supervision, LPIPS/GAN capability, and a cosine-similarity search path. What is still missing is a disciplined encoder architecture search and a geology-aware training objective that makes the latent similarity meaningful to a user.

---

## Agent prompt

You are an engineering assistant working in this repository. Implement the following work end-to-end, using the repository code and documentation as the source of truth.

Perform the work in a disciplined, validation-first manner. Do not make broad speculative changes. Keep the latent interface compatible with the existing search workflow. Do not break checkpoint compatibility, the VAE encoder contract, or the downstream cosine-similarity search logic.

Your task is to carry out the following phases.

### Phase 0: Branch creation and repo hygiene

Before making any model or training changes, prepare the repository state so the work is isolated and recoverable.

1. Check the current git status and identify any uncommitted or local changes in the repo.
2. Review and document all existing modifications, including notebook changes, config edits, plan updates, and any debug or exploratory work.
3. Commit all relevant existing changes with a clear message summarizing the current repository state before the encoder-improvement project begins.
4. If the repo is not already on a dedicated working branch, create a new branch for this effort, for example:
   - `encoder-improvement-2026-08-21`
   - or a similarly specific branch name tied to the geology-aware latent retraining plan
5. Push the branch to GitHub and ensure the remote tracking branch is configured correctly.
6. Record the branch name, commit SHA, and remote URL in the working notes or plan summary so the work can be resumed and reviewed later.
7. Do not start the architecture-search work until the repo is cleanly checked out on the new branch and the current state has been committed and pushed.

Requirements:
- Keep the branch dedicated to this plan only.
- Do not mix unrelated local work into the same branch unless it is required for the project.
- If there are intentionally non-committed or sensitive files, document them explicitly before proceeding.

Acceptance criteria:
- There is a dedicated branch for the encoder-improvement work.
- Existing code changes are documented and committed.
- The branch is pushed to GitHub before any implementation work begins.

---

### Phase 1: Baseline reproduction

1. Reproduce the current training baseline exactly as implemented in the repository.
2. Use the current model in [src/model.py](src/model.py), training loop in [scripts/train.py](scripts/train.py), and dataset pipeline in [scripts/sample_patches.py](scripts/sample_patches.py).
3. Run the model under the current heavy-augmentation regime used by the project and document the baseline results.
4. Collect and save the following baseline artifacts:
   - checkpoint
   - metrics CSV / logs
   - validation outputs
   - representative reconstruction examples
   - latent summary statistics
5. Confirm the baseline remains compatible with the existing tokenizer/search workflow.

Requirements:
- Maintain all existing CLI and config behavior unless a specific change is needed for the new work.
- Do not change the user-facing latent interface or the downstream cosine-similarity API unless absolutely required.

Acceptance criteria:
- A stable baseline is reproducible and saved.
- There is a fixed benchmark for later comparison.
- You have a clear before-state against which the improved architecture and retraining can be measured.

---

### Phase 2: Encoder-search sweep

1. Create a controlled architecture-search process for the encoder side of the 3D VAE.
2. Keep the decoder and latent contract as stable as possible.
3. Search over a bounded set of encoder-side changes such as:
   - encoder depth profile / block scheduling
   - channel growth per stage
   - residual structure vs plain blocks
   - normalization choices
   - activation choices
   - placement of downsampling and feature routing
   - optional lightweight encoder-side enhancement that does not alter the external latent contract
4. Use a semi-random but constrained search loop with fixed seeds and a bounded parameter space.
5. Prioritize changes that improve robustness under heavy augmentation and preserve geologic detail.
6. Rank candidate architectures using a composite objective that includes:
   - reconstruction quality
   - stability under heavy augmentation
   - latent separation on geology labels
   - cosine similarity separation for same-vs-different geology examples
   - tokenizer compatibility and inference stability

Requirements:
- Do not perform arbitrary global redesigns.
- Do not change the external latent dimension or downstream interface unless absolutely necessary.
- Follow the precedent from the upstream repo: stage the experiments, validate with tests, and keep the changes explicit and reproducible.

Acceptance criteria:
- Top 2–3 candidate encoder variants are identified.
- Each candidate is benchmarked against the baseline.
- The winner is chosen based on clear evidence, not anecdotal impression.

---

### Phase 3: Validation gating

Before accepting any encoder variant or retraining change, run the following checks.

1. Heavy-augmentation validation test
   - evaluate each candidate on the same heavy-augmentation benchmark used in the target workflow
   - compare against the baseline under identical conditions

2. Reconstruction validation
   - MSE / MAE / other reconstruction metrics
   - perceptual metrics if enabled
   - preservation of detail under sparse, warped, and mixed inputs

3. Latent-separation validation
   - same-feature vs different-feature cosine similarity distributions
   - measures of latent clustering for geology classes
   - linear probe accuracy or another label-based latent probe

4. Retrieval-quality validation
   - choose reference windows from the validation or synthetic dataset
   - check whether the model retrieves geologically similar windows more reliably than the baseline
   - compare ranking quality using cosine similarity

5. Compatibility validation
   - encoder output shape remains compatible with current loader and tokenizer code
   - checkpoint loading still works
   - inference pipeline still runs without errors
   - user workflow remains intact

6. Stability validation
   - repeat key runs across multiple seeds or repeated runs
   - ensure gains are not one-off results
   - ensure no divergence or collapse during training

Acceptance criteria:
- A candidate is only accepted if it improves the heavy-augmentation benchmark and latent/geology metrics without breaking the tokenizer/search workflow.
- The final encoder version must be demonstrably better than the baseline by the required metrics.

---

### Phase 4: Geology-aware latent retraining

Once the encoder is validated, retrain the model with a geology-aware objective so the latent space reflects geologic similarity.

1. Extend the patch dataset schema to include geology metadata.
   - patch-level feature fractions
   - feature-presence flags
   - primary geology label
   - provenance and normalization metadata
2. Update the dataset loader and training loop to optionally return geology metadata.
3. Keep reconstruction as the backbone objective.
4. Add a geology-aware auxiliary objective on top of reconstruction.
   - feature prediction head or patch-level geology supervision
   - similarity-weighted or supervised contrastive loss on normalized latent vectors
5. Use balanced sampling so each batch contains enough geology-rich examples.
6. Train in phases, not all at once:
   - Phase A: baseline reconstruction
   - Phase B: add geology-aware feature head
   - Phase C: add geology contrastive or similarity-weighted objective at low weight
   - Phase D: increase geology weight only after stability is confirmed
7. Keep the search workflow unchanged at the API level; the improvement should come from better latent geometry, not a different user interface.

Requirements:
- Do not drop the current reconstruction losses unless a clear, validated reason exists.
- Preserve the current latent extraction contract and cosine-similarity use.
- Make the objective explicit and consistent with the geology-search use case.

Acceptance criteria:
- The retrained model produces latent vectors with improved same-vs-different geology separation.
- Cosine similarity is more informative for geology matching in the 3D volume.
- The downstream search workflow shows more convincing matches than the baseline.

---

### Phase 5: Final verification and reporting

1. Run a final comparison against the baseline.
2. Report results in a concise summary covering:
   - baseline metrics
   - winning encoder variant
   - comparison to baseline
   - geology-aware training objective used
   - latent-separation metrics
   - downstream cosine-search improvement
3. Save the final report in the repo under the docs/plans folder with a useful timestamped filename.
4. Provide the exact commands and validation steps that were used.

Acceptance criteria:
- all tests and validation gates pass,
- the final model is both more robust under heavy augmentation and more useful for geology matching in latent space,
- the work is documented in a way another engineer can reproduce it quickly.

---

## Important constraints

- Keep the model backward-compatible with the existing tokenizer and latent-search workflow.
- Do not break checkpoint loading or rescue workflows.
- Do not rely on one metric alone; require a multi-metric gate.
- Keep any architecture search constrained and reproducible.
- Treat the external upstream repo’s session findings as precedent for disciplined, evidence-based architecture experimentation.

## Deliverables

- updated encoder architecture
- any new tests required for validation
- updated training logic for geology-aware objective
- final benchmark comparison against the baseline
- final documentation/report in the plans folder

This is the end-to-end implementation plan to improve the 3D VAE encoder and latent-space usefulness for geology matching. Use the current repository code and document the work as you go.
