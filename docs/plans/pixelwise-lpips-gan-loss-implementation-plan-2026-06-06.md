# Pixelwise + LPIPS + GAN Loss Plan (2026-06-06)

## Context and goal
- Current VAE outputs preserve low-frequency structure but miss important higher-frequency seismic detail (vertical and lateral), including faults and shorter-wavelength events.
- Goal: upgrade training losses to improve both global coherence and fine-detail reconstruction using:
  1. Pixel-wise reconstruction losses
  2. Perceptual LPIPS losses (slice-wise)
  3. GAN/discriminator losses over full cubes
- Branch for implementation work: feat/pixelwise_LPIPS_GAN_loss

## PDF-grounded guidance
- LPIPS paper (arXiv:1801.03924v2): supports deep-feature perceptual distance as a stronger perceptual objective than simple pixel metrics.
- RGT-Est paper: explicitly motivates combining pointwise, perceptual, and adversarial losses for local fidelity + structural consistency + global plausibility in 3D seismic tasks.
- Scheduling guidance from PDFs: no explicit loss-weight schedule prescription was found. Any warmup/ramp policy is engineering judgment.

## Final implementation plan (author+critic agreed)

### Phase 0: prerequisites
- Ensure lpips dependency can be imported in the runtime environment.
- Keep lpips_weight default at 0.0 so existing behavior remains unchanged unless enabled.

### Phase 1: LPIPS integration and numeric safety
- Add LPIPS infrastructure for slice-wise perceptual loss (inline/crossline strategy accepted by user).
- Execute LPIPS in guaranteed FP32 using autocast disabled around LPIPS forward.
- Freeze LPIPS network parameters (`requires_grad=False`) so only VAE receives gradients.
- Preserve gradient flow through reconstruction tensor (no detach on prediction path; detach target path).

### Phase 2: LPIPS slice sizing policy
- Use slice extraction suitable for seismic 3D tensors.
- Handle small slices for AlexNet LPIPS by enforcing minimum spatial size (upsample strategy) where required.
- Keep test coverage for both direct-size and upsample paths.

### Phase 3: objective plumbing
- Add lpips_weight to training objective path and keep GAN optional.
- Validation objective remains weighted objective excluding GAN (as requested).
- Ensure all loss-call sites are updated consistently and maintain baseline equivalence when lpips_weight=0.

### Phase 4: logging and schema-safe resume
- Add LPIPS-related logging fields to console/TensorBoard/CSV.
- Add concrete CSV migration behavior for resumed runs with schema differences.
- Add explicit resume override (resume_epoch) and collision-safe checkpoint handling.

### Phase 5: regression and safety tests
- Add counted-mock test proving LPIPS is not called when lpips_weight=0.
- Add GAN regression test with lpips_weight=0 to prove no behavioral drift in GAN path.
- Add core LPIPS unit tests for shape, dtype, normalization, and non-NaN behavior.

### Phase 6: controlled rollout
- Start with lpips_weight=0 baseline verification.
- Enable small lpips_weight and compare validation objective and qualitative outputs.
- Tune weights only after regression gates pass.

## Two-agent process and probability convergence

### Agents
- Author: Claude Sonnet 4.6
- Constructive critic: GPT-5.4

### Iterations
1. Iteration 1
- Author probabilities: pixel-wise 95%, LPIPS 78%, GAN 88%, overall >70 at 72%
- Critic verdict: REVISE (GAN confidence too low; integration/test gaps)

2. Iteration 2
- Author revised plan with GAN invariants, LPIPS fp32/freeze guidance, CSV handling
- Critic verdict: REVISE (remaining issues on AMP semantics, test slice sizes, CSV edge details, no-op test rigor)

3. Iteration 3
- Author Plan v3.1 resolved all four blockers:
  - explicit LPIPS FP32 execution wrapper
  - valid slice-size/upsample strategy
  - concrete CSV/resume edge-case handling
  - counted-mock no-op regression test
- Author probabilities: pixel-wise 88%, LPIPS 75%, GAN 73%, combined 71%
- Critic final verdict: AGREE
- Critic probabilities: pixel-wise 88%, LPIPS 75%, GAN 73%, combined 71%

## Final sign-off
- Both agents now predict >70% probability for:
  - pixel-wise implementation correctness
  - LPIPS implementation correctness
  - GAN/discriminator integration correctness
- Both agents agree the plan is ready to execute.

## Agent Execution Spec

### Scope and file touchpoints
- Primary implementation file: `scripts/train.py`.
- Expected additional touchpoints:
  - `requirements.txt` or `pyproject.toml` (LPIPS dependency)
  - test files under `tests/` for LPIPS no-op and GAN invariants
  - docs only if CLI help text or usage examples are changed

### CLI and backward-compatibility contract
- Add `--lpips_weight` as `float`, default `0.0`.
- `lpips_weight=0.0` must preserve baseline behavior (bitwise equality is not required, metric-level parity is required).
- Existing checkpoints and resume flows must remain loadable without requiring new fields.

### LPIPS slice and numeric contract
- Compute LPIPS on 2D slices sampled from both inline and crossline directions.
- Execute LPIPS forward in FP32 with autocast disabled for that block.
- Freeze LPIPS backbone parameters (`requires_grad=False`).
- Do not detach reconstructed predictions before LPIPS; detach targets.
- If slice spatial size is below LPIPS minimum expectation, upsample with bilinear interpolation (`align_corners=False`) before LPIPS.

### Logging and CSV migration contract
- Add LPIPS metrics to console, TensorBoard scalar logging, and CSV rows.
- On resume, if historical CSV rows do not include new LPIPS columns, write empty/default values for old rows and continue appending without failure.
- Do not break existing column names used by downstream analysis.

### Acceptance gates (must pass before tuning)
- No-op gate: with `lpips_weight=0.0`, LPIPS forward call count is zero (counted-mock test).
- GAN invariants gate: with `lpips_weight=0.0`, GAN path metrics remain within expected tolerance relative to baseline.
- Stability gate: no NaN/Inf in LPIPS or total loss under mixed precision training.
- Resume gate: training resumes cleanly from pre-LPIPS CSV/checkpoint artifacts.

### Autonomous execution order
1. Add dependency and CLI flag wiring.
2. Implement LPIPS module setup + FP32 forward wrapper.
3. Integrate LPIPS into training and validation objectives per weight.
4. Add logging/CSV schema-safe append behavior.
5. Add/adjust tests for no-op, GAN invariants, and resume compatibility.
6. Run regression gates, then begin low-weight rollout.

## Session Carry-Forward

### Branch and scope guardrails
- Start implementation on `feat/pixelwise_LPIPS_GAN_loss`.
- Do not implement this plan directly on `main`.

### Known unrelated working-tree files
- Ignore unrelated egg-info churn unless explicitly requested to include it:
  - `src/synthoseis_3dvae_poc.egg-info/SOURCES.txt`
  - `src/synthoseis_3dvae_poc.egg-info/top_level.txt`

### Environment bootstrap commands
- Activate environment:
  - `source /Users/donaldpg/synthoseis-3dvae-poc/.venv/bin/activate`
- Validate training CLI after edits:
  - `python scripts/train.py --help`

### Hard acceptance thresholds
- GAN invariants threshold at `lpips_weight=0.0`:
  - Relative drift limit for key GAN metrics (`d_gan_acc`, discriminator loss, generator GAN loss) <= 2% versus pre-change baseline run with same seed/config.
- No-op LPIPS threshold at `lpips_weight=0.0`:
  - LPIPS forward call count must be exactly 0.

### CSV migration rule for resume
- If resuming from historical CSV without LPIPS columns, append using new schema and backfill missing LPIPS fields in legacy rows with empty value (`""`) when rewriting headers is required.
- Resume must not fail due to missing LPIPS columns.

### Safety and rollback behavior
- With `lpips_weight=0.0`, objective behavior must remain baseline-equivalent at metric level.
- If NaN/Inf appears in LPIPS or total loss after enabling LPIPS, disable LPIPS path (`lpips_weight=0.0`) and stop for diagnosis.

### Commit strategy
1. Commit 1: dependency + CLI wiring.
2. Commit 2: LPIPS core integration (FP32 wrapper, slice policy, objective path).
3. Commit 3: logging + CSV migration/resume handling.
4. Commit 4: tests (no-op LPIPS, GAN invariants, resume compatibility).

### Stop conditions for autonomous execution
- Stop and ask for user input if:
  - dependency install/import for LPIPS fails in project environment
  - GAN invariant drift exceeds threshold at `lpips_weight=0.0`
  - CSV migration encounters ambiguous legacy schema not resolvable deterministically
  - training stability fails (persistent NaN/Inf) after one rollback attempt

## Suggested immediate next step
- Implement Phase 1 and Phase 3 together in scripts/train.py first, then run the regression gates before tuning loss weights.
