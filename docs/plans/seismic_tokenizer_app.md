### Seismic Tokenizer Application Plan

## 1) Purpose and Scope
- Branching requirement: all implementation work for this plan must be done on a new git branch created from the current default/base branch before any code edits are made.
- Build a desktop app that lets a user pick a visual seismic pattern (center point of a 32x32x32 cube), compute its latent token, and search for similar patterns in another 3D seismic volume.
- Keep the UI interactive during setup, browsing, and token selection.
- Run full-volume inference as a non-blocking background process, with a temporary progress section in the left menu tree.
- Add a Python entry point at scripts/tokenize.py.

Constraints:
1. Full-volume inference must run in a background process and never block UI.
2. A temporary progress bar must appear in the left menu tree during background inference.

## 2) Existing Code and Constraints
- Reuse preprocessing behavior from docs/train.py and src/augmentations.py:
  - Input scaling mode: divide by standard deviation for inference path.
  - Extrema-only transform: keep_trace_extrema_only.
  - Validation-like input path should be deterministic (no random training augmentations).
- Reuse VAE architecture from src/model.py.
- Load model weights from checkpoints_gan_vwarp2/vae_final.pt.
- Latent output target: 128-D vector.

## 3) UI Framework Evaluation and Recommendation

### Option A: PySide6 + PyVista/VTK (recommended)
- Pros:
  - Native desktop app with responsive widgets, docking, tree panels, and progress controls.
  - Strong control over background workers via QThread/QRunnable + signals.
  - 2D slice rendering and 3D overlays are practical with VTK-backed views.
  - Best match to user preference.
- Cons:
  - More manual wiring than higher-level viewers.

### Option B: Napari plugin app
- Pros:
  - Very fast time-to-first-view for nD data, built-in layer model.
  - Good for scientific image exploration.
- Cons:
  - Harder to deliver custom workflow UX and app-like left menu tree semantics.
  - Plugin distribution and long-term maintenance complexity.

### Option C: Web UI (FastAPI + React/Plotly)
- Pros:
  - Easy remote deployment.
  - Flexible layout and visual styling.
- Cons:
  - Higher engineering overhead for desktop-like interactive volume inspection.
  - GPU/torch process orchestration and file I/O integration is more complex.

### Preferred Option
- Use PySide6 + PyVista/VTK.
- Rationale: best fit for requested desktop behavior, non-blocking background jobs, and seismic slice/overlay workflows.

## 4) Target Architecture

### App Modules
- scripts/tokenize.py
  - CLI launcher for app mode and optional headless inference mode.
- src/tokenizer/ui/
  - main_window.py: split layout, left menu tree, central viewer, toolbar.
  - controls_input.py: input seismic controls.
  - controls_output.py: output/overlay controls.
  - progress_panel.py: temporary tree item with progress bar, ETA, cancel.
  - viewer.py: inline/crossline displays and token marker overlay.
- src/tokenizer/core/
  - io_zarr.py: load/copy/pad/crop zarr volumes.
  - preprocess.py: scaling + extrema-only transforms.
  - model_infer.py: VAE encoder-only wrapper and latent extraction.
  - similarity.py: cosine similarity and taper utilities.
  - search_engine.py: strided subcube scan and accumulation.
  - jobs.py: background worker, progress callbacks, cancellation.
- src/tokenizer/config/
  - defaults.py: default paths, chunk sizes, and runtime parameters.

### Data Flow Summary
1. User loads source volume, navigates slices, and chooses token center (x, y, z).
2. App extracts 32-cube, applies preprocess, and infers a 128-D token.
3. User selects search volume.
4. App prepares temporary padded zarr and output accumulators.
5. Background worker scans volume in 50% overlap strides and computes similarity map.
6. UI receives progress updates and remains responsive.
7. On completion, output is unpadded/cropped to original shape and shown as overlay.

## 5) Core Algorithm Specification

### Search Token Generation
- Input: picked center voxel in source volume.
- Extract cube size 32x32x32 centered at pick with safe boundary handling.
- Preprocess:
  - Convert to float32.
  - Divide by cube stddev if stddev > epsilon, else use epsilon guard.
  - Apply keep_trace_extrema_only.
- Model inference:
  - Use encoder path equivalent to VAE3D encoder output mu.
  - Token = mu flattened to shape (128,).

### Search Volume Preparation
- Copy search input to /tmp/temp_seismic/input_seismic.zarr.
- Pad each axis:
  - +16 at front and +16 at back.
  - Additional tail padding so padded axis length is divisible by 32.
- Chunking target: [16, 16, -1] semantics (map -1 to full axis length in implementation).
- Create output accumulators in temp storage:
  - similarity_sum (float32)
  - weight_sum (float32)

### Background Inference Loop
- Window size: 32.
- Stride: 16 in x, y, z (50% overlap).
- For each subcube:
  - Preprocess as in token path.
  - Infer latent token.
  - Compute cosine similarity with prompt token.
  - Multiply scalar similarity by 3D Hann taper volume (32x32x32).
  - Accumulate into similarity_sum.
  - Accumulate taper into weight_sum.
- Final output before crop: similarity_sum / clip(weight_sum, min=eps).
- Remove all padding and return original search-volume shape.

## 6) Interactivity and Performance Strategy
- Keep UI thread free of blocking work.
- Run full-volume search in dedicated worker process with message-based progress.
- Batch model inference where possible (micro-batches of subcubes) to improve throughput.
- Precompute Hann taper once and reuse.
- Use torch.no_grad() and eval mode.
- Update UI progress at throttled cadence (for example every 100 windows or 200 ms) to reduce signal overhead.

## 7) UI Specification

### Left Panel (collapsible tree, 25% width)
- Input Seismic:
  - Path textbox + file picker button.
  - Clipping slider.
  - Inline/crossline toggle.
  - Slice-position slider(s).
- Output Seismic:
  - Path textbox + file picker button.
  - Overlay colormap selector (rainbow, perceptual).
  - Background clipping slider.
  - Overlay alpha slider.
  - Inline/crossline toggle and slider(s).
- Search Job (temporary, shown only while running):
  - Progress bar.
  - Percent complete.
  - Windows processed / total.
  - ETA.
  - Cancel button.

### Main Display and Toolbar
- Main canvas supports inline and crossline slice views.
- Toolbar controls:
  - Input/output display mode toggle.
  - Token selection mode toggle.
  - Navigation mode (zoom/pan).
  - Mouse wheel zoom.
  - Keyboard up/down for vertical zoom.
- Token visualization:
  - Red sphere at selected xyz.
  - 32x32x32 wireframe box around token cube.

## 8) Reliability and Error Handling
- Validate paths and zarr keys before enabling inference start.
- Validate model checkpoint compatibility (latent dim and architecture).
- Handle stddev near zero during normalization.
- Graceful cancellation: stop loop, flush output, and report canceled state.
- Crash-safe temp handling: create run-scoped temp dirs and clean up on finish/cancel.
- Maintain session log with timings, config, picked coordinates, and output paths.

## 9) Clarification Checklist (Blockers for Autonomous Execution)
- Autonomous execution must not start implementation until all items below are resolved and recorded.
- Required clarifications:
  - Data source contract: zarr key(s), axis order, dtype, and expected metadata.
  - Coordinate convention: mapping between displayed inline/crossline/depth and storage axes.
  - Token definition: confirm deterministic encoder mu is used as the latent token.
  - Normalization guard: confirm epsilon value used when cube stddev is near zero.
  - Chunking rule: confirm fallback behavior when [16, 16, -1] is not directly supported.
  - Output destination policy: temp-only, user-selected path, or both.
  - UI stack confirmation: PySide6 + PyVista/VTK approved as final choice.
  - Branch policy details: confirm base branch name and feature branch naming convention.
  - Confidence gate policy: confirm >0.50 threshold or replace with stricter threshold.
- Resolution rule:
  - Mark each item as Resolved, Assumed, or Deferred.
  - If any item remains Unresolved, autonomous execution is blocked.
  - If an item is Assumed, log owner and rationale before proceeding.

## 10) Staged Implementation Plan with Author/Critic Gates

### Agent Roles (used in every phase)
- Author Agent:
  - Implements phase scope and tests.
  - Reports predicted probability code runs without errors on first use.
- Constructive Critic Agent:
  - Reviews code, tests, and edge cases.
  - Reports independent predicted probability code runs without errors on first use.
- Exit Gate Rule:
  - If either probability <= 0.50, iterate implementation-review-test loop.
  - Continue until both are > 0.50.
  - Record each iteration and confidence deltas.

### Phase 1: Foundation and Contracts
- Goals:
  - Create module skeleton and config schema.
  - Implement deterministic preprocess utilities.
  - Implement encoder-only model wrapper load from checkpoints_gan_vwarp2/vae_final.pt.
  - Add scripts/tokenize.py with basic argument parsing and app bootstrap.
- Tests:
  - Unit: preprocess scaling/extrema behavior matches expected shape/type and deterministic outputs.
  - Unit: encoder wrapper loads checkpoint and emits 128-D latent.
  - Unit: temp path/config validation and defaults.
- Deliverables:
  - Working bootstrap app window.
  - Passing unit tests for preprocessing and model-load path.

### Phase 2: Data I/O and Token Picking UI
- Goals:
  - Implement volume load, slicing, clipping controls, and token pick interaction.
  - Render red marker + 32-cube wireframe.
  - Generate and display token stats for selected location.
- Tests:
  - Unit: cube extraction boundaries and padding edge behavior.
  - Unit: token generation pipeline for known synthetic cube.
  - UI smoke test: input volume load, slider updates, and pick event signal emission.
- Deliverables:
  - User can pick a token from input volume with visible confirmation.

### Phase 3: Search Preparation and Background Job Infrastructure
- Goals:
  - Implement temp zarr copy/padding to /tmp/temp_seismic/input_seismic.zarr.
  - Build cancellable worker job framework.
  - Add temporary progress tree item and binding to worker progress events.
- Tests:
  - Unit: axis padding math (+16/+16 + divisible-by-32 tail pad).
  - Unit: chunk strategy mapping for [16, 16, -1].
  - Integration: worker lifecycle (start/progress/cancel/finish) without UI freeze.
- Deliverables:
  - Non-blocking background framework ready for inference loop.

### Phase 4: Full Similarity Inference Engine
- Goals:
  - Implement sliding-window search loop (stride 16).
  - Add cosine similarity and Hann-weighted accumulation.
  - Produce cropped final output aligned to original search volume shape.
  - Wire overlay rendering in output display controls.
- Tests:
  - Unit: cosine similarity correctness for synthetic latent vectors.
  - Unit: Hann taper shape/value sanity and accumulation normalization.
  - Integration: mini-volume inference regression test with deterministic expected output stats.
  - Performance smoke: verify UI remains responsive during running job.
- Deliverables:
  - End-to-end search result shown as overlay.

### Phase 5: Optimization and Stability Hardening
- Goals:
  - Add micro-batching and optional device tuning (cpu/cuda/mps).
  - Throttle progress events and optimize memory usage.
  - Improve error messages and recovery states.
- Tests:
  - Benchmark tests for throughput and memory envelope on representative mini-volumes.
  - Regression tests for cancellation, restart, and repeated runs.
  - UI smoke test for progress panel add/remove behavior.
- Deliverables:
  - Stable interactive behavior with improved throughput and robust cancellation.

### Phase 6: Final E2E Review, Testing, and Documentation
- Goals:
  - Conduct full end-to-end review and acceptance testing.
  - Validate user workflow: load input, pick token, run search, inspect overlay, cancel/restart.
- Required outputs:
  - End-to-end review report.
  - End-to-end test suite and run results.
  - Code description (module map and runtime flow).
  - Decision log documenting major choices and rationale.
  - User guide (install, launch, operate, troubleshoot).
  - Future work suggestions (distributed inference, multi-token search, cached latent volumes, advanced uncertainty maps).
- Tests:
  - E2E automated test on small known dataset.
  - Manual acceptance checklist for UI and background processing behavior.
- Final Gate:
  - Author and Critic both report > 0.50 first-use success probability.
  - No P0/P1 open defects.

## 11) Definition of Done
- App launches from scripts/tokenize.py.
- User can pick token on source volume and run non-blocking full-volume search.
- Progress appears in temporary left-tree section and disappears on completion/cancel.
- Output overlay aligns with searched volume and supports colormap/alpha controls.
- Tests exist for every phase and pass in CI/local test run.
- Final phase documentation package is complete.

## 12) Branch Setup Checklist
- Create a new branch from base branch before edits:
```bash
git fetch origin
git checkout main
git pull --ff-only
git checkout -b feat/seismic-tokenizer-app
```
- Verify branch and status:
```bash
git branch --show-current
git status
```
- Push branch to remote:
```bash
git push -u origin feat/seismic-tokenizer-app
```
- Open pull request:
```bash
gh pr create --base main --head feat/seismic-tokenizer-app --title "Seismic tokenizer app" --body "Implements staged seismic tokenizer application plan."
```
- Author/Critic workflow note:
  - Both agents work only on this feature branch until all phase gates pass.
