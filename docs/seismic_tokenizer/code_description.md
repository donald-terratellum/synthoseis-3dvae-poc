# Seismic Tokenizer Code Description

## Purpose
This document describes the implemented seismic tokenizer app architecture, runtime flow, and module responsibilities.

## High-level runtime flow
1. Source seismic volume is loaded in the UI.
2. User selects token center `(x, y, z)` (or defaults to current display center).
3. A `32x32x32` cube is extracted and preprocessed (std scaling + extrema-only).
4. Token latent is computed using either:
   - VAE encoder (`mu`) from checkpoint, or
   - pooled fallback latent mode.
5. Search volume is padded and scanned with overlap (`patch=32`, `stride=16`).
6. Cosine similarity is computed per window latent vs token latent.
7. Similarity values are blended into output using 3D Hann taper weighting.
8. Output is unpadded, written to zarr, and loaded as overlay in UI.
9. Background worker streams progress events to the UI progress panel.

## Module map

### Entry points
- `scripts/tokenize.py`
  - CLI commands:
    - `build-token`
    - `search-volume`
    - `ui`

### Core pipeline modules
- `src/tokenizer/core/io_zarr.py`
  - zarr array opening/loading
  - centered cube extraction with boundary-safe zero padding
  - axis/volume padding math
  - temp padded zarr preparation
  - padding removal
- `src/tokenizer/core/preprocess.py`
  - per-cube std normalization
  - extrema-only transform integration
- `src/tokenizer/core/model_adapter.py`
  - deterministic pooled latent adapter
  - VAE encoder checkpoint adapter (`VaeLatentAdapter`)
- `src/tokenizer/core/similarity.py`
  - cosine similarity
  - 3D Hann taper generation
- `src/tokenizer/core/search_engine.py`
  - overlap window iteration
  - batched latent inference path
  - weighted accumulation and normalization
  - progress/cancel callbacks
- `src/tokenizer/core/jobs.py`
  - spawn-based worker execution
  - progress/status/error/artifact event emission
  - cancel handling
  - runtime metrics emission (elapsed and windows/sec)
  - output artifact cleanup helper
- `src/tokenizer/core/batching.py`
  - adaptive batch fallback on retryable memory errors
- `src/tokenizer/core/metrics.py`
  - benchmark report JSON writer used by CLI/worker hooks

### UI modules
- `src/tokenizer/ui/main_window.py`
  - source/output path controls
  - display controls (inline/crossline/z, clip, alpha)
  - search progress/cancel panel
  - Qt signals for source load, output load, token pick, search start/cancel
- `src/tokenizer/ui/controller.py`
  - connects UI signals to core actions
  - source/output load handling
  - display-state synchronization and overlay preview stats
  - background job lifecycle and event polling
- `src/tokenizer/ui/state.py`
  - display and overlay preview state model

## Background process event contract
Worker emits `JobEvent` with kinds:
- `progress`
- `status`
- `artifact`
- `done`
- `error`

UI controller behavior:
- updates progress bar from `progress`
- updates status text from `status`
- loads output zarr from `artifact`
- hides progress panel on `done` or `error`

## Test coverage summary
Current suite validates:
- config/event schema
- CLI smoke behavior
- token extraction/preprocessing behavior
- padding/chunk/window math
- search-volume integration outputs
- engine cosine/Hann/overlap behavior
- UI signal, state sync, progress lifecycle
- optimization and cancellation stability paths
- deterministic regression consistency checks across repeated runs

## Known current boundary
The UI currently exposes overlay preview statistics/state rather than a full rendered seismic overlay canvas. The computational pipeline and state wiring are in place for deeper viewer integration.
