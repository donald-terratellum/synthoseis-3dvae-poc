# Seismic Tokenizer Decisions and Rationale

## 1) UI framework: PySide6 desktop app
Decision:
- Use PySide6 as the primary application framework.

Rationale:
- Matches requested preference.
- Strong event/signal model for non-blocking background processing.
- Good fit for multi-pane controls and interactive scientific workflows.

## 2) Seismic data format: zarr-based workflow
Decision:
- Use zarr for source/search/output volumes and temporary padded artifacts.

Rationale:
- Existing repository already uses zarr.
- Efficient chunked I/O and straightforward interoperability with NumPy.

## 3) Token preprocessing parity
Decision:
- Keep token/search preprocessing aligned with training assumptions:
  - std normalization
  - extrema-only transform.

Rationale:
- Reduces train-infer mismatch risk.
- Keeps latent token semantics consistent with existing model usage.

## 4) Window geometry
Decision:
- Use `patch=32` and `stride=16` (50% overlap).

Rationale:
- Matches stated project requirements.
- Overlap + Hann taper reduces block artifacts.

## 5) Similarity accumulation
Decision:
- Fill each window with cosine scalar and apply 3D Hann taper before accumulation.
- Normalize by accumulated taper weights.

Rationale:
- Produces smooth similarity field with overlap blending.
- Numerically stable and deterministic.

## 6) Background execution model
Decision:
- Use process-based worker runner (`spawn`) with event queues.

Rationale:
- Keeps UI responsive under heavy compute.
- Better isolation for model inference and cancellation handling.

## 7) Latent adapters
Decision:
- Support two latent modes:
  - `vae` (checkpoint-backed encoder `mu`)
  - `pooled` deterministic fallback.

Rationale:
- `vae` provides intended model behavior.
- `pooled` enables deterministic fallback for testing and failure resilience.

## 8) Adaptive batching
Decision:
- Add adaptive batch fallback on retryable memory errors.

Rationale:
- Improves stability across heterogeneous devices and memory limits.
- Avoids hard failures on large default batch size.

## 9) Output cleanup policy
Decision:
- Add cleanup helper for output artifacts and support config for keeping partial output.

Rationale:
- Improves recovery semantics after cancel/error.
- Prevents stale artifacts from misleading users.

## 10) Test strategy
Decision:
- Keep broad unittest coverage across core math, I/O, worker lifecycle, UI signal/state behavior, and optimization paths.

Rationale:
- Supports incremental phased delivery.
- Catches regressions while moving quickly.
