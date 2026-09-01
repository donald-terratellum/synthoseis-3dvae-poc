# Session Summary - 2026-08-31 — Tokenizer z_geo Wiring

## Who made changes

- Primary driver: Donald (direction, decision)
- Implementation partner: GitHub Copilot (code, tests, verification)

## Why

The geology-aware training track selected a winning checkpoint
(`checkpoints/geoaware_v3_phase2_20260831/vae_epoch20.pt`) whose geology projection head
produces a unit-norm `z_geo` embedding optimized for ranking geologically-similar patches
(neighbor_overlap@5 0.139 vs 0.077 `mu` baseline). The `seismic_tokenizer` app, however, still
defaulted to an old checkpoint and always used the raw VAE latent mean (`mu`) for similarity
search. This session wires the app to the winning checkpoint and routes similarity through
`z_geo`, with a toggle to fall back to `mu`.

## What changed

### 1. Default checkpoint + new `embedding_mode` config
[src/tokenizer/config/defaults.py](../../src/tokenizer/config/defaults.py)
- `DEFAULT_MODEL_PATH` → `checkpoints/geoaware_v3_phase2_20260831/vae_epoch20.pt`
  (was `checkpoints_gan_vwarp2/vae_final.pt`).
- Added `RuntimeConfig.embedding_mode: str = "z_geo"` plus validation that it is `"mu"` or
  `"z_geo"`.

### 2. CLI: `--embedding-mode {mu,z_geo}` and routing
[scripts/tokenize.py](../../scripts/tokenize.py)
- Added `--embedding-mode` (default `z_geo`) to `build-token`, `search-volume`, and `ui`.
- `build-token` and `search-volume` now route the token/latent encode through
  `adapter.encode_geo_cube` / `adapter.encode_geo_batch` when `z_geo`, else the original
  `encode_cube` / `encode_batch` (mu).
- `search-volume` benchmark JSON now records `embedding_mode`.
- `ui` passes `embedding_mode` to the controller.

### 3. Background search worker
[src/tokenizer/core/jobs.py](../../src/tokenizer/core/jobs.py)
- `SearchExecutionSpec` gains `embedding_mode: str = "z_geo"`.
- The VAE branch of `_search_worker` selects `adapter.encode_geo_batch` vs `adapter.encode_batch`
  based on `spec.embedding_mode`.

### 4. Desktop UI controller
[src/tokenizer/ui/controller.py](../../src/tokenizer/ui/controller.py)
- `TokenizerController.__init__` gains an `embedding_mode` param (falls back to
  `RuntimeConfig.embedding_mode`).
- Token latent build routes `encode_geo_cube` vs `encode_cube`; `embedding_mode` is passed into
  `SearchExecutionSpec`.

## How it works

The `VaeLatentAdapter` (in `src/tokenizer/core/model_adapter.py`) already exposed
`encode_geo_batch` / `encode_geo_cube`, which run the encoder to `mu` then the trained geology
head to a unit-norm `z_geo` (falling back to L2-normalized `mu` when a checkpoint has no head).
This session added a single `embedding_mode` switch threaded through every similarity entry point
(CLI build-token, CLI search-volume, background job worker, desktop UI) so retrieval uses `z_geo`
by default and can be flipped back to `mu` for comparison. Cosine similarity over unit-norm
`z_geo` is exactly the geological-similarity ranking the training track optimized.

## Verification

- `.venv/bin/python -m unittest tests.test_geology_contrastive tests.test_tokenizer_phase4_engine`
  → 21 tests pass.
- `get_errors` on all four edited files → none.
- Smoke test: `RuntimeConfig()` now defaults to the Phase 2 checkpoint with
  `embedding_mode="z_geo"`; `VaeLatentAdapter` loads it with `geology_projection=True`,
  `patch_shape=(32,32,64)`, and `encode_geo_cube` returns a 64-dim unit-norm vector
  (norm 1.0); `encode_cube` returns 128-dim `mu`.
- `scripts/tokenize.py build-token --help` shows the new `--embedding-mode {mu,z_geo}` flag.

## Artifacts / commits

- Commit `240f76d`: "Wire geology z_geo embeddings into tokenizer app; default to Phase 2
  epoch-20 checkpoint" (this wiring).
- Related same-day commits: `483583a` (Phase 2d resolved doc), `24669d7` +
  `d97b4fa` (improvement plan `docs/training/2026-09-01_geo-aware_improvement_plan.md`).

## Follow-ups

- Optional end-to-end check on a real volume: run `scripts/tokenize.py search-volume` with
  `--embedding-mode z_geo` and confirm the output similarity map ranks geologically-similar
  regions highly (the frozen benchmark already validates the embedding quality offline).
- Future retrieval gains are structural, not more training — see
  `docs/training/2026-09-01_geo-aware_improvement_plan.md` (richer labels + hard mining + bigger
  projection head are the highest-ROI levers).
</content>
