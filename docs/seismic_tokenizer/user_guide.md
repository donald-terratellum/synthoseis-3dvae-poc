# Seismic Tokenizer User Guide

## Prerequisites
1. Use the project uv environment.
2. Ensure dependencies are synced:
```bash
uv sync
```

## Quick start

### 1) Build a token from a source volume
```bash
uv run python scripts/tokenize.py build-token \
  --source /path/to/source.zarr \
  --x 100 --y 120 --z 80 \
  --latent-mode vae
```

Optional useful flags:
- `--key <array_key>`
- `--model-path <checkpoint.pt>`
- `--device auto|cpu|cuda|mps`
- `--latent-mode vae|pooled`

### 2) Run full search-volume inference
```bash
uv run python scripts/tokenize.py search-volume \
  --source /path/to/source.zarr \
  --search /path/to/search.zarr \
  --output /path/to/output_similarity.zarr \
  --latent-mode vae
```

Optional useful flags:
- `--source-key <array_key>`
- `--search-key <array_key>`
- `--temp-zarr /tmp/temp_seismic/input_seismic.zarr`
- `--patch-size 32`
- `--stride 16`
- `--batch-size 32`
- `--x --y --z` (token center override)
- `--benchmark-json /path/to/benchmark.json`

### 2b) Write benchmark report JSON
```bash
uv run python scripts/tokenize.py search-volume \
  --source /path/to/source.zarr \
  --search /path/to/search.zarr \
  --output /path/to/output_similarity.zarr \
  --latent-mode pooled \
  --benchmark-json /tmp/tokenizer_benchmark.json
```

Benchmark JSON includes:
- `elapsed_s`
- `windows_per_sec`
- `total_windows`
- `patch_size`, `stride`, `batch_size`
- `latent_mode`, `device`
- shape metadata (`source_shape`, `search_shape`, `padded_shape`, `output_shape`)
- output stats (`output_min`, `output_max`, `output_mean`, `output_std`)

### 3) Launch the UI
```bash
uv run python scripts/tokenize.py ui --source /path/to/source.zarr
```

## UI workflow
1. Load source volume.
2. Move slice/display controls.
3. Pick token location.
4. Start background search.
5. Watch progress panel and status messages.
6. Load or auto-load output similarity overlay.
7. Adjust output clip and alpha controls.

## Interpreting results
- Higher cosine similarity means stronger latent-pattern match to the selected token.
- Overlay clip/alpha change how strongly output is represented in preview state.
- Overlap + Hann taper smoothing reduces seam artifacts.

## Troubleshooting

### UI does not launch
- Verify PySide6 is installed:
```bash
uv sync
```

### Model load failure
- Check checkpoint path and compatibility with `VAE3D` architecture.
- Try fallback mode:
```bash
--latent-mode pooled
```

### Out-of-memory behavior
- Reduce `--batch-size`.
- Keep `--device auto` unless forcing a specific backend is required.
- Adaptive batching should automatically back off on retryable memory errors.

### Benchmark report is missing
- Confirm `--benchmark-json` path is writable.
- Verify command completed without fatal error.
- If canceled, worker-mode benchmark reports may be emitted from background execution paths.

### Missing/incorrect zarr key
- Provide explicit `--source-key` or `--search-key`.

### Shape mismatch on output load
- Output overlay volume must match source/search shape used for UI overlay.

## Testing commands
Tokenizer-focused suite:
```bash
uv run python -m unittest \
  tests/test_tokenizer_config.py \
  tests/test_tokenizer_events.py \
  tests/test_tokenize_cli.py \
  tests/test_tokenizer_token_picker.py \
  tests/test_tokenizer_phase3_core.py \
  tests/test_tokenizer_phase3_search_volume.py \
  tests/test_tokenizer_phase4_engine.py \
  tests/test_tokenizer_phase5_optimization.py \
  tests/test_tokenizer_regression.py \
  tests/test_tokenizer_ui_smoke.py
```
