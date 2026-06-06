# Pattern Search Application Component

Interactive seismic pattern search application built on the trained VAE latent space.

## What it does
1. Lets a user pick a token location in a source volume.
2. Converts the selected 32x32x32 cube to a latent token.
3. Searches another seismic volume with overlapping windows.
4. Produces cosine-similarity output and exposes overlay controls.
5. Runs full-volume search in a background process with progress updates.
6. Renders inline/crossline/depth slice views with a token marker in the UI.

## Entry point
```bash
uv run python scripts/tokenize.py ui --source /path/to/source.zarr
```

Optional UI flags:
- `--latent-mode vae|pooled`
- `--model-path /path/to/checkpoint.pt` (used when `--latent-mode vae`)
- `--device auto|cpu|cuda|mps`

## CLI modes
```bash
uv run python scripts/tokenize.py build-token ...
uv run python scripts/tokenize.py search-volume ...
uv run python scripts/tokenize.py ui ...
```

## More docs
- docs/seismic_tokenizer/user_guide.md
- docs/seismic_tokenizer/code_description.md
- docs/seismic_tokenizer/decisions_and_rationale.md
- docs/seismic_tokenizer/future_work.md
