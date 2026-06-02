# synthoseis-3dvae-poc

This repository has two components:

1. VAE training component (data preparation + model training): [docs/training/README.md](docs/training/README.md)
2. Pattern search application component: [docs/seismic_tokenizer/README.md](docs/seismic_tokenizer/README.md)

## Environment

```bash
uv sync
```

## Notes

- The training component provides dataset preparation and checkpoint generation.
- The application component uses the trained model for interactive pattern-search workflows.

## Repository Map

- [docs/training/README.md](docs/training/README.md): Training component overview, quickstart, and key files.
- [docs/training/latent_alignment_experiments.md](docs/training/latent_alignment_experiments.md): Latent alignment theory, usage, and ablation guide for detail-focused VAE training.
- [docs/seismic_tokenizer/README.md](docs/seismic_tokenizer/README.md): Application component overview and entrypoints.
- [docs/seismic_tokenizer/user_guide.md](docs/seismic_tokenizer/user_guide.md): End-user commands, UI workflow, troubleshooting, and test commands.
- [docs/seismic_tokenizer/code_description.md](docs/seismic_tokenizer/code_description.md): Architecture and module responsibilities.
- [scripts/sample_patches.py](scripts/sample_patches.py): Build training patch datasets from seismic sources.
- [scripts/train.py](scripts/train.py): VAE training CLI and training loop implementation.
- [scripts/tokenize.py](scripts/tokenize.py): Pattern-search CLI (`build-token`, `search-volume`, `ui`).