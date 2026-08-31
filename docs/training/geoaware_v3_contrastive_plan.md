# Geo-Aware v3: Contrastive Latent Clustering Plan

Goal: make the VAE latent space cluster geologically similar seismic patches so that
cosine similarity between two patch embeddings is a reliable measure of geological
similarity for the `seismic_tokenizer` retrieval workflow (target patch vs. a whole 3D
volume).

## Why v2 plateaued (evidence)

From the frozen 512-sample benchmark (baseline vs. `glw=0.07` vs. `glw=0.10`):

| metric | baseline | runA (0.07) | runB (0.10) |
| --- | --- | --- | --- |
| cosine_separation | 0.01608 | 0.01672 | 0.01330 |
| pair_cosine_correlation | 0.03898 | 0.03964 | 0.03208 |
| neighbor_overlap@5 | 0.02885 | 0.02212 | 0.02404 |
| NDCG@10 | 0.4155 | 0.4192 | 0.4122 |

Raising the geology-loss weight did not help; `0.10` hurt. Root causes:

1. Similarity is read off raw `mu` (128-d), which the reconstruction objective
   (MAE + LPIPS + KL) dominates. Geology weight 0.03-0.10 is far too small to reshape a
   space reconstruction owns.
2. The v2 geology loss regresses the latent pairwise-cosine matrix onto the metadata
   pairwise-cosine matrix. L2-normalized 7-15-d metadata yields a narrow, low-contrast
   target, so gradients push weakly and symmetrically (no margin).
3. No projection head, so the geology objective fights reconstruction on `mu`. The
   earlier latent-alignment loss (2026-06-02) was removed for the same reason.
4. Similarity target collapses structural / stratigraphic / lithologic families into one
   cosine.
5. Sampler background/hard quotas overshoot targets, diluting positive/negative
   structure (fixed separately in `src/geology_sampler.py`).

## Design

Add a dedicated **projection head** `g(mu) -> z_geo` (L2-normalized) and train it with a
**supervised contrastive (SupCon)** objective on strata labels. `mu` stays reconstruction-
only; `z_geo` becomes the geology-similarity embedding used by the tokenizer app.

```
x --Encoder--> mu, logvar --reparam--> z --Decoder--> recon   (reconstruction path, unchanged)
                 |
                 +--proj head g(.)--> z_geo (unit norm) --> SupCon loss / app similarity
```

Key properties:

- The head is separate, so we can use a large contrastive weight without harming
  reconstruction (fixes root causes 1 and 3).
- SupCon gives an explicit pull-positives / push-negatives margin (fixes 2).
- The tokenizer switches similarity search from `mu` to `z_geo` (`encode_geo_embedding`).

## Workstreams (priority order)

### W1 - Projection head + SupCon (core)
- `GeologyProjectionHead`: MLP `latent_dim -> proj_hidden -> proj_dim`, GELU, final
  L2-normalize. Default `proj_hidden=128`, `proj_dim=64`.
- `VAE3D` gains optional head + `encode_geo(mu)` returning unit `z_geo`.
- SupCon loss on `z_geo` using strata labels from `build_multilabel_strata`; temperature
  ~0.1; positives = same stratum, negatives = other strata (hard negatives from sampler).
- CLI: `--geology_projection`, `--geology_proj_dim`, `--geology_proj_hidden`,
  `--geology_contrastive_weight`, `--geology_contrastive_temperature`.

### W2 - Better similarity target
- Start with strata-label SupCon.
- Then add continuous per-family soft targets (structural / stratigraphic / lithologic)
  via RBF kernel on calibrated metadata distance; optionally multiple heads the app can
  weight.

### W3 - Training protocol so the encoder moves
- Warm-start from best reconstruction checkpoint, then two phases:
  - Phase 1: freeze encoder, train head only (high contrastive weight).
  - Phase 2: unfreeze encoder at small LR with reconstruction anchor (keep MAE/LPIPS/KL).
- Keep `kl_end=1e-3` so `mu` is not over-compressed.

### W4 - Metric-learning hygiene
- Sampler quota fix (done) so background/hard fractions hold.
- Hard-negative mining on `z_geo`, temperature tuning, cross-batch negatives / memory bank
  (batch size is only 12).

### W5 - Evaluation upgrades
- Extend `scripts/evaluate_geology_benchmark.py` to also report silhouette score on strata
  and per-family retrieval, plus a UMAP/t-SNE dump.
- KPIs to move: `neighbor_overlap@5/10`, `cosine_separation`, per-family precision@k.
- Guardrail: `mu`-based reconstruction `val_loss` unchanged (geology loss no longer
  perturbs `mu`).

## Experiment sequence

1. Implement W1 (head + SupCon), embeddings normalized, temperature ~0.1.
2. Phase-1 head-only train ~15 epochs from best checkpoint; benchmark on `z_geo`.
3. Phase-2 encoder fine-tune ~20 epochs with reconstruction anchor; benchmark again.
4. Add per-family target (W2); compare.
5. Wire `z_geo` into the tokenizer adapter; re-run the app qualitatively.

Decision rule per step: keep if `neighbor_overlap@5` and `cosine_separation` improve
materially (target n@5 > ~0.10, separation > ~0.05) with the reconstruction guardrail
intact.

## Runnable commands (implemented)

Phase 1 - head-only contrastive training from the best reconstruction checkpoint:

```bash
uv run python scripts/train.py \
  --data data/synth_train_32-32-64.zarr \
  --validation_data data/synth_val_32-32-64.zarr \
  --patch_size 32 32 64 \
  --resume checkpoints/vae_best.pt \
  --geology_projection \
  --geology_proj_hidden 128 \
  --geology_proj_dim 64 \
  --geology_contrastive_weight 1.0 \
  --geology_contrastive_temperature 0.1 \
  --geology_batch_sampler \
  --freeze_encoder \
  --epochs 15 \
  --seed 20260826 \
  --out_dir checkpoints/geoaware_v3_phase1
```

Phase 2 - unfreeze encoder at small LR with the reconstruction anchor retained (drop
`--freeze_encoder`, lower `--encoder_lr_mult`, keep reconstruction/LPIPS/KL as usual).

Benchmark the geology embedding (`z_geo`) on the frozen manifest:

```bash
uv run python scripts/evaluate_geology_benchmark.py \
  --data data/synth_val_32-32-64.zarr \
  --checkpoint checkpoints/geoaware_v3_phase1/vae_best.pt \
  --use_geo_embedding \
  --out_json docs/benchmarks/geoaware_v3_phase1_zgeo_report.json
```

