# Session Summary - 2026-08-31

## Who made changes

- Primary driver: Donald (experiment direction, variant decisions)
- Implementation partner: GitHub Copilot (code, tests, benchmarking, analysis)

## Goal (why this work exists)

Ultimate objective: use the VAE encoder's latent embedding of two seismic patches,
compute cosine similarity between them to judge **geological similarity**, and rank the
most-similar patches throughout a 3D seismic volume inside the `seismic_tokenizer` app.

This session's job: run a staged sequence of geology-aware contrastive-training
experiments on top of the geology projection head (`z_geo`), benchmark each on a frozen
retrieval manifest, and pick the best variant to carry forward.

## Key outcome / DECISION

**Winner: Phase 2 epoch-20** — `checkpoints/geoaware_v3_phase2_20260831/vae_epoch20.pt`

- It is the best model for the *actual* goal (rank retrieval of geologically similar
  patches). It is a complete, fully-trained checkpoint — ready to use now.
- The tokenizer adapter already supports its `z_geo` head via `encode_geo` /
  `encode_geo_batch` / `encode_geo_cube`; select it in the benchmark with
  `--use_geo_embedding`.

### Benchmark comparison (7-key frozen manifest, `z_geo` unless noted)

| metric | mu (baseline) | **P2 e20 (WINNER)** | P2b e20 | P2c e10 | P2c e14 |
|---|---|---|---|---|---|
| neighbor_overlap@5  | 0.077 | **0.139** | 0.092 | 0.100 | 0.085 |
| neighbor_overlap@10 | 0.181 | **0.227** | 0.192 | 0.187 | 0.210 |
| neighbor_overlap@20 | 0.397 | 0.402 | 0.407 | 0.396 | 0.408 |
| cosine_separation   | −0.014 | −0.009 | −0.017 | **+0.012** | **+0.015** |

Reconstruction guardrail for the winner: `val_loss` ≈ 0.128 (unharmed).

## Phases run this session

1. **Phase 1** — frozen encoder + decoder, head-only probe (15 epochs).
   - `--freeze_encoder --freeze_decoder`, contrastive_weight 1.0, temp 0.1.
   - Result: FAILED bar — n@5 0.085, latent cone collapsed. Head alone can't fix it.
2. **Phase 2** — unfrozen encoder (`encoder_lr_mult` small), contrastive only (20 epochs).
   - Result: **BEST** — n@5 0.139, n@10 0.227, val_loss best 0.128. This is the deliverable.
3. **Phase 2b** — added hypersphere uniformity regularizer at weight **0.5** (30 epochs).
   - Result: REGRESSION — uniformity broke the cone (cosines 0.84→~0.00) but DESTROYED
     rank retrieval (n@5 back to ~0.077–0.092). Over-spread.
4. **Phase 2c** — tiny uniformity weight **0.05**, temp 0.2, `encoder_lr_mult` 0.3,
     warm-started from Phase 2 epoch-20 (30 epochs; still running at session end,
     ~epoch 17/30, val_loss best ~0.126).
   - Partial result: FIRST-ever **positive** `cosine_separation` (+0.012…+0.015) — the
     tiny uniformity did exactly what it was designed to do — but rank retrieval at
     epochs 10/14 (n@5 0.085–0.10) still trailed Phase 2. Promising direction, not yet a win.

## Features implemented this session

### Hypersphere uniformity regularizer (anti-collapse) — committed & pushed (`b92624b`)

- `scripts/train.py`: `compute_uniformity_loss(embeddings, labels=None, t=2.0, ignore_label=0)`
  - Wang & Isola uniformity on the unit hypersphere; **excludes background label**.
  - Pairwise squared Euclidean distances via **Gram matrix**
    (`sq_norms.unsqueeze(1)+sq_norms.unsqueeze(0)-2*gram).clamp_min(0`), upper triangle
    via `triu_indices`; `loss = logsumexp(-t*sq_dists) - log(num_pairs)`.
  - Returns a zero scalar if <2 valid samples or non-finite (safe no-op).
  - **MPS note:** `torch.pdist` is NOT implemented on MPS — the Gram-matrix formulation
    is required. Do not reintroduce `pdist`.
- Wired into `train_one_epoch`: new params `geology_uniformity_weight=0.0`,
  `geology_uniformity_t=2.0`; computed once alongside SupCon on `z_geo`; TensorBoard
  scalar `train/geology_uniformity_loss`.
- CLI flags: `--geology_uniformity_weight` (default 0.0), `--geology_uniformity_t`
  (default 2.0). Guard: `--geology_uniformity_weight > 0` requires `--geology_projection`.
- Tests: `tests/test_geology_contrastive.py::UniformityLossTests` (4 tests: collapsed>spread,
  background exclusion, single-valid-sample zero, differentiable). 15 tests total, all pass.

### Earlier safety fixes — committed & pushed (`fbe6f91`)

- Resume allowlist tolerates `geology_head.` prefix.
- Fully-frozen encoder/decoder are set to `eval()` in `train_one_epoch`.

## Key lessons (do not relearn the hard way)

- **cosine_separation ≠ neighbor_overlap.** Absolute contrast (separation) and rank
  retrieval (neighbor overlap) are different metrics. The user's real goal is **rank
  retrieval**. Phase 2's "collapsed cone" was actually fine for ranking. Chasing
  `cosine_separation` via strong uniformity optimized the wrong metric and hurt the goal.
- Uniformity weight 0.5 = over-spread (kills rank retrieval). Weight 0.05 = gentle
  (recovers positive separation, rank retrieval still recovering). The sweet spot for
  keeping BOTH is between 0 and ~0.05 and needs more epochs to judge.
- `vae_best.pt` is selected by **reconstruction val_loss**, NOT geology — always pick the
  geology deliverable by benchmarking specific epoch checkpoints, not by trusting `best`.

## Environment / reproducibility facts

- Mac mini, `device=mps`. Python 3.13 via `.venv/bin/python`. ~5 min/epoch at
  `batch_size=12 number_batches=450`. Tests: `.venv/bin/python -m unittest` (pytest NOT installed).
- Machine timezone: CDT (Central).
- **Metadata keys** (all geology runs use the 7 BACKGROUND keys to match the Run A
  calibration baked into the resume checkpoint — the 15-key default MISMATCHES and errors):
  `meta_fault_fraction meta_fault_intersection_fraction meta_channel_fraction
  meta_channel_core_fraction meta_flat_spot_fraction meta_onlap_fraction
  meta_onlap_variability`
- zsh: unquoted vars do NOT word-split — pass metadata keys inline.
- Benchmark JSONs under `docs/benchmarks/` are gitignored (do not commit them).
- Git branch `encoder-improvement-2026-08-21`, remote `origin`. Pushed through `b92624b`.

---

## PATH FORWARD (for a future autonomous Copilot agent)

You may execute the following without further clarification. Read
`/memories/session/geoaware_v3_progress.md` and this file first for full context.

### Immediate adoption (already-decided deliverable)

- The chosen model is `checkpoints/geoaware_v3_phase2_20260831/vae_epoch20.pt`.
- Verify it still benchmarks at n@5 ≈ 0.139 before wiring it into the app:

```bash
cd /Users/donaldpg/synthoseis-3dvae-poc
.venv/bin/python scripts/evaluate_geology_benchmark.py \
  --checkpoint checkpoints/geoaware_v3_phase2_20260831/vae_epoch20.pt \
  --use_geo_embedding \
  --metadata_keys meta_fault_fraction meta_fault_intersection_fraction \
    meta_channel_fraction meta_channel_core_fraction meta_flat_spot_fraction \
    meta_onlap_fraction meta_onlap_variability \
  --output docs/benchmarks/geoaware_v3_phase2_ep20_zgeo_verify.json
```

- Then wire `z_geo` retrieval into the `seismic_tokenizer` app via
  `VaeLatentAdapter.encode_geo_batch` / `encode_geo_cube` (adapter already supports it).

### Task A — Finish + evaluate Phase 2c (cheap, high-value)

Phase 2c was still training at session end (out_dir
`checkpoints/geoaware_v3_phase2c_20260831`, log `/tmp/geoaware_v3_phase2c.log`, 30 epochs).
When it finishes, benchmark its later epochs (20, 25, 30) with `--use_geo_embedding` and
the 7 keys above. **Decision rule:** Phase 2c only supersedes the winner if a later epoch
reaches `neighbor_overlap@5 ≥ 0.139` AND keeps `val_loss ≲ 0.13`. If it also holds
`cosine_separation ≥ 0`, prefer it (better absolute contrast for cosine thresholds in the app).

### Task B — Low-uniformity sweep (the promising frontier)

Phase 2c showed tiny uniformity recovers positive separation. Sweep uniformity weight in
{0.0, 0.02, 0.05} warm-started from Phase 2 epoch-20, longer (40 epochs), and pick the
epoch that maximizes `neighbor_overlap@5` subject to `cosine_separation ≥ 0`. Template
(fill in `WEIGHT`):

```bash
cd /Users/donaldpg/synthoseis-3dvae-poc
PYTHONUNBUFFERED=1 .venv/bin/python scripts/train.py \
  --data data/synth_train_32-32-64.zarr \
  --validation_data data/synth_val_32-32-64.zarr \
  --patch_size 32 32 64 --batch_size 12 --number_batches 450 \
  --epochs 40 --seed 20260826 \
  --resume checkpoints/geoaware_v3_phase2_20260831/vae_epoch20.pt --resume_epoch 0 \
  --geology_metadata_keys meta_fault_fraction meta_fault_intersection_fraction \
    meta_channel_fraction meta_channel_core_fraction meta_flat_spot_fraction \
    meta_onlap_fraction meta_onlap_variability \
  --geology_projection --geology_proj_hidden 128 --geology_proj_dim 64 \
  --geology_contrastive_weight 0.5 --geology_contrastive_temperature 0.2 \
  --geology_uniformity_weight WEIGHT --geology_uniformity_t 2.0 \
  --geology_batch_sampler --geology_batch_background_fraction 0.05 \
  --geology_batch_hard_fraction 0.30 --encoder_lr_mult 0.3 \
  --augment --vertical_warp_prob 0.5 --mixup_augment_prob 0.0 \
  --early_stopping_patience 999 --save_epoch_checkpoints \
  --out_dir checkpoints/geoaware_v3_uniformity_sweep_WEIGHT_20260901 \
  2>&1 | tee /tmp/geoaware_v3_uniformity_sweep_WEIGHT.log
```

### Task C — "Train fully" the winning recipe (if scaling up)

The winner is already a full training. To extend it, continue Phase 2's contrastive-only
recipe (no uniformity, or ≤0.02) from `vae_epoch20.pt` for more epochs and re-benchmark
each epoch; keep the epoch with the highest `neighbor_overlap@5`.

### Selection metric (authoritative)

Rank by `diagnostics.neighbor_overlap_at_5` on the frozen 7-key manifest (tie-break by
`@10`), with `val_loss ≲ 0.13` as a hard reconstruction guardrail. Treat
`cosine_separation` as a secondary tie-breaker only (helps app cosine thresholds), never
as the primary objective.
