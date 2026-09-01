# Geology-Aware Retrieval — Improvement Plan

Goal: improve latent-space retrieval so patches with similar geology rank near each other by
cosine similarity of the `z_geo` embedding, for the `seismic_tokenizer` similarity search.

This plan is written so a Copilot agentic model can implement each option autonomously (or
nearly so). Each option lists concrete files, steps, expected benefit, loss impact, effort,
risks, and how it combines with the others.

---

## 1. Where we are (baseline & ceiling)

Authoritative metric: `diagnostics.neighbor_overlap_at_5` (n@5) on the frozen 7-key manifest
(`docs/benchmarks/frozen_validation_manifest.json`), tie-broken by n@10. Reconstruction
`val_loss` (~0.125–0.13) is a **guardrail**, not the objective — geology changes must not push
it up materially.

| model | n@5 | n@10 | val_loss |
|---|---|---|---|
| `mu` baseline (no geology head) | 0.077 | 0.181 | ~0.125 |
| **Phase 2 epoch-20 `z_geo` (current deliverable)** | **0.139** | **0.227** | ~0.128 |
| Phase 2b (uniformity 0.5) | 0.092 | 0.192 | 0.125 |
| Phase 2c (uniformity 0.05) | 0.089 | 0.185 | 0.125 |
| Phase 2d (extended 40 ep, no uniformity) | 0.131 | 0.219 | 0.124 |

**Empirically established this cycle:** the current recipe has *converged*. More epochs
(Phase 2d) and anti-collapse regularization (Phase 2b/2c) do **not** move n@5. The remaining
headroom requires a **structural** change from one of the four levers below.

Realistic target: n@5 **0.18–0.25** is plausible with the combined upgrades. n@5 ≈ 1.0 is not
achievable — synthetic labels are noisy and many patches are genuinely ambiguous.

### How to read "benefit" and "loss impact"

- **Δn@5 (primary):** expected absolute change to neighbor_overlap@5. This is the retrieval
  goal. Ranges are rough engineering estimates with a confidence tag (Low/Med/High).
- **val_loss impact (guardrail):** expected effect on reconstruction validation loss. "flat"
  = no meaningful change; "↑ risk" = could regress if not guarded; "↓" = may improve.
- **Contrastive train loss** (the geology SupCon term) will generally *decrease* as retrieval
  improves, but it is not directly comparable across label schemes (changing labels changes
  the loss definition), so we track n@5, not the raw loss, across options.

---

## 2. The four levers (detailed)

### Lever A — More & more diverse training data

**What:** grow the patch corpus, ideally with new synthoseis realizations (new geologies), not
just more patches from the same volumes.

**Why it helps:** contrastive retrieval is data-hungry. The current encoder/head likely overfit
the finite patch set (Phase 2d epoch-40 regressing is a mild overfit signal). More *diverse*
geology broadens the manifold and improves generalization of the ranking.

**How (autonomous):**
- Cheap path (more patches, same volumes): run `scripts/sample_patches.py` with a larger
  `--num-patches` / different `--seed` against existing synthoseis zarr volumes to produce a
  bigger `data/synth_train_*.zarr`. Regenerate the val split with a disjoint seed.
- Better path (new geology): run synthoseis to create new realizations, then
  `scripts/sample_patches.py` on those. Combine the resulting patch stores.
- Derived metadata is computed automatically by `compute_patch_derived_metadata`
  ([scripts/sample_patches.py](../../scripts/sample_patches.py#L149)) — no label work needed.
- Retrain Phase 2 recipe (contrastive-only) on the enlarged set; re-benchmark.

**Effort:** Low (cheap path) / Medium (new realizations, compute-bound).
**Expected benefit:** Δn@5 **+0.01 to +0.03** for 2–4× more *diverse* data (Med confidence).
Merely more patches from the *same* volumes: **+0.00 to +0.01** (Low). Diversity matters more
than raw count.
**val_loss impact:** flat-to-slightly-down (more data usually helps reconstruction too).
**Risks:** must keep the frozen benchmark manifest fixed (do NOT regenerate it) so numbers stay
comparable; keep train/val patch provenance disjoint to avoid leakage.

---

### Lever B — Richer geology labels

**What:** improve the *supervision signal* that defines "similar geology." Today, strata labels
come from thresholding 7 background metadata keys into multi-label signatures
([src/geology_sampler.py](../../src/geology_sampler.py#L18)), and the contrastive target treats
same-signature patches as positives. Richer labels = a more faithful similarity target.

**Why it helps:** this is likely the **highest-leverage** lever. The retrieval ceiling is
partly set by *label quality*: if two geologically-similar patches get different discrete
signatures (or two dissimilar patches collide on one), the contrastive target is wrong and n@5
is capped. Better labels raise the ceiling itself.

**Concrete ideas (increasing sophistication):**
1. **Continuous / soft targets instead of hard strata.** Replace the discrete
   same-signature positive rule with a *soft* similarity target computed from the full metadata
   vector (e.g. cosine or Gaussian-RBF similarity of the per-patch metadata), and train with a
   soft-nearest-neighbor / weighted-SupCon loss. This removes threshold-induced label noise.
   (A metadata-similarity target path already exists in the calibration flow — extend it to a
   soft contrastive target.)
2. **Add more physically-meaningful metadata channels** to
   `compute_patch_derived_metadata`: e.g. dominant dip *magnitude bins*, fault *orientation*
   histogram summaries, channel *sinuosity*, net-to-gross (sand fraction is there — add
   layer-count / cyclicity), unconformity/onlap *angle*, amplitude/impedance texture stats
   (GLCM contrast, spectral centroid). Then include them in the calibration keys.
3. **Hierarchical labels.** Define coarse facies classes (e.g. faulted / channelized / onlap /
   flat) plus fine sub-signatures, and use a hierarchical contrastive loss (positives at both
   levels with different temperatures). Improves both coarse and fine ranking.
4. **De-noise labels with clustering.** Fit a GMM/k-means on the metadata vectors and use
   cluster membership (or soft responsibilities) as the target, which is smoother than
   hand-thresholded signatures.
5. **Curriculum on label confidence.** Weight each patch's contrastive contribution by how
   confidently it belongs to its stratum (distance to threshold / cluster center), so ambiguous
   patches don't inject noise early in training.

**How (autonomous):**
- Idea 2 is the most self-contained: extend `DERIVED_METADATA_KEYS` and
  `compute_patch_derived_metadata`, regenerate patches, add the new keys to the calibration/active
  key set, retrain. New keys must be added to the benchmark's metadata keys consistently.
- Idea 1/4 require a new loss/target path in [scripts/train.py](../../scripts/train.py) — add a
  `--geology_soft_targets` mode that builds a per-batch target similarity matrix from metadata
  and swaps the SupCon positive mask for a soft-weighted objective. Unit-test the new loss.

**Effort:** Low–Medium (idea 2) / Medium (idea 1, 4) / Medium–High (idea 3).
**Expected benefit:** Δn@5 **+0.02 to +0.06** (Med–High confidence) — the biggest single lever,
because it raises the label ceiling rather than just fitting the existing target better.
**val_loss impact:** flat (labels affect the geology head/encoder shaping, not reconstruction).
**Risks:** new metadata must be recomputed for *all* patches AND reflected in the benchmark
manifest's metadata; a soft-target loss needs careful temperature tuning; changing labels makes
raw contrastive-loss values non-comparable across runs (rely on n@5).

---

### Lever C — Harder negative mining (and positive mining)

**What:** improve *which* pairs the contrastive loss sees. The sampler already supports
`hard_fraction` and `hard_top_quantile`
([src/geology_sampler.py](../../src/geology_sampler.py)); this lever pushes it further with
embedding-driven mining.

**Why it helps:** SupCon learns most from *hard* negatives (different stratum but currently
close in `z_geo`) and *hard* positives (same stratum but currently far). Static metadata-based
batch composition (today) is a proxy; mining in the *current embedding space* targets exactly
the pairs the model is getting wrong.

**Negative mining details:**
1. **Semi-hard negatives (safest).** Within a batch, for each anchor, prefer negatives whose
   `z_geo` cosine to the anchor is high but below the nearest positive (the classic
   semi-hard band). Implement as a batch-level reweighting of the SupCon denominator, or as a
   sampler that biases negative selection by a periodically-refreshed embedding index.
2. **Embedding-indexed hard negatives (strongest).** Every N epochs, encode the training set to
   `z_geo`, build a cosine kNN index, and for each anchor sample negatives from its top-k
   *cross-stratum* neighbors. This is the highest-impact mining variant.
3. **Debiased / hardness-weighted contrastive loss.** Replace vanilla SupCon with a
   hardness-weighted variant (weight negatives by `exp(sim/τ_hard)`), which needs no external
   index and is a drop-in loss change.

**Positive mining (yes, appropriate here):**
1. **Hard-positive emphasis.** Up-weight positives that are currently far in `z_geo` (same
   stratum, low cosine) so the head pulls the true tail of each class together — directly
   targets n@5 (near-neighbor recall).
2. **Augmentation-based positives.** Add a second augmented view of each anchor as a guaranteed
   positive (SimCLR-style), combined with the geology-label positives. Cheap, robust, and known
   to sharpen retrieval embeddings.

**How (autonomous):**
- Idea C3 and positive-hardness are pure loss changes in
  [scripts/train.py](../../scripts/train.py) (gate behind `--geology_hard_weighting`,
  `--geology_hard_temp`); add unit tests analogous to `UniformityLossTests`.
- Idea C2 needs a periodic re-encode + kNN refresh hook in the training loop and a sampler that
  consumes a neighbor table; more involved but self-contained.
- Augmentation positives (P2) reuse the existing augmentation pipeline; add a second view in the
  collate/encode path and treat it as a positive.

**Effort:** Low (C3, hard-positive weighting) / Medium (augmentation positives) / Medium–High
(C2 indexed mining).
**Expected benefit:** Δn@5 **+0.02 to +0.05** (Med confidence); indexed hard negatives (C2) at
the top of that range. Augmentation positives: **+0.01 to +0.03** and improves stability.
**val_loss impact:** flat (loss/sampling change only).
**Risks:** overly-aggressive hard mining can collapse or destabilize training (watch n@5 each
epoch and keep a semi-hard band rather than hardest-only); indexed mining adds
periodic-recompute cost.

---

### Lever D — Architecture upgrades

#### D1. Projection head (smallest, safest structural change)

Current head is a 2-layer MLP: `Linear(128→128) → GELU → Linear(128→64) → L2-norm`
([src/model.py](../../src/model.py#L127)).

**Upgrades (each autonomous, backward-compatible via new CLI flags + resume allowlist):**
- **Deeper/wider head:** 3 layers with BatchNorm/LayerNorm between (e.g.
  `128→256 → 256→256 → 256→128`), which is standard for contrastive projection heads and often
  the single easiest win.
- **Add BatchNorm1d** in the head (SimCLR/BYOL found this materially helps contrastive
  embeddings).
- **Larger `proj_dim`** (64 → 128) to reduce crowding of the embedding sphere.
- **Non-collapsing normalization:** keep final L2-norm but add a learnable temperature scale.

**Effort:** Low. **Benefit:** Δn@5 **+0.01 to +0.03** (Med). **val_loss impact:** flat (head is
downstream of `mu`, doesn't touch reconstruction). **Risk:** minimal; add `geology_proj_*` flags
and extend the resume allowlist (already tolerant of `geology_head.` prefix).

#### D2. Encoder capacity / receptive field (larger, higher-risk)

- **Increase `base_ch`** (16 → 24/32) or add a residual encoder block
  (`residual_encoder=True` already exists) to give `mu` more geological expressiveness.
- **Anisotropic / larger patch context** so the encoder sees more lateral geology.

**Effort:** Medium. **Benefit:** Δn@5 **+0.01 to +0.04** (Low–Med). **val_loss impact:** likely
**↓ (improves)** reconstruction too, but larger models are slower and need re-tuning; changes
`mu` so the geology head must be retrained. **Risk:** breaks warm-start from the current
checkpoint (different encoder shape → train from an earlier base or from scratch).

#### D3. Decouple/co-train objectives

- Train the geology head with a **stop-gradient variant** or a small encoder-LR so
  reconstruction (`mu`) stays intact while the head+encoder shape `z_geo`. (Current recipe
  already uses `encoder_lr_mult 0.1` — this is the mild version.)
- Optional **projector + predictor (BYOL-style)** asymmetry to further stabilize.

**Effort:** Medium. **Benefit:** Δn@5 **+0.01 to +0.02** (Low). **val_loss impact:** protects the
guardrail. **Risk:** more moving parts.

---

## 3. Expected-benefit summary

| Lever | Δn@5 (est.) | Confidence | val_loss | Effort |
|---|---|---|---|---|
| A. More/diverse data | +0.01 … +0.03 | Med | flat/↓ | Low–Med |
| **B. Richer labels** | **+0.02 … +0.06** | **Med–High** | flat | Low–Med |
| C. Hard neg/pos mining | +0.02 … +0.05 | Med | flat | Low–Med |
| D1. Bigger proj head | +0.01 … +0.03 | Med | flat | Low |
| D2. Bigger encoder | +0.01 … +0.04 | Low–Med | ↓ (recon) | Med |
| D3. Objective decoupling | +0.01 … +0.02 | Low | protects | Med |

Estimates are **not additive** — gains overlap and saturate. A well-chosen combination
realistically lands n@5 in the **0.18–0.25** band (vs 0.139 today), not the naive sum.

---

## 4. What combines vs. what is exclusive

**Freely combinable (recommended stack):**
- **A + B + C + D1** compose well and are the recommended package. More diverse data (A) makes
  richer labels (B) and mining (C) pay off, and the bigger head (D1) has capacity to exploit
  them. None conflict.
- D3 (mild objective decoupling) is already partly in use and layers on top of all of the above.

**Combine with care / sequencing constraints:**
- **C2 (indexed hard-negative mining) + B (soft labels):** compatible but tune *together* —
  hard mining on top of noisy hard labels amplifies label errors. Do **B first**, then add C.
- **D2 (bigger encoder) is semi-exclusive with warm-starting.** It changes `mu`'s shape/statistics,
  so it invalidates warm-start from `vae_epoch20.pt` and forces retraining the head (and ideally
  the encoder) from an earlier base. Don't combine D2 with "continue from current checkpoint"
  experiments — treat D2 as a fresh training track.
- **Uniformity regularizer:** proven not useful here — do **not** re-enable it in combination
  (keep `--geology_uniformity_weight 0`). It is exclusive with the goal, not additive.

**Mutually exclusive within a lever:** pick one label scheme in B (hard strata *or* soft targets
*or* clustered), one primary negative-mining strategy in C (semi-hard band *or* indexed kNN *or*
hardness-weighted loss) per run to keep attribution clean.

---

## 5. Recommended sequencing (highest ROI first)

Each step is a self-contained, benchmarkable experiment. Keep the frozen manifest fixed and
rank by n@5 (val_loss ≲ 0.13 guardrail).

1. **D1 — bigger projection head** (cheapest structural change; warm-startable). Confirm a small
   lift, lock it in.
2. **B (idea 2) — add richer metadata channels + include in labels.** Highest ceiling-raiser.
3. **C (C3 + hard-positive weighting) — loss-only mining.** Cheap, stacks on B.
4. **A — enlarge/diversify data** (ideally new synthoseis realizations); retrain the stack.
5. **C2 — indexed hard-negative mining** once B/C basics are in and stable.
6. **D2 — bigger encoder** as a separate fresh-training track if 1–5 plateau below target.

After each step: benchmark epochs on the frozen manifest, adopt only if n@5 improves over the
current best (0.139 → new best), and record results in `docs/benchmarks/` + a short session note.

---

## 6. Autonomous implementation checklists (per option)

**D1 (proj head):** edit `GeologyProjectionHead` in [src/model.py](../../src/model.py#L127) to a
configurable depth/width/norm; add `--geology_proj_layers`, `--geology_proj_norm` flags in
[scripts/train.py](../../scripts/train.py); ensure resume allowlist tolerates the new
`geology_head.*` params; add a unit test that the head output is unit-norm and shape-correct;
train Phase-2 recipe; benchmark.

**B-idea2 (richer metadata):** add keys to `DERIVED_METADATA_KEYS` and populate them in
`compute_patch_derived_metadata` ([scripts/sample_patches.py](../../scripts/sample_patches.py#L149));
regenerate `data/synth_train_*`/`synth_val_*`; add the new keys to the active/calibration key
set and to the benchmark `--metadata_keys`; unit-test metadata ranges; retrain; benchmark.

**B-idea1/4 (soft/clustered targets):** add a `--geology_soft_targets` path in
[scripts/train.py](../../scripts/train.py) building a per-batch metadata-similarity matrix and a
soft-weighted contrastive loss; unit-test the loss (soft target reduces to SupCon at hard
limits); train; benchmark.

**C3 / hard-positive (loss-only mining):** add hardness weighting to the contrastive term behind
`--geology_hard_weighting`/`--geology_hard_temp`; unit-test (harder pairs get larger weight);
train; benchmark.

**C2 (indexed hard negatives):** add a periodic (every N epochs) `z_geo` re-encode + cosine kNN
build, and a sampler mode that draws negatives from cross-stratum neighbors; guard cost with a
refresh interval flag; train; benchmark.

**A (data):** run `scripts/sample_patches.py` (bigger `--num-patches` / new seeds, or on new
synthoseis volumes); keep val provenance disjoint; do NOT regenerate the frozen benchmark
manifest; retrain; benchmark.

**D2 (encoder):** raise `base_ch` or set `residual_encoder=True`; note this breaks warm-start —
train as a fresh track; benchmark against the frozen manifest.

---

## 7. Guardrails (apply to every option)

- Frozen benchmark manifest stays fixed; rank by `neighbor_overlap_at_5`, tie-break `@10`.
- Keep reconstruction `val_loss ≲ 0.13`; if an option pushes it up, reduce encoder LR or add a
  reconstruction weight.
- Keep `--geology_uniformity_weight 0` (empirically unhelpful for ranking).
- One primary variable per run for clean attribution; record every result in `docs/benchmarks/`.
- Benchmark CLI flags are `--data`, `--manifest`, `--checkpoint`, `--out_json`,
  `--use_geo_embedding`, `--metadata_keys` (the 7 background keys) — there is **no** `--output`.
</content>
