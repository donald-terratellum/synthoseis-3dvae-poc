# VAE latent shaping for geologic feature discrimination  
_Target: keep current reconstruction quality **and** sharpen latent separation of faults, channels, unconformities, strat boundaries._

Repo: `synthoseis-3dvae-poc` @ `39393e0c5335099134b88f558d4754f4e4cdb286`  

---

## 1. Where the current VAE stands

**Components (from repo structure):**

- **Training pipeline**
  - `scripts/train.py` — CLI + training loop
  - `scripts/sample_patches.py` — builds 3D patch datasets
  - `docs/training/README.md` — training overview
  - `docs/training/latent_alignment_experiments.md` — latent-alignment knobs and ablations
- **Model**
  - Under `src/` (VAE with:
    - 3D conv encoder/decoder
    - KL regularization
    - Pixelwise loss
    - Optional LPIPS + GAN (from “LPIPS GAN mid-session checkpoint” snapshot)
    - Optional deep supervision hooks

**Key point:**  
You already have a **modular loss stack** and **latent-alignment controls**. We’ll extend those rather than bolt on something ad‑hoc.

---

## 2. High-level plan

**Goal:**  
Add **label-aware latent shaping** on top of the existing VAE so that:

- Reconstruction quality is preserved (pixelwise + LPIPS + optional GAN).
- Latent space becomes **geology-structured** via:
  - **Supervised Contrastive Loss (SupCon)** or **N-pair loss** on patch embeddings.
  - **Feature-head BCE** for voxel-wise or patch-wise geologic labels.

**Strategy:**

1. **Add a feature head** on top of the encoder (or latent).
2. **Add a supervised contrastive loss module** that uses your labels.
3. **Wire both into the existing training loop** as optional “latent alignment v2”.
4. **Adjust sampling** in `sample_patches.py` to be label-aware.
5. **Run a staged curriculum** to avoid wrecking reconstruction.

---

## 3. Architecture changes

### 3.1. Feature head for geologic labels

**Where:**  
In `src/` (where the VAE is defined), add a **feature head** module:

- **Input:**  
  - Either:
    - The latent vector `z` (global patch embedding), or
    - A mid-level encoder feature map (for more spatial detail).
- **Output options:**
  - **Patch-level multi-label logits**:  
    - For each patch, predict presence of `{fault, channel, unconformity, strat_boundary}`.
  - **Voxel-level logits** (if you want full segmentation):  
    - 3D conv head upsampling to patch size.

**Recommendation (for first iteration):**

- Start with **patch-level labels**:
  - Define a patch as “fault-positive” if ≥ X% of voxels are labeled fault, etc.
  - This keeps the head small and makes SupCon easier.

**Implementation sketch (conceptual):**

- Add something like `GeologicFeatureHead(nn.Module)`:
  - `forward(z)` → `logits` of shape `[B, num_features]`.
- Integrate into VAE model class:
  - `vae.encode(x)` returns `z`.
  - `vae.feature_logits(z)` returns `logits`.

You already have “latent alignment” hooks; this head can plug into that same pattern.

---

## 4. Loss design

### 4.1. Keep existing reconstruction stack

Retain your current reconstruction losses:

- **Pixelwise loss** (L1 or L2).
- **LPIPS-style perceptual loss** (2D slices or 3D variant).
- **Optional GAN loss** (already wired in your mid-session checkpoint).

These remain the backbone for **visual fidelity**.

### 4.2. Add feature-head BCE

For each patch:

- **Inputs:**
  - `y_patch` — multi-hot vector of geologic features present in the patch.
  - `logits_patch` — output of feature head.

- **Loss:**
  - Weighted **BCE** (or focal loss) to handle imbalance:
    - Faults, channels, unconformities are rarer than background.

Call this:

- `L_feat = BCE_weighted(y_patch, logits_patch)`

### 4.3. Add supervised contrastive loss (SupCon) on latent

**Embedding:**

- Use the **latent vector `z`** as the embedding for contrastive learning.
- Optionally L2-normalize `z` before SupCon.

**Labels:**

- For each patch, define a **primary label**:
  - e.g., `fault`, `channel`, `unconformity`, `strat_boundary`, `background`.
- You can derive this from pixel labels by:
  - Taking the **argmax of voxel fractions** per patch, or
  - Using a priority rule (fault > channel > unconformity > strat > background).

**SupCon idea:**

- In a batch:
  - All patches with the same primary label are **positives** for each other.
  - All others are **negatives**.

**Loss:**

- `L_supcon = SupCon(z_batch, labels_batch)`

### 4.4. Optional: N-pair loss variant

If you prefer N-pair:

- For each anchor patch, pick:
  - One positive (same label).
  - Many negatives (different labels).
- Use N-pair loss instead of SupCon.

But SupCon is more natural with your batch-based training and multiple classes.

### 4.5. Total loss

Extend your existing loss aggregator (you already have latent-alignment knobs) to:

- `L_total =`
  - `λ_pix * L_pixel`
  - `+ λ_lpips * L_lpips`
  - `+ λ_gan * L_gan` (if enabled)
  - `+ λ_kl * L_kl`
  - `+ λ_feat * L_feat`
  - `+ λ_supcon * L_supcon`

**Initial weights (suggested starting point):**

- `λ_pix = 1.0`
- `λ_lpips = 0.1–0.3`
- `λ_gan = 0.0–0.1` (off at first if you want stability)
- `λ_kl = 0.001–0.01`
- `λ_feat = 2.0–5.0` (strong signal)
- `λ_supcon = 0.5–2.0` (start low, ramp up)

---

## 5. Data and sampling changes

### 5.1. Patch-level labels from pixel labels

Extend `scripts/sample_patches.py` (or the dataset class it feeds) to:

- For each 3D patch:
  - Compute **fraction of voxels** for each feature:
    - `f_fault`, `f_channel`, `f_unconformity`, `f_strat`, `f_background`.
  - Build:
    - **Multi-hot vector** for `L_feat`:
      - `y_patch[k] = 1` if `f_k > threshold_k`.
    - **Primary label** for `L_supcon`:
      - `label = argmax_k f_k` (or priority rule).

Store these alongside the patch in your dataset.

### 5.2. Label-aware sampling

Modify your sampling logic to:

- **Over-sample feature-rich patches**:
  - Ensure each batch has a good mix of:
    - Fault-rich
    - Channel-rich
    - Unconformity-rich
    - Strat-boundary-rich
    - Background
- This is crucial for SupCon:
  - Each label needs multiple examples per batch to form positive sets.

You can implement:

- A **sampler** that:
  - Maintains index lists per label.
  - Draws a balanced mini-batch.

---

## 6. Training curriculum

### Phase 0 — Baseline checkpoint (already done)

- You already have a **LPIPS + GAN mid-session checkpoint**.
- This is your **reconstruction baseline**.

### Phase 1 — Add feature-head BCE only

- Enable **feature head** and `L_feat`.
- Keep `λ_supcon = 0` initially.
- Train from the baseline checkpoint.
- Monitor:
  - Reconstruction metrics (L1, LPIPS, GAN discriminator score).
  - Feature-head metrics (patch-level classification accuracy, ROC-AUC).

Goal:  
Ensure reconstruction quality is not significantly degraded while the latent becomes predictive.

### Phase 2 — Turn on SupCon (latent sharpening)

- Enable `L_supcon` with small `λ_supcon` (e.g., 0.5).
- Keep `λ_feat` as in Phase 1.
- Train further from Phase 1 checkpoint.

Monitor:

- Reconstruction metrics vs baseline.
- Latent structure:
  - Linear probe: train a simple classifier on frozen `z` to predict patch labels.
  - Clustering: UMAP / t-SNE of `z` colored by feature type.

If reconstruction degrades:

- Reduce `λ_supcon` or `λ_feat`.
- Optionally:
  - Freeze decoder for a few epochs so encoder/latent adapt first.

### Phase 3 — Optional GAN + stronger SupCon

- If you want maximum realism:
  - Re-enable or increase `λ_gan`.
- If latent clusters are still fuzzy:
  - Gradually increase `λ_supcon`.

---

## 7. How this fits your existing latent-alignment docs

You already have:

- `docs/training/latent_alignment_experiments.md` — describing:
  - Latent-alignment knobs
  - Ablation strategies

Extend that doc with:

- **New section:** “Geologic feature–aware latent alignment”
  - Describe:
    - Feature head
    - SupCon loss
    - Patch-label derivation
    - Curriculum (Phases 1–3)
  - Add experiment IDs:
    - `exp_latent_geo_supcon_v1`
    - `exp_latent_geo_supcon_v2_balanced_sampling`
    - etc.

This keeps your experimentation reproducible and aligned with the rest of the repo.

---

## 8. Concrete next steps checklist

1. **Model:**
   - Add `GeologicFeatureHead` in `src/` and wire it into the VAE.
   - Expose logits in the forward pass or via a method.

2. **Losses:**
   - Implement `FeatureBCELoss` (weighted BCE).
   - Implement `SupConLoss` (batch-based supervised contrastive).
   - Extend your existing loss config to include:
     - `lambda_feat`
     - `lambda_supcon`
     - toggles for enabling them.

3. **Data:**
   - Extend patch dataset to:
     - Compute patch-level multi-hot labels.
     - Compute primary label for SupCon.
   - Implement label-balanced sampler.

4. **Training:**
   - Add CLI flags in `scripts/train.py`:
     - `--enable-feature-head`
     - `--enable-supcon`
     - `--lambda-feat`
     - `--lambda-supcon`
   - Implement the Phase 1 → Phase 2 curriculum as separate configs.

5. **Evaluation:**
   - Add a small script/notebook to:
     - Extract latent `z` for a validation set.
     - Run linear probe classification.
     - Plot UMAP/t-SNE colored by feature type.

---

## 9. Summary

You keep:

- **Current reconstruction performance** (pixelwise + LPIPS + optional GAN).

You add:

- **Feature-head BCE** to make the latent explicitly predictive of geologic features.
- **Supervised Contrastive Loss** on latent embeddings to **sharpen separation** between faults, channels, unconformities, strat boundaries, and background.
- **Label-aware sampling** so each batch is geologically informative.
- A **staged curriculum** so reconstruction quality is preserved while the latent becomes more geologic.

This gives you a VAE whose latent space is not just a compression of seismic, but a **geologic feature manifold** you can probe, cluster, and build tools on top of.

---



# Updated: Geologic-Feature-Aware Latent Shaping Plan  
### For: synthoseis-3dvae-poc  
### Goal: Preserve reconstruction quality **and** sharpen latent separation of geologic features  
### Author: Donald + Copilot (agentic workflow ready)

---

## 1. Objective

Enhance the existing 3D VAE so that its latent space encodes **geologic feature composition**, not just seismic appearance.  
The model should:

- Maintain current reconstruction quality (pixelwise + LPIPS + optional GAN).
- Learn a latent space where **faults, channels, unconformities, strat boundaries, background** form **separable, interpretable clusters**.
- Support **mixed-feature patches** (not just single-dominant examples).
- Use **voxel-level labels** to compute **feature-fraction vectors** per patch.
- Use a **similarity-weighted contrastive loss** to align latent similarity with geologic similarity.

This plan is ready for implementation using VS Code with agentic Copilot.

---

## 2. High-Level Approach

We extend the VAE with three new components:

1. **Patch-level geologic feature vectors**  
   - For each 3D patch, compute a vector:  
     `v = [f_fault, f_channel, f_unconformity, f_strat, f_background]`  
     where each entry is the fraction of voxels with that label.

2. **Feature-head BCE loss**  
   - A small MLP or 1×1×1 conv head predicts the same vector `v_hat` from the latent `z`.  
   - Weighted BCE or focal loss encourages the latent to encode feature presence.

3. **Similarity-weighted contrastive loss**  
   - Build a geologic similarity matrix `S_ij = sim(v_i, v_j)` (dot product or cosine).  
   - Build a latent similarity matrix `L_ij = sim(z_i, z_j)`.  
   - Train the encoder so that `L` matches `S` (diagonal high, off-diagonal low).  
   - This generalizes SupCon/N-pair to multi-hot, continuous labels.

This avoids the need for single-class patches and naturally handles mixtures.


## 3. Implementation Plan

### 3.1. Data Pipeline Updates

Modify `sample_patches.py` and/or the dataset class:

- For each patch:
  - Count voxels per geologic label.
  - Normalize to fractions.
  - Store:
    - `v` (multi-hot fraction vector)
    - `primary_label` (optional; argmax of fractions)
- Add a **label-balanced sampler**:
  - Ensure each batch contains multiple examples of each feature type.
  - Required for stable contrastive learning.

**Output fields per patch:**
```python
{
  "patch": <3D tensor>,
  "feature_fraction": <tensor[K]>,
  "primary_label": <int or None>
}
```

### 3.2. Model Architecture Changes
    Modify the VAE model under src/:
    Add a feature head
    - Input: latent vector z (after reparameterization).
    - Output: predicted feature-fraction vector v_hat.

Example:

```python
class GeologicFeatureHead(nn.Module):
    def __init__(self, latent_dim, num_features):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_features)
        )
    def forward(self, z):
        return self.mlp(z)

```

Integrate into VAE:
- vae.encode(x) → z
- vae.feature_logits(z) → v_hat


## 4. **Loss Functions**
### 4.1. Reconstruction Loss (existing)
- Pixelwise L1 or L2
- LPIPS (2D slices or 3D variant)
- Optional GAN loss
No changes needed.

### 4.2. Feature-Head BCE Loss
For each patch:
    L_feat = BCE_weighted(v, v_hat)

Weights help with class imbalance (faults/channels are rare).


### 4.3. Similarity-Weighted Contrastive Loss (Geologic Contrastive)
This is the core of the plan.
    
Step 1 — Compute geologic similarity
For patches i and j:
    S_ij = dot(v_i, v_j)  # or cosine similarity

Step 2 — Compute latent similarity
    Z_ij = exp(cosine(z_i, z_j) / tau)


Step 3 — Loss
For each anchor i:
    L_geo_i = - sum_j S_ij * log( Z_ij / sum_k Z_ik )

Total:
L_geo = mean(L_geo_i over batch)

Interpretation:
- If two patches have similar geologic composition → pull latent vectors together.
- If two patches differ → push them apart.

This creates a confusion-matrix-like alignment between geologic and latent similarity.

## 5. **Total Loss**
Combine everything:
L_total =
    λ_pix    * L_pixel +
    λ_lpips  * L_lpips +
    λ_gan    * L_gan +
    λ_kl     * L_kl +
    λ_feat   * L_feat +
    λ_geo    * L_geo

Recommended starting weights:
λ_pix = 1.0
λ_lpips = 0.1–0.3
λ_gan = 0.0–0.1
λ_kl = 0.001–0.01
λ_feat = 2.0–5.0
λ_geo = 0.5–2.0


## 6. **Training Curriculum**
Phase 1 — Reconstruction Baseline
- Train VAE with reconstruction + KL only.
- Ensure stable reconstructions.

Phase 2 — Add Feature-Head BCE
- Enable L_feat.
- Keep λ_geo = 0.
- Train until feature-head predictions stabilize.

Phase 3 — Enable Geologic Contrastive Loss
- Turn on L_geo with small weight.
- Gradually increase λ_geo as long as reconstruction quality remains acceptable.

Phase 4 — Optional GAN Refinement
- Re-enable GAN loss for realism.
- Keep contrastive loss active.


## 7. **Evaluation**
### 7.1. Latent Probing
- Freeze encoder.
- Train a linear classifier on z to predict feature fractions or primary label.
- High accuracy → latent is geologically meaningful.

### 7.2. Latent Visualization
- Use UMAP or t-SNE on latent vectors.
- Color by feature type or feature-fraction heatmaps.
- Expect clear separation and smooth transitions.

### 7.3. Reconstruction Quality
- Compare L1, LPIPS, and GAN discriminator scores to baseline.

## 8. **Files to Modify (Implementation Checklist)**
Data
- scripts/sample_patches.py
- src/datasets/*.py

Model
- src/models/vae.py (or equivalent)
- Add GeologicFeatureHead

Losses
- src/losses/feature_bce.py
- src/losses/geologic_contrastive.py

Training
- scripts/train.py

  - Add CLI flags:
    - --enable-feature-head
    - --enable-geologic-contrastive
    - --lambda-feat
    - --lambda-geo

Docs
- Add docs/training/geologic_latent_alignment.md


## 9. **Summary**
This plan upgrades the VAE into a geologic-feature-aware latent model by:
- Using voxel-level labels to compute patch-level feature fractions.
-Adding a feature-head BCE loss to make the latent predictive.
- Adding a similarity-weighted contrastive loss to align latent geometry with geologic similarity.
- Preserving reconstruction quality through staged training and balanced loss weights.

The result is a VAE whose latent space encodes faults, channels, unconformities, strat boundaries, and mixtures thereof — enabling downstream tasks like clustering, retrieval, segmentation, and geologic interpretation.


## 10. **Ready for Agentic Copilot**
This document is structured so you can:
- Open it in VS Code
- Highlight any section
- Tell Copilot:
  - “Implement this module”
  - “Generate the loss function described here”
  - “Modify train.py to support these flags”
  - “Create the dataset changes described above”

Everything is modular, incremental, and testable.
