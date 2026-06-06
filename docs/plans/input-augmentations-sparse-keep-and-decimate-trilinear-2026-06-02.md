# Input Augmentations Plan: Sparse Keep + Decimate Trilinear

Date: 2026-06-02
Status: Approved for implementation
Review Gate: 2 independent reviewers accepted at >=75% confidence

## Objective
Add two new input-only augmentations for training:
1. Random sparse keep in 3D (Poisson-like or uniform-threshold style per sample)
2. Parity decimate + trilinear reconstruction

These augmentations must be mutually exclusive with extrema-only and mutually exclusive with each other on a per-sample basis. Mixup remains allowed after the chosen transform.

## Final Behavior Requirements
- Input-only: both new augmentations apply to `x` only, not `y`.
- Exactly one input transform per sample among:
  - extrema-only
  - sparse keep
  - decimate-trilinear
- Per-sample one-of-three selection uses normalized positive weights.
- Mixup (existing extrema-mixup) still runs after the selected transform when enabled.

## Backward Compatibility
- Preserve current default training behavior via probabilities:
  - `input_extrema_prob = 1.0`
  - `input_sparse_keep_prob = 0.0`
  - `input_decimate_trilinear_prob = 0.0`
- Keep legacy `extrema_only` boolean for compatibility.
- Conflict guard:
  - If `extrema_only=True` and probabilities are not exactly `(1,0,0)`, raise `ValueError` with migration guidance.

## Files to Change
- `src/augmentations.py`
- `scripts/train.py`
- `tests/test_input_augmentations.py` (new)

## CLI / Dataset Parameters
Add CLI args in `scripts/train.py`:
- `--input_extrema_prob` (float, default `1.0`)
- `--input_sparse_keep_prob` (float, default `0.0`)
- `--input_decimate_trilinear_prob` (float, default `0.0`)
- `--sparse_keep_fraction_min` (float, default `0.10`)
- `--sparse_keep_fraction_max` (float, default `0.30`)
- `--sparse_poisson_radius_scale` (float, default `0.85`)

Validation constraints:
- each transform probability in `[0,1]`
- `input_extrema_prob + input_sparse_keep_prob + input_decimate_trilinear_prob > 0`
- `sparse_keep_fraction_min/max` in `[0.01, 1.0]` and `min <= max`
- `sparse_poisson_radius_scale` in `[0.1, 2.0]`

## One-of-Three Selection Semantics
Per sample:
1. Build `probs = [p_extrema, p_sparse, p_decimate]`
2. Keep positive entries only
3. Normalize: `w = probs_pos / sum(probs_pos)`
4. Sample one transform index with `np.random.choice(..., p=w)`
5. Apply exactly one transform to `x`

If only one probability is positive, selection is deterministic.

## Augmentation 1: Random Sparse Keep
### Purpose
Provide much sparser-than-extrema cues so low-level structure is visible while details must be learned.

### Behavior
- Per sample keep fraction `f`:
  - sample uniformly from `[min, max]`
  - if `min == max`, use fixed fraction
- Compute `target_count = clamp(round(f * nvox), 1, nvox)`
- Per sample, choose method 50/50:
  - Poisson-like selector
  - Uniform-threshold selector

### Poisson-like Selector (bounded)
- Radius: `r = radius_scale * cbrt(nvox / target_count)`
- Use spatial hash buckets for neighbor distance checks
- Candidate stream from random voxel permutation
- Hard check cap:
  - `max_checks = min(nvox, max(4096, 8 * target_count))`
- If count is short at cap, random-fill from unseen voxels to reach exact `target_count`

### Uniform-Threshold Selector
- Sample `_p = np.random.uniform(0, 1, shape)`
- Keep voxels where `_p <= keep_fraction`
- Preserve shape and `float32` in the final sparse output

### Output
- Zero all voxels except selected ones
- Preserve shape and `float32`

## Augmentation 2: Decimate + Trilinear Reconstruction
### Purpose
Encourage super-resolution-like detail learning by reconstructing missing samples from decimated lattice values.

### Behavior
- Per sample choose parity `p in {0,1}`
- Anchor lattice keeps indices where all hold: `ix % 2 == p`, `iy % 2 == p`, `iz % 2 == p`
- Reconstruct full volume from anchors with separable linear interpolation:
  - interpolate along `z`, then `y`, then `x`
  - use endpoint clamping (`left/right` in `np.interp`) for stable finite boundaries
- Overwrite anchor voxels with original values after interpolation (exact preservation)

### Output
- Same shape as input
- `float32`
- finite (no NaN/Inf)

## Ordering in Dataset __getitem__
1. Load/scaling
2. Pair geometric augmentations (`x` and `y`) if enabled
3. Trace dropout on `x` if enabled
4. Apply exactly one input transform to `x` (extrema or sparse or decimate)
5. Optional mixup on `x` (unchanged behavior)
6. Return tensors

## Test Plan (unittest)
Create `tests/test_input_augmentations.py` with:
1. Sparse keep count correctness
   - exact/rounded target count behavior
   - edge fractions including `0.01`, `1.0`, and `min == max`
2. Sparse method execution under fixed seed
  - both Poisson-like and uniform-threshold paths execute deterministically
3. Decimate-trilinear correctness
   - anchors preserved exactly for parity `0` and `1`
   - shape/dtype preserved
   - output finite
4. Dataset exclusivity
   - monkeypatch transform funcs and count calls
   - assert exactly one transform is called per sample
5. Backward compatibility
   - legacy `extrema_only=True` path matches prior behavior
6. Conflict guard
   - `extrema_only=True` plus non-default probabilities raises `ValueError`

## Logging and Migration Notes
- Startup logs should print one-of-three probability config and sparse settings.
- If legacy `extrema_only` path is used, print warning recommending new probability knobs.
- CLI help text should explicitly describe one-of-three normalized-weight semantics.

## Non-Functional Guardrails
- No new third-party dependencies.
- Ensure bounded loops (no unbounded retries).
- Keep validation pipeline behavior unchanged by default.
