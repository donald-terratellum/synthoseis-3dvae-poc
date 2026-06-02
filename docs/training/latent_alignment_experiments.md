# Latent Alignment Training Guide

## Goals

- Improve reconstruction detail in predictions, especially fault-like discontinuities and localized vertical offsets.
- Reduce latent mismatch between input, prediction, and label in representative plots.
- Provide a controlled way to bias optimization toward a stronger encoder signal and relatively weaker decoder dynamics.

## Who

- Primary audience: researchers and engineers training the 3D VAE in this repository.
- Secondary audience: reviewers reading experiment rationale and reproducibility notes.

## What

This guide documents two additions in the training loop:

1. Latent consistency losses
- Prediction-label latent alignment.
- Prediction-input latent alignment.

2. Encoder/decoder learning-rate multipliers
- Increase encoder update speed relative to base LR.
- Decrease decoder update speed relative to base LR.

These are optional and default to disabled or neutral values.

## When

Use this recipe when:

- Reconstructions look smooth and miss structure that is present in labels.
- Representative plots show large latent separation between input and prediction and/or between prediction and label.
- MSE appears acceptable, but visual geologic detail quality is still weak.

Do not expect this to solve all detail loss cases by itself. Data quality, augmentation strategy, and model capacity limits still matter.

## How

### New training flags

- --latent_pred_target_weight
- --latent_pred_input_weight
- --latent_alignment_detach_targets / --no_latent_alignment_detach_targets
- --encoder_lr_mult
- --decoder_lr_mult

### Practical interpretation

- --latent_pred_target_weight: pushes encoder(prediction) toward encoder(label).
- --latent_pred_input_weight: pushes encoder(prediction) toward encoder(input).
- --latent_alignment_detach_targets (default): target branches are detached for stability.
- --encoder_lr_mult > 1.0: stronger encoder optimization pressure.
- --decoder_lr_mult < 1.0: relatively weaker decoder optimization dynamics.

### Effective objective

With optional terms enabled, generator optimization is:

L = L_recon + beta * L_KL + lambda_pt * ||E(y_hat) - E(y)||^2 + lambda_pi * ||E(y_hat) - E(x)||^2 + lambda_gan * L_gan

Where:

- x is input
- y_hat is prediction
- y is label
- E is encoder mean branch (mu)

## Usage

### Baseline run (no new terms)

```bash
uv run python scripts/train.py \
  --data data/train_32-32-64.zarr \
  --validation_data data/val_32-32-64.zarr \
  --batch_size 50 \
  --number_batches 100 \
  --epochs 150 \
  --augment \
  --learning_rate 1e-4 \
  --weight_decay 1e-4 \
  --out_dir checkpoints_baseline
```

### Latent alignment only

```bash
uv run python scripts/train.py \
  --data data/train_32-32-64.zarr \
  --validation_data data/val_32-32-64.zarr \
  --batch_size 50 \
  --number_batches 100 \
  --epochs 150 \
  --augment \
  --learning_rate 1e-4 \
  --weight_decay 1e-4 \
  --latent_pred_target_weight 0.05 \
  --latent_pred_input_weight 0.00 \
  --out_dir checkpoints_latent_pt005
```

### Encoder-favored optimization only

```bash
uv run python scripts/train.py \
  --data data/train_32-32-64.zarr \
  --validation_data data/val_32-32-64.zarr \
  --batch_size 50 \
  --number_batches 100 \
  --epochs 150 \
  --augment \
  --learning_rate 1e-4 \
  --weight_decay 1e-4 \
  --encoder_lr_mult 1.5 \
  --decoder_lr_mult 0.75 \
  --out_dir checkpoints_lr_skew_e15_d075
```

### Combined recipe

```bash
uv run python scripts/train.py \
  --data data/train_32-32-64.zarr \
  --validation_data data/val_32-32-64.zarr \
  --batch_size 50 \
  --number_batches 100 \
  --epochs 150 \
  --augment \
  --learning_rate 1e-4 \
  --weight_decay 1e-4 \
  --latent_pred_target_weight 0.05 \
  --latent_pred_input_weight 0.00 \
  --encoder_lr_mult 1.5 \
  --decoder_lr_mult 0.75 \
  --out_dir checkpoints_latent_plus_lr_skew
```

## Experimentation And Ablation Guide

## Experiment goals

- Detect whether fault detail recall improves versus baseline.
- Separate benefit from latent loss versus benefit from optimizer asymmetry.
- Guard against regressions in stability and convergence.

## Recommended experiment matrix

1. A0 Baseline
- latent_pred_target_weight=0.00
- latent_pred_input_weight=0.00
- encoder_lr_mult=1.0
- decoder_lr_mult=1.0

2. A1 Latent target alignment
- latent_pred_target_weight=0.05
- latent_pred_input_weight=0.00
- encoder_lr_mult=1.0
- decoder_lr_mult=1.0

3. A2 LR asymmetry
- latent_pred_target_weight=0.00
- latent_pred_input_weight=0.00
- encoder_lr_mult=1.5
- decoder_lr_mult=0.75

4. A3 Combined
- latent_pred_target_weight=0.05
- latent_pred_input_weight=0.00
- encoder_lr_mult=1.5
- decoder_lr_mult=0.75

Optional sensitivity checks:

- A4 Higher latent pressure: latent_pred_target_weight=0.10
- A5 Add input alignment: latent_pred_input_weight=0.02

## Readouts to track

Primary:

- Validation loss trajectory.
- Representative plots at fixed selected examples over epochs.
- Visual preservation of fault discontinuities.

Secondary:

- train/latent_pred_target_loss and train/latent_pred_input_loss.
- train/encoder_lr and train/decoder_lr.
- GAN metrics when discriminator is enabled.

## Decision criteria

Promote a recipe if:

- Fault detail is consistently clearer in representative plots.
- No major instability (loss spikes/divergence) relative to baseline.
- Validation trend is at least neutral, preferably improved.

Reject or down-weight if:

- Reconstructions become noisy or unstable.
- Validation loss degrades materially.
- Detail gains appear only in isolated examples.

## Theory And Rationale

## Why pixel losses alone can blur detail

Pointwise reconstruction losses (for example MSE) encourage average solutions when uncertainty exists, often smoothing sharp local structure.

## Why latent-space alignment can help

Encouraging prediction latents to match target latents adds structure-level constraints beyond raw amplitude matching.

In this code path, latent alignment is implemented by comparing encoder mu outputs for:

- prediction vs label
- prediction vs input

## Why encoder/decoder LR asymmetry is relevant

A higher encoder LR multiplier and lower decoder LR multiplier biases learning toward representation improvement before decoder adaptation dominates.

## References

1. Larsen et al., Autoencoding beyond pixels using a learned similarity metric, arXiv:1512.09300
- https://arxiv.org/abs/1512.09300

2. Dosovitskiy and Brox, Generating Images with Perceptual Similarity Metrics based on Deep Networks, arXiv:1602.02644
- https://arxiv.org/abs/1602.02644

3. Johnson et al., Perceptual Losses for Real-Time Style Transfer and Super-Resolution, arXiv:1603.08155
- https://arxiv.org/abs/1603.08155

4. Vincent et al., Extracting and Composing Robust Features with Denoising Autoencoders, ICML 2008
- https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf

## Repository Mapping

Relevant implementation locations:

- scripts/train.py
- src/model.py

## Change Log

- Added documentation for latent alignment losses and encoder/decoder LR asymmetry.
- Added reproducible ablation plan and experiment acceptance criteria.
