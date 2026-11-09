Role:
You are my engineering assistant. Follow these instructions exactly.

## Goal
Enhance my 3D seismic VAE codebase so that the model produces a latent representation that can be used as a seismic tokenizer.
The latent vector from a user‑selected (x,y,z) patch must act as a pattern descriptor that can be searched across the entire seismic volume.

## Codebase Context
The repository is here:
https://github.com/donald-terratellum/synthoseis-3dvae-poc/tree/feat/seismic-tokenizer-app

Key components:
- 3D VAE model (models/vae_3d.py)
- Training pipeline (train.py, losses/, data/)
- Tokenizer app logic (tokenizer/)
- Reconstruction + latent extraction CLI tools

The VAE currently reconstructs low‑level patterns well but loses high‑level structure.
We want to improve latent quality without breaking the tokenizer workflow.

## What to Implement
- Add MONAI‑style deep supervision to the decoder:
  - Add prediction heads at multiple decoder levels (1×1×1 conv)
  - Upsample each head to full patch resolution
  - Compute reconstruction loss at each scale
  - Weight losses: [1.0, 0.5, 0.25] (fine → coarse)
  - Combine into a single total loss
- Make deep supervision optional via a config flag
- This must not change the latent interface used by the tokenizer.

## How to Proceed (Phased Approach)
Phase 1 — Codebase Review
- Create and switch to a new branch "feat/vae-deep-supervision" for all code improvements. Starting branch is "feat/seismic-tokenizer-app"
- Inspect the VAE architecture, decoder structure, and where skip connections exist.
- Identify decoder stages where deep supervision heads should attach.
- Review the loss pipeline and determine where multi‑scale losses should be integrated.
- Confirm how latent vectors are extracted for the tokenizer.


Phase 2 — Design the Deep Supervision Integration
- Propose the exact decoder layers to tap.
- Define the prediction heads and upsampling method.
- Define the multi‑scale loss function wrapper.
- Ensure the encoder’s latent output remains unchanged.

Phase 3 — Implement
- Modify the decoder to include auxiliary heads.
- Add a new DeepSupervisionLoss module.
- Add config flags to enable/disable deep supervision.
- Update training loop to compute multi‑scale losses.

Phase 4 — Self‑Review
- Copilot must review its own changes by checking:
  - Does the latent vector remain identical in shape and meaning?
  - Are auxiliary heads only used during training?
  - Are loss weights applied correctly?
  - Does the training loop handle multi‑scale outputs cleanly?
  - Is the code modular, readable, and consistent with the repo style?

Phase 5 — Testing
- Copilot must:
  - Generate a minimal test script that:
    - Runs a forward pass with deep supervision enabled
    - Verifies shapes of all auxiliary outputs
    - Verifies total loss computation
    - Ensures inference mode still returns only the main reconstruction + latent

  - Run a small synthetic training batch to confirm gradients flow.

Phase 6 — Success Criteria
- The update is successful when:
  - Training runs without errors.
  - Latent extraction for the tokenizer is unchanged.
  - Reconstruction quality improves at both coarse and fine scales.
  - Latent vectors show better semantic clustering when tested on sample patches.
  - Code is clean, modular, and documented.

## Additional Requirements
- Do not remove or break existing CLI tools.
- Keep all changes backward‑compatible.
- Provide diffs or full file rewrites when appropriate.
- Explain design decisions as you go.
- After each major change, summarize what was done and what remains.

If you understand these instructions, begin with Phase 1: Codebase Review and describe what you find.