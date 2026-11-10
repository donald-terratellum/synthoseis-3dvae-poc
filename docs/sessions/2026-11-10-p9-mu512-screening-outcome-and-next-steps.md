# P9 mu512 Screening Outcome and Next Steps (2026-11-10)

## Final Status

The seed-20260926 `mu512` screening run completed normally. Stage-1 reconstruction passed
its gate, and all four stage-2 geology checkpoints passed the reconstruction guardrails.
However, none of the stage-2 checkpoints strictly exceeded the same-seed P0b `n@5`
baseline. There is no eligible milestone, so screening failed and the model is not adopted.

**Decision: stop this screening path and retain P0b.** Do not launch an independent-seed
replication or 2048-D training from this result. The 2048-D memory probe passed its measured
batch-12 fit rule, but memory fit alone does not qualify 2048-D for training; the experiment
plan gates it on an eligible 512-D result.

## Results

The same-seed P0b reference is `n@5=0.010937501` and `n@10=0.016666666`, with 768 eligible
queries. Stage-1 `mu512` reconstruction passed: synthetic tokenizer recipe total was 3.21%
lower than P0b and real tokenizer MAE was 6.32% lower than P0b. These are reconstruction
results only; they do not establish retrieval improvement.

| Geology epoch | n@5 | n@10 | Macro AUROC | Reconstruction gate | Retrieval gate |
|---:|---:|---:|---:|---|---|
| 10 | 0.009375 | 0.016276 | 0.7073 | Pass | Fail |
| 20 | 0.009115 | 0.019792 | 0.7161 | Pass | Fail |
| 30 | 0.008333 | 0.020443 | 0.7025 | Pass | Fail |
| 40 | 0.010417 | 0.019792 | 0.6976 | Pass | Fail |

Epoch 40 had the highest `n@5`, but remained below P0b by about 0.000521. The `n@10`
values at epochs 20, 30, and 40 exceed the P0b reference, but `n@10` is only a tie-breaker;
it cannot substitute for the required strict `n@5` improvement. All macro AUROC values were
also below 0.75, the classifier sanity target.

Machine-readable results:

- Decision: `/Volumes/CrucialX9/3dvae-poc_checkpoints/p9_mu512_seed20260926/p9_mu512_decision.json`
- Per-stage comparison: `/Volumes/CrucialX9/3dvae-poc_checkpoints/p9_mu512_seed20260926/latent_size_comparison.csv`
- Memory preflight: `/Volumes/CrucialX9/3dvae-poc_checkpoints/p9_mu512_seed20260926/memory_preflight.json`

## What to Do Next

1. If desired, inspect the completed training curves in TensorBoard:

   ```sh
   uv run tensorboard --logdir /Volumes/CrucialX9/3dvae-poc_checkpoints/p9_mu512_seed20260926
   ```

   This is review of the completed run only; do not extend training based solely on a
   promising-looking curve.
2. Keep P0b as the reference/candidate. Do not select `mu512` based on its reconstruction
   gain or higher `n@10`; it failed the primary retrieval criterion.
3. Do not run seed-2 replication or 2048-D training under the current P9 decision. If a
   follow-up is proposed, first state a revised, testable hypothesis and define a controlled
   experiment with matched controls and a predeclared retrieval gate. Do not silently relax the
   `n@5` criterion or add epochs to this completed screen.

## Run Artifacts and Interpretation

Training checkpoints, TensorBoard events, and evaluation reports are retained under
`/Volumes/CrucialX9/3dvae-poc_checkpoints/p9_mu512_seed20260926`. The runner's completion
message is expected after finalization; it reports the TensorBoard command and confirms that
2048-D training was not launched. No further training action is implied by that message.

The full procedure and settings are in
`docs/plans/2026-09-27__pretrain_v2_encoder_loss_augmentation_transfer_plan.md`, P9.5, and
the mu512 stage explanation is in
`docs/plans/2026-09-27__pretrain_v2_encoder_loss_augmentation_transfer_plan_mu512-details_.html`.
