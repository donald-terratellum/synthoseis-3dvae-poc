# P9 mu512 Screening, Extension, and Next Steps

Updated: 2026-11-11

## Status and decision

The original 40-epoch screen failed its predeclared retrieval gate. A separately
authorized exploratory continuation reached epoch 150 and produced several checkpoints
that numerically exceeded the same-seed P0b reference. That continuation reset optimizer
and scheduler state and was not a matched, predeclared screen; it does not overturn the
original decision or establish an adoptable improvement. Retain P0b as the comparison
reference and keep the extension checkpoints for further evaluation.

The most useful next step is evaluation of existing checkpoints, followed by the
[streamed overlap and semantic retrieval plan](../plans/2026-11-08__streamed_overlap_and_semantic_retrieval_experiment.md).
An optional, bounded P0b continuation is described below if a duration-only question
is worth answering first. No additional training or adoption is authorized by this
summary.

## Original 40-epoch screen

The seed-20260926 `mu512` screening run completed normally. Stage-1 reconstruction passed
its gate, and all four stage-2 geology checkpoints passed the reconstruction guardrails.
However, none of the stage-2 checkpoints strictly exceeded the same-seed P0b `n@5`
baseline. There was no eligible milestone in that screen, and the model was not adopted.

The 2048-D memory probe passed its measured batch-12 fit rule, but memory fit alone does
not qualify 2048-D for training. The original experiment plan gates it on an eligible
512-D screen, which did not occur.

### Original results

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

## Exploratory mu512 phase-2 extension

### Setup and interpretation

The extension resumed **model weights** from the original seed-20260926 stage-2
epoch-40 checkpoint, then trained 110 additional epochs, numbered 41-150, on MPS.
It reinitialized AdamW, the learning-rate scheduler, and best-validation tracking;
it did not restore their epoch-40 state. The logged encoder learning rate restarted
at 0.00005 rather than continuing the original epoch-40 value of 0.0000125. This
changes the optimization trajectory, so the extension is neither an uninterrupted
150-epoch run nor an apples-to-apples continuation of P0b or E1.

The training run used batch size 12 and 450 batches per epoch: 5,400 examples per
epoch and 594,000 additional training examples. Training took approximately 11 hours
20 minutes, or 6.2 minutes per epoch; the earlier same-seed mu128 E1 geology stage
averaged approximately 6.0 minutes per epoch. Evaluation time is additional. The
mu512 model has 13,379,060 parameters, versus 3,793,652 for mu128 E1.

### Retrieval and classification

All rows below use the historical frozen manifest. `n@5` is overlap of the top-five
embedding neighbors with the top-five metadata neighbors, not precision@5. There are
768 eligible geology queries. The extension evaluated every tenth epoch, not every
saved checkpoint.

| Run and stage-2 epoch | n@5 | n@10 | Macro AUROC | Extension reconstruction / retrieval gates |
|---|---:|---:|---:|---|
| Same-seed P0b, 30 | 0.0109375 | 0.016667 | 0.6307 | Reference |
| Same-seed mu128 E1, 30 | 0.0109375 | 0.020182 | 0.7126 | Historical comparison |
| Original mu512, 40 | 0.010417 | 0.019792 | 0.6976 | Pass / fail |
| Extension mu512, 50 | 0.011979 | 0.021094 | 0.7419 | Pass / pass |
| Extension mu512, 60 | 0.007813 | 0.019271 | 0.7000 | Pass / fail |
| Extension mu512, 70 | 0.009115 | 0.020833 | 0.7026 | Pass / fail |
| Extension mu512, 80 | 0.010156 | 0.022135 | 0.7278 | Pass / fail |
| Extension mu512, 90 | 0.012240 | 0.020964 | 0.7245 | Pass / pass |
| Extension mu512, 100 | **0.012500** | 0.021354 | 0.7125 | Pass / pass |
| Extension mu512, 110 | 0.010156 | 0.022786 | 0.7049 | Pass / fail |
| Extension mu512, 120 | 0.010677 | 0.021094 | 0.7028 | Pass / fail |
| Extension mu512, 130 | 0.010417 | 0.022135 | 0.7034 | Pass / fail |
| Extension mu512, 140 | 0.009896 | **0.023177** | 0.7078 | Pass / fail |
| Extension mu512, 150 | 0.009115 | 0.022005 | 0.7214 | Pass / fail |

Epoch 100 is the best **sampled** extension checkpoint by primary n@5, improving on
P0b epoch 30 by 0.0015625, or 14.3% relative. Epochs 50, 90 and 100 exceed that
reference and pass the reconstruction guardrail; eight other sampled extension
checkpoints do not exceed it. This is a fluctuating, small absolute gain, not evidence
that additional epochs monotonically improve retrieval. The automatic extension decision
reports `screening_success: true` and selects epoch 100 under its numerical rule, but
its own continuation context states that the run was exploratory. This must not be
confused with success of the original predeclared screen.

All extension AUROCs miss the 0.75 classifier sanity target; epoch 50 is closest at
0.7419. For context, epoch 100 precision@5 is 0.4167 (bootstrap 95% interval
0.3964-0.4357), compared with P0b's 0.3539 (0.3367-0.3714). These intervals are for
precision@5, **not n@5**, and do not provide a paired significance test of the
selected checkpoint. The historical benchmark and its queries have been inspected
and used to select many experiments and checkpoints; no independent confirmation of
the n@5 gain is available.

### Reconstruction and training behavior

Every evaluated extension milestone passed the full-coverage tokenizer reconstruction
guardrail against both P0b and mu128 E1. At epoch 100, synthetic recipe total was
0.476809 versus P0b 0.493197 and mu128 0.485584: respectively 3.32% and 1.81% lower.
Real MAE was 0.674595 versus 0.710047 and 0.716298: respectively 4.99% and 5.82%
lower. These full-store results cover 5,000 synthetic and 5,000 real patches. The P0b
reconstruction control is `vae_best.pt` (seed-20260926 epoch 40), whereas the P0b
retrieval reference is epoch 30; the mu128 geology reconstruction control is epoch 30.
Do not conflate these checkpoints or the tokenizer scores with training CSV losses.

Synthetic validation loss improved from the original stage-2 best 0.190060 at epoch 39
to 0.185662 at extension epoch 138. Consequently the extension's `vae_best.pt` is
the **epoch-138 synthetic-validation-loss checkpoint**, not the epoch-100 retrieval
checkpoint; epoch 138 was not among the retrieval milestones. Extension epoch 150
ended at synthetic validation loss 0.195920. Training loss fell from 1.495735 at
epoch 41 to 1.370170 at epoch 150, but its objective includes geology terms, unlike
the validation loss, so the two values are not a conventional train/validation gap.

The per-epoch real-validation MAE was 0.738510 at epoch 41 and 0.789895 at epoch 150;
it fluctuated substantially in between. This is a warning to monitor real-domain
behavior, not proof of monotonic overfitting. The repeatedly logged real-test metric
is also not an untouched final test, and the per-epoch real-test subset is not the
same as the full-store tokenizer evaluation. Do not select epochs using real-test
results. The real volumes have already been used as historical guardrails.

### Artifacts

- [Extension comparison CSV](/Volumes/CrucialX9/3dvae-poc_checkpoints/p9_mu512_seed20260926/phase2_extension_110ep/latent_size_comparison.csv)
- [Extension decision and continuation caveat](/Volumes/CrucialX9/3dvae-poc_checkpoints/p9_mu512_seed20260926/phase2_extension_110ep/p9_mu512_decision.json)
- [Extension training metrics](/Volumes/CrucialX9/3dvae-poc_checkpoints/p9_mu512_seed20260926/phase2_extension_110ep/e1_mu512/stage2_geology/training_metrics.csv)
- [Epoch-100 retrieval benchmark](/Volumes/CrucialX9/3dvae-poc_checkpoints/p9_mu512_seed20260926/phase2_extension_110ep/mu512_ep100_zgeo.json)
- [Epoch-100 tokenizer reconstruction evaluation](/Volumes/CrucialX9/3dvae-poc_checkpoints/p9_mu512_seed20260926/phase2_extension_110ep/mu512_ep100_tokenizer.json)

TensorBoard:

```sh
uv run tensorboard --logdir /Volumes/CrucialX9/3dvae-poc_checkpoints/p9_mu512_seed20260926/phase2_extension_110ep
```

## Is there a likely winner over P0b?

**A numerical win already exists; a reliable, independently validated winner does not.**
The extension's epoch-100 checkpoint is the strongest currently observed same-seed
candidate among those discussed here and also reconstructs well. A previous P2
phase-rotation run for seed 20260926 achieved n@5 0.012240 at epoch 20 versus its
P0b 0.010938. However, the other seed peaked at 0.009115 versus its P0b 0.011979;
phase rotation did not replicate. Likewise, P3 stretch won on seed 20260925
(0.013021 versus 0.011979) but lost on seed 20260926 (0.008594 versus 0.010938).
Neither augmentation is a sound recommendation for another unchanged run.

The mu128 E1 pilot reached n@5 0.013802 versus its own seed-20260925 P0b 0.011979,
but its independent seed-20260926 run peaked at 0.010937500, fractionally below
its P0b 0.010937501. It passed reconstruction tolerance, but did not replicate the
strict retrieval gain. P1 loss changes did not produce a repeatable gain; P4 real
mixing failed its first-seed retrieval gate; P5/P6 and the five upstream architecture
bundles failed relevant reconstruction or retrieval gates. The latter five did not
reach geology training, so their retrieval potential was not measured. Classifier
AUROC and reconstruction gains alone do not establish a retrieval winner.

### Recommended order before the new experiment

1. **Evaluate existing candidates before training.** Keep P0b epoch 30, mu512 epoch
   100, mu512 epoch 50, mu512 epoch 138 (`vae_best.pt`), and the same-seed mu128 E1
   epoch 30 as named comparisons. Epoch 138 has not yet been assessed on retrieval;
   evaluating it is diagnostic, not independent validation or permission to pick
   whichever checkpoint happens to win. Check tokenizer preprocessing, embedding
   (`z_geo`), calibration, and split provenance. Compare per-geology cohorts,
   per-source-volume behavior where provenance permits, full-store reconstruction,
   precision/NDCG and hard-negative rates as well as n@5/n@10. Estimate a **paired
   n@5 difference** with uncertainty at the source-volume level where possible;
   query-only intervals can understate correlated-crop uncertainty. Any such analysis
   on the existing repeatedly used validation set remains exploratory.
2. **Prefer moving to the streamed plan rather than an open-ended legacy sweep.**
   Its S0 reference and matched S1 reconstruction-only control separate the new
   data/input contract from the overlap objective. The present metadata-neighbor
   metric is only a proxy for the user's fluid-aware semantic goal. Audit source
   provenance, labels and split exclusions; use validation volumes for selection,
   preserve the 30 newly generated labeled synthetic test volumes for a frozen
   final evaluation, and assess real-domain behavior separately. Existing real test
   volumes are historical guardrails, not a pristine final test. Resolve the plan's
   open contract decisions before implementing or launching it.
3. **Optional P0b duration control, if desired before switching:** seed-20260926
   P0b's best synthetic validation loss was still at epoch 40 (0.218169), but its
   n@5 slipped from 0.010938 at epoch 30 to 0.010677 at epoch 40. Additional
   training is therefore a plausible test, **not a likely proven win**. A bounded
   20-40 additional epochs from each seed's epoch-40 P0b checkpoint, with the original
   data, loss, batch and evaluation contract, would test whether the baseline itself
   benefits from more exposure. At the observed roughly 3.6 minutes per P0b epoch,
   20 extra epochs cost about 73 minutes per seed, plus evaluation and overhead.
   The checkpoints do not restore the original optimizer/scheduler trajectory:
   explicitly record a common restart policy and all parameter-group learning rates,
   scheduler, random seeds and sampling; call the result a warm-start continuation.
   Predeclare milestones, selection and stop rules, and compare each seed against
   its own epoch-30 P0b reference. Do not change augmentation or objective at the
   same time. This is a baseline-duration test, not a clean causal comparison with
   the already-reset mu512 extension.
4. **If the specific mu512 hypothesis is worth a separate legacy experiment:**
   predeclare a matched continuation with P0b and mu512 at equal extra stage-2
   exposure, same data/sampler, loss, reset policy and evaluation cadence. Match
   P0b and mu512 independently for each seed, rather than copying seed-20260926
   weights; a fresh seed-20260925 mu512 stage-1/stage-2 pipeline is required.
   A mu128 E1 arm at the same budget would help distinguish width from architecture
   and duration, but adds cost. Select on validation n@5, retain the +2% matched
   synthetic **and** real reconstruction guardrails, report secondary metrics and
   paired uncertainty, and require independent-seed replication before adoption.
   Because the existing 50/90/100 peaks were selected after inspection, this must be
   a new prospective test, not retroactive validation. This is a lower priority than
   the streamed semantic-retrieval work unless the legacy proxy itself is the goal.

Do not repeat the rejected phase/stretch/loss bundles unchanged, automatically
extend mu512 past 150, or proceed to 2048-D solely because of the exploratory
numerical pass. No earlier result demonstrates that any proposed run will win.

## Run artifacts and background

Original artifacts remain under `/Volumes/CrucialX9/3dvae-poc_checkpoints/p9_mu512_seed20260926`;
the extension is in its `phase2_extension_110ep` subdirectory. The P9 procedure is in
[the pretrain/transfer plan](../plans/2026-09-27__pretrain_v2_encoder_loss_augmentation_transfer_plan.md),
Section 9.12; the [mu512 details](../plans/2026-09-27__pretrain_v2_encoder_loss_augmentation_transfer_plan_mu512-details_.html)
explain the two stages. Earlier comparisons and the replication decision are in
[the WP0-P7 handoff](2026-09-29-wp0-v2-validation-verification-and-next-steps.md) and
[the E1 seed-2 report](2026-10-06-e1-seed2-replication-launch.md).
