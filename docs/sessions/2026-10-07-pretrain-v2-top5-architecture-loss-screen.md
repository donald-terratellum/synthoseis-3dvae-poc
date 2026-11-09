# Pretrain-v2 Top-Five Architecture/Loss Screen (2026-10-07)

## Workbook Ranking

Read `/Volumes/CrucialX9/pretrain_v2_checkpoints/epoch_component_metrics3a.xlsx`, sheet `epoch_components_metrics`. There are 1,425 rows with numeric BA ranks and 10 distinct recipes. Recipes were deduplicated on architecture fields E-H and loss fields K-P, ranked by minimum BA across epochs. LR (I) and derived parameter count (J) were retained for the top recipes but not used to define uniqueness.

| Workbook field | r006 | r004 | r007 | r001 | r011 |
| --- | --- | --- | --- | --- | --- |
| Source row / run date | 1275 / Jun 20 | 1330 / Jun 20 | 2251 / Jun 21 | 1260 / Jun 21 | 2331 / Jun 22 |
| BA rank (best epoch) | 1 (51) | 3 (56) | 3 (86) | 5 (54) | 7 (96) |
| Levels / hidden dims | 3 / 32-64-128 | 3 / 40-80-160 | 4 / 40-74-138-256 | 4 / 40-74-138-256 | 4 / 32-64-128-256 |
| Encoder block schedule | deeper 3-5-8 | deeper 3-5-8 | deepest 3-5-8-5 | deeper 3-4-8-4 | baseline 3-4-6-3 |
| Loss weights MSE, PMSE, MAE, LPIPS, TV, GDL | 0, 0, .995, .005, 0, 0 | 0, 0, .995, .005, .001, 0 | 0, 0, .999, 0, 0, .001 | 0, 0, .995, .005, 0, 0 | 0, 0, .99, 0, 0, .01 |
| Upstream AC at first logged epoch | .374729 @31 | .368422 @29 | .369458 @29 | .375096 @30 | .362386 @30 |
| Upstream AC epoch 20 | not logged | not logged | not logged | not logged | not logged |
| Best upstream AC (epoch) | .326880 (51) | .331173 (55) | .321402 (110) | .327102 (54) | .321349 (106) |
| POC stage-1 best synthetic val loss (epoch) | .389590 (59) | .376074 (57) | .468412 (58) | .481408 (57) | .465316 (57) |
| POC real-test MAE epoch 1 | .773628 | .773214 | .775889 | .774080 | .776784 |
| POC real-test MAE epoch 20 | .745148 | .743400 | .769972 | .769413 | .770124 |
| POC real-test MAE epoch 60 | .714349 | .709430 | .754813 | .757485 | .755391 |
| Full-tokenizer synthetic recipe total vs P0b .493197 | .424017 (-14.03%) | .415439 (-15.77%) | .512175 (+3.85%) | .520795 (+5.60%) | .512488 (+3.91%) |
| Full-tokenizer real MAE vs P0b .710047 | .750956 (+5.76%) | .750653 (+5.72%) | .776562 (+9.37%) | .775895 (+9.27%) | .776366 (+9.34%) |
| +2% reconstruction gate | FAIL | FAIL | FAIL | FAIL | FAIL |
| Geology / frozen-manifest retrieval | skipped / not run | skipped / not run | skipped / not run | skipped / not run | skipped / not run |
| VAE encoder source snapshot | epoch 50 | epoch 55 | epoch 85 | epoch 50 | epoch 95 |

The LR at each workbook row's best BA epoch (column I) was r006 `1.4e-5`, r004 `1.4e-5`,
r007 `1.4e-5`, r001 `2.6405e-5`, and r011 `1.4e-5`; parameter counts (column J) were
17.01M, 26.57M, 74.58M, 72.09M, and 58.92M respectively. Full upstream run-folder IDs and
source paths are in Section 2.1 of the main plan.

### AC Versus POC Real MAE

Both columns are MAE/L1 in name, but their values are not directly comparable:

- Upstream AC evaluates masked real inputs from four `.npy` test volumes using random 128x128x128 crops, global-volume-standard-deviation normalization, and the configured extrema/sparse/decimate plus trace-dropout mask. The upstream U-Net decoder also concatenates multiscale encoder skip features.
- POC `real_test_mae` evaluates clean inputs on 32x32x64 stored patches normalized individually by patch standard deviation. It uses the VAE's 128-D bottleneck and skip-free decoder.
- POC evaluates the first 1,080 of 5,000 stored real-test patches each epoch. Because the store is source-grouped, this subset is all CostaRica; upstream AC's 60 test batches are distributed among all four volumes.
- AC appears only from epochs 29-31 for these selected recipes, after the upstream test loader was introduced. The low epoch-1 rows elsewhere in the workbook are not same-recipe comparisons, and these recipes have no AC at epoch 20.

Therefore the apparent upstream AC (~0.32-0.37 after training) versus POC real MAE (~0.71-0.77) gap cannot diagnose LR or architecture by itself. Compare POC candidates against the same POC baseline/protocol. A direct cross-repo AC comparison would require matching crop size, normalization, masking, volume weighting, and decoder path.

### Completed POC Screen Outcome (2026-10-09)

All five candidates completed 60 reconstruction-only epochs and full exact-tokenizer evaluation
on 5,000 synthetic and 5,000 real patches. The same-seed P0b references were synthetic recipe
total `0.493197` and real MAE `0.710047`. r006/r004 substantially improved synthetic tokenizer
recipe total but exceeded the real-MAE tolerance; r007/r001/r011 exceeded tolerance on both
metrics. **Every candidate failed the +2% gate**, so every geology stage was skipped. No
frozen-manifest geology benchmark or n@5 result exists for this screen; no candidate is eligible
for independent replication or adoption. The machine-readable result is
`/Volumes/CrucialX9/3dvae-poc_checkpoints/pretrain_v2_top5_arch_loss_seed20261008/top5_screen_decision.json`.

The real-MAE regression occurred on all four test source-volume groups for every candidate, not
only on CostaRica. The full tokenizer per-volume reports are under each candidate's
`reconstruction_tokenizer_stage1.json` in the output root.

Recommendation: retain P0b and do not rerun the same bundles with an LR change. First inspect
the per-volume tokenizer reports for a localized real-domain regression and verify the POC real
training/validation sampling and scale contract. Any follow-up should isolate one factor (for
example, real-reconstruction contribution or encoder transfer/adaptation) against a matched P0b
control; do not start geology fine-tuning until the same full reconstruction gate passes. These
bundled upstream U-Net recipes did not establish architecture or retrieval uplift in the VAE.

Kernel schedules G are all 3 at each level. The top BA rows include repeated epochs of the same recipe; the table is five unique architecture/loss combinations, not five rows.

## Transfer and Runner

All five nearest saved upstream checkpoints loaded with this repo's `--init_encoder_from` path. Each matching ResNetV2 VAE produced a finite `(1, 1, 32, 32, 64)` reconstruction and a `(1, 128)` latent. Transfer only the upstream encoder: the upstream U-Net decoder's skips bypass `mu/z_geo`, so each POC VAE decoder is initialized fresh and trained through its latent bottleneck.

`scripts/pretrain_v2_top5_arch_loss_suite.sh` ran the five recipes sequentially with downstream seed 20261008. Each received 60 reconstruction-only epochs with `K=2` real patches per batch and its workbook MAE/LPIPS/TV/GDL weights. All five failed the full 5,000-synthetic/5,000-real exact-tokenizer gate against same-seed P0b, so the planned 40-epoch geology stages and frozen-manifest benchmarks were skipped. A candidate/loss bundle cannot establish architecture-only causality; any future winner needs a matched-loss P0b-encoder control and an independent downstream initialization before adoption.

Checks completed:

- 28 focused architecture/gate tests passed.
- All five actual pretrained encoder transfers and VAE forward checks passed.
- Bash syntax and side-effect-free print-only rendering passed for all five candidates, including source epochs and TV/GDL settings.
- r006 one-epoch MPS smoke used actual source weights and the real training store; reconstruction startup, adapter parity, and one geology fine-tuning epoch completed. Smoke bypassed the full tokenizer gate and is not a result.
- The smoke preceded a training-banner fix; its log displayed default MSE/PMSE fields for `multi_component`. The full-run logger now prints the active MAE/TV/GDL weights plus LPIPS accurately.
- The full five-candidate screen completed on 2026-10-09. All candidates failed the tokenizer real-MAE gate; no geology stages or retrieval benchmarks ran. No candidate is eligible for independent replication, and no adoption is implied.

## Launch and Monitoring

Default checkpoint output: `/Volumes/CrucialX9/3dvae-poc_checkpoints/pretrain_v2_top5_arch_loss_seed20261008`.

Durable log: `/Volumes/CrucialX9/3dvae-poc_logs/pretrain_v2_top5_arch_loss_seed20261008/pretrain_v2_top5_arch_loss_seed20261008.log`.

The completed run log is retained at the paths above. For review, follow/read it with:

```sh
tail -F /Users/donaldpg/synthoseis-3dvae-poc/logs/pretrain_v2_top5_arch_loss_seed20261008.log
```
