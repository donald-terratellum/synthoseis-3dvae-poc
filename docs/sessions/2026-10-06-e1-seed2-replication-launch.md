# E1 Seed-2 Replication Launch (2026-10-06)

## Final Status

The independent seed-20260926 E1 replication completed on 2026-10-07. E0/E1 reconstruction, the exact-tokenizer evaluation, the passing E1 reconstruction gate, 40 epochs of E1 geology fine-tuning, and all four frozen-manifest benchmarks completed. No processes from this run remain active, and no adoption was made.

The run is launched under `caffeinate -i` with the durable external log at `/Volumes/CrucialX9/3dvae-poc_logs/p7_e1_seed20260926_run/p7_e1_seed20260926_replication.log`; `logs/p7_e1_seed20260926_replication.log` is a workspace symlink to it.

Monitor with:

```sh
tail -f /Users/donaldpg/synthoseis-3dvae-poc/logs/p7_e1_seed20260926_replication.log
```

## Runner and Gates

`scripts/p7_e1_seed2_replication.sh` performs fresh paired E0/E1 reconstruction, exact-tokenizer evaluation of same-seed P0b/E0/E1 on 5000 synthetic and 5000 real patches, and stops before geology if E1 exceeds either P0b reconstruction metric by more than 2% or full coverage/contract audit is absent. Only a passing gate starts the 40-epoch E1 geology stage and epoch 10/20/30/40 frozen-manifest benchmarks. The decision JSON requires n@5 to beat the seed-2 P0b epoch-30 baseline; AUROC is recorded separately, and adoption remains manual.

The seed-2 P0b checkpoint and benchmark, all datasets and provenance arrays, validation manifest, MPS availability and external free space passed preflight. The seed-1 geology pilot was not used as initialization.

## Results

The 5000-synthetic/5000-real tokenizer reconstruction comparison passed for E1: synthetic recipe total 0.497442 vs same-seed P0b 0.493197 (+0.86%); real MAE 0.714574 vs 0.710047 (+0.64%). E0 failed the +2% real-MAE guardrail.

| Geology epoch | n@5 | n@10 | Macro AUROC | Precision@5 CI95 | NDCG@5 CI95 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.008073 | 0.016536 | 0.6829 | [0.3530, 0.3895] | [0.3801, 0.4134] |
| 20 | 0.010417 | 0.018099 | 0.7072 | [0.3674, 0.4040] | [0.4001, 0.4295] |
| 30 | 0.010937500 | 0.020182 | 0.7126 | [0.3750, 0.4100] | [0.4038, 0.4347] |
| 40 | 0.008594 | 0.018099 | 0.7037 | [0.3655, 0.4091] | [0.3970, 0.4319] |

Each benchmark has 768 eligible queries. Seed-2 P0b epoch 30 is n@5=0.010937501, n@10=0.016667. E1 epoch 30 is fractionally below the primary n@5 baseline (-9.31e-10), so it fails the strict improvement gate; n@10 is a tie-breaker only. Bootstrap intervals above are for precision@5 and NDCG@5, not n@5. All E1 AUROC values remain below the 0.75 sanity target.

**Decision: do not adopt E1; retain P0b.** Seed 1 cleared its own baseline, but adoption requires both seeds. No third seed, further E1 training, P8 or mu256/512 run is queued.

Machine-readable decision: `/Volumes/CrucialX9/3dvae-poc_checkpoints/p7_e1_seed20260926_replication/seed2_replication_decision.json`.

## Validation

- `tests/test_encoder_architecture.py`: 27 tests passed, including same-seed tokenizer gate requirements and the 2% regression boundary.
- Bash syntax and print-only rendering passed; dry-run created no output paths.
- MPS smoke completed paired one-epoch E0/E1 reconstruction, tokenizer diagnostics/adapter parity, and one E1 geology startup epoch. The smoke bypassed the full-store gate and is not an adoption result.

