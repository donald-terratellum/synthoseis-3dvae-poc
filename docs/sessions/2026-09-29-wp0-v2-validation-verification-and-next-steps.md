# WP0–P7 Transfer Experiments — Verification and Next-Steps Handoff

Started: 2026-09-29
Updated: 2026-10-05
Plan this belongs to: [docs/plans/2026-09-27__pretrain_v2_encoder_loss_augmentation_transfer_plan.md](../plans/2026-09-27__pretrain_v2_encoder_loss_augmentation_transfer_plan.md), Sections 8–9.

Checkpoint-location update (2026-10-05): the user moved most archived checkpoint folders
to `/Volumes/CrucialX9/3dvae-poc_checkpoints`; references to confirmed moved folders now
use that location. P0a, both P0b folders, and P4 remain under local `checkpoints/` as verified
on this date. This is a partial relocation, not a blanket root replacement. Historical log
commands may still show original paths. Runtime runner paths were not changed by this
documentation update; verify their inputs before rerunning older suites. Mount CrucialX9
before accessing archived checkpoints.

## Current status (2026-10-05)

P5a/P5b/P6, the P5a extension, and P7's first-seed reconstruction comparison are complete.
P7 finished both 150-epoch arms and both full-store evaluations, then stopped normally at
the reconstruction gate. That original suite is no longer active. The separately approved
E1 geology pilot is now running on MPS (40 epochs, mu128); no pilot retrieval result or
second seed exists yet. E1 passes the extrema-input gate but both
arms fail the clean-input gate; E0 also fails extrema. Retain P0b and do not silently override
the gate, extend E2 duration, or launch P8. The next recommendation is the no-training
tokenizer input-contract audit in plan Section 9.8 has completed, including all six probes
and full-store evaluation. E1 passes that exact-input reconstruction comparison; no audit
process remains. The user subsequently authorized the 40-epoch E1 geology pilot in Section
9.9 under a prospective exact-tokenizer reconstruction protocol. The pilot is implemented
and active (verification below), not adopted; the original P7 failed gate is retained.
The audit alone did not authorize new training. See the October 5 entries below.

The older verification instructions below are retained as provenance; all of their open items
are complete. The frozen v2 manifest matches the 5,000-patch dataset (`dataset_size=5000`,
2,048 benchmark indices), and P0, P0a, P0b, P1, P2, P3, and P4 have finished.

- P0b remains the control: best n@5 was 0.0120 for seed `20260925` and 0.0109 for seed
  `20260926` (best-per-run mean 0.01146).
- P1 loss recipes L1–L4 were rejected. The strongest provisional result did not reproduce in
  the second seed.
- WP3 phase rotation is implemented with vectorized FFT rotation, triangular angle sampling,
  CLI controls, and unit tests. The full suite passes: 218 tests.
- P2 used `--phase_rotation_prob 1.0 --phase_range -60 0 40`, with each seed warm-started from
  its matching P0b best checkpoint. Best n@5 was 0.0091 and 0.0122 (mean 0.01065), below P0b's
  mean; phase rotation is rejected and must not be carried forward.
- P2 checkpoints are under `/Volumes/CrucialX9/3dvae-poc_checkpoints/p2_phase_seed20260925` and
  `/Volumes/CrucialX9/3dvae-poc_checkpoints/p2_phase_seed20260926`. Reports are
  `docs/benchmarks/p2_phase_seed<seed>_ep<10|20|30|40>_zgeo.json`; the combined log is
  `logs/p2_phase_suite.log`.
- WP3 zoom-in stretch is implemented with label-consistent dip adjustment and no-padding
  enforcement. A real training-Zarr smoke check and all 222 tests pass.
- P3 seed `20260925` improved to n@5 0.0130, but seed `20260926` regressed to 0.0086. The
  best-per-run mean was 0.01081 versus P0b 0.01146, so zoom-in stretch is rejected.
- P3 checkpoints are under `/Volumes/CrucialX9/3dvae-poc_checkpoints/p3_zoom_stretch_seed20260925` and
  `/Volumes/CrucialX9/3dvae-poc_checkpoints/p3_zoom_stretch_seed20260926`; reports follow
  `docs/benchmarks/p3_zoom_stretch_seed<seed>_ep<10|20|30|40>_zgeo.json`.
- WP4 real patch stores contain 10,000 train, 2,500 spatial validation, and 5,000 test patches.
  Train/validation use Netherlands1–4 plus Penobscot3D; test uses CostaRica, Poseidon parts 1/2,
  and cn1. All patches have per-patch standard deviation 1, and source-specific inline origin
  audits confirmed the required 32-sample gap.
- Mixed batches preserve 12 synthetic examples and append `K=2` real examples. Real examples
  contribute reconstruction and KL only; every geology and GAN path uses the synthetic prefix.
- P4 seed `20260925` peaked at n@5 0.01094 and n@10 0.01979 at epoch 20, below its P0b
  control (0.01198 and 0.02214). Best real validation/test MAE was 0.70821/0.67208, but the
  primary retrieval gate failed, so P4 is rejected without replication.
- P4 checkpoints are under `checkpoints/p4_real_mix_seed20260925`; reports follow
  `docs/benchmarks/p4_real_mix_seed20260925_ep<10|20|30|40>_zgeo.json`, and the log is
  `logs/p4_real_mix_seed20260925.log`. The full suite passes: 226 tests.
- WP5/WP6 implementation is complete: configurable E0/E1/E2/E3/E4 models, architecture-aware
  checkpoint/tokenizer config, strict r006 EMA transfer, `zxy` input adapter, and scheduled encoder
  freezing/unfreezing. The real r006 checkpoint loaded 150/150 tensors; an MPS two-epoch smoke run
  and stage-2 resume both succeeded. Focused architecture tests cover all E1–E4 candidates.
- P5/P6 runners are `scripts/p5_encoder_transfer_suite.sh` and
  `scripts/p6_encoder_scratch_suite.sh`, with tail-able logs and print-only command rendering.
  They preserve the P0b recipe, keep phase/stretch off, and use no real training examples; real
  validation/test data are metrics-only.

### Recovery audit and next action (2026-10-03)

The interrupted Copilot session did not interrupt training. P5a finished on October 2;
P5b and P6 finished on October 3. Each completed 60 reconstruction epochs, 40 geology
epochs, and frozen-manifest benchmarks at epochs 10/20/30/40. All 60/40 epoch checkpoints
and both best checkpoints exist per arm. No traceback or interruption marker was found.
No training/evaluation process remains; TensorBoard alone is still running.

| Seed 20260925 arm | Best n@5 epoch | n@5 | n@10 | AUROC at that epoch | Best synthetic val loss | Best real-test MAE |
|---|---:|---:|---:|---:|---:|---:|
| P0b control | 30 | 0.011979 | 0.022135 | 0.6362 | 0.218987 | not logged in control CSV |
| P5a transfer, stage-1 classifier 0.05 | 40 | 0.011719 | 0.019661 | 0.7755 | 0.356375 | 0.717662 |
| P5b transfer, stage-1 classifier 0 | 40 | 0.011198 | 0.020573 | 0.7675 | 0.356796 | 0.710287 |
| P6 scratch, stage-1 classifier 0.05 | 30 | 0.010938 | 0.023177 | 0.7677 | 0.354272 | 0.717474 |

These are single-seed results; best reconstruction metrics need not belong to the best
retrieval epoch. None beats P0b n@5. Best synthetic validation losses are 62-63% worse
than P0b under the same MAE + LPIPS recipe. Classifier gains do not override either gate.
Do not replicate these arms or launch P8 (which requires P5 to win).

The v2 manifest was rechecked: dataset_size=5000, 2048 indices, maximum index 4996.
The full current suite passed 237 tests in 92 seconds; git diff --check passed.
The recovery audit made no code changes. WP4-WP6 code, runners, tests, and documentation
were still uncommitted after commit 24a5971; experiment artifacts remain excluded.

Next: the bounded reconstruction diagnostics defined in the plan's Section 9.2. P5/P6
changed both encoder and decoder versus P0b, so they do not isolate encoder capacity.
Evaluate the same fixed patches, deterministic mean and seeded sampled latents, separate
MAE/LPIPS/KL, axis round-trip, and fixed-patch decoder fitting before further long runs.
Retain P0b. P7 E1/E3 remains prospective; P8 remains gated. WP2 component logging/masking
and orthogonal LPIPS, remaining WP3 augmentations/stack support, and WP8 voxel decoding
are not implied complete by these results.

### Reconstruction diagnostics completed (2026-10-03)

Recovery checkpoint commit `7a0453a` was pushed to origin on the existing
`geology-classifier-2026-09-25` branch before diagnostic implementation. It preserves
WP4-WP6 implementation and the updated handoff without experiment artifacts.

Implemented `scripts/diagnose_p5_p6_reconstruction.py` and
`scripts/p5_p6_reconstruction_diagnostics.sh`, reusing training LPIPS and loss arithmetic.
The runner uses Python module execution to avoid `scripts/tokenize.py` shadowing the
stdlib `tokenize` module (the first path-based launch failed before checkpoint loading).
The repaired runner completed the full default budget: CPU, threads=2, seed=20261003,
32 identical seed-selected patches per split, batch=2, and 40 decoder-fit steps.
Stdout and stderr are in `logs/p5_p6_reconstruction_diagnostics.log`; structured results
are in `logs/p5_p6_reconstruction_diagnostics.json`. These remain local artifacts.

| Arm | Synthetic mean-latent MAE | Real mean-latent MAE | Native two-patch fit initial -> final | Fresh legacy fit initial -> final |
|---|---:|---:|---|---|
| P0b | 0.256640 | 0.708492 | 0.23634 -> 0.09035 | not run |
| P5a | 0.463623 | 0.753793 | 0.36002 -> 0.09077 | 0.46753 -> 0.15307 |
| P5b | 0.485695 | 0.745650 | not run | not run |
| P6 | 0.507599 | 0.749700 | 0.46511 -> 0.09098 | 0.46628 -> 0.14429 |

Interpretation:
- Strict full state-dict loads and asymmetric axis adapter/inverse checks passed.
- Sampled-minus-mean MAE magnitude was below 0.0006; weighted KL below 0.0001.
  The regression is predominantly reconstruction error, not sampling or KL accounting.
- Synthetic mean MAE regressed 81%/89%/98%; real mean MAE regressed 6.4%/5.2%/5.8%.
  These are 32-patch diagnostic deltas, not official full-set acceptance results.
- Native decoders fit these two fixed patches better than independently initialized legacy
  decoders in this short test. Initializations differ; this is not a fair superiority claim,
  and overfitting two patches does not establish generalization. Encoders and original
  checkpoint modules stayed unchanged. No saved weights or datasets were modified.
- Stage 1's best val epoch was 56/60 in every arm. Last-ten versus previous-ten mean
  validation loss improved 4.11% (P5a), 1.59% (P5b), and 2.97% (P6). More reconstruction
  training is plausible, but neither a root cause nor a successful fix has been demonstrated.
- Pylance syntax check, bash syntax, 14 focused tests, full 241-test suite, and diff checks pass.

Previous recommendation (superseded by the completed October 4 extension): extend only P5a reconstruction warmup toward the plan's
150-epoch ceiling, preserving original artifacts and checking component metrics on the
same patches. Require a reconstruction gate before another geology stage. No new long
training, P7/P8, or seed replication was launched during this diagnostic task.

Run again: `bash scripts/p5_p6_reconstruction_diagnostics.sh`.
Observe: `tail -n 50 -F logs/p5_p6_reconstruction_diagnostics.log`.
Overrides: `DEVICE`, `SAMPLES`, `FIT_STEPS`, `BATCH_SIZE`, `SEED`, `LOG_PATH`, `OUT_JSON`,
and `PYTHON`. Default runner output replaces its diagnostic log/report, not training files.

### Approved 150-epoch extension setup (2026-10-03)

The user authorized P5a reconstruction training to 150 total epochs and external-drive
checkpoint/log storage. `scripts/p5a_reconstruction_extension.sh` resumes stage-1 epoch 60
for 90 additional epochs. No stage-2, P7/P8, or replication command is included.

Verified limitation: source checkpoints contain no AdamW moments or scheduler/RNG state.
The experiment continues weights with reset optimizer/scheduler and restored final learning
rates (encoder 2.5e-5; decoder/heads 2.5e-4), not an exact uninterrupted continuation.
Keep this confound in interpreting undertraining. Recipe/architecture and stage-1 classifier
weight 0.05 are otherwise retained; the trunk is not frozen again.

New checkpoints go to `/Volumes/CrucialX9/3dvae-poc_checkpoints/p5a_reconstruction_150_seed20260925`;
the log goes to `/Volumes/CrucialX9/3dvae-poc_logs/p5a_reconstruction_150_seed20260925.log`,
with a workspace symlink in `logs/`. Old artifacts stay in place. CrucialX9 has 886 GiB free,
versus 54 GiB internal. The runner requires 15 GiB output-drive headroom and refuses overwrites.

Before training: 32-patch comparison against P0b's reconstruction-best checkpoint.
During training: component JSONs at absolute epochs 90/120/150.
After training: full 5000 synthetic + 5000 real comparison for P0b, extension best-val, and
epoch 150, with recorded reconstruction gate. No geology stage launches automatically.
Reports and checkpoints are local experiment artifacts. All 243 tests and syntax gates pass.

Monitor: `tail -n 50 -F /Users/donaldpg/synthoseis-3dvae-poc/logs/p5a_reconstruction_150_seed20260925.log`.

Launch verified: detached `nohup bash scripts/p5a_reconstruction_extension.sh` runner PID
98230; training child PID 98294. Preflight and baseline comparison completed, saved
architecture weights loaded, and epoch numbering resumed at 61 with target 150 on MPS.
The process was active at the October 3 launch verification; it has now finished. Additional
launcher output is in the external log folder as `p5a_reconstruction_150_seed20260925.launch.log`.
Current extension code/documentation changes are uncommitted; last pushed diagnostic commit
remains `f872e13`. No original checkpoint/log was relocated.

### Extension outcome and next step (2026-10-04)

Completion verified in the external log: both epoch-150 training and the final full-store
evaluation wrote COMPLETE markers. No training/evaluation/runner process remains. The
extension CSV contains exactly 90 rows (epochs 61-150); all 90 epoch checkpoints exist.
Epoch 150 is also the extension's best-val checkpoint, so the two final report arms are
the same model, not independent results. No additional geology training was launched.

| Full-store clean-input evaluation (5000 patches per split) | P0b recon-best | P5a epoch 150 | Change |
|---|---:|---:|---:|
| Synthetic sampled MAE | 0.254361 | 0.501105 | +97.01% |
| Synthetic recipe total | 0.264555 | 0.519562 | about +96.4% |
| Real sampled MAE | 0.713747 | 0.746218 | +4.55% |

Both final report arms fail +2%. Report path:
`/Volumes/CrucialX9/3dvae-poc_logs/p5a_reconstruction_150_seed20260925_full_reconstruction_gate.json`.

Training-validation CSV loss improved from stage-1 epoch 60's 0.391624 to 0.333777 at
epoch 150, but remains 52.42% above P0b's best 0.218987. The last-ten mean was 0.346810,
versus 0.345549 for epochs 131-140 (0.36% worse), so there is no sustained late-window
improvement. On the identical seed-selected 32 clean patches, epoch 60 -> 150 synthetic
MAE improved only 3.39% (0.558373 -> 0.539430); real MAE improved 1.09%
(0.753218 -> 0.745011). Do not compare those subset numbers with full-store numbers.

**Evaluation caveat discovered and checked:** the diagnostics feed clean inputs, whereas
`train.validate` defaults to extrema-only masking for this recipe. CSV validation also
uses a fixed 1080-patch prefix rather than all 5000 examples. Therefore the final clean-input
report does not reproduce the CSV validation protocol; do not present it as a full masked-input
guardrail certificate. A focused paired check used the same 32 indices, original loader,
extrema-only inputs, no augmentation, and identical seeded latent noise on MPS:
- Synthetic recipe total: P0b 0.249360, P5a150 0.364212 (+46.06%).
- Synthetic MAE: P0b 0.239615, P5a150 0.350843 (+46.42%).
- Real MAE: P0b 0.705859, P5a150 0.757612 (+7.33%).
Thus the regression also persists under matched masked inputs on this subset. This is
not a full masked-store evaluation. Log:
`/Volumes/CrucialX9/3dvae-poc_logs/p5a_reconstruction_150_input_parity_check.log`.

**Decision:** extra reconstruction training with the documented optimizer/scheduler reset
helped somewhat but did not rescue E2. Undertraining alone is not supported as a sufficient
explanation; this does not establish a unique root cause or prove that E2 cannot work.
No further duration-only extension, geology stage, replication, or P8 is recommended.

**Next proposal:** make evaluation input mode explicit (`clean` versus `extrema`), with
matching targets, scaling, selection and batch policy for every control/candidate. Then
prepare P7 E1 residual with the original skip-free decoder, matched against a newly trained
E0 reconstruction control on the same two-stage budget and data. P0b remains the acceptance
reference but is not a fair from-scratch duration control. Gate reconstruction before any
geology stage; replicate only a candidate that improves retrieval. E3 remains a later
spatial-resolution experiment; P8 remains gated on a successful P5 result. These are
proposals, not launched runs. This outcome review changed documentation only, did not
rerun the full test suite, and did not commit or push the uncommitted extension changes.

### Pre-P7 audit complete (2026-10-04)

The interrupted validation was completed: 18 focused tests and all 245 tests pass; Python
syntax, bash syntax and diff checks pass. Executed `bash scripts/pre_p7_readiness_audit.sh`;
the log ends with COMPLETE and all four expected reports exist. No model training occurred.
The script evaluates P0b recon-best and P5a epoch150 on 128 fixed synthetic train examples,
128 fixed validation examples and 128 real-test examples, in both clean/extrema modes.
Input mode is now explicit in evaluator flags/settings/dataset metadata; targets stay clean.

| Synthetic sampled MAE | P0b | P5a150 |
|---|---:|---:|
| Train / clean | 0.286352 | 0.495011 |
| Validation / clean | 0.267464 | 0.500459 |
| Train / extrema | 0.274121 | 0.347024 |
| Validation / extrema | 0.250555 | 0.344187 |

The E2 regression appears on train and val, not simply a low-training/high-validation
overfit pattern. Their sampling distributions differ, so these small probes are not a
generalization certificate. Exposure arithmetic: 5400 examples/epoch, 810000 over 150,
7.5 corpus-equivalent passes. Independent uniform shuffles would imply expected 99.9544%
unique coverage, but actual historical indices/RNG are unavailable and the extension reseeded.
Do not report this expectation as measured coverage.

Log: `logs/pre_p7_readiness_audit_20261004.log`.
Reports: `/Volumes/CrucialX9/3dvae-poc_logs/pre_p7_readiness_audit_20261004/{train,validation}_{clean,extrema}.json`.
Monitor/review: `tail -n 50 -F /Users/donaldpg/synthoseis-3dvae-poc/logs/pre_p7_readiness_audit_20261004.log`.
Reruns require new LOG_PATH/REPORT_DIR because the script refuses overwriting these results.

At audit completion P7 had NOT started and its suite did not exist; this is superseded by
the implementation/launch entry below. The plan now has
a concrete implementation/launch specification in Section 9.6 (matched fresh E0/E1,
legacy decoder, mu128, 150 reconstruction epochs then separately gated 40 geology epochs,
prospective sample coverage and exact-resume state). Section 9.7 describes a proposed
128/256 x classifier-off/on factorial and the production-contract caveat; no larger-latent
training or migration was launched. P0b classifier=0.1/SupCon=0.5 was verified in logs;
it is not a reconstruction-only comparator. Code and documentation remain uncommitted.

### P7 implementation and launch authorization (2026-10-04)

The user authorized starting P7 when ready. Implemented `scripts/p7_e0_e1_suite.sh` and
`scripts/p7_control.py`. E0/E1 share initial FC projections/decoder/heads, with the original
decoder and mu128; parameter counts are 3502436/3793652. Stage 1 is 150 epochs each,
450 batches of 12, run sequentially on MPS, no new data, phase/stretch/real mixing off.
First seed is 20260925. Full 5000-example clean/extrema reconstruction comparisons against
P0b and paired E0 gate both optional 40-epoch geology stages; no automatic seed replication.

New opt-in flags preserve default training behavior: paired index/worker schedules,
consumed-index/source-volume/cumulative coverage JSONs, both-mode train/val/real milestones,
and full epoch-boundary optimizer/scheduler/RNG state. Use `--resume_training_state` only
for same-arm paired reconstruction; geology stage deliberately warm-starts weights.
Atomic epoch/best/last checkpoints contain resumable state after diagnostics and plotting.

Validation: all 249 tests and 22 focused tests pass; syntax, diff, all-stage dry rendering,
and real-data MPS smoke pass. One smoke failed on nonexistent latent/base CLI flags
(removed; train.py hardcodes the required 128/16). Another showed GPU weight differences
also present in two fresh runs, not a lost-RNG resume defect. Resumed/uninterrupted runs
restore RNGs/group configuration exactly and reconstruction outputs within 0.0002 absolute
tolerance; no bit-identical MPS weight claim. Successful log: `logs/p7_smoke_retry2_seed20260925.log`.

New suite root: `/Volumes/CrucialX9/3dvae-poc_checkpoints/p7_e0_e1_seed20260925`.
Aggregate log: `/Volumes/CrucialX9/3dvae-poc_logs/p7_e0_e1_seed20260925.log`, linked into
workspace `logs/`. Separate E0/E1 reconstruction logs are in that external log folder.
Command: `bash scripts/p7_e0_e1_suite.sh`; use nohup for independence from the chat.
Monitor: `tail -n 50 -F /Users/donaldpg/synthoseis-3dvae-poc/logs/p7_e0_e1_seed20260925.log`.
The implementation is validated and authorized; the launch verification below records
the active process, not completion. Changes remain uncommitted and no artifacts are adopted.

Launch verified: `nohup caffeinate -i bash scripts/p7_e0_e1_suite.sh`, detached launcher
PID 30739, idle-sleep inhibitor PID 30760, active E0 MPS training child PID 30795.
Preflight and both shared initializations completed. E0 stage 1 started at epoch 1/150;
E1 has not started and is queued after E0. Full clean/extrema reconstruction gates then
decide whether both geology stages run. No second-seed or larger-mu run is queued.
Launcher output: `/Volumes/CrucialX9/3dvae-poc_logs/p7_e0_e1_seed20260925.launch.log`.
Aggregate tail command above works through the workspace symlink. This launch verification
does not claim any long-run result, completed epoch, or adoption. Latest code remains uncommitted.

### P7 completed and stopped at reconstruction gate (2026-10-05)

Verified aggregate log ends `P7 STOP: reconstruction gate failed; no geology or replication
launched.` Both arms have exactly 150 CSV epoch rows and 150 epoch checkpoints. Both
full-store clean/extrema reports and `reconstruction_gate.json` exist. No P7 process
remains. E0 best CSV val_loss is 0.235184 at epoch 146; E1 is 0.226063 at epoch 149,
versus P0b's historical best 0.218987. These CSV values use masked prefix validation,
not the full-store recipe totals below. All values below are sampled-latent full-store
evaluation of reconstruction-best checkpoints (5000 synthetic and 5000 real examples).

| Input mode | Arm | Synthetic MAE | Synthetic recipe total | Delta total vs P0b | Real-test MAE | Delta MAE vs P0b |
|---|---|---:|---:|---:|---:|---:|
| Clean | P0b | 0.254361 | 0.264555 | reference | 0.713747 | reference |
| Clean | E0 | 0.368546 | 0.386625 | +46.142% | 0.801460 | +12.289% |
| Clean | E1 | 0.410400 | 0.425385 | +60.793% | 0.778106 | +9.017% |
| Extrema | P0b | 0.240898 | 0.250742 | reference | 0.709170 | reference |
| Extrema | E0 | 0.254816 | 0.264147 | +5.346% | 0.727720 | +2.616% |
| Extrema | E1 | 0.244237 | 0.253900 | +1.259% | 0.714787 | +0.792% |

The implemented gate requires both modes and both arms to pass. Decision JSON:
`clean: e0=false/e1=false; extrema: e0=false/e1=true; geology_eligible=false`.
E1 extrema recipe total improves 3.879% versus paired E0 and real MAE improves 1.777%.
E1 clean recipe total is 10.025% worse than E0, while real MAE improves 2.914%.
Residual encoding is therefore promising for the trained masked-input task but is not
an adopted retrieval improvement. No n@5/n@10/AUROC stage-2 results exist for P7.

Coverage was checked from every epoch JSON, not inferred: E0/E1 consumed the exact same
indices for all 150 epochs, 5400 examples each epoch and 810000 exposures per arm.
Both reached 107951/108000 unique patches (99.95463%), leaving 49 unseen; all 180 source
volumes were represented in epoch 150. This removes inadequate corpus index coverage as
a supported explanation for the P7 result, not inadequate optimization time or sampling
distribution mismatch generally. Do not describe 150 partial epochs as 150 full passes.

Artifacts: `/Volumes/CrucialX9/3dvae-poc_checkpoints/p7_e0_e1_seed20260925`, especially
`reconstruction_clean.json`, `reconstruction_extrema.json`, `reconstruction_gate.json`, and
`<e0|e1>/stage1_reconstruction/sampling_epochN.json`. The completed log remains at
`logs/p7_e0_e1_seed20260925.log`; its tail will not show additional progress.

**Next:** audit the actual tokenizer contract before more architecture/capacity training.
`src/tokenizer/core/preprocess.py:preprocess_for_token` divides by each patch's standard
deviation and then keeps extrema; indexing/search/UI use this preprocessor. Current
reconstruction extrema reports mask the baked-in target without this additional per-patch
normalization, so they are not a direct deployment-contract certificate. The clean gate
remains failed and must not be dropped post hoc without explicit protocol approval.

Plan Section 9.8 now specifies an evaluation-only, logged audit on P0b/E0/E1: exact tokenizer
inputs, 128-patch probe, then all validation/real examples, target-scaling conventions,
adapter/encoder parity, components, source-volume metrics and decision branches. No new
training, gate change, second seed or adoption is authorized by recording that specification.
If approved after the audit, a separate E1 geology pilot is more directly motivated by
the current masked-task result than jumping to 512-dimensional mu. The 128/256 factorial
in Section 9.7 remains a separate optional hypothesis test, not a demonstrated fix.

This update concerns only experiment results and next-step instructions. It changes
documentation only; diff validation passes, no full tests were rerun, no code committed
or pushed, and no long experiment was launched by this status review.

### Exact tokenizer contract audit launch history (2026-10-05; completed below)

Implemented `--input_mode tokenizer` and `--contract_audit` in the existing reconstruction
evaluator, reusing preprocess_for_token rather than recreating normalization. Targets remain
clean baked-scaled patches; predictions are not rescaled. Reports record normalization
divisors, source names/indices, MAE/MSE/LPIPS/KL, amplitude, per-volume and volume-balanced
MAE. The evaluator verifies the search-engine default preprocessor and direct mu/z_geo
parity with VaeLatentAdapter on already-prepared input. No retrieval-quality claim is made
for P7 stage-1 projection heads.

Runner: `bash scripts/p7_input_contract_audit.sh`. All 252 tests and 25 focused tests pass,
with bash syntax, editor diagnostics and diff checks clean. The initial synchronous run
stopped without a recorded Python exception after five probes; no result was inferred from
the partial log. Added guarded `RESUME=1` reuse of reports only with identical settings,
all three arms, and passing adapter parity. The audit resumed detached using nohup and
caffeinate: runner PID 78535, full-store evaluator PID 78607 at verification.

Six probes are complete (128 fixed train, val and real examples under clean/extrema/tokenizer).
Provisional tokenizer-mode sampled metrics:

| Probe | Arm | Synthetic MAE | Synthetic recipe total | Real MAE |
|---|---|---:|---:|---:|
| Train | P0b | 0.450784 | 0.467446 | 0.699098 |
| Train | E0 | 0.462251 | 0.478489 | 0.718024 |
| Train | E1 | 0.454629 | 0.470493 | 0.705327 |
| Validation | P0b | 0.477099 | 0.495476 | 0.699098 |
| Validation | E0 | 0.485818 | 0.503900 | 0.718024 |
| Validation | E1 | 0.475830 | 0.493086 | 0.705327 |

These probes favor considering E1, not adopting or launching it: the full 5000-example
synthetic/real evaluation and `decision_summary.json` are still pending. Absolute synthetic
errors rose for all models under per-patch-normalized input versus baked-scaled targets;
this target/input convention must remain explicit in interpreting comparisons.

Log: `logs/p7_input_contract_audit_20261005.log`.
Observe: `tail -n 50 -F /Users/donaldpg/synthoseis-3dvae-poc/logs/p7_input_contract_audit_20261005.log`.
Reports: `/Volumes/CrucialX9/3dvae-poc_logs/p7_input_contract_audit_20261005`.
The bash script appends both stdout/stderr; it does not call train.py or edit P7's gate.
At that progress checkpoint no E1 geology pilot, larger-mu experiment, replication, or P8 had been launched. The
conditional E1 40-epoch training proposal is detailed in plan Section 9.9, requiring the
completed audit and explicit prospective protocol approval first. Changes remain uncommitted.

### Exact tokenizer contract audit completed and pilot recommendation (2026-10-05)

Completion verified: log ends COMPLETE, `full_tokenizer.json` and `decision_summary.json`
exist with full 5000-patch coverage in each split, and no audit/evaluator process remains.
All three checkpoints passed exact mu/z_geo adapter parity (both maximum differences 0).
Targets remained clean baked-scaled data; no output rescaling or model training occurred.

| Checkpoint | Synthetic MAE | Synthetic recipe total | Real MAE | Volume-balanced real MAE |
|---|---:|---:|---:|---:|
| P0b | 0.469916 | 0.488331 | 0.709170 | 0.710936 |
| E0 | 0.479050 | 0.497250 | 0.727720 | 0.728443 |
| E1 | 0.470972 | 0.488592 | 0.714787 | 0.716571 |

E1 versus P0b: synthetic recipe total +0.05344%, real MAE +0.79206% (both within +2%).
E1 versus paired E0: synthetic total -1.74125%, real MAE -1.77715% (both improved).
Decision summary marks `e1_contract_reconstruction_within_2pct=true`, while
`original_clean_extrema_gate_changed=false` and `training_launched=false`.
Real per-volume E1 MAEs: CostaRica 0.680891, Poseidon part1 0.724188, Poseidon part2
0.729447, cn1 0.731758. All are slightly worse than P0b; this is guardrail parity, not
a real-seismic reconstruction improvement. Volume-balanced diagnostics tell the same story.

Learning: the clean-input rejection did not translate into a large relative reconstruction
penalty under actual tokenizer inputs. Residual E1 is competitive with P0b and better than
fresh E0 for this contract. This does not establish the cause of clean-mode behavior, prove
a latent-capacity limit, or measure retrieval improvement. All models' synthetic absolute
MAE is higher under normalized inputs versus baked targets; do not mix these units/modes
with historical masked-prefix CSV metrics or silently redefine targets.

Recommendation: explicitly approve the prospective exact-tokenizer reconstruction protocol
and E1 40-epoch geology pilot specified in Section 9.9. Preserve historical clean/extrema
failures and the original gate JSON. First-seed n@5 must exceed P0b 0.011979 with the
approved reconstruction/real guardrails before second-seed replication; two seeds precede
adoption. Keep mu128; defer 256/512 and P8. The pilot script still needs implementation,
tests, command rendering and MPS smoke after that approval. Nothing new was trained or
launched by this result review. Existing audit log is complete and will not advance.

Artifacts: `/Volumes/CrucialX9/3dvae-poc_logs/p7_input_contract_audit_20261005/full_tokenizer.json`
and `decision_summary.json`; completed progress log is `logs/p7_input_contract_audit_20261005.log`.
This result update changes documentation only. Diff validation passes; implementation's
252-test gate remains the last executable test run. Changes remain uncommitted.

### E1 geology pilot authorized and validated (2026-10-05)

The user requested execution of the recommended mu128 pilot, explicitly approving the
prospective tokenizer-contract reconstruction guardrail. Implemented
`scripts/p7_e1_geology_pilot.sh`: warm-start E1 reconstruction-best epoch149 with fresh
AdamW/plateau state, number epochs reset to 1-40, residual encoder/original decoder,
mu128, classifier0.1/SupCon0.5/T0.2, original presence sampler, canonical synthetic batch12,
450 batches/epoch, no phase/stretch/real mixing. P7's original gate is not modified.

Passed 25 focused tests, all 252 tests, bash syntax, print-only command rendering, editor
diagnostics and real-data two-batch MPS smoke. Smoke met sampler presence quotas and
wrote clean/extrema/tokenizer diagnostics. Smoke log:
`logs/p7_e1_geology_pilot_smoke_seed20260925.log`. No smoke result qualifies for adoption.

Checkpoint output: `/Volumes/CrucialX9/3dvae-poc_checkpoints/p7_e1_geology_pilot_seed20260925`.
Log: `/Volumes/CrucialX9/3dvae-poc_logs/p7_e1_geology_pilot_seed20260925.log`, workspace
symlink `logs/p7_e1_geology_pilot_seed20260925.log`. External drive had 799 GiB free at
preflight. Script refuses existing output/logs and verifies source/audit/targets/provenance,
manifest/control references and MPS. Original data/checkpoints/manifest/gate are preserved.

After training, benchmark epoch10/20/30/40 z_geo and natural-prevalence classifier metrics,
then full tokenizer reconstruction for all four epochs against P0b. `pilot_decision.json`
filters reconstruction/real guardrails and strict n@5 improvement over the exact P0b epoch30
reference, ranking by n@5/n@10 only. It does not launch a second seed or adopt a model.
Warm-start checkpoints do not claim full geology optimizer/RNG continuation support.

Observe: `tail -n 50 -F /Users/donaldpg/synthoseis-3dvae-poc/logs/p7_e1_geology_pilot_seed20260925.log`.
Run: `bash scripts/p7_e1_geology_pilot.sh`; PRINT_COMMANDS=1 and SMOKE=1 are supported.
Pilot completion and launch outcome are recorded below. No prior failed P7 result is erased.

### E1 geology pilot completed; second-seed recommendation (2026-10-06)

The first-seed 40-epoch pilot and post-training evaluation completed; no process remains.
Epoch40 is the best n@5 checkpoint. Frozen-manifest eligible query_count=768.

| Epoch | n@5 | n@10 | Macro AUROC | Macro F1 |
|---:|---:|---:|---:|---:|
| 10 | 0.010937 | 0.019141 | 0.6909 | 0.3350 |
| 20 | 0.011458 | 0.017839 | 0.7218 | 0.3432 |
| 30 | 0.011458 | 0.023698 | 0.7353 | 0.3555 |
| 40 | **0.013802** | 0.023177 | 0.7412 | 0.3571 |

Seed1 P0b epoch30 is n@5=0.011979/n@10=0.022135; epoch40 pilot gains are +15.21%/+4.71%.
Seed2 P0b epoch30 reference is 0.0109375/0.0166667. The pilot has no bootstrap interval
for n@5 itself. Available epoch40 precision@5 CI95 is [0.396589,0.435814] and NDCG@5
CI95 [0.440451,0.476648], n=768; these are different metrics, not n@5 uncertainty.
Macro AUROC 0.7412 misses the 0.75 sanity target by 0.0088 and must remain reported.

All epoch10/20/30/40 checkpoints pass the full tokenizer reconstruction guardrail.
At epoch40, synthetic recipe total 0.488592 vs P0b 0.488331 (+0.05344%); real MAE
0.714787 vs 0.709170 (+0.79206%). E1 also improves over P7 E0 by 1.74125% synthetic
recipe total and 1.77715% real MAE. Volume-balanced real MAE is 0.716571; all four
source-volume MAEs are slightly worse than P0b. Thus guardrail parity passes, but this
pilot does not improve real reconstruction and has not yet replicated.

Recommendation: replicate the full E1 pipeline on fresh seed20260926, from scratch, rather
than adopt, warm-start from seed1, jump to 256/512 mu, or proceed to P8. Same-seed P0b
epoch30 reference exists (n@5 0.0109375/n@10 0.0166667). The second seed must get its own
fresh E1 reconstruction stage and exact-tokenizer full-store gate before a geology stage;
then use the same 40-epoch recipe and frozen manifest. Require each seed to beat its own
P0b n@5 reference and pass reconstruction before considering adoption. This tests pipeline
repeatability, not isolated E1 causality, because no matched E0 geology stage exists. If
the second seed misses either gate, stop and retain P0b. Report AUROC sanity gap; do not
replace n@5 with classifier score. No second seed has launched yet.

The plan's Section 9.10 contains the detailed seed2 protocol, paths and gates. The current
pilot script is fixed to seed20260925, so implement/validate a seed-parameterized replication
runner before launch. Pilot artifacts: `/Volumes/CrucialX9/3dvae-poc_checkpoints/p7_e1_geology_pilot_seed20260925`;
log/report paths are recorded in Section 9.9. No weights adopted. Documentation-only review;
diff check passes; the 252-test implementation gate remains the last full test run.

### Previous next action (2026-10-01; superseded)

P1–P4 produced no repeatable primary-metric gain, so retain unchanged P0b. Do not replicate P4
or carry real mixing, phase rotation, or zoom-in stretch into P5/P6. Implementation and focused
validation gates pass. The inspected P5a/P5b/P6 seed `20260925` sequence launched on 2026-10-01,
running sequentially on MPS; P5a stage 1 is active. Monitor with
`tail -f logs/p5a_seed20260925.log`; later arms write `logs/p5b_seed20260925.log` and
`logs/p6_seed20260925.log`. P5a completed: its best n@5 was **0.01172** at epoch 40 (n@10
0.01966), below the P0b seed-1 control (0.01198 / 0.02214); classifier macro AUROC reached
0.775, and best real-test MAE was 0.71766. P5b stage 1 is underway at epoch 11/60, with the
encoder unfrozen after its five-epoch warmup; P6 has not started. These are provisional single-
seed results, not adoption decisions. Generated stores, logs, checkpoints, and benchmark reports
remain experiment artifacts and should not be committed unless explicitly requested.

This file is written so a lower-cost agentic model can execute it with minimal judgment calls.
Every step has an exact command and an unambiguous pass/fail check. Do the steps in order; do
not skip the verification step even though it looks like the pipeline already finished.

Run everything from the repo root with the venv active:

```bash
cd /Users/donaldpg/synthoseis-3dvae-poc
source .venv/bin/activate
```

---

## 1. Already verified (2026-09-29) — no action needed

These were checked and passed. Re-run only if you suspect something changed:

1. All 25 volumes exist:
   ```bash
   ls /Volumes/CrucialX9/fake_data_validation_v2 | grep -c '^seismic__'
   # Expect: 25
   ls /Volumes/CrucialX9/fake_data_validation_v2 | grep '^seismic__' | sed 's/.*run_//' | sort -n | sed -n '1p;$p'
   # Expect: 5000
   #         5024
   ```
2. No partial/leftover generation folders:
   ```bash
   ls /Volumes/CrucialX9/fake_data_validation_v2 | grep temp_folder
   # Expect: no output
   ```
3. Every volume has all 8 required label arrays (checked with a Python loop over
   `zarr.open(...).keys()` against `fault_segments_id`, `fault_intersection_segments`,
   `faults/faulted_channel_labels`, `closure_segments_id`, `onlap_segments`,
   `faulted_lithology`, `flat_spot`, `geologic_age_faulted`) — zero volumes missing anything.
4. Disk: `/Volumes/CrucialX9/fake_data_validation_v2` is 78G; CrucialX9 has 886Gi free of
   1.8Ti (53% used). No disk pressure.
5. `logs/wp0_generate_v2_validation_3.log` shows the pipeline reached the end: all 25 volumes
   scanned, `Wrote 5000 patches to data/synth_val_v2_32-32-64.zarr`,
   `Wrote 5000 patches to data/synth_val_v2_uniform_32-32-64.zarr`,
   `Wrote benchmark report: docs/benchmarks/adopted_v2_reference.json`.

---

## 2. OPEN ITEM — verify the manifest is not stale (do this first, do not skip)

**Why this matters:** before the real 25-volume run, a small throwaway test was run with only
1 volume present (`run_5000`), to sanity-check the generation script. That test sampled only
200 patches and ran `scripts/evaluate_geology_benchmark.py`, which — if
`docs/benchmarks/frozen_validation_manifest_v2.json` did not already exist — would have created
it with `dataset_size: 200`. After the test, the placeholder data files and benchmark JSON were
deleted, but the manifest file itself was not deleted in that cleanup.

`scripts/evaluate_geology_benchmark.py`'s `_load_or_create_manifest()` (around line 105) does
this:

```python
if manifest_path.exists():
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "indices" not in payload:
        raise ValueError(...)
    return payload   # returns AS-IS — no check that dataset_size matches the current dataset
```

So if the stale 200-row manifest was still on disk when the real 25-volume run reached its
benchmark step, `docs/benchmarks/adopted_v2_reference.json` would have been computed against a
200-example index space, not the real 5000-patch v2 validation set — silently, with no error.
A first look at the current manifest showed `dataset_size: 5000` (which would mean it's fine),
but this was not conclusively confirmed before this handoff, so **do not trust it — verify it
directly with the commands below.**

### 2.1 Run this exact check

```bash
.venv/bin/python -c "
import json, zarr
m = json.load(open('docs/benchmarks/frozen_validation_manifest_v2.json'))
z = zarr.open('data/synth_val_v2_32-32-64.zarr', mode='r')
real_size = int(z['patches'].shape[0])
print('manifest dataset_size:', m['dataset_size'])
print('real dataset size:', real_size)
print('manifest benchmark_size:', m['benchmark_size'], 'n_indices:', len(m['indices']))
print('max index in manifest:', max(m['indices']))
print('RESULT:', 'MATCH' if m['dataset_size'] == real_size else 'MISMATCH_STALE_MANIFEST')
"
```

### 2.2 Decision

- **Prints `RESULT: MATCH`** → the manifest is correct and reflects the real 5000-patch v2
  dataset. Do nothing further here. Go to Section 3.
- **Prints `RESULT: MISMATCH_STALE_MANIFEST`** → the manifest is stale. Fix it:

  ```bash
  rm -f docs/benchmarks/frozen_validation_manifest_v2.json docs/benchmarks/adopted_v2_reference.json

  .venv/bin/python scripts/evaluate_geology_benchmark.py \
    --data data/synth_val_v2_32-32-64.zarr \
    --manifest docs/benchmarks/frozen_validation_manifest_v2.json \
    --benchmark_size 2048 \
    --seed 20260927 \
    --checkpoint /Volumes/CrucialX9/3dvae-poc_checkpoints/geoaware_v3_phase2_20260831/vae_epoch20.pt \
    --use_geo_embedding \
    --metadata_keys meta_fault_fraction meta_fault_intersection_fraction \
      meta_channel_fraction meta_channel_core_fraction \
      meta_flat_spot_fraction meta_onlap_fraction meta_onlap_variability \
    --out_json docs/benchmarks/adopted_v2_reference.json
  ```

  Then re-run the check in 2.1 and confirm it now prints `RESULT: MATCH` before continuing.

### 2.3 Report the eligible-query count

The plan (Section 8) wants this reported: "2048 examples should give about 4x the 52 eligible
queries of the old manifest."

```bash
.venv/bin/python -c "
import json
d = json.load(open('docs/benchmarks/adopted_v2_reference.json'))
print(json.dumps(d.get('diagnostics', {}), indent=2))
"
```

Record the printed `neighbor_overlap_at_5` / `neighbor_overlap_at_10` values — these are the
**P0 reference numbers** (adopted checkpoint, re-benchmarked on the v2 manifest).

---

## 3. Next steps per the plan (Section 9)

Only proceed here after Section 2 confirms `RESULT: MATCH` (or has been fixed to match).

### 3.1 Finish P0: also benchmark the R2 seed-1 checkpoint on the v2 manifest

The adopted-checkpoint half of P0 is done (Section 2.3 above). The plan also wants the R2
seed-1 experimental checkpoint re-benchmarked on the same v2 manifest, for the full P0 baseline
pair:

```bash
.venv/bin/python scripts/evaluate_geology_benchmark.py \
  --data data/synth_val_v2_32-32-64.zarr \
  --manifest docs/benchmarks/frozen_validation_manifest_v2.json \
  --checkpoint checkpoints/geoaware_p0a_dipfix_r2/vae_epoch30.pt \
  --use_geo_embedding \
  --metadata_keys meta_fault_fraction meta_fault_intersection_fraction \
    meta_channel_fraction meta_channel_core_fraction \
    meta_flat_spot_fraction meta_onlap_fraction meta_onlap_variability \
  --out_json docs/benchmarks/r2_seed1_ep30_v2_reference.json
```

(`checkpoints/geoaware_p0a_dipfix_r2/vae_epoch30.pt` is the P0a run from 2026-09-28 — old
manifest n@5 was 0.1346 there. This step re-measures the same checkpoint on the new v2
manifest so both P0 numbers are on a comparable basis.)

### 3.2 P0b: rerun the 180-volume training baseline on the v2 validation set

The data split is now fixed: training uses only the 180 datasets under
`/Volumes/CrucialX9/fake_data` with `--exclude_dir validation`; validation uses the 25 new
datasets under `/Volumes/CrucialX9/fake_data_validation_v2`. P0b therefore reruns the R2 recipe on the
existing 180-volume training patches, with the new v2 validation set and two seeds.

Use `data/synth_val_v2_uniform_32-32-64.zarr` as the validation input. It contains the v2
source-volume provenance, so it is the correct zarr path for `--disjoint_from` if a new training
set is sampled. The existing `data/synth_train_anchored_32-32-64.zarr` is already the correct
108,000-patch, label-complete training set and does not need to be recreated.

If regeneration is genuinely needed, use the corrected command:

```bash
.venv/bin/python scripts/sample_patches.py \
  --source /Volumes/CrucialX9/fake_data \
  --exclude_dir validation \
  --disjoint_from data/synth_val_v2_uniform_32-32-64.zarr \
  --patch_size 32 32 64 --n_patches 108000 --n_per_volume 600 \
  --seismic_key seismicCubes_cumsum_fullstack --geoscore_key geologic_score \
  --sampling_mode class_anchored \
  --class_quotas fault=0.10 fault_x=0.10 flat_spot=0.10 channel=0.10 closure=0.10 onlap=0.10 sand=0.05 \
  --background_fraction 0.25 --max_patches_per_object 24 --presence_min_voxels 32 \
  --label_z_offset 1 --dip_sample_count 512 --seed 20260929 \
  --out data/synth_train_anchored_32-32-64.zarr
```

Do not run that sampling command unless the existing 180-volume training set is missing or
needs regeneration. Then train the R2 recipe on
`data/synth_train_anchored_32-32-64.zarr`, 2 seeds, validating against
`data/synth_val_v2_uniform_32-32-64.zarr`, and benchmark both against
`docs/benchmarks/frozen_validation_manifest_v2.json`. Compare n@5 against the P0/P0a numbers
from Sections 2.3 and 3.1 — this establishes the clean v2 holdout baseline the plan calls P0b.

### 3.3 After P0b: proceed to P1

Per the plan, P1 is a loss-recipe sweep (L1–L4) warm-started from P0b's best recipe. Only start
this after P0b has run and been compared; do not skip ahead.

---

## 4. Optional: record the outcome in repo memory

If you have access to the memory tool for this repo, append a short note to
`/memories/repo/geoaware_model.md` under the "Pretrain-v2 transfer plan" section once Sections
2 and 3.1 are done, e.g.: "v2 manifest confirmed correct (dataset_size=5000); P0 adopted n@5=X,
R2 seed1 ep30 n@5=Y (v2 manifest)." This is optional and does not block continuing to P0b.

---

## Not something an agent can do: switching the chat model

The request to "switch to auto model selection for the 10% discount" is a client-side setting
in the chat UI's model picker (the dropdown next to the chat input), not something any tool
available to an agent can change. This must be done directly in the editor by the user.
