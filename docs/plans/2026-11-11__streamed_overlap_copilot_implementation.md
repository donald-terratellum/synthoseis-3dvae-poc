# Copilot handoff: streamed overlap and directional semantic retrieval

Updated: 2026-11-22.

## Local implementation status

The refreshed local audit `streamed-overlap-target-audit-20261122-03` inspected
180 training and 25 validation sources. All 205 have the required semantic arrays;
`faulted_net_to_gross`, channel labels and closure arrays are present, while the
separate `faults/faulted_channel_ng` array is absent. The current neighboring generator
inserts channel N/G into `faulted_net_to_gross` before closure/property construction;
13 synthetic end-to-end tests passed. This verifies current code behavior, not the
unknown generation-time worktree/command of every saved source.

The owner approved a five-sample, nearest-edge depth smoother blended with the original
seismic before strict adjacent-sample Z-axis extrema, same-index non-padded label
support, closure fractions as a coarse operational fluid proxy despite provenance
uncertainty, source-specific filter metadata, and the proposed evaluation success
criteria. Added-noise augmentation uses an independent 12-60 dB range; source `sn_db`
describes noise already present in stored data and is not the augmentation setting.
Different deepest Z extents are not by themselves a problem: use fully supported
same-index windows without padding, clipping, or shifts. The owner reports that
`fake_data` was recreated and is keeping the regenerating `fake_data_validation_v2`
separate; do not read it until regeneration is complete, then audit separately before
considering it for any use. The existing validation set remains primary. The executable target,
target-yield preflight and frozen evaluation manifest remain to be implemented and
checked. Gate 1 remains closed until those checks pass; the separate training-launch
approval gate remains closed as well.

Local implementation is on `experiment/streamed-overlap-copilot-implementation`.
The audit tooling and local source inventory exist; pair-target generation, training,
resume, evaluation, completed-step CSV and automatic trainer/TensorBoard integration
remain subsequent work. Do not treat this handoff or the current tests as permission
to launch training.

The cloud prompt below is a future handoff, not a claim that this branch or private
audit artifacts have been published. Continue from the published implementation branch
if it is later made available; do not recreate or overwrite its audit work.

### Available Bash runner and monitoring commands

Run from the repository root. All actual Python commands, audit subprocesses and
TensorBoard use the repository's **uv-managed `.venv`**, not system Python.
The Bash wrapper defaults to `.venv/bin/python` after changing to the repository root.
Use `PYTHON` only to explicitly select another environment; it is not changed automatically.

Fresh audit (metadata and bounded examples, **not training**):

```bash
bash scripts/streamed_overlap_experiment.sh --mode audit \
  --config scripts/streamed_overlap_audit.example.json \
  --run-id streamed-overlap-audit-review-01
```

For any existing run, use its actual ID and suite directory. The following completed
local audit has a run manifest and can be inspected now:

```bash
RUN_ID=streamed-overlap-target-audit-20261111-04
RUN_DIR="/Volumes/CrucialX9/3dvae-poc_checkpoints/streamed_overlap/$RUN_ID"

# Follow combined preflight/audit/service stdout and stderr.
tail -n 100 -F "logs/$RUN_ID/training.log"

# In a separate terminal: start a persistent, run-owned TensorBoard.
bash scripts/streamed_overlap_experiment.sh --mode tensorboard \
  --run-id "$RUN_ID" --run-dir "$RUN_DIR"

# Check HTTP readiness, suite identity and PID ownership.
bash scripts/streamed_overlap_experiment.sh --mode tensorboard-status \
  --run-id "$RUN_ID" --run-dir "$RUN_DIR"

# Verify ownership, then stop only the recorded TensorBoard PID.
bash scripts/streamed_overlap_experiment.sh --mode tensorboard-stop \
  --run-id "$RUN_ID" --run-dir "$RUN_DIR"
```

Open `http://127.0.0.1:6006` after successful start. **Audit runs write no training
TensorBoard events**, so their dashboards do not show training progress. The server
does not fabricate events or start training. Future trainer integration must write
the required real epoch/batch tags.

The equivalent standalone command, only when port 6006 is free, is:

```bash
.venv/bin/python -m tensorboard.main \
  --logdir "$RUN_DIR" --host 127.0.0.1 --port 6006
```

This standalone foreground command is **not owned by the runner**; stop it with
Ctrl-C in its terminal. Never stop an unrelated process to free the port. The runner
refuses an occupied port instead of silently choosing another one.

Every audit creates a fresh `logs/<run-id>/training.log` and `run_manifest.json`.
The manifest records attempt ID, exact command, interpreter/library versions, platform,
git revision/status/diff hash, resolved config and config/artifact hashes, return code
and terminal status. Interrupted/failed audits are not marked training-complete.
It does not claim model-device/memory or full-state-resume validation.
Config/interpreter preflight errors are logged; existing outputs are never overwritten.
Service actions append to the existing combined log and maintain `tensorboard.log`,
`tensorboard.pid`, `tensorboard_owner.json` and a fsynced `tensorboard_history.jsonl`.
A file lock serializes service actions; PID/start-time/UID/command plus HTTP logdir
verification prevents claiming or stopping an unrelated listener. Successful start
separates the service lifetime from the invoking shell; failed startup cleans up only
its own newly spawned child. `--print-commands` creates nothing and starts nothing.

As of 2026-11-22, the repository `.venv` passed **21 audit/target tests and four
subtests**; shell syntax and `git diff --check` passed. The neighboring generator's
own `.venv` passed **13 channel end-to-end tests**. The refreshed read-only audit
`streamed-overlap-target-audit-20261122-03` inventoried 180 training and 25 validation
sources and recorded 44 source-SNR metadata warnings. No training or final-test crops
were read.

Port 6006 was already occupied and was left untouched. A real TensorBoard lifecycle
was separately verified on an explicitly selected temporary test port: readiness,
survival after starter exit, Bash status and owned shutdown. That test server was
stopped. This did not alter the runner's default port or run any model.

## Paste-ready GitHub Copilot prompt

```text
Implement the approved streamed-overlap and directional fluid-aware semantic-retrieval
experiment in donald-terratellum/synthoseis-3dvae-poc.

Start from a published branch containing this implementation branch's audit work and
the current planning documents. Do not start from main: it lacks the current
experiment work. The local branch and uncommitted changes are not automatically
available to a cloud agent; the owner must publish the intended work first.
Verify the required files and actual base revision before implementing.

Read and follow:
1. docs/plans/2026-11-08__streamed_overlap_and_semantic_retrieval_experiment.md
2. docs/plans/2026-11-11__streamed_overlap_copilot_implementation.md
3. docs/plans/2026-11-11__streamed_overlap_target_contract_proposal.md
4. docs/sessions/2026-11-10-p9-mu512-screening-outcome-and-next-steps.md

Create a separate implementation branch and draft PR targeting the selected experiment
base. Implement the opt-in code, synthetic-fixture tests, audit and shell-runner tooling.
Use fresh E1 mu128 and three matched arms: reconstruction-only, overlap-only and
overlap+semantic. Stream original sources; never create a training-patch corpus.
Preserve production tokenizer defaults and historical benchmarks.

Do not ask again about settled decisions. Implement the exact owner-approved rules in
the current execution plan and target-contract proposal: strict local extrema, native
same-index label support, the closure-mask fraction proxy, source-specific filter
metadata, and the approved evaluation criteria. Unknown generation-time dirty-worktree
provenance is a documented limitation, not a reason to block the proxy. The streamed
target, validity/yield preflight and frozen evaluation manifest still need implementation.
Do not claim the experiment is training-ready or infer a launch budget.

The 2026-11-22 local audit inventories the current training/validation cohort. Use the
dated regeneration-check instructions only as historical context. Do not treat the
recorded SHA or current generator tests as proof of the worktree that produced each
saved volume. Do not modify, delete, regenerate or overwrite source data without
separate owner approval.

Cloud checks use tiny fabricated fixtures only. Do not upload or request private Zarr
cubes/checkpoints, assume /Volumes is mounted, run long training, use final-test data
for tuning, or claim synthetic tests establish MPS or source-schema correctness.

The runner must capture stdout+stderr under logs/<run-id>, fsync one completed-step CSV
row per successful optimizer step, retain epoch CSVs, save full epoch-boundary resume
state with replay reconciliation, and start/verify a run-owned localhost TensorBoard
server that remains running after local training ends.

Follow the ordered milestones and acceptance criteria in the handoff. Make focused
commits with the repository's co-author trailer. Report completed cloud checks, exact
local audit/smoke commands, unresolved approval gates, and the next required action.
No training or model adoption is authorized by this prompt.
```

## 1. Can implementation be performed on GitHub?

**Yes for repository code, fixture tests, runner integration and PR review; not for
complete local certification or production training with the current environment.**

The repository's GitHub default branch is `main`. The complete saved mu512 work is
on `experiment/mu512-latent-scaling`, verified at commit `41a8f8a` before these planning
edits. Cloud execution must start from a branch containing this plan revision, not
blindly from that older hash. Publishing these documents is a separate owner action;
this planning request does not commit, push, create an issue or open a PR.

| GitHub/cloud work | Local-only prerequisites in this project |
|---|---|
| Model heads, loss integration, sampler/manifests, logging/state code | Actual `/Volumes/CrucialX9/fake_data` and checkpoint access |
| Fabricated Zarr/metadata fixtures and CPU regression tests | Age/fluid interpretation, generator/key/axis/alignment audit |
| Shell runner and fake-process service tests | Real streamed-crop MPS memory/throughput and noise-spectrum checks |
| Draft PR, documentation and reproducible commands | Target-contract approval and launch-budget approval |
| Target logic once the approved contract is supplied | Sequential long training and persistent local TensorBoard |

Do not assume a cloud environment has Mac MPS, the external drive, a GPU, an interactive
browser, or a persistent server reachable from the user's Mac. Do not create a self-hosted
runner, network tunnel, credentials or data-upload workflow as a workaround. A remote
custom environment would need separate setup/authorization; it is not part of this task.

GitHub provides cloud and Actions-based agent environments, with differing lifecycle
and customization support; choose the permitted environment when assigning the task.
See [GitHub's environment overview](https://docs.github.com/en/copilot/concepts/copilot-surfaces/copilot-on-github).
Availability depends on the account and organization; no availability claim is a successful
test of this repository's data/hardware setup.

## 2. Governing requirements

Use [the execution plan](2026-11-08__streamed_overlap_and_semantic_retrieval_experiment.md)
as the authoritative source. This document sequences the work, not a second set of
scientific defaults. If later changes conflict, stop and obtain one consolidated
clarification rather than silently choose a different objective.

Important approved decisions:

- Two distinct outputs: physical canonical overlap fraction and directional,
  fluid-adjusted observable-event coverage, with A as the reference. This replaces
  equal-weight layer coverage; the event ratio is not an exact layer count.
- Smooth each seismic trace with `uniform_filter1d(size=5, axis=2, mode="nearest")`,
  blend `0.2 * original + 0.8 * filtered`, then count strict adjacent-sample
  Z-axis peaks/troughs in the shared region relative to the full A depth over shared
  XY, before augmentation; no prominence threshold. Use same-index label support and
  exclude padded indices 1500:1510. Apply the
  execution plan's window-level oil/gas/other closure-fraction correction as a coarse
  proxy, not physical-fluid truth. Disjoint or different-source pairs have zero
  coverage. Do not compare numeric age IDs across independent sources.
- Calibrated cosine overlap on mu; an ordered learned semantic head on mu_A/mu_B.
  Select by directional pair-head ranking and measure mu-cosine retrieval independently.
  A later spatial-alignment model is deferred.
- Train each arm end-to-end: one combined enabled-objective loss, one backward pass,
  and one optimizer step per batch. Do not detach/cache mu or freeze the shared encoder.
  Reconstruction/KL and enabled pair-head gradients must update the same encoder; the
  lower encoder learning-rate multiplier slows updates but does not freeze it.
- Fresh E1 residual mu128, skip-free decoder, 32x32x64, shared encoder; no trained
  P0b/mu512 initialization and no production search scoring changes.
- Stream volume-balanced original fullstack inputs, 12 pairs per batch: 6 same-source
  overlap, 3 same-volume disjoint, 3 cross-volume; two zero-shift pairs; +/-25% XYZ offsets.
- Rotation and scaling on every view, independent flips/swap and source-band depth
  noise (probability 0.5; additional noise SNR uniform 12-60 dB), not arbitrary
  phase/polarity invariance. Apply tokenizer normalization/extrema input; reconstruct
  noisy augmented dense A. Synthetic training only.
- Three sequential matched arms; epochs/batches are specified at launch, not guessed.
- Durable batch CSV plus existing epoch CSV/TensorBoard; epoch-state resume with explicit
  replay history; logs under repository `logs/`, checkpoints on CrucialX9.
- At every 100 completed steps and epoch boundary, run fixed deterministic no-gradient
  train/validation monitor batches and report component losses to stdout and TensorBoard.
- Evaluation criteria: at least 200 eligible positive queries overall and 20 queries
  from five volumes per predeclared cohort; +0.01 pair-head NDCG@10 over repeated
  random ranking, positive lower 95% source-cluster bootstrap bound, no cohort loss
  worse than 0.02, and independent-seed replication.

The [mu512 assessment](../sessions/2026-11-10-p9-mu512-screening-outcome-and-next-steps.md)
explains why numerical extension wins do not authorize a wider starting model or another
legacy sweep. P0b and mu512 are named evaluation references only.

## 3. Branch and publication procedure

Owner preparation, before assigning the cloud task:

1. Review the revised plans, audit code and intended branch contents. Preserve intended
   work using explicit paths; never add datasets, private labels, logs, checkpoints or
   environment secrets. Local ignored or uncommitted files are not visible in the cloud.
2. With owner approval, publish a branch containing the reviewed plans and audit work.
   Verify the committed content, not merely the presence of files in the local worktree.
3. Select that branch explicitly for the Copilot task/issue. If the assignment UI cannot
   select it, arrange the supported base-branch workflow first; do not let Copilot
   silently rebuild the experiment from `main`.

Implementation agent:

1. Confirm required plan/code files are present and record the actual base revision.
2. Use a fresh implementation branch (for example `experiment/streamed-overlap-retrieval`
   locally, or the branch generated by GitHub Copilot). Do not reset/reuse an unrelated
   existing branch; preserve other contributors' changes.
3. Open a draft PR targeting the selected experiment base, not an unreviewed merge into
   `main`. Do not launch production training from PR automation.
4. Use focused commits with:

   ```text
   Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
   ```

5. The existing CI workflow filters PRs to `main` and installs runtime requirements, not
   explicit development test dependencies. Do not claim that checks ran automatically
   on an experiment-targeted PR. As part of this implementation, scope CI triggering
   to the chosen experiment base as well, and ensure the test runner used by that
   workflow is installed via the existing project development dependencies. Keep
   local-volume/MPS tests explicitly separate from portable fixture tests.

## 4. Ordered implementation milestones

### M1: inspect and implement portable source/audit infrastructure

Read prior art in:

- [trainer](../../scripts/train.py), [model](../../src/model.py)
- [source sampler](../../scripts/sample_patches.py)
- [paired/state helpers](../../scripts/p7_control.py)
- [tokenizer preprocessing](../../src/tokenizer/core/preprocess.py)
- [adapter](../../src/tokenizer/core/model_adapter.py) and
  [search engine](../../src/tokenizer/core/search_engine.py)
- [historical evaluator](../../scripts/evaluate_geology_benchmark.py)
- [reconstruction diagnostics](../../scripts/diagnose_p5_p6_reconstruction.py)
- Existing model/sampler/latent-width tests.

Reuse contracts/helpers where safe; do not copy broad catches or invalid-input defaults
into the new path. Keep source streaming and paired objectives opt-in, with old behavior
unchanged. Add tiny manufactured source stores with known axis mappings, layers, fluids,
parameter-file bands, missing fields and excluded/symlinked source directories.

Implement `scripts/streamed_overlap_experiment.sh --mode audit` and a structured local
audit report. The output must identify unapproved semantics instead of claiming fabricated
fixtures validate real source labels. Print the exact local command for the owner.

### M2: implement the approved target contract

Use the local audit and reviewed code evidence to implement a versioned target contract
with explicit source validity and invalid reasons. Generate targets before augmentation,
use canonical A-reference supports, and stream full-depth seismic events from Zarr in
XY chunks. Add focused fixtures for extrema, age support, fluid-fraction edge cases,
pair order, zero-target categories and unavailable denominators/empty support. Run a
read-only target-yield preflight on training/validation sources only; do not read final
test crops or train. Freeze evaluation cohorts/manifests only after the yield is known.

The owner-approved fluid proxy must remain a proxy: unknown generation provenance is
recorded, not silently elevated to physical truth or used to invalidate every source.
Share only non-sensitive rules and manufactured fixtures with any future cloud agent;
never upload private datasets or raw source crops. If implementation evidence requires
changing an approved scientific choice, stop and ask before changing it.

### M3: model/trainer integration and durable recovery

Implement the mu overlap calibration and directional semantic head with saved configs;
test that each enabled loss reaches the shared encoder and its head, along with pair
order and adapter parity. Match initialized VAE state and source streams across
S1/S2/S3. Ensure extra head initialization and evaluation do not consume the
data/augmentation RNG stream.

Reuse reconstruction/LPIPS/KL functions. Log all component losses and all group LRs.
Extend the streamed path's full-state save/resume rather than relaxing old resume guards
globally. One successful optimizer step produces one fsynced journal event; crash
recovery preserves stale-attempt rows and explicitly reconciles replay/committed exposure.

### M4: evaluation and local execution runner

Implement deterministic validation, distinct pair-head versus cosine ranking metrics,
approved eligibility/selection, separate best checkpoint identities and bounded exact
CLI searches. Reuse tokenizer preprocessing and adapter/search paths. Stream selected
historical benchmark rows rather than materialize the full patch store.

Implement all shell modes and failure contracts in execution-plan section 11. Prove
combined log capture, child/log error propagation, output refusal and side-effect-free
print mode. Use fake processes and tiny fixtures for cloud TensorBoard lifecycle tests;
the real local readiness check must verify actual HTTP serving and persistence.

### M5: portable tests and honest cloud completion

Run targeted tests first, then the existing portable suite. Use existing dependency
manifests; install only the dependencies needed by the chosen validation command.
Report environment/dependency failures distinctly from test failures. Do not satisfy a
local-data test with zero-filled or success-shaped fallback results.

The PR report must include:

- Implementation and test summary with exact commands/results.
- Any `LOCAL_TARGET_VALIDATION_PENDING` or local hardware/data checks still outstanding.
- Approved-contract hash/status; never mark an approval true just because a file exists.
- Exact local audit, smoke, print-only, train/resume and evaluation commands.
- Output paths, CSV schema/recovery policy, TensorBoard URL/PID/stop instructions.
- No claims of completed training, benchmark uplift, MPS fit or adoption without evidence.

### M6: local readiness and training approval

Check out the reviewed implementation locally without discarding other work. Run the
approved target audit checks and a few actual streamed training/validation pairs through
full batch-12-pair MPS forward/backward, preprocessing, targets, logs, CSV and recovery.
Measure memory including decoded chunks/cache, noise filtering and evaluation buffers.
Check actual source/holdout exclusions and TensorBoard serving after the trainer exits.

Present one launch review with epochs/batches per epoch, total exposure/time estimate,
resolved config, exact shell command, approval/manifest hashes, resource evidence and
remaining risks. Do not start until approved. Once approved, run only that sequential
budget and capture outputs as specified. Do not silently add epochs, seeds, arms,
failed-gate bypasses or final-test evaluations.

## 5. Definition of done

**Cloud implementation done:** portable code/tests and runner are committed in the PR;
all unapproved/local prerequisites are explicit. This is not training readiness.

**Local training-ready:** target contract approved, full code/tests pass, real streamed
MPS smoke and recovery/logging/TensorBoard checks pass, and exact launch budget is approved.

**Experiment complete:** approved sequential run and required evaluations finish;
batch/epoch journals and checkpoints are persistent; failures and selection decisions
are explicit; TensorBoard remains available with a verified run-owned PID and shutdown
instructions. Training completion does not mean model adoption.

Do not replace the approved thresholds or source/filter rules, invent additional
scientific targets, or guess a training budget to bypass the implementation and
training-launch gates.
