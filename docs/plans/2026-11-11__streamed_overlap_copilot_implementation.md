# Copilot handoff: streamed overlap and directional semantic retrieval

Updated: 2026-11-21.

## Local implementation status

The owner reports all SynthoSeis datasets have now been updated. Reinventory and
audit the current training and validation sources; completion is reported, not yet
independently verified. The November 15 audit identified updated training runs
1041-1214, superseding the earlier 48-hour-only cohort rule for that snapshot. Apply the
[regeneration-check instructions](2026-11-11__streamed_overlap_target_contract_proposal.md#2026-11-15-regeneration-check-and-copilot-instructions)
and read the [channel/closure follow-up](2026-11-11__streamed_overlap_target_contract_proposal.md#2026-11-15-channelclosure-follow-up)
before making claims about updated data. As of that historical snapshot, updated training
data has unresolved alignment and synthesis/label questions, and no validation
sources met that recency criterion. Older validation observations are not proof about
updated validation. Do not treat those historical findings as a current inventory.
The owner's November 21 decisions replace the old layer/analog target with directional
fluid-adjusted observable-event coverage; the remaining target details and the launch
approval remain separate gates.

Local implementation is on `experiment/streamed-overlap-retrieval`, based on planning
commit `1e7e642`. The owner authorized audit tooling and local training/validation
inspection only, ending with the [unapproved target-contract proposal](2026-11-11__streamed_overlap_target_contract_proposal.md).
The portable audit fixtures and actual bounded local audit passed. Gate 1 remains
closed: this is **not** approval to implement guessed semantic labels or run training.
The shell runner now supports audit plus owned TensorBoard start/status/stop.
M2 targets, M3 trainer/recovery and M4 evaluation/completed-step CSV and automatic
training/TensorBoard integration remain subsequent work. Train/resume/smoke/evaluate
remain blocked; monitoring support is not production-training readiness.

The owner clarified that channel labels `> 0.5` override background lithology as sand;
closure masks indicate trap/fill geometry, not exhaustive lithology. The revised
proposal and schema-2 audit separate corrected facies, closure context and final-fluid
consistency. Channels retain identity; oil/gas closures modify fill within sand,
including channels. Do not invalidate non-trapping sand because closure masks are
absent or adopt the neighboring generator's late brine-channel override as the target
rule. Verify historical synthesis agrees with the owner's hierarchy before training.

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

Local validation used the uv environment: **35 audit/runner/legacy sampler tests
passed**, and the mounted-source audit produced the new attempt manifest.
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

Do not ask again about settled decisions. Do not guess unresolved geology semantics,
target thresholds or launch budgets. Exact semantic-target rules require a local audit
and owner approval. If no approved target contract is available, implement the audit
and infrastructure first, fail closed for semantic training, and report
LOCAL_TARGET_VALIDATION_PENDING. Do not claim the experiment is training-ready.

The owner reports dataset replacement is complete. Reinventory the current cohorts;
use the dated regeneration-check instructions in the target-contract proposal as
historical context, separately inventory current source and validation cohorts,
and verify actual generator provenance, alignment and fluid synthesis for the relevant
cohorts. Do not treat older validation results, timestamps, or a recorded SHA alone as
proof that replacement is complete or that stored waveforms match a target. Do not
modify, delete, regenerate or overwrite source data without separate owner approval.

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
- Count peaks and troughs in the shared region relative to the full A depth over
  shared XY, before augmentation; apply the execution plan's window-level
  oil/gas/other correction. Disjoint or different-source pairs have zero coverage.
  Do not compare numeric age IDs across independent sources.
- Calibrated cosine overlap on mu; an ordered learned semantic head on mu_A/mu_B.
  Select by directional pair-head ranking and measure mu-cosine retrieval independently.
  A later spatial-alignment model is deferred.
- Fresh E1 residual mu128, skip-free decoder, 32x32x64, shared encoder; no trained
  P0b/mu512 initialization and no production search scoring changes.
- Stream volume-balanced original fullstack inputs, 12 pairs per batch: 6 same-source
  overlap, 3 same-volume disjoint, 3 cross-volume; two zero-shift pairs; +/-25% XYZ offsets.
- Rotation and scaling on every view, independent flips/swap and source-band depth noise, not arbitrary
  phase/polarity invariance. Apply tokenizer normalization/extrema input; reconstruct
  noisy augmented dense A. Synthetic training only.
- Three sequential matched arms; epochs/batches are specified at launch, not guessed.
- Durable batch CSV plus existing epoch CSV/TensorBoard; epoch-state resume with explicit
  replay history; logs under repository `logs/`, checkpoints on CrucialX9.

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

### M2: obtain local target-contract approval

The local assistant/user runs the read-only audit against the updated mounted sources.
Audit training/validation examples, not final-test model performance. Present one
approval form covering the unresolved contract listed in execution-plan sections 3
and 7: event-detection and boundary settings, age-range validity and alignment,
source fluid provenance, noise filter and metric/adoption thresholds. Preserve the
owner's settled event ratio and simple window-level fluid formula.

The owner supplies a versioned approved contract to the implementation branch. Share
only approved non-sensitive rules and fabricated fixture cases with the cloud agent;
do not upload private datasets or raw source crops. If the audit discovers a need to
change an approved scientific choice, ask rather than conceal that scope change.

This milestone is a hard gate. Cloud infrastructure can proceed in parallel where
independent, but semantic target implementation cannot guess the missing formulas.
After approval, implement exact deterministic target rules and their regression cases.

### M3: model/trainer integration and durable recovery

Implement the mu overlap calibration and directional semantic head with saved configs;
test gradients, pair order and adapter parity. Match initialized VAE state and source
streams across S1/S2/S3. Ensure extra head initialization and evaluation do not consume
the data/augmentation RNG stream.

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

No threshold, source-schema rule, frequency sampling interval, scientific target or
training budget should be invented to avoid the two planned approval gates.
