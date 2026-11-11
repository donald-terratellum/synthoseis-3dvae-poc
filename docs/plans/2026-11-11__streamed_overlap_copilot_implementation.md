# Copilot handoff: streamed overlap and directional semantic retrieval

Updated: 2026-11-11.

## Paste-ready GitHub Copilot prompt

```text
Implement the approved streamed-overlap and directional fluid-aware semantic-retrieval
experiment in donald-terratellum/synthoseis-3dvae-poc.

Start from experiment/mu512-latent-scaling, including the planning documents in this
handoff. Do not start from main: it lacks the current experiment work. At preparation
time the saved base was 41a8f8a; the owner must publish the updated planning documents
before assigning this cloud task. Verify those files exist on the selected base.

Read and follow:
1. docs/plans/2026-11-08__streamed_overlap_and_semantic_retrieval_experiment.md
2. docs/plans/2026-11-11__streamed_overlap_copilot_implementation.md
3. docs/sessions/2026-11-10-p9-mu512-screening-outcome-and-next-steps.md

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

- Two distinct outputs: physical canonical overlap and directional common-layer
  coverage with fluid-compatible partial credit.
- Equal weight per distinct A layer; tolerate thickness differences; measure NTG
  differences separately without penalizing coverage.
- Same-realization physical layer identity and cross-volume analog patterns have
  separate labels/evaluation. Do not compare numeric age IDs across independent sources.
- Calibrated cosine overlap on mu; an ordered learned semantic head on mu_A/mu_B.
  Measure mu-cosine retrieval independently. A later spatial-alignment model is deferred.
- Fresh E1 residual mu128, skip-free decoder, 32x32x64, shared encoder; no trained
  P0b/mu512 initialization and no production search scoring changes.
- Stream volume-balanced original fullstack inputs, 12 pairs per batch: 6 same-source
  overlap, 3 same-volume disjoint, 3 cross-volume; two zero-shift pairs; +/-25% XYZ offsets.
- Approved rotation/flips/swap/scales and source-band depth noise, not arbitrary
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

1. Review the revised plan and this handoff. Preserve intended work using explicit paths;
   never add datasets, private labels, logs, checkpoints or environment secrets.
2. Commit/push the planning documents to `experiment/mu512-latent-scaling`, or publish
   an explicitly named successor planning branch containing the same base work.
   Do not assume the cloud agent can see local uncommitted edits.
   This checkout's `.git/info/exclude` ignores new files under `docs/plans/`. The
   revised existing plan is tracked, but this new handoff is not. After review,
   explicitly force-add only this intended Markdown file; do not remove the folder
   exclusion or force-add its other contents:

   ```sh
   git add -u -- docs/plans/2026-11-08__streamed_overlap_and_semantic_retrieval_experiment.md
   git add -f -- docs/plans/2026-11-11__streamed_overlap_copilot_implementation.md
   git diff --cached --check
   git diff --cached --stat
   # Review the staged content, then commit with the co-author trailer and push
   # the selected planning branch. Do not execute this step without owner approval.
   ```

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

The local assistant/user runs the read-only audit against mounted sources. Audit
training/validation examples, not final-test model performance. Present one approval
form covering the unresolved contract listed in execution-plan sections 3 and 7:
denominator/background, layer identity/analog correspondence, 3D/partial-fluid credit,
validity/no-reservoir behavior, noise filter and metric/adoption thresholds.

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
