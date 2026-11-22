# Copilot handoff: streamed overlap and directional semantic retrieval

Updated: 2026-11-22.

## Local continuation: post-selection evaluation

The shell now connects evaluation of an **already completed** matched suite to
the evaluation-only runner, combined log capture and log confirmation:

```sh
bash scripts/streamed_overlap_experiment.sh --mode evaluate \
  --config <prepared-suite-inputs.json> --run-dir <completed-suite-directory> \
  --run-id <new-evaluation-attempt-id> --config-sha256 <prepared-config-sha256> \
  --mps-memory-fraction <limit> --min-suite-free-bytes <bytes> \
  --min-log-free-bytes <bytes> --acknowledge-evaluation-resources
```

The acknowledgement means the caller has reviewed memory, concurrent tokenizer
subprocesses, disk, logs and reserve; the runner does **not** establish aggregate
memory headroom. The disk floors are caller estimates, not reservations, and
must account for combined use when paths share storage. Evaluation can write
reports, journal recovery records and TensorBoard events; it does not start
TensorBoard, is not atomic, and cannot train or finish an incomplete arm.
The output includes final S2-vs-S1 and S3-vs-S2 mu-cosine NDCG comparisons with
source-cluster intervals, separately from the S3 pair-head-vs-random metric.
These are validation proxy measurements, not proof of geological understanding
or an adoption decision. Summary output is preliminary until log confirmation
and report checks succeed. Independent-seed replication, reconstruction
guardrails and untouched final-test evaluation remain separate.

No real suite was evaluated. Shell train/resume remain blocked and production
training still requires separate approval. Later entries describe historical
implementation states, not the current availability of evaluation mode.

Journal recovery now reads completed rows incrementally and hashes recorded torn
tails in bounded chunks instead of loading and splitting the whole CSV. It still
retains all parsed rows and event IDs; reconciliation may hold both old and new
rows, and the suite opens three journals. Long-run memory is therefore not bounded
or approved by this change.

On this Mac, opening a synthetic 10,000-row journal made from a real disposable
smoke row (6,987,872 bytes) used 41,873,298 bytes of peak traced Python allocations
before the change and 28,202,281 bytes afterward. These are `tracemalloc` readings,
not total RSS, MPS use, reconciliation peaks or a full-suite memory forecast.
No training or evaluation ran during that recovery change.

The record-bound suite can now resume a narrowly defined interrupted bootstrap:
after the suite-input record was saved, but before all three arm directories
were created. It recreates only a missing suffix of arm directories after
locking existing journals and verifying they have no recorded or incomplete
steps, unexpected files, checkpoints or incompatible recovery records.
Evaluation-only mode cannot use this recovery path. Interruptions inside an
arm's directory/journal creation, a missing suite-input record, or other
unexpected files still require investigation; no files are removed or
overwritten. This is a resume capability, not a training authorization.
Initialization, epoch and runner fixtures passed 96 tests and 50 subtests;
the partial-bootstrap fixture also exercised MPS checkpoint initialization
with a synthetic sampler and a mocked suite coordinator. No training ran.

The future logged runner now has an evaluation-only child in
`scripts/streamed_overlap_evaluate_child.py`. It checks the pinned record and
saved suite inputs, requires an explicit MPS memory fraction in `(0, 0.5]`
without CPU fallback, and calls the existing suite runner with `resume=True`
and `evaluate_only=True`. On completion it prints checkpoint identities and
the separate encoder-cosine and S3 pair-head report hashes, not an adoption
decision. A changed input record or invalid returned report fails.

This child is a runner component, not a complete launch command: it does not
capture a durable combined log, start TensorBoard, establish disk or memory
headroom, or verify historical measurement provenance. It can write evaluation
reports and journal recovery records and is not atomic.

The runner's `evaluate-attempt` action now invokes that child and creates an
attempt manifest and exclusive receipt. A successful child remains
`evaluation_child_complete_log_unconfirmed` until `confirm-evaluation-log`
rechecks the record, receipt, final checkpoint hashes and separate reports,
then syncs the combined log. The caller must invoke confirmation only after
the stdout/stderr capture pipeline succeeds; confirmation cannot independently
prove that `tee` captured all output. Interruption and child failures are
recorded without reporting completion. Before launching the child,
`evaluate-attempt` now requires explicit `--min-suite-free-bytes` and
`--min-log-free-bytes` floors, measures available bytes at both existing
directories, records those measurements in the attempt manifest and fails if
either falls below its floor. The caller must choose floors covering all
remaining evaluation writes, logs, retries, filesystem overhead and reserve;
when paths share a filesystem or storage pool, both floors must account for
their combined use. These floors are not verified output estimates, the
measurements are not reservations, and concurrent use can invalidate them.

Checking the checkpoints again adds disk I/O. The MPS fraction is per process,
not a combined memory bound: bounded tokenizer searches start separate
processes with their own frozen MPS setting. Memory headroom is still
unverified. These actions are components for future shell integration, not
resource approval: shell train/resume/evaluate remain blocked, and no training
is authorized. No real evaluation has been run.
Runner, initialization and epoch fixtures passed 95 tests and 50 subtests using
`.venv/bin/python -m pytest -q tests/test_streamed_overlap_runner.py
tests/test_streamed_overlap_initialization.py tests/test_streamed_overlap_epoch_runner.py`.
They used synthetic files and mocked child execution; no real suite was evaluated.

For a prospective fresh suite, review storage against a prepared suite record:

```sh
.venv/bin/python -m scripts.streamed_overlap_storage \
  --suite-record <prepared-record.json> --suite-dir <new-suite-directory> \
  --config-sha256 <prepared-config-sha256> \
  --s1-checkpoint-upper-bytes <bytes> --s2-checkpoint-upper-bytes <bytes> \
  --s3-checkpoint-upper-bytes <bytes> --other-output-bytes <bytes> \
  --free-reserve-bytes <bytes>
```

The command verifies the supplied record hash, destination, matching and
evaluation settings before checking its filesystem. It counts separate initial
and last checkpoints, best aliases, up to one retained selection snapshot per
epoch per arm, and one additional temporary checkpoint copy for sequential
publication. The caller supplies positive per-arm checkpoint size bounds, an
allowance for all other outputs on that filesystem, and a free-space reserve.
Insufficient space fails; the JSON report is printed, not saved automatically.

This is a read-only estimate for a clean fresh run, not a reservation or launch
gate. The provided bounds are not verified; they must reflect full optimizer
state, other outputs and filesystem overhead. It does not reverify source
volumes, historical files or MPS. Interrupted attempts, retries, concurrent
disk use and logs on other filesystems need separate allowances and checks.
It neither estimates memory or runtime nor authorizes training; the shell modes
remain blocked. No epoch count or byte allowance has been chosen.
Epoch, initialization and runner fixtures passed 87 tests and 37 subtests using
`.venv/bin/python -m pytest -q tests/test_streamed_overlap_epoch_runner.py
tests/test_streamed_overlap_initialization.py tests/test_streamed_overlap_runner.py`.
The record-bound command used synthetic fixtures and a temporary filesystem;
capacity boundaries were mocked, not measured on the production destination.

Completed suites now write `pair_head_metric_checks.json` alongside, but
separately from, `mu_cosine_comparisons.json`. It records the existing S3
pair-head metric checks for both the final-epoch and retrieval-selected
checkpoints, after checking the matched-arm records, checkpoint and evaluation
identities, report hashes, validation cohort and recomputed summaries. A failed
single-seed check is `FAIL`; a passing one remains `PENDING` because independent
replication is not supplied. The selected checkpoint was chosen on the same
validation set. This checks the head against the random baseline, not S3 versus
S2 encoder representations, and is not an adoption decision. Reconstruction
guardrails and independent-seed replication remain separate. Existing reports
are reused only when their contents agree; publication is not atomic across the
suite's reports.

The metric, initialization and epoch fixtures passed 76 tests and 31 subtests
using `.venv/bin/python -m pytest -q tests/test_streamed_overlap_metrics.py
tests/test_streamed_overlap_initialization.py tests/test_streamed_overlap_epoch_runner.py`.
These are synthetic fixtures; no real suite was trained or evaluated.

Before opening any arm, the prepared-suite coordinator now rechecks the full
frozen synthetic and real reconstruction guardrail stores, reconstruction
runtime and LPIPS state. It uses the existing guardrail wall-time setting and
logs the observed verification time. A missing or changed store, incompatible
runtime, or expired check stops the suite before training; fresh suite
initialization may already have written checkpoints. The check runs on fresh,
resume and evaluation-only suite calls, and is repeated during reconstruction
evaluation. It reads the historical guardrail stores, not the new final-test
set; it does not verify complete training volumes or guarantee inputs cannot
change afterward. The wall limit is checked between reads, not a hard watchdog.
The low-level single-arm runner does not perform this suite-level preflight.

The guardrail, initialization, epoch, integrity and search tests passed
102 tests and 35 subtests using `.venv/bin/python -m pytest -q
tests/test_streamed_overlap_guardrails.py tests/test_streamed_overlap_initialization.py
tests/test_streamed_overlap_epoch_runner.py tests/test_streamed_overlap_integrity.py
tests/test_streamed_overlap_search.py`. The store checks used tiny synthetic
Zarr fixtures; no real-data preflight, training or evaluation ran.

Suite preparation and the programmatic suite and arm runners now verify the
historical P0b report and checkpoint before training can start or resume. The
check rehashes both files and requires the report's checkpoint identity, frozen
protocol, complete synthetic and real inputs, and finite metrics to match the
reconstruction guardrail. Missing, changed or incompatible references fail
before a suite is initialized or an arm is restored. Evaluation uses the same
check. This checks file identity and report consistency, not independent
measurement provenance; it can also require reading the checkpoint more than
once. It does not authorize training or change the semantic target.
The focused preparation, epoch, initialization, guardrail and runner tests
passed 104 tests and 37 subtests using `.venv/bin/python -m pytest -q
tests/test_streamed_overlap_epoch_runner.py tests/test_streamed_overlap_guardrails.py
tests/test_streamed_overlap_initialization.py tests/test_streamed_overlap_runner.py`.
These used fixture data; no real preparation, training or evaluation ran.

The record-driven suite now has a programmatic evaluation-only option:
`run_suite_from_record(..., resume=True, evaluate_only=True)`. It requires an
existing initialized suite, the pinned inputs, and final-epoch checkpoints for
all three arms. It checks each checkpoint's suite identity and journal watermark
before evaluating any arm, then rechecks each arm before restoring it. An
incomplete arm cannot enter the training loop. Normal checkpoint restoration and
evaluation still perform further checks and may fail after an earlier arm has
produced reports; this is not an atomic publication of all results.

Evaluation can write journal recovery records, TensorBoard events and evaluation
reports; it is not read-only and can require substantial MPS time and disk
space. Its caller still needs resource controls and durable logging. The shell
train/resume/evaluate modes remain blocked. The existing matched-stream checks
and separate encoder μ-cosine and S3 pair-head results remain in force; the
event-coverage proxy alone does not establish geological understanding.
Checkpoint, matching, initialization and evaluation fixtures passed 108 tests
and 42 subtests using `.venv/bin/python -m pytest -q
tests/test_streamed_overlap_epoch_runner.py tests/test_streamed_overlap_initialization.py
tests/test_streamed_overlap_checkpoint.py tests/test_streamed_overlap_evaluation.py
tests/test_streamed_overlap_matching.py`. The new evaluation-only fixtures use
synthetic checkpoints and mock actual evaluation and MPS runtime; no real suite
was evaluated or trained.

The existing shell runner can now capture suite-input preparation and its
preflight output under a new run ID:

```sh
bash scripts/streamed_overlap_experiment.sh --mode prepare \
  --config <private-suite-settings.json> --run-id <new-preparation-id> \
  --output-root <private-output-directory> --mps-memory-fraction <preparation-limit> \
  --max-wall-seconds <preparation-limit> [--print-commands]
```

The private output directory must already exist. The output is the suite-input
JSON file `<private-output-directory>/<run-id>`;
stdout and stderr go to `logs/<run-id>/training.log`. The attempt manifest
records the input and output hashes and the logical suite-input hash, and marks
preparation as **not training-authorized**. A failed child or changed input
config is recorded as failure. Successful preparation is marked complete only
after `tee` exits successfully and the log is synced; otherwise a finished
child remains log-unconfirmed. Print-only creates nothing. This mode does not
start TensorBoard, run an optimizer, perform full launch checks or enable the
blocked train/resume/evaluate shell modes. Runner, audit and epoch fixtures
passed 64 tests and 24 subtests using `.venv/bin/python -m pytest -q
tests/test_streamed_overlap_runner.py tests/test_streamed_overlap_audit.py
tests/test_streamed_overlap_epoch_runner.py` in 5.97 seconds. An invalid-config
fixture exercised the real shell, runner and preparation child; no real-data
preparation or training ran.

The suite-input record can now be prepared without training:

```sh
.venv/bin/python -u -m scripts.streamed_overlap_prepare \
  --config <private-suite-settings.json> --output <new-private-suite-inputs.json> \
  --mps-memory-fraction <preparation-limit> --max-wall-seconds <preparation-limit>
```

The JSON must contain absolute `suite_dir`, `source_root`, `validation_manifest`,
`search_manifest`, `diagnostic_manifest` and `guardrail_manifest` paths; the
explicit `source_sha256`, `target_sha256` and `validation_manifest_sha256`;
`matching_config`, `epoch_validation`, `formal_config` and `guardrail_config`
objects; and a `historical` object with absolute `report_path` and
`checkpoint_path`, `report_sha256`, `checkpoint_sha256` and `checkpoint_label`.
`source_sha256` is the sampler's source-inventory hash (also reported as
`source_manifest_sha256` by the no-optimizer smoke), not a whole-volume hash.
The three sidecar paths refer to their existing frozen JSON objects. The command
checks their configuration bindings, prepares the fixed monitors from streamed
train and validation pairs, verifies the historical reference as described
above, and writes the record and prints its logical hash and workload. It does
not create the suite directory, run an optimizer step, or
approve training. The preparation MPS fraction and time limit are not a launch
resource approval. The time limit is checked after monitor preparation, not a
hard watchdog, and excludes initial configuration checks. Full validation
fingerprinting, historical measurement provenance, resource checks and launch
approval remain outstanding.
The portable streamed regression passed 243 tests, with one skipped and 59
subtests, using `.venv/bin/python -m pytest -q tests/test_streamed_*.py` in
46.30 seconds. The preparation fixture used synthetic batches with sampler and
MPS allocation mocked; no real-data preparation or training ran.

Here "training budget" means epochs and batches per epoch—not cloud charges.
Because training draws streamed pairs, those explicit counts specify how much
work each matched arm performs; they have not been chosen here.

`run_suite_from_record` now connects a pinned `suite_input_record` JSON file to
the existing fresh/resume suite runner. It requires the separately supplied
configuration hash and the same suite destination, reconstructs the frozen
epoch and evaluation arguments, and delegates to `run_suite`, which rechecks the
saved record on resume. This is a programmatic runner primitive, not a shell
launch mode: calling it can train, and its caller must enforce launch approval,
resource limits and durable logging. Those gates remain incomplete; do not call
it for production training yet. The focused initialization, epoch and step tests
passed 73 tests, with one skipped and 29 subtests, using `.venv/bin/python -m pytest -q
tests/test_streamed_overlap_initialization.py tests/test_streamed_overlap_epoch_runner.py
tests/test_streamed_overlap_step.py` in 32.25 seconds. The record-driven MPS
fixture mocked epoch-input preparation and arm execution; it did not train or
read real source data.

The bounded tokenizer-search manifest can now be frozen from an explicit private
settings file, without model inference or training:

```sh
.venv/bin/python -u -m scripts.streamed_overlap_search \
  --source-root <primary-source-root> --manifest <bound-validation-manifest.json> \
  --settings <private-search-settings.json> --output <new-private-search-manifest.json> \
  --max-bind-wall-seconds <binding-limit>
```

The settings JSON must contain `plans` (a list of objects with `query_id`,
`region_origin` and `region_shape`) and explicit `stride`, `batch_size`,
`max_region_bytes`, `max_windows`, `max_wall_seconds`, `device` and
`mps_memory_fraction`. Its `max_wall_seconds` is the later search-run limit;
the separate command-line limit applies to binding. Select and freeze queries,
regions and budgets before seeing model results. The command rechecks the
primary validation fingerprints and source inventory, verifies the bounded
regions and writes only a new manifest outside the source root. The binding
wall limit rejects late results but does not interrupt an individual read or
the full fingerprint check. It excludes validation-v2 and does not authorize
training. Search manifests made with an older search implementation must be
regenerated because the implementation fingerprint includes this script.
The search, diagnostic, epoch and suite-initialization fixture tests passed 74
tests and 27 subtests using `.venv/bin/python -m pytest -q
tests/test_streamed_overlap_search.py tests/test_streamed_overlap_diagnostics.py
tests/test_streamed_overlap_epoch_runner.py tests/test_streamed_overlap_initialization.py`
in 32.19 seconds. The command was not run on real data; the fixture mocks the
validation inventory and full-input verification. No settings were selected.

`run_suite` now binds and saves its resolved suite inputs before initializing an
arm. The record includes the matching and evaluation settings, frozen manifests,
historical-reference identities, resolved input/output paths and non-config
checkpoint hashes. Resume requires the same record; a changed or missing record
stops before journals or checkpoints are opened. The hash detects accidental
changes, not coordinated replacement of artifacts.

To prepare the binding without training, first use `prepare_epoch_inputs` and
merge its returned monitor and epoch-validation hashes into the other input
hashes. Call `suite_input_record` with the returned validator and the same suite
arguments, then set `hashes["config"] = content_hash(record)` before calling
`run_suite`. The suite writes `resolved_suite_inputs.json` and binds its hash
into each checkpoint. Changing the suite path also changes the hash. Existing
experimental suites without this record cannot use this suite runner to resume.
This record is not a complete launch configuration: resource limits, approval,
logging and service ownership remain external, and train/resume/evaluate shell
modes remain blocked. No training or launch approval is implied.
The portable streamed suite passed 239 tests, with one skipped and 59 subtests,
using `.venv/bin/python -m pytest -q tests/test_streamed_*.py` in 46.69 seconds.
The suite workflow fixture mocked arm execution; no real-data training or
evaluation was run for this change.

`run_suite` can now construct its fixed train/validation monitors, frozen
epoch validator and LPIPS directly from explicit `epoch_validation` settings:
`reconstruction_query_ids`, `encode_batch_size`, `reconstruction_batch_size`
and `max_wall_seconds`. Monitor samplers use `monitor_seed` and are separate
from the training sampler; the validator uses `validation_seed`. LPIPS must
match the frozen guardrail fingerprint. The existing prepared-input interface
is still supported, but cannot be mixed with these settings.

Before freezing the complete launch configuration, callers can use
`prepare_epoch_inputs` with the same paths, matching configuration, settings,
guardrail manifest, and source/target/validation-manifest hashes. Its fourth
return value supplies the `monitor` and `epoch_validation` hashes to bind into
the suite hashes. `run_suite` reconstructs and checks both hashes on fresh runs
and resumes; changed monitor batches or validation settings fail instead of
silently changing the monitors. Preparation reads bounded training and primary-validation pairs; it
does not train or run validation. The caller must enforce MPS memory limits,
resource and approval gates before invoking it. This is not a runnable launch
command: the complete configuration, other frozen inputs, durable launch log,
resource evidence and explicit training approval are still outstanding.

Synthetic sampler, epoch, suite and joint-step tests passed 77 tests, with one
skipped and 29 subtests, using `.venv/bin/python -m pytest -q
tests/test_streamed_overlap_sampler.py tests/test_streamed_overlap_epoch_runner.py
tests/test_streamed_overlap_initialization.py tests/test_streamed_overlap_step.py`.
They took 39.98 seconds and 799,916,032 bytes peak RSS including startup.
The new preparation integration used real sampler/augmentation code over
synthetic arrays, with LPIPS and MPS mocked; it is not a training-resource
measurement. No real data or optimizer step was used for this change.

After all three arms complete, the suite now writes
`mu_cosine_comparisons.json` in the suite directory using existing checkpoint
evaluations. It reports S2-S1, S3-S1 and S3-S2 NDCG@10 differences with paired
source-cluster intervals, separately from the S3 pair-head result. It compares
the matched final-epoch checkpoints, reconstruction-selected checkpoints for
all arms, and retrieval-selected S3 against each reconstruction-selected baseline.
The report identifies each checkpoint and its selection criterion. Different selected epochs,
and especially different selection criteria, limit interpretation of differences;
event coverage is a geological proxy, not proof of geological understanding.
The comparison uses matched recorded streams and the frozen validation cohort;
it is not an adoption decision. Resuming verifies the same saved comparison.
No training or evaluation on real data was run for this change.
Focused metric, suite, epoch and joint-step tests passed 78 tests, with one
skipped and 29 subtests, using `.venv/bin/python -m pytest -q
tests/test_streamed_overlap_metrics.py tests/test_streamed_overlap_initialization.py
tests/test_streamed_overlap_epoch_runner.py tests/test_streamed_overlap_step.py`.
They took 33.13 seconds and 702,021,632 bytes peak RSS including startup; these
fixture-test measurements are not training-resource estimates.

`run_suite` now connects fresh S1/S2/S3 checkpoint preparation and journal
ownership to the sequential coordinator, and reopens the same suite for resume.
It can finish an interrupted bootstrap when all three journals exist and have
no completed or torn steps; otherwise it stops without replacing checkpoints.
It closes journal locks on success or failure. The caller still supplies the
frozen inputs and must enforce launch approval, resource limits and log capture.
The shell train/resume/evaluate modes remain blocked.

A synthetic MPS/ExFAT fixture exercised fresh preparation, reopening and
interrupted bootstrap without an optimizer step. Arm execution and input
validation were mocked. It passed in 6.94 seconds including startup, with
658,980,864 bytes peak RSS; this is not a training-resource estimate. Temporary
files and the empty test-created PyTorch cache directory were removed. The
focused suite workflow, epoch and step tests passed 69 tests, with one skipped
and 29 subtests, using `.venv/bin/python -m pytest -q
tests/test_streamed_overlap_initialization.py tests/test_streamed_overlap_epoch_runner.py
tests/test_streamed_overlap_step.py`.

The prepared-suite coordinator can now construct each arm's fresh E1 model,
optimizer, scheduler, seeded training sampler and TensorBoard writer itself,
then release the writer before starting the next arm. It creates a missing
epoch CSV only for an untouched epoch-zero arm; a missing CSV for an advanced
or interrupted arm is an error, not a new log. Existing CSVs are preserved.
The per-arm runner still restores the checkpoint and reconciles the journal.
This connects the existing training pieces, but the caller must still prepare
the suite, locks, manifests and fixed inputs. The shell's train/resume/evaluate
modes remain blocked; this is not a launch command or training approval.

The focused runtime, recovery and joint-step fixture tests passed 67 tests,
with one skipped and 29 subtests, using `.venv/bin/python -m pytest -q
tests/test_streamed_overlap_initialization.py tests/test_streamed_overlap_epoch_runner.py
tests/test_streamed_overlap_step.py`. The suite integration fixture mocked arm
execution and sampler/model construction. A separate synthetic MPS test used
the real S3 model, optimizer and TensorBoard writer with a mocked sampler and
no optimizer step. Those two focused tests took 4.49 seconds and 524,959,744
bytes peak RSS including startup; this is not a training-resource estimate.
No real data was used for this wiring.

The prepared-suite coordinator now checks the formal-evaluation,
reconstruction-guardrail and historical-reference hashes against the
checkpoint-bound suite hashes before opening an arm runtime or reading
validation inputs. It also requires the formal final epoch to match the
matching configuration and verifies that configuration's hash. The same
evaluation-binding checks remain in the per-arm path. The historical
reference files and real guardrail stores are not read by this check; their
post-selection verification remains separate. These checks do not bind the
complete resolved launch configuration or authorize training.

The portable streamed suite passed 231 tests, with one skipped and 59 subtests,
using `.venv/bin/python -m pytest -q tests/test_streamed_*.py`; it took 37.26
seconds wall time and 711,049,216 bytes peak RSS. These are fixture-test
measurements, not training estimates. Train, resume and evaluate remain blocked.

Before opening any arm runtime, the prepared-suite coordinator now also
revalidates each frozen bounded-search region and the diagnostic label
fingerprints using the same verifiers used during evaluation. It checks the
validation manifest and source inventory again afterward. The diagnostic
verification uses the configured diagnostic wall limit; search verification
can also take time and read the bounded regions. Unavailable diagnostic
windows retain their recorded unavailable status. These checks do not certify
whole sources, prevent later changes, establish resource readiness, or authorize
training; evaluation still repeats its own checks.

The search verifier was extracted from the search runner, changing the
implementation fingerprint. Previously frozen search manifests therefore fail
verification and must be regenerated from explicitly selected settings before
use; none were generated here. The diagnostic implementation did not change.
No real-data search or diagnostic preflight was run on this continuation.
The portable streamed suite passed 231 tests, with one skipped and 54 subtests,
using `.venv/bin/python -m pytest -q tests/test_streamed_*.py`; it took 36.87
seconds wall time and 714,080,256 bytes peak RSS. These test measurements are
not training-resource estimates. The shell's train, resume and evaluate modes
remain blocked.

The prepared-suite coordinator now validates the bounded-search manifest's
schema, content hash and nonempty search list before opening an arm runtime.
It prints a `suite_workload` report calculated from explicit matching settings
and that search list: training exposure, monitors, epoch validation, formal
evaluations, S3 diagnostics and tokenizer searches. The count includes the
extra selected-checkpoint evaluations and does not double-count a fifth/final
epoch or a monitor due both at step 100 and an epoch boundary. These are counts
for a clean fresh run, not remaining resume work; retries and incomplete
attempts can add work. They do not validate search regions, estimate disk,
memory or runtime, choose an epoch/batch budget, or authorize training.

The portable streamed suite passed 230 tests, with one skipped and 54 subtests,
using `.venv/bin/python -m pytest -q tests/test_streamed_*.py`.

The prepared-suite coordinator now revalidates the frozen primary-validation
manifest, target implementation, source metadata, and recorded input
fingerprints before opening the first arm runtime. It rechecks the manifest
and source inventory after fingerprinting; changed inputs stop the suite.
This is a read-only preflight, not a full-source audit, launch authorization,
or a replacement for the checks performed during evaluation. Budget this
additional preflight on each suite invocation, including resume.

The local read-only preflight verified 3,300 recorded windows and 300
full-depth event regions in 198.82 seconds wall time at 1,000,718,336 bytes
peak RSS. It used the existing primary-validation manifest, not validation-v2
or final-test crops. Fingerprints do not cover whole source cubes or all
diagnostic labels; this is not a training-resource estimate. The portable
streamed suite passed 227 tests, with one skipped and 54 subtests, using
`.venv/bin/python -m pytest -q tests/test_streamed_*.py`.

The prepared-suite and per-arm paths now verify fixed monitor inputs, the
epoch-validator configuration and manifest/source paths, and the LPIPS state
against the frozen reconstruction guardrail before opening an arm runtime or
restoring a checkpoint. The validator also detects settings changed after
construction before evaluation. These checks do not replace the complete
resolved launch configuration, source-data revalidation, resource gates or
separate training approval.

Three synthetic input-gate and suite-order tests passed on ExFAT in 3.84 seconds
wall time at 512,360,448 bytes peak RSS. LPIPS fingerprinting and arm execution
were mocked; no training or real data was used, and temporary files were removed.
This is not a training-resource estimate. The portable streamed suite passed
225 tests, with one skipped and 54 subtests, using
`.venv/bin/python -m pytest -q tests/test_streamed_*.py`.

The no-training `prepare_suite_initial_checkpoints` helper now prepares fresh
S1/S2/S3 E1 checkpoints sequentially from an explicitly seeded sampler and
device, then verifies and persists initialization evidence and suite metadata.
It requires three separate, locked, reconciled, empty journals and verifies all
existing arm state before writing. It can recover an interrupted fresh bootstrap
without replacing valid checkpoints; missing files after evidence publication,
partial records or an advanced arm fail closed. An advanced suite must use
resume, not bootstrap. The caller still creates and owns the directories and
journals. The helper does not bind the full launch configuration, validate
resource gates, call the coordinator, or authorize training.

Three synthetic CPU/MPS bootstrap and recovery tests passed on ExFAT in 9.95
seconds wall time at 625,344,512 bytes peak RSS. Optimizer steps were prohibited
in the successful bootstrap checks, no real data was used, and temporary files
were removed. This is not a training-resource estimate. The portable streamed
suite passed 223 tests, with one skipped and 54 subtests, using
`.venv/bin/python -m pytest -q tests/test_streamed_*.py`.

An opt-in `run_suite_epochs` coordinator now calls the existing per-arm runner
in S1, S2, S3 order. It verifies the pinned suite metadata and all three locked
journals before opening an arm runtime, releases each runtime context before the
next, and supplies S1's reconstruction reference to S2 and S3. An arm failure
stops later arms, including when a runtime context suppresses an exception.
On retry the existing per-arm checkpoint recovery is responsible for avoiding
repeated committed updates. Numeric guardrail failures remain reported results,
not automatic adoption decisions.

The caller still owns journal locks, runtime construction and cleanup. This
prepared-suite coordinator does not bootstrap the suite, bind the complete
resolved launch configuration, enforce resource and logging gates, or authorize
training. It is not connected to the shell's blocked train/resume/evaluate modes.
Two synthetic ExFAT tests passed in 5.44 seconds wall time at 525,467,648 bytes
peak RSS; arm execution was mocked, no optimizer step occurred, and temporary
files were removed. This is not a training-resource estimate. The portable
streamed suite passed 220 tests, with one skipped and 54 subtests, using
`.venv/bin/python -m pytest -q tests/test_streamed_*.py`.

The per-arm runner now requires verified `suite_metadata.json` before restoring
or advancing an arm, and requires the caller's initialization pin to match the
pin in that metadata. The evidence file must be the suite's
`initialization_evidence.json`; missing, mismatched or changed metadata and
changed initial snapshots stop before restore. The gate is checked on resume
as well. It does not create metadata, establish the complete suite configuration,
or replace the sequential suite runner and training approval.

The synthetic restore-gate test passed on ExFAT in 3.76 seconds wall time at
479,133,696 bytes peak RSS; restore and training were mocked and temporary files
were removed. This is not a training-resource estimate. The portable streamed
suite passed 218 tests, with one skipped and 54 subtests, using
`.venv/bin/python -m pytest -q tests/test_streamed_*.py`.

An opt-in suite metadata helper now persists the initialization evidence pin,
matching configuration and checkpoint hash bindings in `suite_metadata.json`.
It verifies the evidence and retained initial snapshots before publication and
on reuse, including after an arm's `last.pt` advances. Existing partial or
mismatched metadata is not overwritten. Its exclusive publication is fsynced
but is not atomic on ExFAT; a hard interruption can leave a partial file that
must be investigated. The self-hash detects accidental changes, not coordinated
replacement of metadata and evidence. This helper does not persist the complete
resolved suite configuration, prove the initialization was used, or authorize
training. The future suite runner must persist this metadata before any arm
advances and verify it on resume, bind all launch settings and artifact paths,
and enforce the remaining gates.

Three synthetic metadata tests passed on ExFAT in 4.56 seconds wall time and
481,558,528 bytes peak RSS. No training or optimizer step occurred. The initial
cleanup command reported a nonempty temporary directory because PyTorch created
an empty cache directory; both directories were subsequently removed. This is
not a training-resource estimate. The portable streamed suite passed 218 tests,
with one skipped and 54 subtests, using
`.venv/bin/python -m pytest -q tests/test_streamed_*.py`.

The per-arm helper now verifies S2 and S3's completed recorded pair and
augmentation streams against S1 before selected-checkpoint comparisons. It
requires the initial checkpoint hashes from the pinned suite evidence, retains
a hash-bound result under the candidate's evaluation directory, and rechecks
the evidence before reusing a completed result. Mismatch, changed evidence,
invalid or duplicate records stop the arm rather than producing a comparison.
Interrupted publication is recoverable without another optimizer step. S1 has
no matched-reference result. These are recorded-stream checks only: they do not
prove source contents or actual tensors, verify all intermediate RNG states or
recorded optimizer rules, establish adoption readiness, or replace the suite
runner and its durable run metadata.

Three synthetic matched-record and recovery tests passed on ExFAT in 0.92
seconds, at 318,832,640 bytes peak RSS and 475,136 bytes MPS driver allocation
sampled afterward. They did not train or score real data; temporary files were
removed. These are not training-resource estimates.
The portable streamed suite passed 215 tests, with one skipped and 54 subtests,
using `.venv/bin/python -m pytest -q tests/test_streamed_*.py`.

The per-arm epoch helper now requires the caller's pinned suite-initialization
evidence before any restore or optimizer step. It verifies all three retained
initial snapshots and requires the current arm's `initial.pt`, `last.pt` and
journal to share an arm directory. S2/S3 must use an S1 reconstruction reference
from the initialized suite's S1 directory. Invalid or missing evidence stops before
restore. The helper also validates the frozen search/diagnostic manifest identities
and evaluation seeds, counts and wall limits before restore rather than discovering
invalid settings after an epoch. The suite runner must still persist the evidence pin
in durable run metadata, construct the complete frozen configuration, and verify recorded
streams; this gate alone does not establish a matched trajectory or authorize
training. Synthetic ExFAT testing of the gate took 2.15 seconds, with 443,613,184
bytes peak RSS and 475,136 bytes MPS driver allocation sampled afterward. Restore
and training were mocked; temporary files were removed. These are not training
resource estimates.
The portable streamed suite passed 211 tests, with one skipped and 54 subtests,
using `.venv/bin/python -m pytest -q tests/test_streamed_*.py`.

The opt-in suite-initialization preflight can now persist
`initialization_evidence.json` with the matching configuration, checkpoint hash
bindings and hash-bound preflight report. A separate verifier checks a caller-pinned
evidence hash and each retained initial snapshot even after `last.pt` has advanced.
The eventual suite runner must persist that evidence hash in its own durable run
metadata, call these helpers before training and on resume, and bind the complete
suite configuration; this evidence alone does not establish that the recorded
initialization was used, verify later streams or authorize training.

Evidence publication is exclusive and fsynced, but not atomic on ExFAT. A hard
interruption can leave a partial file: later attempts fail closed without overwriting
it, and it must be investigated. In-process cleanup checks the current identity
of the still-open file before removing it. Six synthetic ExFAT/MPS evidence and
recovery tests passed in 7.60 seconds with 465,747,968 bytes peak RSS and
75,972,608 bytes MPS driver allocation sampled afterward; no optimizer step or
training occurred. Temporary files were removed. These are not training estimates.
The portable streamed suite passed 210 tests, with one skipped and 46 subtests,
using `.venv/bin/python -m pytest -q tests/test_streamed_*.py`.

The opt-in, read-only `verify_suite_initialization` preflight checks S1, S2
and S3 together before any arm advances. It requires separate arm directories,
locked and reconciled empty journals, fresh configuration-bound E1 snapshots,
identical per-arm epoch-zero initial/last checkpoints, the approved fresh
optimizer/scheduler/stopping state and matching experiment-seeded training RNG
states. It rechecks checkpoint, journal and recovery-record identities before
returning a hash-bound `MATCHED_INITIALIZATION_ONLY` report. The eventual suite
runner must persist that report before training and independently verify the
retained snapshots and committed streams on resume; this preflight is not yet
wired into the suite runner and does not check source contents, full suite
configuration, later trajectories or training authorization.

The synthetic CPU/MPS initialization tests passed on ExFAT with no optimizer
step: 20 tests in 7.90 seconds, 493,944,832 bytes peak RSS and 75,972,608 bytes
MPS driver allocation sampled afterward. Temporary test files were removed.
These measurements are not training resource estimates.
The portable streamed suite passed 205 tests, with one skipped and 44 subtests,
using `.venv/bin/python -m pytest -q tests/test_streamed_*.py`.

The no-training bootstrap now seeds the initial Python, NumPy, CPU and, for an
MPS arm, MPS training RNGs from the bound `experiment_seed`. It isolates this
operation from the caller's RNG states and restores the model's training modes,
including on failure. Different ambient RNG states therefore do not change
the initial VAE-sampling state of freshly prepared matched arms. Existing initial
snapshots must have the expected seeded state or initialization fails without
replacing them; a snapshot made by the earlier ambient-RNG bootstrap may require
investigation rather than automatic reuse. The sampler, model/head and evaluation
seeds remain separately configured. Synthetic CPU and local MPS checks compare
S1 and S3 without an optimizer step. The suite runner must still bootstrap and
verify every arm, persist the full configuration and check recorded matched
streams; this is not training approval or evidence of matched final trajectories.
On the actual ExFAT volume, a new file's reported inode changed after its first
write. Publication cleanup now compares the destination against the still-open
file handle after writing, rather than a pre-write inode. The focused synthetic
CPU/MPS initialization tests passed on ExFAT in 3.49 seconds, with 479,559,680
bytes peak RSS and 42,418,176 bytes MPS driver allocation sampled afterward.
Temporary files were removed. These measurements are not training estimates.
The portable streamed suite passed 200 tests, with one skipped and 42 subtests,
using `.venv/bin/python -m pytest -q tests/test_streamed_*.py`.

An opt-in `prepare_arm_initial_checkpoints` helper now prepares a fresh arm
without an optimizer step. It requires a locked, reconciled, empty step journal,
the hash-bound matching configuration, fresh seeded E1 model and sampler, and
fresh approved optimizer, scheduler and early-stopping state. It publishes
separate byte-identical epoch-zero `initial.pt` and `last.pt`, verifies them,
and reuses a verified initial snapshot if `last.pt` was never created. Existing
unrelated or mismatched checkpoints are not overwritten. Source and run
directories must be separate, and validation-v2 remains excluded.

The checkpoint volume is ExFAT: hard links and exclusive rename to a new name
are unavailable. Publication uses exclusive file creation, fsync and digest
verification instead. This is **not atomic**: a hard interruption can leave
partial checkpoints or temporary files. Partial checkpoints fail closed and
are preserved for investigation; do not assume automatic recovery or delete
them without checking their provenance. In-process failures clean up only the
destination inode created by that attempt. A disposable synthetic ExFAT smoke
completed in 0.40 seconds at 418,873,344 bytes peak RSS, with two distinct
15,250,225-byte checkpoints, zero optimizer steps, and subsequent cleanup.
This is not a training-resource estimate. The helper is not a suite runner or
a training authorization; it does not persist the complete suite configuration.
The portable streamed suite passed 195 tests, with one skipped and 42 subtests,
using `.venv/bin/python -m pytest -q tests/test_streamed_*.py`; `.venv/bin/uv`
was unavailable.

The per-arm runner and matched-record verifier now require a hash-bound
`streamed-matching-config-v1`, stored in each checkpoint as `hashes["matching"]`.
It explicitly names the run and experiment seed, model/head/sampler seeds,
monitor/validation seeds, final epoch, batches per epoch and sparse-view
probability. The arm runner checks its runtime arguments against this configuration
before restore or training; the verifier checks both initial/final checkpoint
bindings and the recorded final run, seed and exposure. Before every restore,
the arm helper also requires a separately retained fresh E1 initial checkpoint,
and requires `last.pt` to be byte-identical to it at epoch zero. A retained
snapshot alone does not prove a resumed trajectory actually started from it.
No budget or probability default is selected. The matching configuration is not
the complete suite configuration; the future suite runner must persist it,
retain initial checkpoints, and invoke the verifier. These checks do not
authorize a launch.
The portable streamed suite passed 184 tests, with one skipped and 42 subtests.

An opt-in `verify_matched_arm_records` helper now checks a candidate S2/S3
against S1 using retained epoch-zero and reconciled final checkpoints plus
committed journal provenance. It verifies the specified fresh E1 weights and
initial sampler seed, matching configuration, exposure, recorded pair and
augmentation streams, and initial/final RNG states. Replayed stale records
are excluded. Its result is limited to recorded evidence: it does not prove
the source data or actual tensors were unchanged, verify every intermediate
RNG state, or authorize adoption. The suite runner must retain each arm's
initial checkpoint, bind the seeds to its frozen configuration, and invoke
this verifier; the helper does not launch training. Only synthetic evidence
was checked in this continuation. It does not check the recorded optimizer
rules or certify that the matching configuration is bound to the full suite configuration.
The portable streamed suite passed 178 tests, with one skipped and 21 subtests.

The per-arm epoch helper now requires the frozen reconstruction configuration
and historical reference hashes before restoring or training. After final
selection, it runs the recoverable reconstruction comparison for S1/S2's
reconstruction selection and for both S3 selections. S2/S3 require an S1
reconstruction reference; numerical failures remain visible in the returned
comparison records, not adoption decisions. An interrupted guardrail can be
retried from the completed epoch without another optimizer step. This is
synthetic-tested per-arm wiring only: the suite runner must still establish
matched initialization and stream provenance and pass verified references.
No historical or real guardrail data was scored in this continuation.
The portable streamed suite passed 166 tests, with one skipped and 21 subtests.

The selected-checkpoint orchestration now also has an opt-in
`complete_selected_reconstruction` primitive. It requires the selected formal
evaluation, a reconstruction-manifest hash, device and wall limit bound to both
the final and selected checkpoints as `hashes["reconstruction_guardrail"]`.
It measures the selected immutable snapshot with the frozen full-store producer,
records a hash-bound completion, and rechecks the stores, runtime and LPIPS state
before reusing a completed measurement. Interrupted attempts are preserved.
Synthetic tests cover the producer and recovery; this continuation did not measure
any real or historical guardrail data.

An opt-in `complete_reconstruction_comparison` primitive now checks the
selected measurement against the frozen P0b reconstruction reference, and also
against the selected S1 reconstruction reference for S2/S3. The historical report
hash, checkpoint hash and label must be bound to both final and selected checkpoints
as `hashes["reconstruction_reference"]`. S1 and the candidate must share run,
source, target, validation and experiment-seed identities. The helper verifies
the selected measurement artifacts and preserves a hash-bound numeric comparison.
Its `PASS_NUMERIC_ONLY` result does not verify matched initialization or sampling
streams, certify historical provenance, or authorize adoption. Synthetic tests
exercise the comparison; no actual reference was remeasured or rescored. The
measurement and comparison helpers are not yet wired into a suite runner.

The epoch orchestration now also runs frozen retrieval, bounded tokenizer search
and, for S3, diagnostic evaluation on the actual selected reconstruction checkpoint
and selected S3 retrieval checkpoint after the final epoch. It verifies the
reconciled final checkpoint, live selection history, immutable snapshot and alias
before evaluation. Interrupted attempts remain on disk; completed reports and
search outputs are checked before reuse. These are opt-in primitives with synthetic
tests, not a fully wired suite runner. Suite-level guardrail integration, launch inputs, budgeting, and the train/resume/evaluate shell
modes remain incomplete and blocked. Selected evaluation can repeat an evaluation
already performed at the final epoch, so budget that work explicitly.

On this continuation, the existing target preflight sampled one set of four
diagnostic pairs from each of 180 training and 25 primary-validation sources.
All 820 targets were available; this is a bounded diagnostic, not representative
yield or a retrieval-query floor. It took 99.47 seconds and 963,723,264 bytes
peak RSS. The existing bound primary-validation manifest
`13a5ef4f27c1ec0ab9da53168ee1e765f2e8cc94c74e3761c3e573e9c4c9d6c7`
verified 3,300 input windows and 300 full-depth event regions in 204.14 seconds
and 1,003,323,392 bytes peak RSS. These fingerprints do not verify whole stores
or generation-time provenance.

The no-optimizer real-data MPS smoke
`streamed-overlap-no-optimizer-mps-smoke-20261122-04` passed with 12 train and 12
validation pairs, all coverage targets available, and nonzero finite S3 encoder
and head gradients. It used the existing smoke-only sparse probability and 0.5
MPS memory fraction; no optimizer step or training occurred. Including startup,
it took 16.25 seconds, with 1,916,125,184 bytes peak RSS and 2,476,277,760 bytes
MPS driver allocation sampled afterward. These are not a full-training memory or
throughput estimate. The portable streamed tests passed: 152 tests, one skipped
and 21 subtests. The training-launch gate remains closed.

## Resume locally after the 2026-11-22 session

Reconnect from the MacBook using VS Code **Remote-SSH: Connect to Host**, select
the existing Mac mini SSH host, and reopen
`/Users/donaldpg/synthoseis-3dvae-poc`. Keep the Mac mini powered on, connected to
the network and awake; ensure the CrucialX9 volume is mounted. On the Mac mini,
`pmset -g` reported system sleep set to 0 and a local SSH connection received an
OpenSSH banner. These checks do not prove that the MacBook can reach the host after
a network change, sleep or reboot. If reconnection fails, check the network and
macOS System Settings > General > Sharing > Remote Login on the Mac mini; do not
change machine-wide settings or stop unrelated processes without approval.

Start a new chat in the remote workspace and paste:

```text
Continue the streamed overlap experiment locally on the Mac mini. First inspect
the branch and worktree; read the authoritative 2026-11-08 execution plan, the
target-contract proposal, and the 2026-11-11 Copilot handoff, especially its
2026-11-22 resume section. Continue on
experiment/streamed-overlap-copilot-implementation without changing production
tokenizer defaults. Preserve existing untracked work and private data. Finish
the remaining runner and evaluation work, validate it, and stop before production
training for separate launch approval. Do not select an epoch/batch budget.
```

Verify the starting point without altering the worktree:

```sh
git --no-pager status --short --branch
git --no-pager log -5 --oneline
test -d /Volumes/CrucialX9/fake_data && echo "primary sources mounted"
```

The branch includes `afe3c4b` (bounded search and diagnostics) and `09116ae`
(recoverable epoch-evaluation primitives). The earlier commit `afe3c4b` has a
mistyped Copilot co-author email; do not rewrite history without explicit approval.
Preserve the pre-existing untracked `studies/` and
`docs/sessions/2026-11-15-latent-mu-cosine-study.md`; do not stage or alter them.
Private manifests, checkpoints and audit outputs remain on the external volume and
are not in GitHub. `.venv/bin/uv` was unavailable; use `.venv/bin/python`, not
system Python.

Training is **not ready or authorized**. The bootstrap and prepared-suite
coordinator are not a complete launch runner. Next implement the complete
configuration-bound train/resume and logging integration, frozen bounded-search
and diagnostics inputs, and disk/memory budgeting. Then run the permitted integrated real-data MPS
smoke and present proposed launch commands, measurements and remaining approval
gates. A matched streamed S1 reference and independent-seed replication require
later training. No production thin-run threshold or training budget has been chosen.

The primary bound validation manifest is
`/Volumes/CrucialX9/3dvae-poc_checkpoints/streamed_overlap/validation-manifest-seed-20260926-20261121-v4/validation_manifest.json`
with logical SHA-256
`13a5ef4f27c1ec0ab9da53168ee1e765f2e8cc94c74e3761c3e573e9c4c9d6c7`.
Reverify it before scoring. Do not read or audit the regenerating
`/Volumes/CrucialX9/fake_data_validation_v2` until completion is confirmed and it
is separately audited; never use final-test crops for training or target-yield
preflight. Keep the fresh E1 mu128, shared trainable encoder and joint optimization
contract in the authoritative plan.

Most recent portable regression: 186 passed, one skipped and 17 subtests passed;
subsequent focused diagnostics/search/epoch tests passed (17 tests and three
subtests). A disposable real-data bounded-search MPS smoke passed in 244.96 seconds,
with 1,253,982,208 bytes parent peak RSS; tokenizer, window-score and Hann-map
hashes were independently verified. These smoke settings and measurements are not
a production budget or peak child/MPS memory forecast. Pylance requests timed out.

## Local implementation status

### Incremental target, manifest, and model implementation

The opt-in `scripts.streamed_overlap_targets` module now computes canonical overlap
and directional event-coverage targets on original seismic and same-index labels.
Unsupported windows and unusable semantic inputs have explicit unavailable reasons.
The sampler and opt-in joint-step primitives use these targets; the full training
runner is not integrated. The coarse closure fractions remain a proxy, and
the saved sources' generation-time worktrees remain unknown.

The diagnostic yield sampler reads only discovered train/validation sources. For each
source and requested sample it checks zero-shift, Z+16, same-source disjoint and
cross-source pairs. It is not the eventual batch sampler or retrieval candidate
manifest. For example, with a new private output path outside the source root:

```sh
.venv/bin/python -m scripts.streamed_overlap_yield \
  --source-root /Volumes/CrucialX9/fake_data \
  --output <new-private-yield-report.json> --seed 20260926 \
  --samples-per-source 1
```

The stricter diagnostic sampled all 180 train and 25 existing validation
sources: all 820 sampled pair targets were available. The 50 positive validation
**pairs** do not establish the required 200 eligible retrieval **queries**, and this
small sample is not a representative yield estimate. It does not freeze candidates,
prove eligibility, or open either approval gate. The stricter run took about 101
seconds and reached about 863 MiB peak RSS on this Mac. The private output is not
committed.

The model-independent validation manifest can be generated with:

```sh
.venv/bin/python -m scripts.streamed_overlap_manifest \
  --source-root /Volumes/CrucialX9/fake_data \
  --output <new-private-manifest.json> --seed 20260926 \
  --queries-per-cohort-source 6
```

The local candidate manifest contains 300 eligible queries: 150 from 25 sources in
each of the predeclared shallower/deeper query-depth cohorts. It fixes three overlapping,
three same-source disjoint and four cross-source candidates per query, with shuffled
candidate order and known relevance; ties use candidate ID. Its content hash is
`8d299e067dd54758021ae083c57b4df8da2755a954124eb4169c4fbd1e43ba3b`.
Generation took about 270 seconds and 907 MiB peak RSS. The private manifest is
stored on the checkpoint volume, not committed. Its source inventory records metadata,
not cryptographic hashes of the complete source data; source-content integrity still
needs verification before model evaluation.

The opt-in pair model and augmentation helpers are implemented and tested, including
joint-step primitives. They use a fresh matched residual mu128 VAE, a shared trainable
encoder, calibrated overlap and ordered coverage heads, and a combined loss. Views
can use extrema or sparse input masks while reconstructing normalized dense augmented
A. A guarded local S3 forward/backward smoke used 12 real train pairs, source-band
noise and LPIPS, without an optimizer step. It took about 10.5 seconds wall time;
peak process RSS was 1.16 GiB, total peak memory footprint about 3.69 GiB, and MPS
driver allocation after backward was 2.37 GiB. This is not a training-throughput
estimate and did not test journal, resume, evaluation or TensorBoard integration.

The source-bound validation manifest adds hashes for 3,300 canonical input windows
and 300 full-depth event regions. The bound manifest's content hash is
`13a5ef4f27c1ec0ab9da53168ee1e765f2e8cc94c74e3761c3e573e9c4c9d6c7`.
Binding took 422 seconds and about 949 MiB peak RSS; a separate verification took
123 seconds and about 965 MiB peak RSS. The files are private on the checkpoint
volume. Verify again immediately before evaluation; these hashes do not cover whole
source cubes, other diagnostic labels, or generation provenance.

```sh
.venv/bin/python -m scripts.streamed_overlap_integrity --mode verify \
  --source-root /Volumes/CrucialX9/fake_data \
  --manifest <private-bound-validation-manifest.json> \
  --output <new-private-verification-report.json>
```

The opt-in single-worker pair sampler now draws fresh volume-balanced pairs, keeps
separate sampling and augmentation RNG states, bounds open source handles, computes
all canonical targets before augmentation, and rejects train/validation recorded
geology-seed overlap. A restored sampler reproduces both pair specifications and
augmented inputs. The sparse-input probability is an explicit parameter, not a
training default or an approved launch setting.

With the real sampler, separate train and primary-validation 12-pair S3 MPS
forward/backward checks each contained six overlapping, three disjoint and three
cross-source pairs, with two zero-shift pairs and 12 valid coverage labels. Neither
performed an optimizer step. Including process startup, the train check took 13.14 s,
1.13 GiB peak RSS and 3.79 GiB peak memory footprint; the validation check took
11.74 s, 1.58 GiB peak RSS and 3.80 GiB peak memory footprint. MPS driver allocation
after backward was approximately 2.30 GiB for each. These checks used a 0.5 MPS
memory-fraction guard and a test-only sparse-input probability of 0.5. They do not
establish training throughput, full training-memory headroom, or approval of that
probability. They did not exercise an optimizer, journal, recovery, evaluation or
TensorBoard integration.

The existing trainer now has an opt-in single-batch joint-step primitive. Synthetic
tests exercise one combined backward/optimizer step for each arm and verify encoder
and enabled-head updates. The associated journal fsyncs each completed CSV row, uses
a single-writer lock, preserves torn bytes in place with immutable recovery records,
and identifies stale and replayed events relative to a checkpoint. Per-step source
and augmentation provenance is written separately. Atomic epoch-state primitives
capture the model, optimizer, approved scheduler, RNGs, sampler position, selection
state, hashes and journal watermark; synthetic tests reproduce the joint step after
recovery. A failed optimizer or journal write requires a fresh recovery attempt.
These are primitives, not an integrated training or resume runner.

The opt-in epoch helper now draws fresh batches, logs step and epoch components,
noise and throughput, runs fixed train/validation monitors every 100 steps and at
epoch boundaries, and schedules on validation MAE plus weighted LPIPS and KL only.
Monitor inputs, labels and transforms are hash-bound; inference restores model modes
and Python, NumPy, CPU and device RNG state. An initial full-state checkpoint supports
replay even if the first epoch is interrupted. Synthetic tests cover epoch checkpoint
failure, retained stale CSV rows, torn-tail recovery and exact model replay. The
epoch CSV associates each row with its completed batch event so recovery records
identify stale attempts; preserved torn bytes require the recovery records when
parsing history.

A separate real-data MPS monitor smoke used 12 train and 12 primary-validation pairs,
with no optimizer step. Including startup and loading, it took 17.83 seconds,
1.64 GiB peak RSS and 2.88 GiB peak memory footprint; monitor inference itself took
0.49 seconds and sampled MPS driver allocation afterward was 1.29 GiB. CPU and MPS
RNGs and training mode were restored. The check used a 0.5 MPS memory-fraction
guard and a test-only sparse-input probability of 0.5; its untrained-head loss
values are not retrieval results. This does not establish full training headroom.

The opt-in ranking calculation module now handles linear-gain NDCG@10 with
candidate-ID tie-breaking, frozen eligibility checks, an explicitly seeded repeated
uniform-random baseline, paired source-cluster bootstrap, separate same/different
source candidate summaries, and matched-S1 mu-cosine comparisons. Its retrieval
metric-only gate compares each depth cohort with the same random baseline and
requires independent-seed replication; absent replication remains pending. Seeds
and repetition counts are explicit, not approved launch settings. Synthetic tests
exercise the calculations and the existing bound manifest passes the 300-query
eligibility check without reading source cubes.

An opt-in checkpoint retrieval evaluator now hashes a completed epoch checkpoint,
matches its arm and step to the locked journal and obtains seed and attempt from it, rejects
discarded recovery events, and verifies the bound validation inputs immediately
before scoring. Each seismic window is checked again when read. Mu-cosine uses the
tokenizer's actual preprocessing and similarity function. Directional head scoring
requires S3, committed valid coverage supervision and nonzero saved optimizer
gradient evidence. Its explicit batch size and wall-time limit must be configured
before use; timeouts fail rather than returning partial rankings. Synthetic tests
exercise the evaluator; five focused tests took 1.70 seconds and about 336 MiB
peak RSS including interpreter startup, using tiny fixture models and arrays.
These are not real-data evaluation resource estimates. No real checkpoint or model
has been scored.

An opt-in epoch callback now verifies the same frozen inputs, evaluates
tokenizer-aligned mu-cosine on the frozen queries, and computes sampled
reconstruction MAE, LPIPS and KL on an explicitly fixed subset of query IDs. Its
manifest, query IDs, batch sizes, seed and wall-time limit have a configuration
hash that must match the epoch helper's checkpoint hashes. Validation metrics are
stored in the full-state checkpoint and the helper returns its hash and journal
identity. A separate helper supports a create-exclusive export of a completed VAE
checkpoint to the existing tokenizer adapter format; it does not change tokenizer
defaults or run a tokenizer search.

The numerical reconstruction guardrail compares fully covered, hash-identified
synthetic and real measurement records using the fixed +2% recipe and MAE
thresholds and matched sampling protocol. A separate opt-in full-store producer
can now freeze and reread every baked `divide_by_std` patch, fingerprint tokenizer
inputs and dense targets, and measure sampled reconstruction using the actual
tokenizer adapter or a journal-bound streamed checkpoint. It binds the device,
LPIPS weights and implementation, relevant code and package versions, rejects
changed inputs/checkpoints and restores inference RNG state. It uses a fixed
one-patch-at-a-time sampling protocol, with no augmentation of guardrail targets.
The numerical comparison remains **not** an adoption decision: dataset/reference
approval, source provenance and selection provenance are not established by these
helpers. No final synthetic-test data was accessed. The recorded hashes cover decoded
float32 patches and tokenizer inputs, not complete source-store or generation provenance.

The owner subsequently approved auditing the historical synthetic/real candidates
and freezing/measuring them if suitable. The historical synthetic patch store has
5,000 baked-scaled patches from 25 historical source paths and records the old
`label_z_offset=1`; those derived labels are not used in reconstruction and are
not suitable as the new semantic targets. The separate historical real test store
has 5,000 patches from four real volumes and is strictly a post-selection
reconstruction guardrail, not an epoch-selection or untouched-test dataset.
Archived original generation logs identify horizon, fault, closure and fan seeds
for all 25 historical synthetic sources; none matches the corresponding seeds in
the current 180-source training inventory snapshot. The archived seed blocks do
not establish channel-simulation seeds, exact generator worktrees, or complete
independence. Thus this is a historical reconstruction control, not a certificate
of current semantic-label validity or fully verified geological provenance.

The approved historical reconstruction manifest was refreshed at private checkpoint
volume path `historical-reconstruction-guardrail-20261122-04/manifest.json`, hash
`c7cf3ca0b9fd0307f348821a4e0b160475cdfa88452312e237358f2c741ad719`.
It fingerprints all 5,000 synthetic and 5,000 real inputs/targets with sampled
tokenizer reconstruction on MPS, seed 20260926 and a fixed LPIPS implementation.
The evaluator implementation is fingerprinted, so its overlap-diagnostic change
required this refreshed manifest; all patch fingerprints match the earlier `-01`
manifest. Freezing took 27.27 seconds including startup and 836,321,280 bytes peak RSS.
The historical P0b reconstruction-best checkpoint is seed-20260926 epoch 40,
SHA-256 `090e6f14e9f68941b468a7189e22a7c2dfc41de8571efa7e9ccff5273c5f439b`;
its retrieval reference is a different checkpoint. A matched streamed S1 reference
does not exist yet. The regenerated validation-v2 sources remain off limits until
the owner confirms completion and they are separately audited.

The full P0b reference measurement was written privately as
`historical-reconstruction-guardrail-20261122-04/p0b_measurement.json` (file SHA-256
`d8a2a60a879edf94dd0e7c2f9a9d1430227b30916df5bfcac6a3c57b6b6df3aa`).
The matched protocol produced synthetic recipe total 0.4931984 and real MAE
0.7100480 across all 10,000 patches. The producer's numeric/identity validation
passed; this is a reference measurement, not an experiment result or adoption.
All per-patch metrics exactly match the earlier `-01` reference. This measurement
took 208.07 seconds including startup, 953,303,040 bytes peak RSS and
93,143,040 bytes MPS driver allocation sampled afterward. The 0.5 MPS
memory-fraction guard is not a full-training memory estimate.

The checkpoint retrieval evaluator now also reports directional event-coverage
errors by pair category and per-pair closure-proxy fluid mismatch where the frozen
target records it. The existing frozen primary manifest's 1,800 recorded fluid
triples passed the diagnostic's numeric checks without scoring a model or rereading
source cubes. Missing fluid fractions are explicitly unavailable. NTG,
thin-layer and no-reservoir diagnostics require separately fingerprinted
inputs. Opt-in checkpoint scoring can also report physical-overlap MAE, RMSE and
bias by pair category and fixed overlap bins, against a zero constant and an explicit
waveform baseline: original-window mean-centered Pearson correlation mapped to
`[0, 1]`. Nonconstant windows and a journal/optimizer-verified trained S2/S3
overlap head are required. These diagnostics are numeric only and do not establish
retrieval performance or adoption.

An opt-in checkpoint selector records immutable journal-linked selection events
and publishes separate best-reconstruction and best-retrieval aliases for
`last.pt`. It excludes discarded recovery events, repairs interrupted alias
publication on the next selection, verifies retrieval eligibility and trained-head
evidence, and never treats checkpoint ranking as adoption. Historical best
snapshots are retained as verified copies so earlier selections can be restored
after recovery; their disk consumption can grow throughout a run and must be
budgeted before launch. The checkpoint volume does not support hard links; a local
probe confirmed file/directory fsync and atomic replacement.

The independent-seed provenance check, frozen diagnostic/search launch inputs and
full runner wiring remain. Keep launch modes blocked.

The opt-in bounded-search primitive now freezes primary-validation regions, query
identities, tokenizer implementation hashes, stride, batch size, window count,
memory limits and timeout. It verifies source fingerprints and runs the actual
tokenizer CLI in explicit VAE/mu/cosine mode. An optional CLI JSONL stream preserves
individual window scores without changing the existing Hann output; reports distinguish
the best fully contained window and query rank from the Hann-blended voxel peak.
A disposable real-data MPS check used one canonical-sized validation region, 45 padded
windows and one fully contained window. It passed in 272.40 seconds including
source verification and startup. Parent peak RSS was 1,249,230,848 bytes; this is not
a peak measurement for the CLI child or a launch estimate. The untrained checkpoint
and one-window ranking are a plumbing check, not a scientific result. An initial
attempt failed because the MPS low watermark exceeded the configured high watermark;
the corrected subprocess sets both explicitly. A second check after adding output
fingerprints passed in 244.96 seconds, with 1,253,982,208 bytes parent peak RSS;
this is still not a peak measurement for the CLI child. No production search
manifest has been approved, and no search is yet invoked by the training runner.

Separately, a diagnostic sidecar can fingerprint native-index NTG, lithology and
channel-label windows and reverify them before reporting directional pair errors.
It reports mean stored NTG over corrected rock, corrected-sand absence, and complete
within-window corrected-sand runs below a required predeclared voxel threshold.
Window-edge runs are censored. These are diagnostic proxies, not proven geological
layers or proof of reservoir absence. Missing, unsupported and invalid arrays retain
explicit reasons; a water-only window has no NTG-over-rock value. The sidecar does
not verify checkpoint provenance, has not yet been frozen for the primary cohort,
and is not yet connected to a production suite runner.
The sidecar can be bound or verified read-only with explicit settings, for example:

```sh
.venv/bin/python -m scripts.streamed_overlap_diagnostics --mode bind \
  --source-root /Volumes/CrucialX9/fake_data --manifest <bound-validation-manifest.json> \
  --output <new-private-sidecar.json> --thin-max-voxels <predeclared-threshold> \
  --max-wall-seconds <limit>
```

Verification uses `--mode verify --bound <private-sidecar.json>` instead of
`--thin-max-voxels`. Neither mode uses final-test crops or trains a model. The wall
limit rejects an over-budget result but does not interrupt an in-progress source
verification or individual array read.

An opt-in epoch orchestration helper now restores an arm's full checkpoint and
requires the last committed epoch's evaluation to finish before training advances.
It schedules checkpoint-bound directional evaluation, bounded tokenizer search
and diagnostic reporting every five epochs and at the final epoch, then publishes
the appropriate reconstruction and (S3 only) retrieval selection. Completion
records are immutable and hash-bound; interrupted evaluation attempts are preserved
and retried, and completed reports, tokenizer checkpoint, window scores and voxel
maps are checked on recovery. S1/S2 cosine reports are never passed as pair-head
retrieval selections. This is a tested orchestration primitive, not a fully wired
suite runner, launch authorization or evidence that final selected checkpoints
have received all post-selection searches and guardrails. Train/resume/evaluate
shell modes remain blocked.
The existing interrupted-checkpoint/replay fixture now also runs on MPS: it uses
a small synthetic S3 model and synthetic pairs, exercises optimizer steps, epoch
logging, an injected checkpoint failure and deterministic replay, and never opens
source data. With explicit `STREAMED_MPS_SMOKE=1`, the MPS fixture took 10.08
seconds including startup and 636,141,568 bytes peak RSS while the portable
regression suite was also running. It is skipped without that opt-in. This is
not the required full-size, real-data end-to-end MPS runner smoke or an approval
to train.

```sh
STREAMED_MPS_SMOKE=1 .venv/bin/python -m pytest -q \
  tests/test_streamed_overlap_step.py -k synthetic_epoch_recovery_on_mps
```
The standalone disposable real-data integration smoke now also exercises a full-size
fresh S3 E1 model on MPS: one 12-pair optimizer step, an injected interruption,
epoch-boundary checkpoint restoration and replay, fixed train/validation monitors,
the source-verified 300-query primary validation callback, epoch CSV/TensorBoard,
and reconstruction checkpoint selection. It explicitly does not score a pair head
as retrieval or establish a matched S1 reference. It is not wired into the production
runner and does not implement its train/resume/evaluate modes.

Run `disposable-integration-smoke-20261122-02` used a 0.5 MPS memory fraction, a
single fixed validation reconstruction query and encode batch size 4, all smoke-only
settings. It passed in 287.93 seconds including startup, with 1,945,108,480 bytes
peak RSS, 4,281,207,304 bytes peak macOS memory footprint and 2,329,804,800 bytes
MPS driver allocation sampled afterward. The last measurement is not a peak-MPS
measurement; this is not a full-training resource forecast. The selected
reconstruction alias matched `last.pt`; the persisted report SHA-256 is
`e18323a07a1dcbeccfa9eb59310dc23ae6ffa253374aa8ede5663c7715d296b1`.
The 900-second wall-time setting rejects a late result but is not a hard watchdog.

To repeat with a new disposable output directory, provide a private JSON config
containing `source_root`, `seed`, `sparse_probability`, `mps_memory_fraction`,
`max_wall_seconds`, `validation_manifest`, `validation_manifest_sha256`,
`reconstruction_query_ids`, `encode_batch_size` and `reconstruction_batch_size`:

```sh
.venv/bin/python -u -m scripts.streamed_overlap_integration_smoke \
  --config <private-smoke-config.json> --output <new-disposable-output-directory>
```

The code checks the frozen manifest and query IDs before an optimizer step, excludes
the regenerating validation-v2 dataset, and rejects MPS CPU fallback. Smoke budgets
and outputs must not be used as approved production settings or scientific results.
Fourteen synthetic evaluator, selection and guardrail tests used temporary files
on the checkpoint volume and took 4.06 seconds and about 343 MiB peak RSS
including startup. This is not a real-data runtime or memory estimate.
A separate synthetic CPU smoke exercised full-store freezing, two repeat measurements,
the real tokenizer adapter and cached `SliceLPIPSLoss`, using one artificial patch
per domain and an untrained tiny fixture checkpoint. It took 1.85 seconds, about
794 MiB peak RSS and 724 MiB peak memory footprint including startup; it performed
no optimizer step and is neither a guardrail result nor a full-suite/MPS estimate.
The sampler's source-manifest hash covers inventory metadata, not full
training-source contents. The journal currently reads completed CSV rows into
memory; its long-run memory behavior must be assessed before launch. Keep train,
resume and evaluate modes blocked. No training has been authorized.

The wrapper now supports a distinct no-optimizer MPS smoke. Run
`streamed-overlap-no-optimizer-mps-smoke-20261122-03` exercised real train and
primary-validation batches (12 pairs each), sampler replay, a combined S3 backward
pass with finite nonzero encoder and head gradients, monitor state restoration and
TensorBoard event writing. The report hash, smoke source hash and event file were
verified. Including startup it took 13.36 seconds, with 1,919,434,752 bytes peak
RSS and 2,476,277,760 bytes MPS driver allocation sampled after monitoring. These
are neither peak MPS usage nor full-training resource estimates. This test performed
no optimizer step and did not exercise epoch orchestration, checkpoint recovery or
retrieval scoring. Its sparse probability and 0.5 MPS memory fraction are smoke-only
settings, not approved training settings. The wall-time limit is checked between
stages, not a hard watchdog. Source discovery prunes directories named
`fake_data_validation_v2` without descending into them.

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
considering it for any use. The existing validation set remains primary. Target and
diagnostic yield checks and the model-independent candidate manifest now exist, with
the source-integrity limitations above. The separate training-launch approval gate
remains closed.

Local implementation is on `experiment/streamed-overlap-copilot-implementation`.
The audit tooling, pair-target generation and local source inventory exist; integrated
training, resume, evaluation, completed-step CSV and automatic trainer/TensorBoard
integration remain subsequent work. Do not treat this handoff or the current tests as
permission to launch training.

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

Fresh no-optimizer smoke (requires MPS and a new run ID):

```bash
bash scripts/streamed_overlap_experiment.sh --mode smoke \
  --config scripts/streamed_overlap_smoke.example.json \
  --run-id <new-smoke-run-id>
```

The smoke creates private output and a run manifest; `--print-commands` is
side-effect-free. It is not a training-readiness or recovery test.

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
