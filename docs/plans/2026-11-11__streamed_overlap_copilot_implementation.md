# Copilot handoff: streamed overlap and directional semantic retrieval

Updated: 2026-11-22.

## Local continuation: post-selection evaluation

The selected-checkpoint orchestration now also has an opt-in
`complete_selected_reconstruction` primitive. It requires the selected formal
evaluation, a reconstruction-manifest hash, device and wall limit bound to both
the final and selected checkpoints as `hashes["reconstruction_guardrail"]`.
It measures the selected immutable snapshot with the frozen full-store producer,
records a hash-bound completion, and rechecks the stores, runtime and LPIPS state
before reusing a completed measurement. Interrupted attempts are preserved.
Synthetic tests cover the producer and recovery; this continuation did not measure
any real or historical guardrail data. The helper is not yet wired into the suite
runner or matched S1/P0b comparisons, and its result is not an adoption decision.

The epoch orchestration now also runs frozen retrieval, bounded tokenizer search
and, for S3, diagnostic evaluation on the actual selected reconstruction checkpoint
and selected S3 retrieval checkpoint after the final epoch. It verifies the
reconciled final checkpoint, live selection history, immutable snapshot and alias
before evaluation. Interrupted attempts remain on disk; completed reports and
search outputs are checked before reuse. These are opt-in primitives with synthetic
tests, not a fully wired suite runner. Reconstruction guardrail comparisons on
selected checkpoints, launch inputs, budgeting, and the train/resume/evaluate shell
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

Training is **not ready or authorized**. The epoch helper is not the sequential
S1/S2/S3 suite runner. Next implement and verify the actual configuration-bound
suite runner, durable train/resume and logging integration, selected-checkpoint
reconstruction comparisons, frozen bounded-search and diagnostics
inputs, and disk/memory budgeting. Then run the permitted integrated real-data MPS
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
