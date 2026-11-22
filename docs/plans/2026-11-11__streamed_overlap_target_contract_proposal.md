# Streamed retrieval: local audit and target-contract proposal

Status: **LOCAL_TARGET_VALIDATION_PENDING**. This records the owner's scientific
decisions and the bounded local evidence; the streaming target, yield preflight and
evaluation manifest still need implementation and checks. It is not permission to
launch training.

## 2026-11-22 owner decisions and current status

The owner reports that all SynthoSeis datasets have been updated; the November 15
audit observations below are historical and have not yet been revalidated. The owner
also replaced the proposed equal-weight distinct-layer and cross-volume analog tasks
with directional **fluid-adjusted observable-event coverage**, selected by the ordered
pair head. The [governing execution plan](2026-11-08__streamed_overlap_and_semantic_retrieval_experiment.md#3-approved-scientific-contract)
records the current event ratio, zero-target cases and window-level fluid formula.
Any older statements below about equal-weight layers, cross-volume analog labels or
an ongoing replacement are historical, not instructions for the new target. On
November 22 the owner approved the smoothed strict adjacent-sample detector, native same-index
label support, closure-mask fractions as a coarse operational fluid proxy despite
unknown per-source generation-time worktrees, source-specific filter metadata, and the
proposed evaluation success criteria. The executable target and yield checks remain
to be implemented. The separate training-launch gate also remains closed.

The owner reports that all datasets under `/Volumes/CrucialX9/fake_data` were
recreated, but SHA values and timestamps do not unambiguously distinguish the initial
and recreated outputs. Treat those fields as recorded metadata, not conclusive age or
generation evidence. `/Volumes/CrucialX9/fake_data_validation_v2` is being regenerated;
the existing validation cohort remains primary, and v2 is a separate optional cohort
that must not be read, audited, or mixed until regeneration is complete. Audit it
separately before considering it for any use.

Revision: incorporates the owner's November 11 clarification of channel sand and
closure geometry. The earlier recommendation to invalidate a whole semantic pair
merely because sand has no closure-mask hit is **withdrawn**.
The owner's subsequent hierarchy also supersedes the proposed unconditional brine
channel override: channels retain identity, while oil/gas closures modify fluid
within sand, including channel sand.

## 2026-11-15 channel/closure follow-up

**Historical evidence only.** The 2026-11-22 recursive inventory in §7 supersedes
this section's earlier claims about which arrays exist across the source collection.
The three sampled volumes remain useful label-consistency evidence, not an inventory
of all refreshed sources.

The owner clarified that updated training sources are runs **1041 through 1214**;
this explicit run range supersedes using a rolling 48-hour cutoff as the sole
definition of the updated cohort. Replacement remains in progress outside that range.
This follow-up scanned every stored label voxel in three selected, non-random
channel-positive runs, in bounded 50 x 50 x 1510 chunks:

| Run | Channel/levee voxels | Gas closure in channel | Oil closure in channel | Brine closure in channel |
|---|---:|---:|---:|---:|
| 1165 | 751,246 | 87,104 | 185,063 | 0 |
| 1181 | 1,228,819 | 5,078 | 13,714 | 11,237 |
| 1186 | 222,737 | 10,299 | 25,934 | 0 |

Across each of the three complete 300 x 300 x 1510 label grids, gas/oil/brine
closure masks were binary and mutually exclusive; every fluid-mask voxel was
inside raw `faulted_lithology >= 0.5`, had a positive closure-segment ID and a
closure-type mask, and the respective unions agreed. All channel/levee-labelled
voxels had raw lithology exactly 1.0; none were labelled raw water or shale.
All 46, 79 and 42 closure IDs in runs 1165, 1181 and 1186 respectively had
exactly one fluid and one closure type. Both channel-label values 1 and 2
occurred. This is positive evidence of
**internal label consistency**, not proof of actual fluid fill or of every
source in the updated range.

The local SynthoSeis generator is a Git checkout at
`/Users/donaldpg/synthoseis/synthoseis`; the earlier statement that the
neighboring generator was not a Git repository was incorrect. Its HEAD matches
the recorded `bd51313...` SHA but **Faults.py and Seismic.py have uncommitted
changes**, so the SHA alone does not identify the code used for regeneration.
The current working-tree code copies channel N/G into the faulted property model
before closure finding and mixes the selected fluid once; it removes the
historical late brine-channel remixture. Seven focused generator tests passed at that
historical review; a later 2026-11-22 run passed 13 channel end-to-end tests, including
gas/oil/brine with both inverse-velocity and Backus mixing and N/G insertion. This
supports that the current path works on synthetic fixtures, **not**
that these specific saved cubes were generated by this exact dirty worktree.

At the earlier review date, the selected output stores did not contain
`faults/faulted_channel_ng`, `faulted_net_to_gross`, or `rho`/`vp`/`vs`; thus actual N/G and synthesized
fluid properties cannot be compared voxel by voxel from these stores alone.
Absence of a closure mask is not absence of channel sand; open channel fluid
is a synthesis-policy inference, not a stored brine-closure label.

**Historical N/G inventory follow-up (superseded by §7):** On 2026-11-15 all **174** owner-identified training
stores, runs 1041-1214, were recursively enumerated, including nested groups.
All had the same 19 array keys and none had an array named for N/G, including
`faults/faulted_channel_ng` or `faulted_net_to_gross`. No other immediate Zarr
store was present in these source directories. The stored channel array has
categorical labels, not per-voxel N/G; `geologic_score` is an aggregate including
a thresholded channel-N/G contribution, not an N/G volume. Seismic is influenced
by N/G but is not a direct retained N/G label.

The current generator creates both N/G arrays during generation, but its
`--zarr-out segmentation` allowlist omits them and its pruning step removes
non-allowlisted arrays. This explains how the observed output schema can arise;
the output keys alone do not prove which command or generator worktree was used.
For future provenance-pinned generation, retain both arrays explicitly (or use
`--zarr-out all` after reviewing the much larger storage requirements), then
verify their keys, shapes, ranges and channel-voxel values before pruning/copying.
This was a historical follow-up. Do not modify or regenerate existing sources without
owner approval.

The current generator inserts channel/levee labels and N/G over host stratigraphy;
the label values distinguish channel sand (1) from levee (2). In an examined
run-1186 section at Y=98, X=105, a contiguous channel spans Z=993..1001 while
stored host age changes from approximately 181.224 to 182.873. The stored age
therefore must not automatically be interpreted as the depositional identity of
the channel fill. This observation does not by itself prove how much older rock
was eroded, nor validate varying N/G: pre-insertion lithology, erosional surfaces,
and per-voxel N/G were not available in the selected output stores.

Detailed scan: local session artifact `channel_closure_scan_20261115.json`.
Representative label-grid sections are in
`channel_closure_representative_sections.png` in the same session artifacts.
They do not require the unverified seismic-to-label Z offset.

**Historical provenance recommendation:** generation-time commits and dirty-diff/source
hashes would be needed to prove how any particular saved source was made. On
2026-11-22 the owner approved the closure masks as a coarse operational target proxy
despite this uncertainty. Keep the limitation explicit; do not describe the closure
fractions as verified physical-fluid truth. The local generator's current synthetic
tests support its N/G and closure insertion path but do not retroactively establish
the exact worktree used for each saved volume.

## 2026-11-15 regeneration check and Copilot instructions

**Status: unresolved, not a failed verdict on the replacement generator.** The owner
initially used datestamps newer than 48 hours as an update indicator; the later
explicit run range above supersedes that as the cohort definition. At approximately
16:46 local time on
November 15, using each uniquely paired `model_parameters_*.txt` file's filesystem
modification time as the datestamp and a cutoff of November 13 16:46 CST:

- The fresh whole-root inventory had 180 training and 25 validation stores. Of these,
  **93 training and zero validation** stores met the 48-hour criterion; all 93 had
  the inspected semantic arrays and recorded SHA
  `bd51313ebfb620800bb2973c2cea28f4b9325a15`. Their matched config and store-directory
  timestamps also met the cutoff at inspection. The other 87 training and 25 validation
  stores are outside this historical recent cohort. Some are now known by the owner
  to be updated under the run-range clarification. Filesystem timestamps classify
  that snapshot, not completion, content identity, or a clean generator worktree.
- All 25 validation stores were outside that window. Five lacked the inspected
  geology/fluid arrays. These are **older-store findings**, not a defect established
  in regenerated validation. Completion and schema of updated validation remain unknown.
- The 93 recent sources had fullstack shape `(300,300,1499)` and inspected semantic
  label shapes `(300,300,1510)`; the difference alone neither proves nor
  disproves an offset. Eight recent training stores were separately sampled with
  `geologic_age_faulted` and the existing correlation diagnostic. Envelope best offsets
  were `[4, 3, 4, 7, 2, 3, 15, 15]`; its aggregate `verified` flag was false.
  Correlation is a heuristic: this result means the mapping is **not verified**, not
  that any particular offset is definitively wrong. The earlier eight-volume
  validation report concerns older stores and cannot certify the recent cohort.
- A separate six-store recent training diagnostic used bounded fluid-seeking probes
  and four views per source at the unverified `+1` hypothesis. Two sources included
  channel/hydrocarbon-closure overlaps: run 1165 disjoint view had 8,453 voxels;
  run 1186 A and Z+16 views had 673 and 609 respectively. Views can overlap, so
  these counts are not a prevalence estimate or distinct-voxel total. A recorded
  historical SHA has a known late channel-mixing rule, but does not prove the
  generation worktree was clean or establish the actual final elastic properties
  in these new stores. The synthesis/target question remains unresolved.

Local evidence is in the new audit directory
`/Volumes/CrucialX9/3dvae-poc_checkpoints/streamed_overlap/regenerated-source-audit-20261115-03`:
`source_manifest.json`, `audit_report.json`, `recent_training_age_alignment.json`,
and `recent_training_fluid_diagnostic.json`. The fresh root audit sampled only one
recent training source; the last two reports explicitly sample the recent cohort.
These snapshots may become stale while regeneration continues. Source metadata changed
for run 1214 between the two fresh whole-root audits; a manifest is not a content
freeze. No final-test crops or training were used.

### Historical next-agent instructions

The following November 15 instructions describe the earlier audit and target.
Use them for provenance and alignment context only; the November 21 decisions and
the governing execution plan override their outdated cohort and target wording.

```text
Keep LOCAL_TARGET_VALIDATION_PENDING; do not run production training or silently
substitute overlap-only training for the requested two-output experiment.

1. Reinventory after the owner's ongoing source replacement. Respect the owner's
   explicit updated run range 1041-1214; report actual modification times and, where
   useful, the inspection time, timezone and 48-hour cutoff separately. Record each
   uniquely paired model_parameters_*.txt's actual
   time, source/split, configuration and parameter hashes, array keys/shapes/chunks,
   recorded generator SHA and completion evidence. Compare config/store timestamps;
   investigate mismatches and sources changing during inspection instead of assuming
   that a recent timestamp proves a complete, stable store. Preserve separate recent,
   updated, unconfirmed and incomplete cohorts; never infer completion or generator
   identity from the parent directory or recency alone. Recompute as work advances.
2. Do not modify, overwrite, remove or regenerate source data as part of this audit.
   Coordinate with the owner before any generator change or replacement run. Obtain
   the exact generator revision, dirty-worktree diff or source fingerprint and
   generation parameters for each relevant cohort; recorded Git SHA alone cannot
   rule out local edits. Unknown or mixed provenance remains explicitly unverified.
3. Trace XYZ axis and depth indexing from the actual generator through padding,
   convolution/filtering and final fullstack/label writes for each source/key/version.
   Build deterministic small known-index fixtures that exercise boundaries and any
   offsets, and compare their outputs with stored training and validation examples.
   Supplement with bounded numerical/visual checks across representative updated
   sources. Do not infer +1 from shape difference or promote a weak correlation
   optimum to verified truth. Record evidence, exceptions and any per-version mapping.
4. For hydrocarbon-closure/channel intersections and non-channel controls, compare
   actual source-version mixing order and resulting elastic properties/seismic against
   the owner's hierarchy: channels remain channels; oil/gas fills apply within
   corrected sand. Use reproducible manufactured endmember cases plus provenance-
   matched source evidence; closure masks alone do not prove final fluid. If behavior
   conflicts, present a source-generator fix and validation/regeneration plan for
   owner approval, preserving unaffected outputs and versioning the generator. If
   actual behavior cannot be established, retain explicit unknown/invalid reasons
   and quantify usable-target yield; do not claim clean fill or drop layers silently.
5. Once updated validation stores exist and are stable, audit their schemas, provenance,
   alignment and fluid cases separately. For any still-incomplete stores, propose
   explicit exclusion or reconstruction-only use, including split/yield consequences;
   do not fabricate missing masks or assume replacement has already repaired them.
6. Reconcile results with the governing plan and present a single reviewed contract
   covering layer identity, directional correspondence, fluid/unknown handling,
   thresholds, and candidate/yield rules. Obtain owner approval before implementing
   unresolved semantic targets. Complete the planned pair losses/heads, streaming
   sampler, tests, recovery, logging and local device/resource smoke checks; the
   present runner deliberately blocks train/resume/smoke/evaluate. Then present
   actual launch settings and obtain the separate training-launch approval.
```

Independent code/fixture infrastructure work may proceed only within the existing
handoff's authorization. Neither a source audit, the exploratory mu study, nor a
generic approval flag substitutes for the complete target contract and launch gates.

## 1. Completed scope

Implemented read-only source discovery, metadata inventory, strict parameter-file/config
mapping, bounded crop/column reads, canonical physical-overlap checks, numeric evidence,
and self-contained HTML trace plots. The audit runner captures combined stdout/stderr
in `logs/<run-id>/training.log`, refuses existing outputs, and propagates failures.
Train/resume/smoke/evaluate modes deliberately fail closed. The runner now also supports
owned TensorBoard start/status/stop; see the [Bash commands](2026-11-11__streamed_overlap_copilot_implementation.md).

No model was loaded or trained. No final-test crops or predictions were read. No training
patch corpus was created. Original trainer/model/tokenizer behavior is unchanged.
Automatic training/TensorBoard integration, completed-optimizer-step CSV journaling,
full-state resume, augmentations, and matched training arms remain later milestones;
their absence here is not a waiver
of the governing plan. There are no optimizer steps to journal or visualize in this audit.

## 2. Reproduce and inspect

From the repository root, choose a **new** run ID:

```sh
bash scripts/streamed_overlap_experiment.sh \
  --mode audit \
  --config scripts/streamed_overlap_audit.example.json \
  --run-id streamed-overlap-target-audit-review-01
```

Add `--print-commands` for a no-side-effect command preview.
The default evidence parent is
`/Volumes/CrucialX9/3dvae-poc_checkpoints/streamed_overlap`.
The config explicitly labels `+1` as an alignment **hypothesis**, not verified truth.
The interpreter defaults to `.venv/bin/python`; `PYTHON` can select another executable.

Historical local run (superseded by section 7):
`streamed-overlap-target-audit-20261111-04` (report schema 2).
It includes the owner's clarified hierarchy and a runner attempt manifest. Earlier
run `-03` is preserved; its embedded fluid-policy text predates that hierarchy and
is superseded by this document and the updated audit writer.
Its directory contains:

- `source_manifest.json`: all training/validation metadata, explicit missing-array issues,
  parameter/config hashes, generator SHAs/seeds, shapes/chunks/dtypes.
- `audit_report.json`: exact bounded-example origins, full-crop statistics, nine lateral
  trace profiles per view, read budgets, timings, software/script identity, limitations.
- `audit_examples.html`: offline plots of age, lithology, fluid masks and seismic.
- `target_contract.json`: machine-readable **unapproved** contract and unresolved rules.
- `resolved_audit_config.json`: the exact inspection configuration.

The matching combined log is
`logs/streamed-overlap-target-audit-20261111-04/training.log`.
Audits `-01` and `-02` are retained. Run `-03` adds channel-corrected lithology,
raw-versus-corrected counts, channel/hydrocarbon-closure disagreements and channel
trace overlays. Earlier "sand without fluid" counts mean **sand outside closure
masks**, not missing lithology labels or established unknown physical fluid.

Manifest SHA256 for `-04` (same source inventory as `-03`):
`27450fe0aa0582b764d280b0d62d77710c1376dbbf0af884dae3a4a1e6534265`.
Runtime reports/plots stay local and ignored by Git; do not upload private trace evidence
as cloud fixtures. The actual generator checkout contains uncommitted changes, so its
current code cannot be identified by the historical SHA alone.

## 3. Historical audit observations and consequences

The following table and examples describe the earlier `-04` audit, **not** the
48-hour regenerated cohort above. Keep them for historical comparison only.
| Observation | Assessment |
|---|---|
| 180/180 training sources have the inspected semantic arrays | Suitable for further source-contract validation, not proof that targets are correct |
| 20/25 validation sources have those arrays | The five early validation sources must not receive invented zero labels; reconstruction-only use needs explicit policy |
| All 205 seismic arrays are `(300,300,1499)`, chunks `(50,50,1499)` | Small crops decode full-depth chunks; crop byte counts are not physical I/O |
| All 205 parameter/config pairs parsed; `digi=4`, filter order 4 | Per-source band limits exist; verified filter/key/units semantics are still required |
| Observed low/high band-limit extremes: 3.03 Hz / 44.92 Hz | Do not replace these with one fallback global band |
| Three generator SHAs, with 127/73/5 sources | Historical generation provenance is heterogeneous; current generator code is contextual evidence only |
| Age values are almost entirely noninteger in rock examples | Rounding or flooring is a target proposal, not an already established layer-ID contract |
| Substantial diagnostic sand has no gas/oil/brine closure mask | Expected for non-trapping sand; closure absence is not missing lithology and must not remove layers |
| Channel sand overrides the background lithology where `faulted_channel_labels > 0.5` | Apply this override before sand/shale/water diagnostics; retain raw disagreement counts |
| The historical committed seismic-property generator applies a late brine channel override | The current dirty worktree removes it; determine which code generated each store rather than overriding the owner's target semantics |
| No conflicts/nonbinary fluid values in the 24 inspected views | Only a bounded observation; add explicit invalid/unknown handling regardless |
| Water-only, shale-only, channel-positive, gas, oil and brine examples occur | These require different validity/cohort handling; no-reservoir must not become an automatic perfect match |
| No fault-positive voxels in these examples | Fault behavior remains unvalidated; absence in this sample is not absence in the source |

Bounded example anchors (origins are seismic XYZ; labels use the unverified `+1`):

| Source suffix / view | Origin | Evidence |
|---|---|---|
| train 1041 / A | `(61,59,431)` | 31.9% water, sand/rock fraction 0.358, 15,989 diagnostic sand voxels without fluid masks |
| train 1130 / A | `(135,136,832)` | 642 gas-mask voxels, 16,872 diagnostic sand voxels without fluid masks |
| train 1220 / A | `(60,62,1371)` | Rock without a fluid-mask hit; inspect alongside its water control |
| validation 1221 / A | `(61,59,381)` | 2,624 oil-closure-mask voxels overlap channels; not proof of final oil-filled channel properties |
| validation 1230 / A and disjoint B | A `(136,136,369)`, B `(136,136,433)` | Gas-closure mask in A (1,031 voxels), brine-closure mask in B (987); sand fractions 0.333 and 0.112 |
| validation 1040 / disjoint B | `(61,60,1224)` | All rock, no diagnostic sand: shale-only/no-reservoir case |

These are full-crop voxel statistics, not central-trace estimates. The HTML plots show
nine XY locations; they do not prove a 3D semantic correspondence algorithm. Adjacent
views expose truncated top/bottom support, but exact counts of missing/extra **layers**
cannot be certified before defining layer extraction. Different sand fractions are
not measurements of true NTG or proof of NTG-invariant matching. Cross-volume analog
identity, faults, erosion and thickness-invariance still need reviewed cases.

### Verified interpretation and revised voxel labels

The owner's clarified lithology contract is:

```text
channel = faults/faulted_channel_labels > 0.5
sand = (faulted_lithology >= 0.5) OR channel
rock = (faulted_lithology >= 0) OR channel
shale = rock AND NOT sand
water = NOT rock
```

The background lithology threshold remains the existing diagnostic threshold, not a
new NTG or partial-voxel weighting rule. Channels take precedence even over a raw
water label; record such inconsistencies explicitly rather than hiding them.
Closure masks do not create sand, remove shale, define the layer denominator, or
determine whether a layer exists. Use corrected lithology for facies support and
retain age/geometry separately for correspondence.

The authoritative target hierarchy is:

1. `faulted_lithology` supplies background sand versus shale.
2. `faults/faulted_channel_labels > 0.5` overrides background lithology; channel
   identity is retained as a separate categorical/geometric attribute, not collapsed
   into generic stratigraphic sand.
3. `oil_closures` and `gas_closures` modify fluid fill **within corrected sand**,
   including channel sand. Oil-filled and gas-filled channels remain channels.
   Closure geometry alone does not establish fluid fill in all trapping geometries.

Read-only inspection of the neighboring generator supports this separation:

- `Closures_3d.py:300-338` constructs a reservoir mask and finds structurally trapped
  regions. `Closures_3d.py:416-463` assigns gas/oil/brine to surviving closure components.
  Those arrays are trap/fill indicators, not exhaustive sand/shale masks.
- The historical committed `Seismic.py` applies a later brine channel/levee mix after
  oil/gas, while the current dirty working tree removes that second mix. A positive
  oil/gas closure mask alone cannot establish which synthesis path generated a cube;
  see the newer channel/closure follow-up above.

The previous proposed channel-brine precedence is withdrawn. The historical code's
brine override is a **synthesis/target consistency issue**, not permission to change
the owner's hierarchy. Before training, verify the historical generator behavior and
whether seismic properties agree with the intended oil/gas channel labels. Do not
silently rewrite channel fluid to brine or train on a known contradictory target.
Overlapping incompatible closure masks and fluid outside explicitly labeled portions
remain separate unresolved policies. A positive brine-closure mask documents a brine
trap, not the whole background-brine population; water/shale are not labeled brine.

The rerun found **zero channel-added sand voxels in these 24 views**: their channel
voxels already exceeded the raw lithology threshold. This does not establish that the
override is unnecessary elsewhere. It also found **10,917 channel/hydrocarbon-closure
overlaps across the views** (overlapping views count repeated physical voxels).
This makes synthesis/target consistency important in the actual examples, not just
fabricated fixtures. The hierarchy assigns these overlaps hydrocarbon fill within
channels; the audit does not yet certify matching final seismic properties.

### Resource measurements, not training-performance predictions

The fourth audit performed 306 bounded reads: 16,330,176 requested voxels,
41,227,776 requested bytes (39.3 MiB), and a conservative cumulative decoded-chunk
estimate of 4,344,180,000 bytes (about 105.4 times requested bytes). The largest
estimated single decode was 60,400,000 bytes (57.6 MiB). Peak process RSS was 254.8 MiB.
Inventory took 7.81 seconds; inventory plus example inspection took 12.28 seconds,
before report serialization. Each of six sources has A, Z+16, Z+64 and a water-containing
control, with three thin full-depth probes.

These are one-process, potentially warm-cache audit timings. They exclude training,
augmentation, LPIPS, backward passes, search evaluation and journal fsync. They cannot
justify a batch budget, MPS throughput forecast, or changed crop/batch sizes.

## 4. Observable-event target and remaining review

The November 21 owner decisions supersede this document's earlier distinct-layer
proposal. Use the complete definition in the
[execution plan](2026-11-08__streamed_overlap_and_semantic_retrieval_experiment.md#event-coverage-and-fluid-adjustment).
In brief, count A's seismic peaks and troughs in the shared XYZ region divided by
those across A's full depth over the shared XY footprint, and multiply by the approved
window-level oil/gas/other fraction correction. Both targets use canonical supports
before augmentation; same-source disjoint and different-source pairs have zero
event coverage. An overlapping pair with no denominator events is unavailable.

The owner chose a simple correction, not per-layer fluid correspondence. It can miss
fluid changes that preserve aggregate percentages, and the all-voxel denominator
means the remainder is not an exhaustive physical brine label. On November 22 the
owner approved closure masks as the operational proxy despite unknown per-source
generation-time dirty-worktree provenance. Do not describe the target as physical
fluid truth or infer brine from closure absence. The owner also approved smoothed strict
adjacent-sample extrema, native same-index support excluding padding, source-specific
filter metadata, and the evaluation success criteria below. Remaining implementation
work is the streamed target, explicit invalid reasons, target-yield preflight and
frozen evaluation manifest/cohorts.

## 5. Noise and evaluation proposal

Keep the approved added-noise SNR range 12-60 dB and independent A/B nuisance transforms.
For each source, use the actual `bandpass_bandlimits` in its text metadata, plus the
matched JSON `bandwidth_low`/`bandwidth_high` ranges and `bandwidth_ord`. The 205
inspected sources use 4 ms and order 4; actual cutoff pairs are inside the configured
ranges (low 3.03-6.00 Hz; high 22.69-44.92 Hz). Reproduce the local generator's
depth-only Butterworth/Gustafsson filter and pole-decay context without importing the
neighboring checkout at training time. The `.txt` `sn_db` describes noise already
present in the stored source; do not use it as an augmentation setting: 44/205 values
fall outside JSON's source SNR range, while
filter settings remain valid. Non-finite or zero pre-noise RMS has undefined SNR and
must get an explicit invalid reason.

Owner-approved evaluation/adoption rules:

- Make **directional pair-head NDCG@10** the primary retrieval selection measure;
  score mu-cosine separately. A better head alone is not evidence that the cosine
  embedding improved.
- Use continuous approved directional coverage as graded relevance, linear DCG gain,
  and only queries with a known positive and an evaluable top-10 candidate set.
  Invalid/unknown pair relevance must never be treated as zero. Freeze candidate
  eligibility independently of the model so missing labels cannot improve its ranking.
- Report same-source and different-source cohorts separately; different-source
  examples have zero event-coverage targets, not cross-volume analog labels.
- Require at least **200 eligible positive queries overall**, and at least **20 queries
  from five source volumes** in each predeclared positive-query cohort. If the floor
  cannot be met, stop and report it; do not relax it after seeing results.
- Require at least **+0.01 absolute** primary NDCG@10 over a repeated uniform-random
  ranking baseline, with a positive lower 95% paired source-cluster bootstrap bound;
  no mandatory cohort may regress by more than **0.02** absolute NDCG@10. Repeat the
  direction of gain with an independent seed. Compare mu-cosine separately against
  the matched S1 reconstruction-only mu baseline; never score an untrained head.
- Retain the already approved **+2% maximum** matched synthetic tokenizer recipe and
  real-tokenizer MAE regressions. Legacy n@5 is continuity evidence, not this new target.

The numerical criteria above are owner-approved success rules, not quantities inferred
from the audit. Candidate/query manifests, cohort membership, bootstrap repetitions,
validity yield, no-reservoir handling and spatial tolerances still need to be fixed in
the implementation before evaluation. Epochs and batches per epoch remain unspecified;
these rules are not training authorization.

## 6. Consolidated gate-1 review and next work

Owner decisions now cover the detector, edge treatment, native index support,
closure-proxy policy, source filter metadata and evaluation criteria. Implementation
may proceed on that basis. The next readiness gate is the versioned executable target,
focused semantic tests, source/target validity and yield, and a frozen evaluation
manifest/cohort. Do not accept a generic `approved: true` flag as a substitute for
these checks. Current tests validate the portable audit and generator channel path,
not the complete pair target, MPS training, TensorBoard lifecycle or scientific success.

**Implementation is authorized; production training, baseline extension and replication
are not.** Commit and push scoped code/tests/docs on the task branch; keep local audit
outputs, datasets and unrelated work out of commits.

## 7. Refreshed source inventory (2026-11-22)

Read-only run `streamed-overlap-target-audit-20261122-03` used the existing local
train/validation source root and configuration. It inspected metadata for 205 sources:
180 training and 25 validation. All 205 have complete required semantic arrays and no
inventory issues. No final-test crops were read. The run remains
`LOCAL_TARGET_VALIDATION_PENDING` because the semantic target and yield preflight are
not yet implemented; production training is explicitly disallowed. The local manifest
SHA256 is `094c5c801452f3380d0ef29664f57195dfdbfab59d565fb35d7ad6b6a22349bd`.

The expanded manifest inventories 20 recursive array keys per source. Across all
sources, the four seismic variants have shape `(300, 300, 1499)`, chunks
`(50, 50, 1499)`, and `float32` dtype. Semantic/geometry arrays have shape
`(300, 300, 1510)` and chunks `(50, 50, 1510)`. `geologic_age_faulted` and
`faulted_lithology` are `float32`; oil, gas, and brine closures and channel labels are
`uint8`. `faulted_net_to_gross` is present in all 205 stores, as are
`faults/faulted_channel_labels`; `faults/faulted_channel_ng` is absent from all 205.
`closure_segments_id` is `uint16` in 204 stores and `uint8` in one. See the local
`source_manifest.json` for the full per-array inventory.

The neighboring generator's current `Faults._apply_channel_properties` writes
channel N/G into `faulted_net_to_gross` before closure detection and changes qualifying
channel voxels to reservoir lithology. `Closures3D` then finds traps from that
post-insertion lithology. Its focused synthetic end-to-end module passed **13 tests**,
including channel N/G insertion, segmentation-output retention, and oil/gas/brine
closure-property cases under inverse-velocity and Backus mixing. This verifies the
current code path on synthetic fixtures. Because the saved stores omit the separate
channel N/G array and do not record each generation-time dirty worktree/command, it
does not independently verify every saved channel N/G value or prove the exact code
used to generate any individual volume. Three earlier full-volume label scans did
find channel/oil and channel/gas closure overlaps; those are positive examples, not
an all-source audit.

The audit snapshot records generator SHA
`bd51313ebfb620800bb2973c2cea28f4b9325a15` for all stores; parameter/config file
modification times range from November 16 through November 20 UTC. The recorded commit
is available and an ancestor of the neighboring checkout's current
`6ae391fe9f18a3af0293e60b781f32aeb81a430b` HEAD. The current checkout has an
uncommitted 9,513-byte diff in `Faults.py` and `Seismic.py`; those files' modification
times precede the observed source metadata times. This makes the dirty code a plausible
generation-time context, not proof: per-run generation commands and dirty-worktree
fingerprints were not stored. The owner confirms the current `fake_data` sources were
recreated, but the available SHA/timestamp fields cannot establish which generation
produced an individual source. A local, observation-only fingerprint is preserved
beside the audit output; it is not source-level provenance and must not be used to
claim that a specific store came from that worktree. The current inventory's uniform
Z dimensions are observations, not a requirement for every source. The owner directs
that differing deepest extents be ignored absent strong evidence of an alignment
problem; use same-index supported windows, without global padding, clipping, or
shifting, and mark windows unavailable when their full support is absent.

The per-source `.txt` `bandpass_bandlimits` values all fall within their matched JSON
`bandwidth_low`/`bandwidth_high` ranges. Across the 205 sources, low cutoffs are
3.03-6.00 Hz, high cutoffs 22.69-44.92 Hz, sample spacing is 4 ms and filter order is
4. The `.txt` `sn_db` describes noise already present in the stored source, not a
filter value or extra-noise augmentation setting; 44 values fall outside their JSON `signal_to_noise_ratio_db`
range and are retained as warnings: 37 in training and seven in validation. Actual
filter metadata remains parseable for all 205 sources.

The refreshed inventory resolves array availability and uniformity. The owner-approved
detector, index support, fluid proxy, filter metadata and evaluation criteria are now
recorded below and in the execution plan. The exact generation-time dirty worktree for
each saved source remains unknown; this is a documented proxy limitation, not a veto.
The streamed target, source/target validity yield and frozen evaluation manifest still
need implementation and checks. No target preflight or model training has run.

## 8. Bounded detector and filter-context diagnostics

Run `streamed-overlap-target-audit-20261121-03` contains bounded detector and filter
diagnostics.
It selected 12 training and 6 validation sources spread across each split, read one
random full-depth `(50,50,Z)` seismic chunk per source, and sampled 25 traces per source:
450 traces total and 269,820,000 requested bytes. No final-test source, semantic target
generation, or model training was involved. The SciPy sensitivity results are local in
`event_detector_diagnostics.json`; the later strict-extrema counts are in
`local_extrema_diagnostics.json` beside it.

The earlier sensitivity probe used SciPy local maxima on each full original trace and on its negation
(troughs), varying prominence as a fraction of that trace's standard deviation and
minimum peak spacing. Counts are peaks plus troughs per 1,499-sample trace:

| Prominence | Train p10 / median / p90 | Validation p10 / median / p90 | Median events per 64 samples (train / validation) |
|---:|---:|---:|---:|
| 0.25 SD | 112 / 150 / 164 | 110 / 160 / 211 | 6.4 / 6.8 |
| 0.50 SD | 94 / 117 / 134 | 82 / 123 / 166 | 5.0 / 5.2 |
| 0.75 SD | 77 / 98 / 113 | 67 / 99 / 139 | 4.2 / 4.2 |
| 1.00 SD | 62 / 83 / 97 | 51 / 84 / 120 | 3.5 / 3.6 |

This table is historical sensitivity evidence only; the owner did not select a
prominence detector. No sampled full trace had a zero denominator at these candidate settings. Changing
minimum spacing from 2 to 4 samples barely changed counts; spacing 6 removed only a
small number. At 0.50 SD and spacing 3, the median event density in the first/last 64
samples was 0.25 of interior density in training and 0.44 in validation; the probe did
not show an event-density spike at source ends. This is a small, deliberately bounded
sensitivity sample, not ground truth or an estimate of final pair-target yield.

The owner's selected detector first applies
`scipy.ndimage.uniform_filter1d(size=5, axis=2, mode="nearest")` to each full seismic
trace and blends the filtered and original values as `0.2 * original + 0.8 * filtered`.
On that blended trace, detect strict adjacent-sample extrema along Z, without
prominence or spacing thresholds. For each XY chunk, compare the center
`volume[:,:,1:-1]` with its immediate shallower and deeper neighbors; use `np.where`
for `center > both` peaks and `center < both` troughs, then add one to returned Z
indices. Combine peak and trough `(i,j,k)` coordinates. This excludes the first and
last source samples naturally. Stream full-depth source traces in XY chunks, count
events in A's shared XYZ support, and divide by events over A's full depth on those
same shared-XY traces. The overlap boundary does not create extrema. Explicitly mark
non-finite input or a zero full-depth denominator unavailable.

The approved smoothed detector was checked on the same one full-depth `(50,50,Z)`
chunk from each of 12 training and 6 validation sources: 45,000 traces and 269.82 MB
requested. No sampled trace had zero combined extrema. Peak-plus-trough counts per
trace had training p10 / median / p90 of 197 / 215 / 243 and validation p10 / median /
p90 of 183.9 / 217 / 239. Median densities were about 9.18 and 9.26 events per 64
samples. These replace the previous raw-detector counts, which are not applicable to
the approved smoothed detector. This bounded sample does not estimate final
pair-target yield. The updated private diagnostic is
`local_extrema_smoothed_diagnostics.json` beside the 2026-11-22 audit output.

### Generator-derived index relationship

The 205 source configs record `cube_shape=(300,300,1500)`, `pad_samples=10`,
`infill_factor=10`, 4 ms sampling and order-4 filters. The current generator constructs
faulted age/lithology/N/G on the `(300,300,1510)` padded grid. Its property builder uses
those voxel indices, then trims elastic-property volumes to the first 1,500 samples.
Zoeppritz reflectivity has 1,499 Z samples because each reflectivity is computed between
adjacent property samples `z` and `z+1`; filtering and cumulative summation retain this
length. The plausible coordinate convention is therefore native XYZ, with a seismic
sample index anchoring the corresponding same-index property/label cell interval;
never use the padded label tail `1500:1510`. Peaks and troughs in the filtered,
cumulative trace are not one-to-one geological interfaces.

The owner approved native same-index support for each seismic/property/label crop.
Exclude padded label indices 1500:1510 and do not apply a global shift. This is based
on the current generator's construction path; exact per-source generation history is
unknown. The earlier correlation heuristic sampled eight sources per split: envelope
peaks averaged at offset +2 in both splits, but signed-derivative curves averaged at
+1 in training and -4 in validation, with many per-source estimates at the ±15-sample
search boundary. Those estimates are not a reason to shift the approved native mapping.

### Fluid validity, noise, and evaluation decisions

- **Fluid proxy:** all 205 sources have `faulted_net_to_gross`, channel labels, and the
  three closure arrays, but no `faults/faulted_channel_ng`. The owner approved the
  oil/gas/other window-fraction correction as a coarse operational proxy despite the
  unknown per-source generation-time worktree. The remainder is mathematical, not an
  exhaustive brine label; closure masks are not asserted to be measured final fluid.
  Three selected full-volume scans found binary, mutually exclusive closure masks and
  channel/hydrocarbon overlaps, but did not check all sources or final elastic
  properties. The current generator inserts channel N/G into `faulted_net_to_gross`
  and derives closures from post-insertion reservoir lithology; 13 synthetic
  end-to-end tests passed. Because the separate channel N/G array is absent from the
  saved sources and generation provenance is incomplete, code/test evidence does not
  independently validate every source's historical channel N/G.
- **Added noise:** source JSON bandwidth fields are ranges, while the matched
  `model_parameters` text file gives the actual sampled `bandpass_bandlimits`; JSON
  gives `bandwidth_ord`. All 205 actual bands are within their configured ranges
  (low 3.03-6.00 Hz, high 22.69-44.92 Hz), at 4 ms and order 4. The generator uses
  depth-only zero-phase Butterworth/Gustafsson filtering. Its pole-decay calculation
  gives per-source impulse-response lengths 446-853 samples (median 589; about
  1.8-3.4 seconds). Keep the approved probability 0.5 and 12-60 dB augmentation SNR;
  generate independent noise after spatial transforms, use that source's exact band
  and order plus computed context, then scale to `10**(-SNR/20)` times pre-noise RMS.
  This gives RMS ratios of about 0.251 to 0.001. Do not re-filter the seismic signal.
  The text `sn_db` describes noise already present in the stored source, not
  augmentation SNR; 44/205 values fall outside JSON's source SNR range, so record a
  warning without substituting it or rejecting otherwise valid filter metadata.
  Non-finite/zero RMS has undefined SNR and requires an explicit invalid reason.
- **Evaluation:** directional pair-head NDCG@10 is primary; report mu-cosine
  separately and also report physical-overlap and event-fraction MAE/RMSE/bias/
  calibration. Use linear DCG gain, a frozen model-independent candidate manifest,
  known relevance labels, and queries with at least one valid positive. Cross-volume
  and same-source-disjoint pairs remain zero-target negatives, not positive-query
  cohorts. The owner-approved pilot floor is 200 eligible positive queries overall
  and at least 20 queries from five source volumes in each predeclared positive-query
  cohort. Require +0.01 absolute pair-head NDCG@10 over a repeated uniform-random
  ranking baseline, a positive lower 95% paired source-cluster bootstrap bound, no
  mandatory cohort loss greater than 0.02, and repeat the direction of gain with an
  independent seed. Compare mu-cosine against matched S1 reconstruction-only mu;
  never score an untrained head. Preserve the separate approved +2% reconstruction
  guardrail. These rules do not authorize a training launch.

The extrema diagnostic does not measure valid-target yield of actual sampled pairs,
age support, closure-proxy coverage or retrieval queries. Implement a read-only
target-yield preflight against the approved rules; require the stated query floor and
frozen cohort manifest before any training authorization.
