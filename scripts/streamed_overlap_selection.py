"""Recovery-aware selection of distinct streamed reconstruction and retrieval checkpoints."""

import hashlib
import json
import math
import os
from pathlib import Path
import uuid

from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_evaluation import (
    _checkpoint_state, _identity, _require_trained_coverage,
)
from scripts.streamed_overlap_integrity import load_manifest
from scripts.streamed_overlap_journal import BatchJournal, _atomic_record, _sync_directory
from scripts.streamed_overlap_manifest import COHORTS, MIN_COHORT_QUERIES, MIN_COHORT_SOURCES, MIN_QUERIES
from scripts.streamed_overlap_metrics import _eligible_queries
from scripts.streamed_overlap_model import fresh_e1_model


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_verified(source: Path, destination: Path, expected: str) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        digest = hashlib.sha256()
        with source.open("rb") as reader, temporary.open("xb") as writer:
            while chunk := reader.read(1024 * 1024):
                writer.write(chunk)
                digest.update(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        if digest.hexdigest() != expected:
            raise ValueError("Checkpoint changed while making a verified copy.")
        os.replace(temporary, destination)
        _sync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _reconstruction(state: dict, manifest_hash: str) -> float:
    hashes = state.get("hashes", {})
    validation = state.get("selection_state", {}).get("_epoch_validation")
    if (not isinstance(validation, dict) or validation.get("split") != "validation"
            or validation.get("manifest_sha256") != manifest_hash
            or validation.get("epoch_validation_sha256") != hashes.get("epoch_validation")
            or not isinstance(hashes.get("epoch_validation"), str)):
        raise ValueError("Checkpoint lacks matching frozen epoch validation.")
    numbers = []
    for key in ("reconstruction_mae", "lpips_loss", "kl"):
        value = validation.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError("Checkpoint reconstruction metrics are invalid.")
        numbers.append(float(value))
    return numbers[0] + 0.1 * numbers[1] + 0.001 * numbers[2]


def _retrieval(report: dict, identity: dict, manifest: dict, queries: list[dict], state: dict,
               journal: BatchJournal) -> float:
    verification = report.get("input_verification")
    scorer = report.get("scorers", {}).get("pair_head")
    summary = scorer.get("summary") if isinstance(scorer, dict) else None
    rows = scorer.get("rows") if isinstance(scorer, dict) else None
    fingerprints = manifest["input_integrity"]["fingerprints"]
    if (identity["arm"] != "S3" or report.get("schema") != "streamed-retrieval-evaluation-v1"
            or report.get("checkpoint") != identity or not isinstance(verification, dict)
            or verification.get("status") != "VALIDATION_INPUTS_VERIFIED"
            or verification.get("manifest_sha256") != identity["validation_manifest_sha256"]
            or verification.get("verified_windows") != len(fingerprints["windows"])
            or verification.get("verified_full_depth_event_regions") != len(fingerprints["event_traces"])
            or not isinstance(summary, dict) or summary.get("scorer") != "pair_head"
            or summary.get("arm") != "S3" or summary.get("manifest_sha256") != identity["validation_manifest_sha256"]
            or summary.get("experiment_seed") != identity["experiment_seed"]
            or summary.get("eligible_queries") != len(queries)
            or len(queries) < MIN_QUERIES or not isinstance(rows, list) or len(rows) != len(queries)):
        raise ValueError("Retrieval selection requires a verified S3 frozen-manifest evaluation.")
    expected = {query["id"]: (query["window"]["source_id"], query["cohort"]) for query in queries}
    seen = set()
    values = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Retrieval query results are invalid.")
        query_id = row.get("query_id")
        value = row.get("ndcg")
        if (not isinstance(query_id, str) or query_id in seen
                or expected.get(query_id) != (row.get("source_id"), row.get("cohort"))
                or isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("Retrieval query results differ from the frozen cohort.")
        seen.add(query_id)
        values.append(float(value))
    if seen != set(expected):
        raise ValueError("Retrieval query coverage is incomplete.")
    for cohort in COHORTS:
        subset = [row for row in rows if row["cohort"] == cohort]
        if len(subset) < MIN_COHORT_QUERIES or len({row["source_id"] for row in subset}) < MIN_COHORT_SOURCES:
            raise ValueError("Retrieval cohort eligibility floor is not met.")
    value = sum(values) / len(values)
    if not isinstance(summary.get("ndcg_at_10"), (int, float)) or not math.isclose(
        value, summary["ndcg_at_10"], rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("Retrieval summary disagrees with query results.")
    model = fresh_e1_model(identity["arm"], model_seed=0, head_seed=0)
    model.load_state_dict(state["model"], strict=True)
    _require_trained_coverage(state, model, journal, identity)
    return value


def _records(directory: Path, journal: BatchJournal, identity: dict) -> list[dict]:
    records = []
    for path in directory.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(record, dict) or record.get("schema") != "streamed-selection-event-v1"
                or record.get("sha256") != content_hash({key: value for key, value in record.items() if key != "sha256"})):
            raise ValueError(f"Invalid selection event: {path}")
        records.append(record)
    records.sort(key=lambda item: item["sequence"])
    if [record["sequence"] for record in records] != list(range(1, len(records) + 1)):
        raise ValueError("Selection event sequence is incomplete or duplicated.")
    stale = {event for record in journal._records() for event in record.get("uncommitted_event_ids", [])}
    journal_rows = {row["event_id"]: row for row in journal.rows}
    live = []
    for record in records:
        item = record["checkpoint"]
        row = journal_rows.get(item["journal_event_id"])
        if row is None:
            raise ValueError("Selection references an unknown journal event.")
        if any(item.get(key) != identity[key] for key in (
            "arm", "run_id", "experiment_seed", "config_sha256", "source_sha256",
            "target_sha256", "validation_manifest_sha256", "epoch_validation_sha256",
        )) or (row["arm"], row["run_id"], row["seed"], row["attempt_id"], row["logical_step"]) != (
            item["arm"], item["run_id"], str(item["experiment_seed"]), item["attempt_id"], str(item["step"])
        ):
            raise ValueError("Selection event does not match the journal or active run.")
        for key in ("reconstruction", "retrieval"):
            value = record[key]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value < 0
                                      or (key == "retrieval" and value > 1)):
                raise ValueError("Selection event contains an invalid metric.")
        if item["journal_event_id"] not in stale:
            live.append(record)
    return live


def _best(records: list[dict], metric: str) -> dict | None:
    available = [record for record in records if record[metric] is not None]
    if not available:
        return None
    return min(available, key=lambda record: record[metric]) if metric == "reconstruction" else max(
        available, key=lambda record: record[metric]
    )


def _publish_alias(alias: Path, selected: dict | None, snapshots: Path, allowed: set[str]) -> None:
    if alias.is_symlink() or (alias.exists() and not alias.is_file()):
        raise ValueError(f"Selection alias is not a regular file: {alias}")
    if alias.exists() and _digest(alias) not in allowed:
        raise ValueError(f"Refusing to overwrite an unrelated selection alias: {alias}")
    if selected is None:
        if alias.exists():
            alias.unlink()
            _sync_directory(alias.parent)
        return
    expected = selected["checkpoint"]["checkpoint_sha256"]
    snapshot = snapshots / f"{expected}.pt"
    if not snapshot.is_file() or snapshot.is_symlink() or _digest(snapshot) != expected:
        raise ValueError("Selected immutable checkpoint snapshot is missing or changed.")
    if alias.exists() and _digest(alias) == expected:
        return
    _copy_verified(snapshot, alias, expected)


def select_checkpoints(
    checkpoint_path: Path, journal: BatchJournal, manifest_path: Path, *, retrieval_report: dict | None,
) -> dict:
    """Record a completed epoch and publish separate best aliases; this is not adoption."""
    if checkpoint_path.name != "last.pt" or checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValueError("Selection requires the run's regular last.pt checkpoint.")
    manifest = load_manifest(manifest_path)
    queries = _eligible_queries(manifest)
    state, checkpoint_hash = _checkpoint_state(checkpoint_path)
    identity = _identity(state, checkpoint_hash, journal, manifest["manifest_sha256"])
    if (journal.position is None or journal.position.logical_step != identity["step"]
            or journal.position.cumulative_exposure != 12 * identity["step"]):
        raise ValueError("Selection requires the currently reconciled checkpoint trajectory.")
    reconstruction = _reconstruction(state, manifest["manifest_sha256"])
    retrieval = None if retrieval_report is None else _retrieval(
        retrieval_report, identity, manifest, queries, state, journal,
    )
    identity = {**identity, "epoch_validation_sha256": state["hashes"]["epoch_validation"]}
    root = checkpoint_path.parent
    selection = root / "selection"
    if selection.is_symlink() or (selection.exists() and not selection.is_dir()):
        raise ValueError("Selection directory must be a regular run-owned directory.")
    if not selection.exists():
        if any((root / name).exists() or (root / name).is_symlink()
               for name in ("best_reconstruction.pt", "best_retrieval.pt")):
            raise ValueError("Refusing to appropriate existing checkpoint aliases.")
        selection.mkdir()
        (selection / "events").mkdir()
        (selection / "snapshots").mkdir()
        _sync_directory(selection)
        _sync_directory(root)
    events, snapshots = selection / "events", selection / "snapshots"
    if any(path.is_symlink() or not path.is_dir() for path in (events, snapshots)):
        raise ValueError("Selection event and snapshot directories must exist.")
    live = _records(events, journal, identity)
    if live and identity["step"] < max(record["checkpoint"]["step"] for record in live):
        raise ValueError("Selection cannot move backwards without journal reconciliation.")
    if any(record["checkpoint"]["journal_event_id"] == identity["journal_event_id"]
           and record["checkpoint"]["checkpoint_sha256"] != checkpoint_hash for record in live):
        raise ValueError("A journal event cannot identify different checkpoint contents.")
    same_checkpoint = [
        record for record in live if record["checkpoint"]["journal_event_id"] == identity["journal_event_id"]
    ]
    if any(record["reconstruction"] != reconstruction for record in same_checkpoint):
        raise ValueError("Checkpoint reconstruction selection metric changed.")
    previous_retrieval = [record["retrieval"] for record in same_checkpoint if record["retrieval"] is not None]
    if retrieval is not None and previous_retrieval and any(
        value != retrieval for value in previous_retrieval
    ):
        raise ValueError("Checkpoint retrieval selection metric changed.")
    previous_rec, previous_ret = _best(live, "reconstruction"), _best(live, "retrieval")
    improve_rec = previous_rec is None or reconstruction < previous_rec["reconstruction"]
    improve_ret = retrieval is not None and (previous_ret is None or retrieval > previous_ret["retrieval"])
    if improve_rec or improve_ret:
        snapshot = snapshots / f"{checkpoint_hash}.pt"
        if snapshot.is_symlink():
            raise ValueError("Checkpoint snapshot must not be a symlink.")
        if not snapshot.exists():
            _copy_verified(checkpoint_path, snapshot, checkpoint_hash)
        if _digest(snapshot) != checkpoint_hash:
            raise ValueError("Checkpoint snapshot differs from its evaluated hash.")
    if not same_checkpoint or (retrieval is not None and not previous_retrieval):
        count = len(list(events.glob("*.json")))
        record = {
            "schema": "streamed-selection-event-v1", "sequence": count + 1,
            "checkpoint": identity, "reconstruction": reconstruction, "retrieval": retrieval,
        }
        record["sha256"] = content_hash(record)
        _atomic_record(events, record)
    live = _records(events, journal, identity)
    known = {item["checkpoint"]["checkpoint_sha256"] for item in live}
    # Stale selections are still run-owned aliases and may need reconciliation.
    for path in events.glob("*.json"):
        known.add(json.loads(path.read_text(encoding="utf-8"))["checkpoint"]["checkpoint_sha256"])
    best_reconstruction, best_retrieval = _best(live, "reconstruction"), _best(live, "retrieval")
    _publish_alias(root / "best_reconstruction.pt", best_reconstruction, snapshots, known)
    _publish_alias(root / "best_retrieval.pt", best_retrieval, snapshots, known)
    return {
        "last": identity,
        "best_reconstruction": best_reconstruction,
        "best_retrieval": best_retrieval,
        "limitation": "Checkpoint ranking is not experiment success or adoption.",
    }
