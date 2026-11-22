"""Recoverable post-epoch validation, search and checkpoint selection."""

import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
import zarr

from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_checkpoint import restore_epoch_state
from scripts.streamed_overlap_diagnostics import diagnostic_report
from scripts.streamed_overlap_evaluation import _checkpoint_state, _identity, evaluate_checkpoint
from scripts.streamed_overlap_integrity import _digest_array, load_manifest, open_validation_sources
from scripts.streamed_overlap_journal import _atomic_record, _sync_directory
from scripts.streamed_overlap_search import run_bounded_searches
from scripts.streamed_overlap_selection import _best, _records, select_checkpoints
from scripts.train import train_streamed_epoch


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _save_report(directory: Path, name: str, report: dict) -> dict:
    path = directory / name
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, allow_nan=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _sync_directory(directory)
    return {"path": str(path.relative_to(directory.parent)), "sha256": _digest(path)}


def _completed(directory: Path, identity: dict, config_hash: str, due: bool, search_manifest: dict) -> dict | None:
    events = directory / "completed"
    if not events.exists():
        return None
    if events.is_symlink() or not events.is_dir():
        raise ValueError("Evaluation completion records must be in a regular directory.")
    matched = []
    for path in events.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record.get("schema") != "streamed-epoch-evaluation-completion-v1"
                or record.get("sha256") != content_hash({
                    key: value for key, value in record.items() if key != "sha256"
                })):
            raise ValueError("Evaluation completion record is invalid.")
        if record.get("checkpoint") != identity or record.get("config_sha256") != config_hash:
            continue
        artifacts = record.get("artifacts")
        expected = {"selection"}
        if due:
            expected.update({"retrieval", "tokenizer_search"})
            if identity["arm"] == "S3":
                expected.add("diagnostics")
        if (record.get("formal_evaluation_due") is not due
                or not isinstance(artifacts, dict) or set(artifacts) != expected):
            raise ValueError("Evaluation completion record lacks required artifacts.")
        for item in artifacts.values():
            path = directory / item["path"]
            if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(directory.resolve())
                    or _digest(path) != item["sha256"]):
                raise ValueError("Completed epoch evaluation artifact is missing or changed.")
        if due:
            report_path = directory / artifacts["tokenizer_search"]["path"]
            report = json.loads(report_path.read_text(encoding="utf-8"))
            search_root = report_path.parent / "tokenizer_search"
            if (not isinstance(report.get("searches"), list)
                    or len(report["searches"]) != len(search_manifest["searches"])
                    or _digest(search_root / "tokenizer.pt") != report["checkpoint"]["tokenizer_checkpoint_sha256"]):
                raise ValueError("Completed tokenizer checkpoint is missing or changed.")
            for index, result in enumerate(report["searches"]):
                folder = search_root / f"search-{index:04d}"
                if (_digest(folder / "scores.jsonl") != result["window_scores_sha256"]
                        or _digest_array(np.asarray(zarr.open_group(
                            str(folder / "similarity.zarr"), mode="r")["data"])) != result["hann_similarity_sha256"]):
                    raise ValueError("Completed tokenizer search output is missing or changed.")
        matched.append(record)
    if len(matched) > 1:
        raise ValueError("Duplicate completion records for the same checkpoint.")
    return matched[0] if matched else None


def _validate_formal_config(search_manifest: dict, diagnostic_manifest: dict, config: dict) -> None:
    required = {
        "final_epoch", "device", "inference_seed", "encode_batch_size", "max_wall_seconds",
        "random_seed", "random_repeats", "bootstrap_seed", "bootstrap_repeats",
        "diagnostic_max_wall_seconds", "search_manifest_sha256", "diagnostic_manifest_sha256",
    }
    if (not isinstance(config, dict) or set(config) != required
            or isinstance(config["final_epoch"], bool) or not isinstance(config["final_epoch"], int)
            or config["final_epoch"] < 1 or config["device"] != "mps"
            or search_manifest.get("search_manifest_sha256") != config["search_manifest_sha256"]
            or search_manifest.get("search_manifest_sha256") != content_hash({
                key: value for key, value in search_manifest.items() if key != "search_manifest_sha256"
            })
            or diagnostic_manifest.get("diagnostic_manifest_sha256") != config["diagnostic_manifest_sha256"]
            or diagnostic_manifest.get("diagnostic_manifest_sha256") != content_hash({
                key: value for key, value in diagnostic_manifest.items() if key != "diagnostic_manifest_sha256"
            })):
        raise ValueError("An explicit frozen MPS evaluation configuration is required.")


def _formal_reports(
    checkpoint_path, journal, manifest_path, root, manifest, identity,
    search_manifest, diagnostic_manifest, config, attempt,
) -> tuple[dict, dict]:
    arm = identity["arm"]
    report = evaluate_checkpoint(
        checkpoint_path, journal, manifest_path, root, device=config["device"],
        inference_seed=config["inference_seed"], encode_batch_size=config["encode_batch_size"],
        max_wall_seconds=config["max_wall_seconds"], random_seed=config["random_seed"],
        random_repeats=config["random_repeats"], bootstrap_seed=config["bootstrap_seed"],
        bootstrap_repeats=config["bootstrap_repeats"], score_pair_head=arm == "S3",
        score_overlap=arm in {"S2", "S3"},
    )
    if report["checkpoint"] != identity:
        raise ValueError("Formal evaluation checkpoint differs from the selected checkpoint.")
    artifacts = {"retrieval": _save_report(attempt, "retrieval.json", report)}
    groups = open_validation_sources(root, manifest)
    search = run_bounded_searches(
        checkpoint_path, journal, manifest_path, root, manifest, groups,
        search_manifest, attempt / "tokenizer_search",
    )
    if search["checkpoint"]["checkpoint"] != identity:
        raise ValueError("Bounded search checkpoint differs from the selected checkpoint.")
    artifacts["tokenizer_search"] = _save_report(attempt, "tokenizer_search.json", search)
    if arm == "S3":
        rows = report["scorers"]["pair_head"]["event_coverage_diagnostics"]["pairs"]
        predictions = {}
        for row in rows:
            predictions.setdefault(row["query_id"], {})[row["candidate_id"]] = row["prediction"]
        diagnostics = diagnostic_report(
            manifest, groups, diagnostic_manifest, predictions,
            max_wall_seconds=config["diagnostic_max_wall_seconds"],
        )
        artifacts["diagnostics"] = _save_report(attempt, "diagnostics.json", diagnostics)
    return report, artifacts


def complete_epoch(
    checkpoint_path: Path, journal, manifest_path: Path, source_root: Path,
    search_manifest: dict, diagnostic_manifest: dict, config: dict, *, evaluation_dir: Path,
) -> dict:
    """Complete or recover a checkpoint's mandatory evaluation before the next epoch."""
    _validate_formal_config(search_manifest, diagnostic_manifest, config)
    root = source_root.resolve(strict=True)
    forbidden = Path("/Volumes/CrucialX9/fake_data_validation_v2")
    if ("fake_data_validation_v2" in root.parts or forbidden.is_relative_to(root)):
        raise ValueError("The regenerating validation-v2 dataset is not permitted.")
    if checkpoint_path.name != "last.pt" or not evaluation_dir.is_dir() or evaluation_dir.is_symlink():
        raise ValueError("Epoch completion requires last.pt and a regular evaluation directory.")
    manifest = load_manifest(manifest_path)
    state, checkpoint_hash = _checkpoint_state(checkpoint_path)
    identity = _identity(state, checkpoint_hash, journal, manifest["manifest_sha256"])
    config_hash = content_hash(config)
    if (state["hashes"].get("formal_evaluation") != config_hash
            or identity["epoch"] > config["final_epoch"]):
        raise ValueError("Checkpoint and frozen evaluation configuration disagree.")
    due = identity["epoch"] % 5 == 0 or identity["epoch"] == config["final_epoch"]
    existing = _completed(evaluation_dir, identity, config_hash, due, search_manifest)
    if existing is not None:
        return existing
    events = evaluation_dir / "completed"
    events.mkdir(exist_ok=True)
    _sync_directory(evaluation_dir)
    attempt = evaluation_dir / f"epoch-{identity['epoch']:06d}-{uuid.uuid4().hex}"
    attempt.mkdir()
    _sync_directory(evaluation_dir)
    started = time.monotonic()
    artifacts = {}
    report = None
    if due:
        report, artifacts = _formal_reports(
            checkpoint_path, journal, manifest_path, root, manifest, identity,
            search_manifest, diagnostic_manifest, config, attempt,
        )
    selection = select_checkpoints(
        checkpoint_path, journal, manifest_path,
        retrieval_report=report if identity["arm"] == "S3" else None,
    )
    artifacts["selection"] = _save_report(attempt, "selection.json", selection)
    if _digest(checkpoint_path) != checkpoint_hash:
        raise ValueError("Checkpoint changed during epoch evaluation.")
    completion = {
        "schema": "streamed-epoch-evaluation-completion-v1", "checkpoint": identity,
        "config_sha256": config_hash, "formal_evaluation_due": due, "artifacts": artifacts,
        "elapsed_seconds": time.monotonic() - started,
    }
    completion["sha256"] = content_hash(completion)
    _atomic_record(events, completion)
    return completion


def complete_selected_checkpoint(
    checkpoint_path: Path, journal, manifest_path: Path, source_root: Path,
    search_manifest: dict, diagnostic_manifest: dict, config: dict, *,
    selection_kind: str, evaluation_dir: Path,
) -> dict:
    """Run frozen formal evaluations on a selected checkpoint after the final epoch.

    This does not measure reconstruction guardrails or authorize adoption.
    """
    _validate_formal_config(search_manifest, diagnostic_manifest, config)
    if selection_kind not in {"reconstruction", "retrieval"}:
        raise ValueError("Selection kind must be reconstruction or retrieval.")
    root = source_root.resolve(strict=True)
    forbidden = Path("/Volumes/CrucialX9/fake_data_validation_v2")
    if "fake_data_validation_v2" in root.parts or forbidden.is_relative_to(root):
        raise ValueError("The regenerating validation-v2 dataset is not permitted.")
    if (checkpoint_path.name != "last.pt" or checkpoint_path.is_symlink()
            or not checkpoint_path.is_file() or evaluation_dir.is_symlink()
            or not evaluation_dir.is_dir()):
        raise ValueError("Post-selection evaluation requires last.pt and a regular evaluation directory.")
    manifest = load_manifest(manifest_path)
    state, last_hash = _checkpoint_state(checkpoint_path)
    last_identity = _identity(state, last_hash, journal, manifest["manifest_sha256"])
    if (state["hashes"].get("formal_evaluation") != content_hash(config)
            or last_identity["epoch"] != config["final_epoch"]
            or journal.position is None
            or journal.position.logical_step != last_identity["step"]
            or journal.position.cumulative_exposure != 12 * last_identity["step"]):
        raise ValueError("Post-selection evaluation requires the reconciled final epoch and frozen configuration.")
    if _completed(evaluation_dir, last_identity, content_hash(config), True, search_manifest) is None:
        raise ValueError("Final epoch evaluation must be complete before post-selection evaluation.")
    selection = checkpoint_path.parent / "selection"
    events, snapshots = selection / "events", selection / "snapshots"
    if any(path.is_symlink() or not path.is_dir() for path in (selection, events, snapshots)):
        raise ValueError("Selection events and snapshots must be regular run-owned directories.")
    active_identity = {**last_identity, "epoch_validation_sha256": state["hashes"]["epoch_validation"]}
    selected = _best(_records(events, journal, active_identity), selection_kind)
    if selected is None or (selection_kind == "retrieval" and last_identity["arm"] != "S3"):
        raise ValueError("The requested checkpoint selection is unavailable.")
    digest = selected["checkpoint"]["checkpoint_sha256"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        raise ValueError("Selected checkpoint hash is invalid.")
    snapshot = snapshots / f"{digest}.pt"
    alias = checkpoint_path.parent / f"best_{selection_kind}.pt"
    if any(path.is_symlink() or not path.is_file() or _digest(path) != digest for path in (snapshot, alias)):
        raise ValueError("The selected checkpoint snapshot or alias is missing or changed.")
    selected_state, selected_hash = _checkpoint_state(snapshot)
    identity = _identity(selected_state, selected_hash, journal, manifest["manifest_sha256"])
    if ({**identity, "epoch_validation_sha256": selected_state["hashes"]["epoch_validation"]}
            != selected["checkpoint"]
            or selected_state["hashes"].get("formal_evaluation") != content_hash(config)):
        raise ValueError("Selected checkpoint identity or configuration disagrees with selection history.")
    directory = evaluation_dir / f"selected-{selection_kind}"
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise ValueError("Post-selection evaluation directory must be a regular directory.")
    directory.mkdir(exist_ok=True)
    _sync_directory(evaluation_dir)
    config_hash = content_hash({"formal_evaluation": config, "selection_kind": selection_kind})
    existing = _completed(directory, identity, config_hash, True, search_manifest)
    if existing is not None:
        return existing
    completed = directory / "completed"
    completed.mkdir(exist_ok=True)
    _sync_directory(directory)
    attempt = directory / f"checkpoint-{identity['epoch']:06d}-{uuid.uuid4().hex}"
    attempt.mkdir()
    _sync_directory(directory)
    started = time.monotonic()
    _, artifacts = _formal_reports(
        snapshot, journal, manifest_path, root, manifest, identity,
        search_manifest, diagnostic_manifest, config, attempt,
    )
    artifacts["selection"] = _save_report(attempt, "selection.json", selected)
    if _digest(snapshot) != digest or _digest(alias) != digest or _digest(checkpoint_path) != last_hash:
        raise ValueError("Checkpoint changed during post-selection evaluation.")
    completion = {
        "schema": "streamed-epoch-evaluation-completion-v1", "checkpoint": identity,
        "config_sha256": config_hash, "formal_evaluation_due": True, "artifacts": artifacts,
        "elapsed_seconds": time.monotonic() - started,
        "limitation": "Post-selection validation only; reconstruction guardrails and adoption remain separate.",
    }
    completion["sha256"] = content_hash(completion)
    _atomic_record(completed, completion)
    return completion


def run_arm_epochs(
    model, optimizer, scheduler, sampler, journal, lpips_loss_fn, writer, *,
    run_id: str, seed: int, attempt_id: str, batches_per_epoch: int, sparse_probability: float,
    monitor_batches: dict, monitor_seed: int, validation_seed: int, validate_epoch,
    epoch_csv: Path, checkpoint_path: Path, hashes: dict, early_stopping,
    manifest_path: Path, source_root: Path, search_manifest: dict, diagnostic_manifest: dict,
    formal_config: dict, evaluation_dir: Path,
) -> dict:
    """Resume one arm from its full epoch state and finish evaluation before advancing."""
    if hashes.get("formal_evaluation") != content_hash(formal_config):
        raise ValueError("The arm must use its frozen formal evaluation configuration.")
    if (isinstance(formal_config.get("final_epoch"), bool)
            or not isinstance(formal_config.get("final_epoch"), int)
            or formal_config["final_epoch"] < 1):
        raise ValueError("An explicit positive final epoch is required.")
    restored = restore_epoch_state(
        checkpoint_path, model=model, optimizer=optimizer, scheduler=scheduler, sampler=sampler,
        early_stopping=early_stopping, journal=journal, expected_hashes=hashes,
        expected_batches_per_epoch=batches_per_epoch, attempt_id=attempt_id,
    )
    completed = restored["completed_epoch"]
    if completed > formal_config["final_epoch"]:
        raise ValueError("Checkpoint is beyond the approved final epoch.")
    selection_state = restored["selection_state"]

    def finish():
        return complete_epoch(
            checkpoint_path, journal, manifest_path, source_root, search_manifest,
            diagnostic_manifest, formal_config, evaluation_dir=evaluation_dir,
        )

    last_completion = finish() if completed else None
    for epoch in range(completed + 1, formal_config["final_epoch"] + 1):
        validation = train_streamed_epoch(
            model, optimizer, scheduler, sampler, journal, lpips_loss_fn, "mps", writer,
            run_id=run_id, seed=seed, epoch=epoch, batches_per_epoch=batches_per_epoch,
            sparse_probability=sparse_probability, monitor_batches=monitor_batches,
            monitor_seed=monitor_seed, validation_seed=validation_seed,
            validate_epoch=validate_epoch, epoch_csv=epoch_csv, checkpoint_path=checkpoint_path,
            hashes=hashes, early_stopping=early_stopping, selection_state=selection_state,
        )
        selection_state = {**selection_state, "_epoch_validation": validation}
        last_completion = finish()
    if last_completion is None:
        raise ValueError("The arm did not complete an epoch.")
    selected_evaluations = {}
    kinds = ("reconstruction", "retrieval") if model.arm == "S3" else ("reconstruction",)
    for kind in kinds:
        selected_evaluations[kind] = complete_selected_checkpoint(
            checkpoint_path, journal, manifest_path, source_root, search_manifest,
            diagnostic_manifest, formal_config, selection_kind=kind, evaluation_dir=evaluation_dir,
        )
    return {"last_epoch": last_completion, "selected_evaluations": selected_evaluations}
