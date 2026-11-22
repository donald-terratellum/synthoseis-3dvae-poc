"""Recoverable post-epoch validation, search and checkpoint selection."""

import hashlib
import json
import math
import os
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
import time
from typing import BinaryIO, Callable
import uuid

import numpy as np
import torch
import zarr
from torch.utils.tensorboard import SummaryWriter

from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_checkpoint import _validate_hashes, restore_epoch_state
from scripts.streamed_overlap_diagnostics import diagnostic_report, verify_diagnostics
from scripts.streamed_overlap_evaluation import (
    StreamedEpochValidator, _checkpoint_state, _identity, _validate_evaluation_parameters, evaluate_checkpoint,
)
from scripts.streamed_overlap_guardrails import (
    _manifest as _guardrail_manifest, _measurement, _sha256, compare_reconstruction_guardrails,
)
from scripts.streamed_overlap_integrity import (
    _digest_array, load_manifest, open_validation_sources, verify_validation_preflight,
)
from scripts.streamed_overlap_initialization import prepare_suite_initial_checkpoints
from scripts.streamed_overlap_journal import BatchJournal, STEP_FIELDS, _atomic_record, _publish_exclusive, _sync_directory
from scripts.streamed_overlap_matching import (
    ArmEvidence, _permitted_file, matching_config_hash, verify_arm_initial_checkpoint,
    verify_matched_arm_records, verify_saved_suite_initialization, verify_suite_run_metadata,
)
from scripts.streamed_overlap_metrics import (
    QueryRanking, RankingReport, _eligible_queries, compare_matched_cosine, summarize_rankings,
)
from scripts.streamed_overlap_model import StreamedPairVAE, build_optimizer, fresh_e1_model
from scripts.streamed_overlap_reconstruction import (
    _runtime as _reconstruction_runtime, _verify_stores, lpips_fingerprint, measure_reconstruction,
)
from scripts.streamed_overlap_search import run_bounded_searches, verify_search_manifest
from scripts.streamed_overlap_sampler import StreamedPairSampler
from scripts.streamed_overlap_selection import _best, _records, select_checkpoints
from scripts.train import (
    EarlyStoppingState, SliceLPIPSLoss, initialize_streamed_epoch_metrics, streamed_monitor_hash,
    train_streamed_epoch,
)
from src.tokenizer.core.model_adapter import choose_torch_device


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
            or not isinstance(search_manifest, dict) or not isinstance(diagnostic_manifest, dict)
            or search_manifest.get("search_manifest_sha256") != config["search_manifest_sha256"]
            or search_manifest.get("search_manifest_sha256") != content_hash({
                key: value for key, value in search_manifest.items() if key != "search_manifest_sha256"
            })
            or diagnostic_manifest.get("diagnostic_manifest_sha256") != config["diagnostic_manifest_sha256"]
            or diagnostic_manifest.get("diagnostic_manifest_sha256") != content_hash({
                key: value for key, value in diagnostic_manifest.items() if key != "diagnostic_manifest_sha256"
            })):
        raise ValueError("An explicit frozen MPS evaluation configuration is required.")
    _validate_evaluation_parameters(
        inference_seed=config["inference_seed"], encode_batch_size=config["encode_batch_size"],
        max_wall_seconds=config["max_wall_seconds"], random_seed=config["random_seed"],
        random_repeats=config["random_repeats"], bootstrap_seed=config["bootstrap_seed"],
        bootstrap_repeats=config["bootstrap_repeats"],
    )
    diagnostic_limit = config["diagnostic_max_wall_seconds"]
    if (isinstance(diagnostic_limit, bool) or not isinstance(diagnostic_limit, (int, float))
            or not math.isfinite(diagnostic_limit) or diagnostic_limit <= 0):
        raise ValueError("An explicit positive diagnostic wall-time limit is required.")


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


def _validate_reconstruction_config(guardrail_manifest: dict, guardrail_config: dict, formal_config: dict) -> dict:
    required = {"manifest_sha256", "device", "max_wall_seconds"}
    if (not isinstance(formal_config, dict) or not isinstance(guardrail_config, dict)
            or set(guardrail_config) != required or not isinstance(guardrail_manifest, dict)
            or guardrail_config["device"] != formal_config.get("device")
            or isinstance(guardrail_config["max_wall_seconds"], bool)
            or not isinstance(guardrail_config["max_wall_seconds"], (int, float))
            or not math.isfinite(guardrail_config["max_wall_seconds"])
            or guardrail_config["max_wall_seconds"] <= 0):
        raise ValueError("An explicit frozen reconstruction guardrail configuration is required.")
    expected = _guardrail_manifest(guardrail_manifest)
    if (guardrail_manifest["manifest_sha256"] != guardrail_config["manifest_sha256"]
            or guardrail_manifest["protocol"].get("device") != guardrail_config["device"]):
        raise ValueError("Reconstruction manifest and frozen configuration disagree.")
    return expected


def complete_selected_reconstruction(
    checkpoint_path: Path, journal, manifest_path: Path, source_root: Path,
    search_manifest: dict, diagnostic_manifest: dict, formal_config: dict, *,
    selection_kind: str, evaluation_dir: Path, guardrail_manifest: dict,
    guardrail_config: dict, lpips_loss_fn,
) -> dict:
    """Measure a verified selection on the frozen reconstruction guardrail inputs."""
    expected = _validate_reconstruction_config(guardrail_manifest, guardrail_config, formal_config)
    config_hash = content_hash(guardrail_config)
    last_state, last_hash = _checkpoint_state(checkpoint_path)
    if last_state.get("hashes", {}).get("reconstruction_guardrail") != config_hash:
        raise ValueError("Final checkpoint and frozen reconstruction configuration disagree.")
    selected = complete_selected_checkpoint(
        checkpoint_path, journal, manifest_path, source_root, search_manifest,
        diagnostic_manifest, formal_config, selection_kind=selection_kind, evaluation_dir=evaluation_dir,
    )
    identity = selected["checkpoint"]
    snapshot = checkpoint_path.parent / "selection" / "snapshots" / f"{identity['checkpoint_sha256']}.pt"
    state, digest = _checkpoint_state(snapshot)
    if (digest != identity["checkpoint_sha256"]
            or state["hashes"].get("reconstruction_guardrail") != config_hash):
        raise ValueError("Selected checkpoint and frozen reconstruction configuration disagree.")
    alias = checkpoint_path.parent / f"best_{selection_kind}.pt"

    def verify_checkpoints():
        if any(path.is_symlink() or not path.is_file() or _digest(path) != expected_hash
               for path, expected_hash in ((checkpoint_path, last_hash), (snapshot, digest), (alias, digest))):
            raise ValueError("Final or selected checkpoint changed during reconstruction measurement.")

    verify_checkpoints()
    directory = evaluation_dir / f"guardrail-{selection_kind}"
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise ValueError("Reconstruction measurement directory must be a regular directory.")
    directory.mkdir(exist_ok=True)
    _sync_directory(evaluation_dir)
    events = directory / "completed"
    if events.is_symlink() or (events.exists() and not events.is_dir()):
        raise ValueError("Reconstruction completion records must be in a regular directory.")
    events.mkdir(exist_ok=True)
    _sync_directory(directory)
    matches = []
    for path in events.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(record, dict) or record.get("schema") != "streamed-reconstruction-completion-v1"
                or record.get("sha256") != content_hash({
                    key: value for key, value in record.items() if key != "sha256"
                })):
            raise ValueError("Reconstruction completion record is invalid.")
        if record.get("checkpoint") == identity and record.get("config_sha256") == config_hash:
            matches.append(record)
    if len(matches) > 1:
        raise ValueError("Duplicate reconstruction completion records for the selected checkpoint.")
    if matches:
        record = matches[0]
        artifact = record["measurement"]
        path = directory / artifact["path"]
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(directory.resolve())
                or _digest(path) != artifact["sha256"]):
            raise ValueError("Completed reconstruction measurement is missing or changed.")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("checkpoint") != identity:
            raise ValueError("Reconstruction measurement checkpoint differs from the selection.")
        _measurement(report, guardrail_manifest, expected, identity["arm"])
        deadline = time.monotonic() + guardrail_config["max_wall_seconds"]
        protocol = guardrail_manifest["protocol"]
        device = choose_torch_device(guardrail_config["device"])
        if (protocol.get("runtime") != _reconstruction_runtime()
                or lpips_fingerprint(lpips_loss_fn) != protocol["lpips_sha256"]
                or {parameter.device.type for parameter in lpips_loss_fn.parameters()} != {device.type}):
            raise ValueError("Reconstruction runtime or LPIPS differs from the frozen protocol.")
        _verify_stores(guardrail_manifest, deadline)
        verify_checkpoints()
        return record
    attempt = directory / f"checkpoint-{identity['epoch']:06d}-{uuid.uuid4().hex}"
    attempt.mkdir()
    _sync_directory(directory)
    report = measure_reconstruction(
        guardrail_manifest, snapshot, arm=identity["arm"], device=guardrail_config["device"],
        lpips_loss_fn=lpips_loss_fn, max_wall_seconds=guardrail_config["max_wall_seconds"],
        journal=journal, validation_manifest_path=manifest_path,
    )
    if report.get("checkpoint") != identity:
        raise ValueError("Reconstruction measurement checkpoint differs from the selection.")
    _measurement(report, guardrail_manifest, expected, identity["arm"])
    artifact = _save_report(attempt, "measurement.json", report)
    verify_checkpoints()
    record = {
        "schema": "streamed-reconstruction-completion-v1", "checkpoint": identity,
        "config_sha256": config_hash, "measurement": artifact,
        "limitation": "Measurement only; matched references, numerical comparison and adoption remain separate.",
    }
    record["sha256"] = content_hash(record)
    _atomic_record(events, record)
    return record


@dataclass(frozen=True)
class SelectedReconstruction:
    checkpoint_path: Path
    journal: BatchJournal
    manifest_path: Path
    source_root: Path
    search_manifest: dict
    diagnostic_manifest: dict
    formal_config: dict
    evaluation_dir: Path
    selection_kind: str


@dataclass(frozen=True)
class HistoricalReconstruction:
    report_path: Path
    report_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_label: str


@dataclass(frozen=True)
class SuiteInitialization:
    evidence_path: Path
    evidence_sha256: str
    initial_checkpoints: dict[str, Path]


@dataclass(frozen=True)
class ArmRuntime:
    model: StreamedPairVAE
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau
    sampler: StreamedPairSampler
    early_stopping: EarlyStoppingState
    writer: SummaryWriter


@contextmanager
def _arm_runtime(
    arm: str, initial: Path, journal: BatchJournal, source_root: Path, matching_config: dict, hashes: dict,
):
    sampler = StreamedPairSampler(source_root, "train", matching_config["sampler_seed"])
    if sampler.source_manifest_hash != hashes["source"]:
        raise ValueError("Training sampler does not match the frozen source manifest.")
    epoch_csv = initial.parent / "training_metrics.csv"
    if not epoch_csv.exists() and not epoch_csv.is_symlink():
        state, _ = _checkpoint_state(initial.with_name("last.pt"))
        if state.get("completed_epoch") != 0 or journal.rows:
            raise FileNotFoundError("Cannot recreate a missing epoch CSV for an advanced or interrupted arm.")
        initialize_streamed_epoch_metrics(epoch_csv)
    if epoch_csv.is_symlink() or not epoch_csv.is_file():
        raise ValueError("The epoch CSV must be a regular file.")
    tensorboard = initial.parent / "tensorboard"
    if tensorboard.is_symlink() or (tensorboard.exists() and not tensorboard.is_dir()):
        raise ValueError("TensorBoard output must be a regular directory.")
    model = fresh_e1_model(arm, matching_config["model_seed"], matching_config["head_seed"]).to("mps")
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=6, min_lr=1e-5)
    with SummaryWriter(log_dir=str(tensorboard)) as writer:
        yield ArmRuntime(model, optimizer, scheduler, sampler, EarlyStoppingState(float("inf"), 0), writer)


def suite_workload(matching_config: dict, search_manifest: dict) -> dict:
    """Count clean-run exposure and scheduled evaluations without authorizing a launch."""
    config_hash = matching_config_hash(matching_config)
    if (not isinstance(search_manifest, dict)
            or search_manifest.get("schema") != "streamed-bounded-tokenizer-search-v1"
            or not isinstance(search_manifest.get("searches"), list) or not search_manifest["searches"]
            or search_manifest.get("search_manifest_sha256") != content_hash({
                key: value for key, value in search_manifest.items() if key != "search_manifest_sha256"
            })):
        raise ValueError("A nonempty frozen bounded-search manifest is required.")
    epochs = matching_config["final_epoch"]
    batches = matching_config["batches_per_epoch"]
    steps = epochs * batches
    due_epochs = epochs // 5 + int(epochs % 5 != 0)
    monitors = epochs + steps // 100 - epochs // (100 // math.gcd(100, batches))
    epoch_formal = 3 * due_epochs
    selected_formal = 4
    return {
        "schema": "streamed-suite-workload-v1",
        "matching_config_sha256": config_hash,
        "search_manifest_sha256": search_manifest["search_manifest_sha256"],
        "per_arm": {
            "epochs": epochs, "optimizer_steps": steps, "training_pairs": 12 * steps,
            "training_encoder_views": 24 * steps, "training_reconstructed_a": 12 * steps,
            "epoch_validations": epochs, "monitor_runs": monitors,
            "epoch_formal_evaluations": due_epochs,
        },
        "suite": {
            "optimizer_steps": 3 * steps, "training_pairs": 36 * steps,
            "training_encoder_views": 72 * steps, "training_reconstructed_a": 36 * steps,
            "epoch_validations": 3 * epochs, "monitor_runs": 3 * monitors,
            "epoch_formal_evaluations": epoch_formal, "selected_formal_evaluations": selected_formal,
            "s3_diagnostic_evaluations": due_epochs + 2,
            "tokenizer_searches": (epoch_formal + selected_formal) * len(search_manifest["searches"]),
        },
        "limitation": (
            "Clean fresh run only; retries, incomplete attempts, I/O, storage, wall time and memory are not "
            "estimated. Counts do not authorize training."
        ),
    }


def _validate_epoch_inputs(
    monitor_batches: dict, validate_epoch, lpips_loss_fn, *, hashes: dict,
    manifest_path: Path, source_root: Path, guardrail_manifest: dict,
) -> None:
    if (not isinstance(validate_epoch, StreamedEpochValidator)
            or not isinstance(manifest_path, Path) or not isinstance(source_root, Path)
            or not isinstance(validate_epoch.manifest_path, Path) or not isinstance(validate_epoch.source_root, Path)):
        raise ValueError("A frozen epoch validator and explicit input paths are required.")
    paths = (manifest_path, source_root, validate_epoch.manifest_path, validate_epoch.source_root)
    if (any("fake_data_validation_v2" in path.parts or "fake_data_validation_v2" in path.resolve().parts
            for path in paths)
            or manifest_path.resolve() != validate_epoch.manifest_path.resolve()
            or source_root.resolve() != validate_epoch.source_root.resolve()
            or validate_epoch.manifest_sha256 != hashes.get("validation_manifest")
            or validate_epoch.configuration_hash() != validate_epoch.config_sha256
            or validate_epoch.config_sha256 != hashes.get("epoch_validation")
            or validate_epoch.lpips_loss_fn is not lpips_loss_fn):
        raise ValueError("Epoch validation inputs or configuration differ from the frozen suite.")
    if hashes.get("monitor") != streamed_monitor_hash(monitor_batches):
        raise ValueError("Fixed monitor inputs differ from the frozen monitor hash.")
    if lpips_fingerprint(lpips_loss_fn) != guardrail_manifest["protocol"]["lpips_sha256"]:
        raise ValueError("LPIPS does not match the frozen reconstruction guardrail.")


def prepare_epoch_inputs(
    *, manifest_path: Path, source_root: Path, hashes: dict, guardrail_manifest: dict,
    matching_config: dict, epoch_validation: dict,
) -> tuple[dict, StreamedEpochValidator, SliceLPIPSLoss, dict[str, str]]:
    """Build fixed monitors and epoch validation without advancing a training sampler."""
    matching_config_hash(matching_config)
    required = {"reconstruction_query_ids", "encode_batch_size", "reconstruction_batch_size", "max_wall_seconds"}
    if not isinstance(epoch_validation, dict) or set(epoch_validation) != required:
        raise ValueError("Explicit epoch validation queries, batch sizes and wall limit are required.")
    ids = epoch_validation["reconstruction_query_ids"]
    if (not isinstance(ids, list) or not ids or any(not isinstance(item, str) or not item for item in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Reconstruction query IDs must be an explicit list of unique nonempty strings.")
    if (not isinstance(manifest_path, Path) or not isinstance(source_root, Path)
            or "fake_data_validation_v2" in manifest_path.parts or "fake_data_validation_v2" in source_root.parts):
        raise ValueError("Explicit primary validation and source paths are required.")
    root = source_root.resolve(strict=True)
    path = manifest_path.resolve(strict=True)
    forbidden = Path("/Volumes/CrucialX9/fake_data_validation_v2")
    if ("fake_data_validation_v2" in root.parts or "fake_data_validation_v2" in path.parts
            or forbidden.is_relative_to(root) or not root.is_dir() or not path.is_file()
            or path.is_relative_to(root)):
        raise ValueError("Validation manifest and source paths must exclude validation-v2 and be separate.")
    manifest = load_manifest(path)
    if (manifest.get("manifest_sha256") != hashes.get("validation_manifest")
            or manifest.get("target_implementation_sha256") != hashes.get("target")):
        raise ValueError("The validation manifest or target differs from the frozen suite.")
    if not set(ids) <= {query["id"] for query in _eligible_queries(manifest)}:
        raise ValueError("Reconstruction queries must belong to the frozen eligible manifest.")
    if not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("Preparing streamed epoch inputs requires MPS without CPU fallback.")
    with torch.random.fork_rng(devices=[]):
        lpips = SliceLPIPSLoss().to("mps").eval()
    if lpips_fingerprint(lpips) != guardrail_manifest["protocol"]["lpips_sha256"]:
        raise ValueError("LPIPS does not match the frozen reconstruction guardrail.")
    validator = StreamedEpochValidator(
        manifest_path, source_root, manifest_sha256=manifest["manifest_sha256"],
        reconstruction_query_ids=tuple(ids), reconstruction_seed=matching_config["validation_seed"],
        encode_batch_size=epoch_validation["encode_batch_size"],
        reconstruction_batch_size=epoch_validation["reconstruction_batch_size"],
        max_wall_seconds=epoch_validation["max_wall_seconds"], lpips_loss_fn=lpips,
    )
    samplers = {
        split: StreamedPairSampler(root, split, matching_config["monitor_seed"])
        for split in ("train", "validation")
    }
    if any(sampler.source_manifest_hash != hashes.get("source") for sampler in samplers.values()):
        raise ValueError("Monitor samplers do not match the frozen source manifest.")
    batches = {
        split: sampler.load_batch(sampler.next_specs(), sparse_probability=matching_config["sparse_probability"])
        for split, sampler in samplers.items()
    }
    observed = {"monitor": streamed_monitor_hash(batches), "epoch_validation": validator.config_sha256}
    for key, value in observed.items():
        if key in hashes and hashes[key] != value:
            raise ValueError(f"Prepared {key} differs from the frozen suite.")
    return batches, validator, lpips, observed


def _reference_hash(historical: HistoricalReconstruction) -> str:
    if (not isinstance(historical, HistoricalReconstruction)
            or not _sha256(historical.report_sha256) or not _sha256(historical.checkpoint_sha256)
            or not isinstance(historical.checkpoint_label, str) or not historical.checkpoint_label):
        raise ValueError("An explicit historical reconstruction reference is required.")
    return content_hash({
        "report_sha256": historical.report_sha256,
        "checkpoint_sha256": historical.checkpoint_sha256,
        "checkpoint_label": historical.checkpoint_label,
    })


def _validate_evaluation_bindings(
    hashes: dict, formal_config: dict, guardrail_config: dict, historical: HistoricalReconstruction,
) -> None:
    if hashes.get("formal_evaluation") != content_hash(formal_config):
        raise ValueError("The arm must use its frozen formal evaluation configuration.")
    if (hashes.get("reconstruction_guardrail") != content_hash(guardrail_config)
            or hashes.get("reconstruction_reference") != _reference_hash(historical)):
        raise ValueError("The arm must bind the frozen reconstruction configuration and reference.")


def complete_reconstruction_comparison(
    candidate: SelectedReconstruction, historical: HistoricalReconstruction, *,
    s1: SelectedReconstruction | None, guardrail_manifest: dict, guardrail_config: dict, lpips_loss_fn,
) -> dict:
    """Compare verified selected measurements; this is not an adoption decision."""
    reference_hash = _reference_hash(historical)

    def verify_historical():
        for path, digest in ((historical.report_path, historical.report_sha256),
                             (historical.checkpoint_path, historical.checkpoint_sha256)):
            if ("fake_data_validation_v2" in path.parts or "fake_data_validation_v2" in path.resolve().parts
                    or path.is_symlink() or not path.is_file() or _digest(path) != digest):
                raise ValueError("Historical reconstruction report or checkpoint is missing or changed.")

    verify_historical()
    historical_report = json.loads(historical.report_path.read_text(encoding="utf-8"))
    historical_identity = historical_report.get("checkpoint") if isinstance(historical_report, dict) else None
    if (not isinstance(historical_identity, dict) or historical_identity.get("arm") != "P0b"
            or historical_identity.get("checkpoint_sha256") != historical.checkpoint_sha256
            or historical_identity.get("checkpoint_label") != historical.checkpoint_label):
        raise ValueError("Historical reconstruction report does not match the frozen reference.")

    def check_reference_binding(selection: SelectedReconstruction) -> dict:
        if any("fake_data_validation_v2" in path.parts or "fake_data_validation_v2" in path.resolve().parts
               for path in (
            selection.checkpoint_path, selection.manifest_path, selection.evaluation_dir,
        )):
            raise ValueError("The regenerating validation-v2 dataset is not permitted.")
        state, _ = _checkpoint_state(selection.checkpoint_path)
        if state.get("hashes", {}).get("reconstruction_reference") != reference_hash:
            raise ValueError("Checkpoint does not bind the historical reconstruction reference.")
        return state

    state = check_reference_binding(candidate)
    arm = state.get("arm")
    if arm not in {"S1", "S2", "S3"} or (arm == "S1") != (s1 is None):
        raise ValueError("S1 needs P0b only; S2 and S3 require both matched S1 and P0b.")
    if s1 is not None:
        if s1.selection_kind != "reconstruction" or check_reference_binding(s1).get("arm") != "S1":
            raise ValueError("The matched S1 reference must be the reconstruction selection.")

    def verified_measurement(selection: SelectedReconstruction) -> tuple[dict, str, Path]:
        record = complete_selected_reconstruction(
            selection.checkpoint_path, selection.journal, selection.manifest_path, selection.source_root,
            selection.search_manifest, selection.diagnostic_manifest, selection.formal_config,
            selection_kind=selection.selection_kind, evaluation_dir=selection.evaluation_dir,
            guardrail_manifest=guardrail_manifest, guardrail_config=guardrail_config,
            lpips_loss_fn=lpips_loss_fn,
        )
        directory = selection.evaluation_dir / f"guardrail-{selection.selection_kind}"
        artifact = record["measurement"]
        path = directory / artifact["path"]
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(directory.resolve())
                or _digest(path) != artifact["sha256"]):
            raise ValueError("Selected reconstruction measurement is missing or changed.")
        report = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(report, dict) or report.get("checkpoint") != record["checkpoint"]
                or _digest(path) != artifact["sha256"]):
            raise ValueError("Selected reconstruction report identity or contents changed.")
        snapshot = (selection.checkpoint_path.parent / "selection" / "snapshots"
                    / f"{record['checkpoint']['checkpoint_sha256']}.pt")
        selected_state, digest = _checkpoint_state(snapshot)
        if (digest != record["checkpoint"]["checkpoint_sha256"]
                or selected_state.get("hashes", {}).get("reconstruction_reference") != reference_hash):
            raise ValueError("Selected checkpoint does not bind the historical reconstruction reference.")
        return report, artifact["sha256"], path

    candidate_report, candidate_hash, candidate_path = verified_measurement(candidate)
    if candidate_report["checkpoint"]["arm"] != arm:
        raise ValueError("Selected reconstruction report has the wrong arm.")
    references = {"P0b": historical_report}
    inputs = {"candidate": candidate_hash, "P0b": historical.report_sha256}
    s1_report: dict | None = None
    s1_path: Path | None = None
    if s1 is not None:
        s1_report, s1_hash, s1_path = verified_measurement(s1)
        candidate_identity, s1_identity = candidate_report["checkpoint"], s1_report["checkpoint"]
        if (s1_identity.get("arm") != "S1" or any(
            candidate_identity.get(key) != s1_identity.get(key) or not candidate_identity.get(key)
            for key in ("run_id", "source_sha256", "target_sha256", "validation_manifest_sha256")
        )):
            raise ValueError("The S1 reference does not match the candidate run, source, target or validation.")
        references["S1"], inputs["S1"] = s1_report, s1_hash
    comparison = compare_reconstruction_guardrails(guardrail_manifest, candidate_report, references)
    verify_historical()

    def verify_selected(selection: SelectedReconstruction, report: dict, path: Path, digest: str) -> None:
        selected = complete_selected_checkpoint(
            selection.checkpoint_path, selection.journal, selection.manifest_path, selection.source_root,
            selection.search_manifest, selection.diagnostic_manifest, selection.formal_config,
            selection_kind=selection.selection_kind, evaluation_dir=selection.evaluation_dir,
        )
        if (selected["checkpoint"] != report["checkpoint"] or path.is_symlink()
                or not path.is_file() or _digest(path) != digest):
            raise ValueError("Selected checkpoint or reconstruction measurement changed during comparison.")

    verify_selected(candidate, candidate_report, candidate_path, candidate_hash)
    if s1 is not None and s1_report is not None and s1_path is not None:
        verify_selected(s1, s1_report, s1_path, inputs["S1"])
    directory = candidate.evaluation_dir / f"guardrail-{candidate.selection_kind}" / "comparisons"
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise ValueError("Reconstruction comparison directory must be a regular directory.")
    directory.mkdir(exist_ok=True)
    _sync_directory(directory.parent)
    record = {
        "schema": "streamed-reconstruction-comparison-v1",
        "checkpoint": candidate_report["checkpoint"], "manifest_sha256": guardrail_manifest["manifest_sha256"],
        "reference_sha256": reference_hash, "measurement_sha256": inputs,
        "reference_checkpoints": {name: report["checkpoint"] for name, report in references.items()},
        "comparison": comparison,
        "limitation": (
            "Numeric guardrail only; matched initialization/streams, reference provenance and adoption "
            "require separate verification."
        ),
    }
    record["sha256"] = content_hash(record)
    matches = []
    for path in directory.glob("*.json"):
        previous = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(previous, dict) or previous.get("schema") != record["schema"]
                or previous.get("sha256") != content_hash({
                    key: value for key, value in previous.items() if key != "sha256"
                })):
            raise ValueError("Reconstruction comparison record is invalid.")
        if (previous.get("checkpoint") == record["checkpoint"]
                and previous.get("reference_sha256") == reference_hash
                and previous.get("measurement_sha256") == inputs):
            matches.append(previous)
    if len(matches) > 1 or (matches and matches[0] != record):
        raise ValueError("Reconstruction comparison records are duplicated or inconsistent.")
    if matches:
        return matches[0]
    _atomic_record(directory, record)
    return record


def complete_matched_arm(
    reference: ArmEvidence, candidate: ArmEvidence, *, matching_config: dict, evaluation_dir: Path,
    expected_initial_sha256: dict[str, str],
) -> dict:
    """Persist or reverify matched recorded streams; this is not an adoption decision."""
    if (not isinstance(candidate, ArmEvidence) or not isinstance(candidate.journal, BatchJournal)
            or "fake_data_validation_v2" in evaluation_dir.parts or evaluation_dir.is_symlink()
            or "fake_data_validation_v2" in evaluation_dir.resolve().parts or not evaluation_dir.is_dir()
            or not evaluation_dir.resolve().is_relative_to(candidate.journal.path.parent.resolve())):
        raise ValueError("Matched evidence requires a regular candidate-owned evaluation directory.")
    report = verify_matched_arm_records(reference, candidate, matching_config=matching_config)
    if report["initial_checkpoint_sha256"] != expected_initial_sha256:
        raise ValueError("Matched streams do not use the pinned suite initial checkpoints.")
    directory = evaluation_dir / "matched_streams"
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise ValueError("Matched-stream records require a regular directory.")
    directory.mkdir(exist_ok=True)
    _sync_directory(evaluation_dir)

    def existing() -> list[Path]:
        if (directory.is_symlink() or "fake_data_validation_v2" in directory.resolve().parts
                or not directory.is_dir()):
            raise ValueError("Matched-stream records require a regular directory.")
        paths = []
        for path in directory.glob("*.json"):
            _permitted_file(path)
            previous = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(previous, dict) or previous.get("schema") != "streamed-matched-arm-records-v2"
                    or previous.get("sha256") != content_hash({
                        key: value for key, value in previous.items() if key != "sha256"
                    }) or previous != report):
                raise ValueError("Saved matched-stream record is invalid or differs from current evidence.")
            paths.append(path)
        if len(paths) > 1:
            raise ValueError("Matched-stream records are duplicated.")
        return paths

    paths = existing()
    path = paths[0] if paths else _atomic_record(directory, report)
    if (verify_matched_arm_records(reference, candidate, matching_config=matching_config) != report
            or existing() != [path]):
        raise ValueError("Matched-stream evidence changed during publication.")
    return {"path": str(path.relative_to(evaluation_dir)), "record": report}


def run_arm_epochs(
    model, optimizer, scheduler, sampler, journal, lpips_loss_fn, writer, *,
    run_id: str, seed: int, attempt_id: str, batches_per_epoch: int, sparse_probability: float,
    monitor_batches: dict, monitor_seed: int, validation_seed: int, validate_epoch,
    epoch_csv: Path, checkpoint_path: Path, hashes: dict, early_stopping,
    manifest_path: Path, source_root: Path, search_manifest: dict, diagnostic_manifest: dict,
    formal_config: dict, evaluation_dir: Path, guardrail_manifest: dict, guardrail_config: dict,
    historical: HistoricalReconstruction, s1: SelectedReconstruction | None, matching_config: dict,
    initial_checkpoint: Path, suite_initialization: SuiteInitialization,
) -> dict:
    """Resume one arm from its full epoch state and finish evaluation before advancing."""
    _validate_evaluation_bindings(hashes, formal_config, guardrail_config, historical)
    if (isinstance(formal_config.get("final_epoch"), bool)
            or not isinstance(formal_config.get("final_epoch"), int)
            or formal_config["final_epoch"] < 1):
        raise ValueError("An explicit positive final epoch is required.")
    if (hashes.get("matching") != matching_config_hash(matching_config)
            or not isinstance(run_id, str) or run_id != matching_config["run_id"]
            or any(isinstance(value, bool) or not isinstance(value, int) or value != matching_config[key]
                   for key, value in (
                       ("experiment_seed", seed), ("batches_per_epoch", batches_per_epoch),
                       ("monitor_seed", monitor_seed), ("validation_seed", validation_seed),
                   ))
            or isinstance(sparse_probability, bool) or not isinstance(sparse_probability, (int, float))
            or not math.isfinite(sparse_probability)
            or sparse_probability != matching_config["sparse_probability"]
            or formal_config["final_epoch"] != matching_config["final_epoch"]):
        raise ValueError("Arm arguments must match the frozen matching configuration.")
    _validate_formal_config(search_manifest, diagnostic_manifest, formal_config)
    _validate_reconstruction_config(guardrail_manifest, guardrail_config, formal_config)
    _validate_epoch_inputs(
        monitor_batches, validate_epoch, lpips_loss_fn, hashes=hashes,
        manifest_path=manifest_path, source_root=source_root, guardrail_manifest=guardrail_manifest,
    )
    if (model.arm not in {"S1", "S2", "S3"}
            or (model.arm == "S1" and s1 is not None)
            or (model.arm in {"S2", "S3"} and (
                not isinstance(s1, SelectedReconstruction) or s1.selection_kind != "reconstruction"
            ))):
        raise ValueError("S1 needs P0b only; S2 and S3 require the selected S1 reconstruction reference.")
    if (not isinstance(suite_initialization, SuiteInitialization)
            or not isinstance(suite_initialization.initial_checkpoints, dict)
            or suite_initialization.initial_checkpoints.get(model.arm) != initial_checkpoint
            or initial_checkpoint.name != "initial.pt"
            or checkpoint_path != initial_checkpoint.with_name("last.pt")
            or not isinstance(suite_initialization.evidence_path, Path)
            or suite_initialization.evidence_path.name != "initialization_evidence.json"
            or not isinstance(getattr(journal, "path", None), Path)
            or journal.path.parent != initial_checkpoint.parent):
        raise ValueError("The arm must use its retained suite initialization checkpoint.")
    metadata = verify_suite_run_metadata(
        suite_initialization.evidence_path.parent, suite_initialization.initial_checkpoints,
        matching_config=matching_config, expected_hashes=hashes,
    )
    if metadata["initialization_sha256"] != suite_initialization.evidence_sha256:
        raise ValueError("The arm must use the initialization pin recorded in suite metadata.")
    suite_record = verify_saved_suite_initialization(
        suite_initialization.evidence_path, suite_initialization.initial_checkpoints,
        matching_config=matching_config, expected_hashes=hashes,
        expected_evidence_sha256=suite_initialization.evidence_sha256,
    )
    if s1 is not None:
        s1_initial = suite_initialization.initial_checkpoints["S1"]
        if (s1.checkpoint_path != s1_initial.with_name("last.pt")
                or not isinstance(getattr(s1.journal, "path", None), Path)
                or s1.journal.path.parent != s1_initial.parent):
            raise ValueError("The S1 reconstruction reference must belong to this initialized suite.")
    verify_arm_initial_checkpoint(
        initial_checkpoint, checkpoint_path, arm=model.arm,
        matching_config=matching_config, expected_hashes=hashes,
    )
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
    matched_streams = None
    if s1 is not None:
        initial_hashes = suite_record["initialization"]["initial_checkpoint_sha256"]
        matched_streams = complete_matched_arm(
            ArmEvidence(suite_initialization.initial_checkpoints["S1"], s1.checkpoint_path, s1.journal),
            ArmEvidence(initial_checkpoint, checkpoint_path, journal),
            matching_config=matching_config, evaluation_dir=evaluation_dir,
            expected_initial_sha256={"S1": initial_hashes["S1"], model.arm: initial_hashes[model.arm]},
        )
    selected_evaluations = {}
    kinds = ("reconstruction", "retrieval") if model.arm == "S3" else ("reconstruction",)
    for kind in kinds:
        selected_evaluations[kind] = complete_selected_checkpoint(
            checkpoint_path, journal, manifest_path, source_root, search_manifest,
            diagnostic_manifest, formal_config, selection_kind=kind, evaluation_dir=evaluation_dir,
        )
    comparisons = {}
    for kind in kinds:
        candidate = SelectedReconstruction(
            checkpoint_path, journal, manifest_path, source_root, search_manifest, diagnostic_manifest,
            formal_config, evaluation_dir, kind,
        )
        comparisons[kind] = complete_reconstruction_comparison(
            candidate, historical, s1=s1, guardrail_manifest=guardrail_manifest,
            guardrail_config=guardrail_config, lpips_loss_fn=lpips_loss_fn,
        )
    result = {
        "last_epoch": last_completion, "selected_evaluations": selected_evaluations,
        "reconstruction_comparisons": comparisons,
    }
    if matched_streams is not None:
        result["matched_streams"] = matched_streams
    return result


def _validate_suite_inputs(
    *, monitor_batches: dict, validate_epoch, lpips_loss_fn,
    manifest_path: Path, source_root: Path, search_manifest: dict, diagnostic_manifest: dict,
    formal_config: dict, guardrail_manifest: dict, guardrail_config: dict,
    historical: HistoricalReconstruction, matching_config: dict, hashes: dict,
) -> dict:
    _validate_hashes(hashes)
    _validate_evaluation_bindings(hashes, formal_config, guardrail_config, historical)
    _validate_formal_config(search_manifest, diagnostic_manifest, formal_config)
    _validate_reconstruction_config(guardrail_manifest, guardrail_config, formal_config)
    _validate_epoch_inputs(
        monitor_batches, validate_epoch, lpips_loss_fn, hashes=hashes,
        manifest_path=manifest_path, source_root=source_root, guardrail_manifest=guardrail_manifest,
    )
    workload = suite_workload(matching_config, search_manifest)
    if (hashes.get("matching") != matching_config_hash(matching_config)
            or formal_config["final_epoch"] != matching_config["final_epoch"]):
        raise ValueError("Evaluation and suite budget must match the frozen matching configuration.")
    return workload


def suite_input_record(
    suite_dir: Path, *, validate_epoch: StreamedEpochValidator, manifest_path: Path, source_root: Path,
    search_manifest: dict, diagnostic_manifest: dict, formal_config: dict, guardrail_manifest: dict,
    guardrail_config: dict, historical: HistoricalReconstruction, matching_config: dict, hashes: dict,
) -> dict:
    """Describe the suite inputs for hashing; this does not approve or launch training."""
    if not isinstance(validate_epoch, StreamedEpochValidator):
        raise ValueError("A prepared epoch validator is required to describe suite inputs.")
    _reference_hash(historical)
    record = {
        "schema": "streamed-suite-inputs-v1",
        "suite_dir": str(suite_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "source_root": str(source_root.resolve()),
        "matching_config": matching_config,
        "epoch_validation": {
            "reconstruction_query_ids": list(validate_epoch.reconstruction_query_ids),
            "encode_batch_size": validate_epoch.encode_batch_size,
            "reconstruction_batch_size": validate_epoch.reconstruction_batch_size,
            "max_wall_seconds": validate_epoch.max_wall_seconds,
        },
        "search_manifest": search_manifest,
        "diagnostic_manifest": diagnostic_manifest,
        "formal_config": formal_config,
        "guardrail_manifest": guardrail_manifest,
        "guardrail_config": guardrail_config,
        "historical": {
            "report_path": str(historical.report_path.resolve()), "report_sha256": historical.report_sha256,
            "checkpoint_path": str(historical.checkpoint_path.resolve()),
            "checkpoint_sha256": historical.checkpoint_sha256, "checkpoint_label": historical.checkpoint_label,
        },
        "input_hashes": {key: value for key, value in hashes.items() if key != "config"},
    }
    return json.loads(json.dumps(record, sort_keys=True, allow_nan=False))


def _bind_suite_inputs(suite_dir: Path, record: dict, *, resume: bool) -> None:
    path = suite_dir / "resolved_suite_inputs.json"
    if resume:
        _permitted_file(path)
        if content_hash(json.loads(path.read_text(encoding="utf-8"))) != content_hash(record):
            raise ValueError("Saved suite inputs differ from the requested configuration.")
    else:
        data = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()

        def write(handle: BinaryIO) -> None:
            handle.write(data)

        _publish_exclusive(path, write)


def _suite_cosine_comparisons(results: dict, initialization: SuiteInitialization, manifest: dict,
                              formal_config: dict) -> dict:
    """Compare selected encoder rankings, separately from the trained S3 pair head."""
    for arm in ("S2", "S3"):
        matched = results[arm]["matched_streams"]["record"]
        if (matched.get("status") != "MATCHED_RECORDED_STREAMS_ONLY"
                or matched.get("sha256") != content_hash({k: v for k, v in matched.items() if k != "sha256"})
                or matched.get("reference") != results["S1"]["last_epoch"]["checkpoint"]
                or matched.get("candidate") != results[arm]["last_epoch"]["checkpoint"]):
            raise ValueError("Cosine comparisons require the verified matched arm records.")

    queries = _eligible_queries(manifest)
    expected = {q["id"]: (q["window"]["source_id"], q["cohort"]) for q in queries}
    fingerprints = manifest["input_integrity"]["fingerprints"]
    rankings = {}
    sources = {}
    for arm, kind in (("S1", "final"), ("S2", "final"), ("S3", "final"),
                      ("S1", "reconstruction"), ("S2", "reconstruction"),
                      ("S3", "reconstruction"), ("S3", "retrieval")):
        completion = (results[arm]["last_epoch"] if kind == "final"
                      else results[arm]["selected_evaluations"][kind])
        identity = completion["checkpoint"]
        directory = initialization.initial_checkpoints[arm].parent / "evaluation"
        config_hash = content_hash(formal_config)
        if kind != "final":
            directory = directory / f"selected-{kind}"
            config_hash = content_hash({"formal_evaluation": formal_config, "selection_kind": kind})
        artifact = completion["artifacts"]["retrieval"]
        path = directory / artifact["path"]
        if (completion.get("sha256") != content_hash({k: v for k, v in completion.items() if k != "sha256"})
                or completion.get("config_sha256") != config_hash
                or identity.get("arm") != arm or identity.get("validation_manifest_sha256") != manifest["manifest_sha256"]
                or path.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(directory.resolve()) or _digest(path) != artifact["sha256"]):
            raise ValueError("Selected cosine evaluation is missing, changed or mismatched.")
        report = json.loads(path.read_text(encoding="utf-8"))
        verification = report["input_verification"]
        if (report.get("schema") != "streamed-retrieval-evaluation-v1" or report.get("checkpoint") != identity
                or verification.get("status") != "VALIDATION_INPUTS_VERIFIED"
                or verification.get("manifest_sha256") != manifest["manifest_sha256"]
                or verification.get("verified_windows") != len(fingerprints["windows"])
                or verification.get("verified_full_depth_event_regions") != len(fingerprints["event_traces"])):
            raise ValueError("Selected cosine evaluation lacks verified validation inputs.")
        score = report["scorers"]["mu_cosine"]
        summary = score["summary"]
        ranking = RankingReport(
            manifest["manifest_sha256"], "mu_cosine", arm, identity["experiment_seed"],
            formal_config["random_seed"], formal_config["random_repeats"],
            tuple(QueryRanking(**row) for row in score["rows"]),
        )
        if ({row.query_id: (row.source_id, row.cohort) for row in ranking.rows} != expected
                or len(ranking.rows) != len(expected)
                or summarize_rankings(ranking, bootstrap_seed=formal_config["bootstrap_seed"],
                                      bootstrap_repeats=formal_config["bootstrap_repeats"]) != summary):
            raise ValueError("Selected cosine rankings do not match the frozen validation cohort.")
        final = results[arm]["last_epoch"]["checkpoint"]
        if any(identity[key] != final[key] for key in (
            "arm", "run_id", "experiment_seed", "config_sha256", "source_sha256",
            "target_sha256", "validation_manifest_sha256",
        )):
            raise ValueError("Selected cosine checkpoint does not belong to the matched arm.")
        key = f"{arm}_{kind}"
        rankings[key] = ranking
        sources[key] = {"checkpoint": identity, "retrieval_sha256": artifact["sha256"],
                        "ndcg_at_10": summary["ndcg_at_10"]}

    comparisons = {}
    for candidate, baseline in (
        ("S2_final", "S1_final"),
        ("S3_final", "S1_final"),
        ("S3_final", "S2_final"),
        ("S2_reconstruction", "S1_reconstruction"),
        ("S3_reconstruction", "S1_reconstruction"),
        ("S3_reconstruction", "S2_reconstruction"),
        ("S3_retrieval", "S1_reconstruction"),
        ("S3_retrieval", "S2_reconstruction"),
    ):
        comparisons[f"{candidate}_vs_{baseline}"] = compare_matched_cosine(
            rankings[candidate], rankings[baseline], bootstrap_seed=formal_config["bootstrap_seed"],
            bootstrap_repeats=formal_config["bootstrap_repeats"],
        )
    result = {
        "schema": "streamed-suite-cosine-comparisons-v1", "manifest_sha256": manifest["manifest_sha256"],
        "sources": sources, "comparisons": comparisons,
        "limitation": (
            "Validation event-coverage proxy only, not proof of geological understanding. "
            "Checkpoints can differ in epoch; retrieval-selected S3 and reconstruction-selected baselines "
            "use different selection criteria. This is not an adoption decision."
        ),
    }
    result["sha256"] = content_hash(result)
    path = initialization.evidence_path.parent / "mu_cosine_comparisons.json"
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("Cosine comparison output must be a regular file.")
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != result:
            raise ValueError("Saved cosine comparison differs from current evaluation.")
    else:
        _save_report(path.parent, path.name, result)
    return result


def run_suite_epochs(
    journals: dict[str, BatchJournal],
    runtime_factory: Callable[[str], AbstractContextManager[ArmRuntime]] | None = None, *,
    attempt_id: str, monitor_batches: dict, validate_epoch, lpips_loss_fn,
    manifest_path: Path, source_root: Path, search_manifest: dict, diagnostic_manifest: dict,
    formal_config: dict, guardrail_manifest: dict, guardrail_config: dict,
    historical: HistoricalReconstruction, matching_config: dict, hashes: dict,
    suite_initialization: SuiteInitialization,
) -> dict[str, dict]:
    """Run prepared arms in order; launch approval and resource gates remain external."""
    if (not isinstance(suite_initialization, SuiteInitialization)
            or not isinstance(suite_initialization.evidence_path, Path)
            or suite_initialization.evidence_path.name != "initialization_evidence.json"
            or not isinstance(journals, dict) or set(journals) != {"S1", "S2", "S3"}
            or (runtime_factory is not None and not callable(runtime_factory)) or not isinstance(attempt_id, str)
            or not attempt_id or any(character in attempt_id for character in "\r\n")):
        raise ValueError("A prepared suite, three journals and an explicit attempt are required.")
    metadata = verify_suite_run_metadata(
        suite_initialization.evidence_path.parent, suite_initialization.initial_checkpoints,
        matching_config=matching_config, expected_hashes=hashes,
    )
    if metadata["initialization_sha256"] != suite_initialization.evidence_sha256:
        raise ValueError("The suite must use the initialization pin recorded in suite metadata.")
    workload = _validate_suite_inputs(
        monitor_batches=monitor_batches, validate_epoch=validate_epoch, lpips_loss_fn=lpips_loss_fn,
        manifest_path=manifest_path, source_root=source_root, search_manifest=search_manifest,
        diagnostic_manifest=diagnostic_manifest, formal_config=formal_config, guardrail_manifest=guardrail_manifest,
        guardrail_config=guardrail_config, historical=historical, matching_config=matching_config, hashes=hashes,
    )
    for arm in ("S1", "S2", "S3"):
        journal = journals[arm]
        initial = suite_initialization.initial_checkpoints[arm]
        evaluation = initial.parent / "evaluation"
        if (not isinstance(journal, BatchJournal) or journal.lock_fd == -1 or journal.fields != STEP_FIELDS
                or journal.path != initial.parent / "batch_metrics.csv"
                or evaluation.is_symlink() or not evaluation.is_dir()):
            raise ValueError("Each arm requires its locked journal and regular evaluation directory.")
        _permitted_file(journal.path)

    verify_validation_preflight(
        manifest_path, source_root, expected_manifest_sha256=hashes["validation_manifest"],
        expected_target_sha256=hashes["target"],
    )
    manifest = load_manifest(manifest_path)
    if (manifest.get("manifest_sha256") != hashes["validation_manifest"]
            or manifest.get("target_implementation_sha256") != hashes["target"]):
        raise ValueError("Validation manifest changed before evaluation-input preflight.")
    root = source_root.resolve(strict=True)
    forbidden = Path("/Volumes/CrucialX9/fake_data_validation_v2")
    if "fake_data_validation_v2" in root.parts or forbidden.is_relative_to(root):
        raise ValueError("Evaluation-input preflight must exclude validation-v2.")
    groups = open_validation_sources(root, manifest)
    verify_search_manifest(manifest, groups, search_manifest)
    verify_diagnostics(
        manifest, groups, diagnostic_manifest, max_wall_seconds=formal_config["diagnostic_max_wall_seconds"],
    )
    if load_manifest(manifest_path) != manifest or source_root.resolve(strict=True) != root:
        raise ValueError("Validation manifest or source path changed during evaluation-input preflight.")
    open_validation_sources(root, manifest)
    del groups
    print(json.dumps(workload, sort_keys=True), flush=True)
    s1_initial = suite_initialization.initial_checkpoints["S1"]
    s1 = SelectedReconstruction(
        s1_initial.with_name("last.pt"), journals["S1"], manifest_path, source_root,
        search_manifest, diagnostic_manifest, formal_config, s1_initial.parent / "evaluation", "reconstruction",
    )
    results = {}
    for arm in ("S1", "S2", "S3"):
        initial = suite_initialization.initial_checkpoints[arm]
        completed = False
        context = (_arm_runtime(arm, initial, journals[arm], source_root, matching_config, hashes)
                   if runtime_factory is None else runtime_factory(arm))
        with context as runtime:
            if not isinstance(runtime, ArmRuntime) or runtime.model.arm != arm:
                raise ValueError("The runtime factory must provide the requested arm.")
            results[arm] = run_arm_epochs(
                runtime.model, runtime.optimizer, runtime.scheduler, runtime.sampler, journals[arm],
                lpips_loss_fn, runtime.writer,
                run_id=matching_config["run_id"], seed=matching_config["experiment_seed"],
                attempt_id=attempt_id, batches_per_epoch=matching_config["batches_per_epoch"],
                sparse_probability=matching_config["sparse_probability"], monitor_batches=monitor_batches,
                monitor_seed=matching_config["monitor_seed"], validation_seed=matching_config["validation_seed"],
                validate_epoch=validate_epoch, epoch_csv=initial.parent / "training_metrics.csv",
                checkpoint_path=initial.with_name("last.pt"), hashes=hashes, early_stopping=runtime.early_stopping,
                manifest_path=manifest_path, source_root=source_root, search_manifest=search_manifest,
                diagnostic_manifest=diagnostic_manifest, formal_config=formal_config,
                evaluation_dir=initial.parent / "evaluation", guardrail_manifest=guardrail_manifest,
                guardrail_config=guardrail_config, historical=historical, s1=None if arm == "S1" else s1,
                matching_config=matching_config, initial_checkpoint=initial, suite_initialization=suite_initialization,
            )
            completed = True
        if not completed:
            raise RuntimeError(f"{arm} runtime suppressed an arm failure.")
        del runtime
    results["S3"]["mu_cosine_comparisons"] = _suite_cosine_comparisons(
        results, suite_initialization, manifest, formal_config,
    )
    return results


def run_suite(
    suite_dir: Path, *, resume: bool, attempt_id: str, monitor_batches: dict | None = None,
    validate_epoch=None, lpips_loss_fn=None, epoch_validation: dict | None = None,
    manifest_path: Path, source_root: Path, search_manifest: dict, diagnostic_manifest: dict,
    formal_config: dict, guardrail_manifest: dict, guardrail_config: dict,
    historical: HistoricalReconstruction, matching_config: dict, hashes: dict,
) -> dict[str, dict]:
    """Prepare a fresh suite or reopen an existing one, then run its arms in order.

    The caller must enforce launch approval, resource limits and log capture.
    """
    if not isinstance(resume, bool) or not isinstance(attempt_id, str) or not attempt_id or any(
        character in attempt_id for character in "\r\n"
    ):
        raise ValueError("An explicit fresh/resume mode and single-line attempt ID are required.")
    if epoch_validation is not None:
        if any(value is not None for value in (monitor_batches, validate_epoch, lpips_loss_fn)):
            raise ValueError("Provide either epoch validation settings or prepared epoch inputs, not both.")
        monitor_batches, validate_epoch, lpips_loss_fn, _ = prepare_epoch_inputs(
            manifest_path=manifest_path, source_root=source_root, hashes=hashes,
            guardrail_manifest=guardrail_manifest, matching_config=matching_config,
            epoch_validation=epoch_validation,
        )
    if not isinstance(monitor_batches, dict):
        raise ValueError("Fixed monitor batches or explicit epoch validation settings are required.")
    _validate_suite_inputs(
        monitor_batches=monitor_batches, validate_epoch=validate_epoch, lpips_loss_fn=lpips_loss_fn,
        manifest_path=manifest_path, source_root=source_root, search_manifest=search_manifest,
        diagnostic_manifest=diagnostic_manifest, formal_config=formal_config, guardrail_manifest=guardrail_manifest,
        guardrail_config=guardrail_config, historical=historical, matching_config=matching_config, hashes=hashes,
    )
    if not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("The streamed suite requires MPS without CPU fallback.")
    source = source_root.resolve(strict=True)
    destination = suite_dir.resolve()
    forbidden = Path("/Volumes/CrucialX9/fake_data_validation_v2")
    if (not source.is_dir() or "fake_data_validation_v2" in source_root.parts
            or "fake_data_validation_v2" in source.parts or forbidden.is_relative_to(source)
            or "fake_data_validation_v2" in suite_dir.parts or "fake_data_validation_v2" in destination.parts
            or forbidden.is_relative_to(destination) or destination.is_relative_to(forbidden)
            or destination.is_relative_to(source) or source.is_relative_to(destination)):
        raise ValueError("Suite and source paths must be separate and outside validation-v2.")

    if not isinstance(validate_epoch, StreamedEpochValidator):
        raise ValueError("A frozen epoch validator is required to describe suite inputs.")
    record = suite_input_record(
        suite_dir, validate_epoch=validate_epoch, manifest_path=manifest_path, source_root=source_root,
        search_manifest=search_manifest, diagnostic_manifest=diagnostic_manifest, formal_config=formal_config,
        guardrail_manifest=guardrail_manifest, guardrail_config=guardrail_config, historical=historical,
        matching_config=matching_config, hashes=hashes,
    )
    if content_hash(record) != hashes["config"]:
        raise ValueError("Suite inputs do not match the frozen configuration hash.")
    sampler = None
    if not resume:
        sampler = StreamedPairSampler(source, "train", matching_config["sampler_seed"])
        if sampler.source_manifest_hash != hashes["source"]:
            raise ValueError("Training sampler does not match the frozen source manifest.")
        suite_dir.mkdir()
        _sync_directory(suite_dir.parent)
    _bind_suite_inputs(suite_dir, record, resume=resume)

    with ExitStack() as stack:
        journals = {}
        for arm in ("S1", "S2", "S3"):
            directory = suite_dir / arm
            if not resume:
                directory.mkdir()
                (directory / "evaluation").mkdir()
                _sync_directory(directory)
                _sync_directory(suite_dir)
            journal = BatchJournal(directory / "batch_metrics.csv", STEP_FIELDS, create=not resume)
            stack.callback(journal.close)
            if not resume:
                journal.reconcile(watermark=None, attempt_id=f"bootstrap:{attempt_id}",
                                  expected_step=0, expected_exposure=0)
            journals[arm] = journal
        metadata_path = suite_dir / "suite_metadata.json"
        if resume and (metadata_path.exists() or metadata_path.is_symlink()):
            checkpoints = {arm: suite_dir / arm / "initial.pt" for arm in ("S1", "S2", "S3")}
            metadata = verify_suite_run_metadata(
                suite_dir, checkpoints, matching_config=matching_config, expected_hashes=hashes,
            )
            initialization = SuiteInitialization(
                suite_dir / "initialization_evidence.json", metadata["initialization_sha256"], checkpoints,
            )
        else:
            if resume:
                if any(journal._read() or journal.rows for journal in journals.values()):
                    raise ValueError("A suite with recorded or incomplete steps cannot be reinitialized.")
                for journal in journals.values():
                    journal.reconcile(watermark=None, attempt_id=f"bootstrap:{attempt_id}",
                                      expected_step=0, expected_exposure=0)
            if sampler is None:
                sampler = StreamedPairSampler(source, "train", matching_config["sampler_seed"])
            if sampler.source_manifest_hash != hashes["source"]:
                raise ValueError("Training sampler does not match the frozen source manifest.")
            prepared = prepare_suite_initial_checkpoints(
                suite_dir, journals, sampler=sampler, device="mps", hashes=hashes,
                matching_config=matching_config, source_root=source,
            )
            initialization = SuiteInitialization(
                prepared["evidence_path"], prepared["evidence_sha256"], prepared["initial_checkpoints"],
            )
        return run_suite_epochs(
            journals, attempt_id=attempt_id, monitor_batches=monitor_batches, validate_epoch=validate_epoch,
            lpips_loss_fn=lpips_loss_fn, manifest_path=manifest_path, source_root=source,
            search_manifest=search_manifest, diagnostic_manifest=diagnostic_manifest,
            formal_config=formal_config, guardrail_manifest=guardrail_manifest, guardrail_config=guardrail_config,
            historical=historical, matching_config=matching_config, hashes=hashes,
            suite_initialization=initialization,
        )
