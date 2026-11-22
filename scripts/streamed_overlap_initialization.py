"""Publish fresh, recoverable arm checkpoints; this module never starts training."""

import hashlib
import json
from pathlib import Path
import uuid

import torch

from scripts.p7_control import capture_training_state
from scripts.streamed_overlap_checkpoint import (
    _model_device, _validate_hashes, _validate_scheduler, save_epoch_state,
)
from scripts.streamed_overlap_evaluation import _checkpoint_state
from scripts.streamed_overlap_journal import BatchJournal, STEP_FIELDS, _publish_exclusive, _sync_directory
from scripts.streamed_overlap_matching import (
    ArmEvidence, _digest, _initial_state, _permitted_file, _recovery_files, _rngs_equal,
    _validate_suite_metadata_dir, matching_config_hash, save_suite_initialization, save_suite_run_metadata,
    verify_arm_initial_checkpoint, verify_saved_suite_initialization, verify_suite_run_metadata,
)
from scripts.streamed_overlap_model import StreamedPairVAE, build_optimizer, fresh_e1_model
from scripts.train import EarlyStoppingState, streamed_inference_context


def _publish_copy(source: Path, destination: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with source.open("rb") as reader:
        def write(writer):
            while chunk := reader.read(1024 * 1024):
                writer.write(chunk)
                digest.update(chunk)

        def verify():
            if (digest.hexdigest() != expected or _digest(source) != expected
                    or _digest(destination) != expected):
                raise ValueError("Initial checkpoint changed during publication.")

        _publish_exclusive(destination, write, verify)


def prepare_arm_initial_checkpoints(
    *, model: StreamedPairVAE, optimizer, scheduler, sampler, early_stopping, journal: BatchJournal,
    hashes: dict, matching_config: dict, source_root: Path,
) -> dict:
    """Create or recover a separate initial snapshot and identical epoch-zero last checkpoint."""
    config_hash = matching_config_hash(matching_config)
    _validate_hashes(hashes)
    if hashes.get("matching") != config_hash:
        raise ValueError("The initial checkpoint must bind the matching configuration.")
    if not isinstance(model, StreamedPairVAE):
        raise ValueError("A fresh streamed E1 model is required.")
    if (journal.lock_fd == -1 or journal.fields != STEP_FIELDS or journal.position is None
            or journal.attempt_id is None or journal.position.logical_step != 0
            or journal.position.cumulative_exposure != 0):
        raise ValueError("Initialization requires an empty, reconciled, locked step journal.")
    root = journal.path.parent
    if ("fake_data_validation_v2" in root.parts or "fake_data_validation_v2" in source_root.parts
            or root.is_symlink() or "fake_data_validation_v2" in root.resolve().parts or not root.is_dir()):
        raise ValueError("Initialization requires a permitted regular run directory.")
    _permitted_file(journal.path)
    if journal.recovery_dir.is_symlink() or not journal.recovery_dir.is_dir():
        raise ValueError("Journal recovery must be in a regular run-owned directory.")
    resolved_source = source_root.resolve(strict=True)
    resolved_root = root.resolve(strict=True)
    if ("fake_data_validation_v2" in resolved_source.parts or not resolved_source.is_dir()
            or resolved_root.is_relative_to(resolved_source) or resolved_source.is_relative_to(resolved_root)):
        raise ValueError("The run directory must be separate from source data and validation-v2.")
    if journal._read() or journal.rows:
        raise ValueError("Initialization requires an empty, reconciled, locked step journal.")

    if type(optimizer) is not torch.optim.AdamW:
        raise ValueError("The approved fresh AdamW optimizer is required.")
    expected_optimizer = build_optimizer(model)
    if (optimizer.state_dict() != expected_optimizer.state_dict()
            or len(optimizer.param_groups) != len(expected_optimizer.param_groups)
            or any(len(actual["params"]) != len(expected["params"])
                   or any(left is not right for left, right in zip(actual["params"], expected["params"]))
                   for actual, expected in zip(optimizer.param_groups, expected_optimizer.param_groups))):
        raise ValueError("The fresh optimizer must use the approved parameter groups and settings.")
    _validate_scheduler(scheduler, optimizer)
    expected_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        expected_optimizer, factor=0.5, patience=6, min_lr=1e-5,
    )
    if scheduler.state_dict() != expected_scheduler.state_dict():
        raise ValueError("The reconstruction scheduler must be fresh.")
    if vars(early_stopping) != {"best_val_loss": float("inf"), "epochs_without_improvement": 0}:
        raise ValueError("The early-stopping state must be fresh.")

    device = _model_device(model)
    draft = {
        "arm": model.arm, "device": device, "completed_epoch": 0,
        "batches_per_epoch": matching_config["batches_per_epoch"], "hashes": hashes,
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "training": {"optimizer": optimizer.state_dict()}, "sampler": sampler.state_dict(),
        "journal": {"event_id": None, "logical_step": 0, "cumulative_exposure": 0}, "selection_state": {},
    }
    _initial_state(
        draft, model.arm, draft, matching_config["model_seed"], matching_config["head_seed"],
        matching_config["sampler_seed"],
    )
    initial, last = root / "initial.pt", root / "last.pt"
    initial_exists = initial.exists() or initial.is_symlink()
    last_exists = last.exists() or last.is_symlink()
    if last_exists and not initial_exists:
        raise FileExistsError("Refusing an existing last checkpoint without its initial snapshot.")
    with streamed_inference_context(model, device, matching_config["experiment_seed"]):
        expected_rng = capture_training_state(optimizer, scheduler, None, early_stopping)
        if not initial_exists:
            temporary = root / f".initial-{uuid.uuid4().hex}.pt"
            try:
                save_epoch_state(
                    temporary, model=model, optimizer=optimizer, scheduler=scheduler, sampler=sampler,
                    early_stopping=early_stopping, journal=journal, hashes=hashes, completed_epoch=0,
                    batches_per_epoch=matching_config["batches_per_epoch"], selection_state={},
                )
                state, digest = _checkpoint_state(temporary)
                _initial_state(
                    state, model.arm, draft, matching_config["model_seed"], matching_config["head_seed"],
                    matching_config["sampler_seed"],
                )
                if not _rngs_equal(state["training"], expected_rng, device):
                    raise ValueError("Initial checkpoint RNG state does not match the experiment seed.")
                _publish_copy(temporary, initial, digest)
            finally:
                if temporary.exists():
                    temporary.unlink()
                    _sync_directory(root)
        else:
            _permitted_file(initial)
            state, digest = _checkpoint_state(initial)
            _initial_state(
                state, model.arm, draft, matching_config["model_seed"], matching_config["head_seed"],
                matching_config["sampler_seed"],
            )
            if not _rngs_equal(state["training"], expected_rng, device):
                raise ValueError("Initial checkpoint RNG state does not match the experiment seed.")
    if (state["training"]["optimizer"] != optimizer.state_dict()
            or state["training"]["scheduler"] != scheduler.state_dict()
            or state["training"]["early_stopping"] != vars(early_stopping)):
        raise ValueError("Retained initial checkpoint has different optimizer, scheduler or stopping state.")
    if not last_exists:
        _publish_copy(initial, last, digest)
    _permitted_file(last)
    last_state, last_hash = _checkpoint_state(last)
    if type(last_state.get("completed_epoch")) is not int or last_state["completed_epoch"] != 0 or last_hash != digest:
        raise ValueError("Existing last checkpoint does not match the retained epoch-zero snapshot.")
    verified = verify_arm_initial_checkpoint(
        initial, last, arm=model.arm, matching_config=matching_config, expected_hashes=hashes,
    )
    if verified != digest:
        raise ValueError("Initial checkpoint changed during initialization.")
    return {
        "initial_checkpoint": initial, "checkpoint_path": last,
        "initial_checkpoint_sha256": digest, "matching_config_sha256": config_hash,
    }


def prepare_suite_initial_checkpoints(
    suite_dir: Path, journals: dict[str, BatchJournal], *, sampler, device: str,
    hashes: dict, matching_config: dict, source_root: Path,
) -> dict:
    """Prepare and pin three fresh arms without training; advanced suites must use resume."""
    _validate_suite_metadata_dir(suite_dir)
    _validate_hashes(hashes)
    if (hashes.get("matching") != matching_config_hash(matching_config)
            or not isinstance(journals, dict) or set(journals) != {"S1", "S2", "S3"}
            or device not in {"cpu", "mps"}):
        raise ValueError("A bound configuration, explicit device and three arm journals are required.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for an MPS suite initialization.")
    source = source_root.resolve(strict=True)
    root = suite_dir.resolve(strict=True)
    if ("fake_data_validation_v2" in source_root.parts or "fake_data_validation_v2" in source.parts
            or not source.is_dir() or root.is_relative_to(source) or source.is_relative_to(root)):
        raise ValueError("Suite and source directories must be separate and outside validation-v2.")

    evidence_path = suite_dir / "initialization_evidence.json"
    metadata_path = suite_dir / "suite_metadata.json"
    sealed = any(path.exists() or path.is_symlink() for path in (evidence_path, metadata_path))
    checkpoints = {}
    roots = set()
    for arm in ("S1", "S2", "S3"):
        journal = journals[arm]
        if (not isinstance(journal, BatchJournal) or journal.lock_fd == -1 or journal.fields != STEP_FIELDS
                or journal.position is None or journal.attempt_id is None or journal.position.logical_step != 0
                or journal.position.cumulative_exposure != 0):
            raise ValueError("Each arm requires an empty, reconciled, locked step journal.")
        directory = journal.path.parent
        if (directory.is_symlink() or not directory.is_dir() or directory.resolve().parent != root
                or directory.resolve() in roots or journal.path.name != "batch_metrics.csv"):
            raise ValueError("Each arm requires its own regular directory inside the suite.")
        roots.add(directory.resolve())
        _permitted_file(journal.path)
        _recovery_files(journal)
        if journal._read() or journal.rows:
            raise ValueError("Each arm requires an empty, reconciled, locked step journal.")
        initial, last = directory / "initial.pt", directory / "last.pt"
        initial_exists = initial.exists() or initial.is_symlink()
        last_exists = last.exists() or last.is_symlink()
        if (last_exists and not initial_exists) or (sealed and not (initial_exists and last_exists)):
            raise ValueError("Existing suite evidence requires all retained initial and last checkpoints.")
        if initial_exists:
            _permitted_file(initial)
            state, _ = _checkpoint_state(initial)
            if (state.get("hashes") != hashes or state.get("device") != device
                    or state.get("batches_per_epoch") != matching_config["batches_per_epoch"]):
                raise ValueError("Existing initial checkpoint does not match the requested suite.")
            _initial_state(
                state, arm, state, matching_config["model_seed"], matching_config["head_seed"],
                matching_config["sampler_seed"],
            )
        if last_exists:
            verify_arm_initial_checkpoint(
                initial, last, arm=arm, matching_config=matching_config, expected_hashes=hashes,
            )
            state, _ = _checkpoint_state(last)
            if type(state.get("completed_epoch")) is not int or state["completed_epoch"] != 0:
                raise ValueError("An advanced suite must use resume, not initialization.")
        checkpoints[arm] = initial

    if metadata_path.exists() or metadata_path.is_symlink():
        verify_suite_run_metadata(suite_dir, checkpoints, matching_config=matching_config, expected_hashes=hashes)
    elif evidence_path.exists() or evidence_path.is_symlink():
        _permitted_file(evidence_path)
        saved = json.loads(evidence_path.read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            raise ValueError("Existing suite initialization evidence is invalid.")
        pin = saved.get("sha256")
        if not isinstance(pin, str):
            raise ValueError("Existing suite initialization evidence is invalid.")
        verify_saved_suite_initialization(
            evidence_path, checkpoints, matching_config=matching_config, expected_hashes=hashes,
            expected_evidence_sha256=pin,
        )

    arms = {}
    for arm in ("S1", "S2", "S3"):
        model = fresh_e1_model(arm, matching_config["model_seed"], matching_config["head_seed"]).to(device)
        optimizer = build_optimizer(model)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=6, min_lr=1e-5)
        result = prepare_arm_initial_checkpoints(
            model=model, optimizer=optimizer, scheduler=scheduler, sampler=sampler,
            early_stopping=EarlyStoppingState(float("inf"), 0), journal=journals[arm],
            hashes=hashes, matching_config=matching_config, source_root=source_root,
        )
        arms[arm] = ArmEvidence(result["initial_checkpoint"], result["checkpoint_path"], journals[arm])
        del scheduler, optimizer, model
    saved = save_suite_initialization(
        suite_dir, arms, matching_config=matching_config, expected_hashes=hashes, source_root=source_root,
    )
    metadata = save_suite_run_metadata(
        suite_dir, checkpoints, matching_config=matching_config, expected_hashes=hashes,
        initialization_sha256=saved["evidence_sha256"],
    )
    return {
        "evidence_path": saved["path"], "evidence_sha256": saved["evidence_sha256"],
        "initial_checkpoints": checkpoints, "metadata_path": metadata["path"],
        "metadata_sha256": metadata["metadata_sha256"],
    }
