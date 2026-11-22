"""Publish fresh, recoverable arm checkpoints; this module never starts training."""

import hashlib
import os
from pathlib import Path
import uuid

import torch

from scripts.streamed_overlap_checkpoint import (
    _model_device, _validate_hashes, _validate_scheduler, save_epoch_state,
)
from scripts.streamed_overlap_evaluation import _checkpoint_state
from scripts.streamed_overlap_journal import BatchJournal, STEP_FIELDS, _sync_directory
from scripts.streamed_overlap_matching import (
    _digest, _initial_state, _permitted_file, matching_config_hash, verify_arm_initial_checkpoint,
)
from scripts.streamed_overlap_model import StreamedPairVAE, build_optimizer


def _publish_copy(source: Path, destination: Path, expected: str) -> None:
    created = None
    complete = False
    try:
        digest = hashlib.sha256()
        with source.open("rb") as reader, destination.open("xb") as writer:
            created = os.fstat(writer.fileno())
            while chunk := reader.read(1024 * 1024):
                writer.write(chunk)
                digest.update(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        if (digest.hexdigest() != expected or _digest(source) != expected
                or _digest(destination) != expected):
            raise ValueError("Initial checkpoint changed during publication.")
        _sync_directory(destination.parent)
        complete = True
    finally:
        if created is not None and not complete and destination.exists() and not destination.is_symlink():
            current = destination.stat()
            if (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino):
                destination.unlink()
                _sync_directory(destination.parent)


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
