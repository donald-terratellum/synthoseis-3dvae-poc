"""Atomic epoch-state checkpoints for the opt-in streamed experiment."""

import json
import os
from pathlib import Path
import uuid

import torch

from scripts.p7_control import capture_training_state, restore_training_state
from scripts.streamed_overlap_journal import BatchJournal, _sync_directory


def _validate_scheduler(scheduler, optimizer) -> None:
    if (not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
            or scheduler.optimizer is not optimizer or scheduler.mode != "min"
            or scheduler.factor != 0.5 or scheduler.patience != 6
            or any(rate != 1e-5 for rate in scheduler.min_lrs)):
        raise ValueError("Checkpoint scheduler must match the approved reconstruction scheduler.")


def _model_device(model) -> str:
    devices = {parameter.device.type for parameter in model.parameters()}
    if len(devices) != 1:
        raise ValueError("Checkpoint model parameters must use one device type.")
    return devices.pop()


def _validate_hashes(hashes: dict[str, str]) -> None:
    if not {"config", "source", "target", "validation_manifest"} <= hashes.keys() or any(
                         not key or not isinstance(value, str) or len(value) != 64
                         or any(char not in "0123456789abcdef" for char in value)
                         for key, value in hashes.items()):
        raise ValueError("Configuration and manifest hashes must be lowercase SHA256 values.")


def save_epoch_state(
    path: Path, *, model, optimizer, scheduler, sampler, early_stopping,
    journal: BatchJournal, hashes: dict[str, str], completed_epoch: int, batches_per_epoch: int,
    selection_state: dict,
) -> None:
    if journal.position is None or completed_epoch < 0 or batches_per_epoch < 1:
        raise ValueError("A completed, reconciled epoch and journal watermark are required.")
    if completed_epoch == 0:
        if journal.rows or journal.position.logical_step != 0 or journal.position.cumulative_exposure != 0:
            raise ValueError("An initial checkpoint requires an empty journal.")
        event_id = None
    else:
        if (not journal.rows
                or journal.rows[-1]["logical_step"] != str(journal.position.logical_step)
                or journal.rows[-1]["cumulative_exposure"] != str(journal.position.cumulative_exposure)
                or journal.rows[-1].get("epoch") != str(completed_epoch)
                or journal.rows[-1].get("batch_index") != str(batches_per_epoch)
                or ("arm" in journal.rows[-1] and journal.rows[-1]["arm"] != model.arm)):
            raise ValueError("A completed, reconciled epoch and journal watermark are required.")
        event_id = journal.rows[-1]["event_id"]
    _validate_hashes(hashes)
    sampler_state = sampler.state_dict()
    if hashes["source"] != sampler_state.get("source_manifest_hash"):
        raise ValueError("Checkpoint source identity is incompatible with the sampler.")
    _validate_scheduler(scheduler, optimizer)
    json.dumps(selection_state, allow_nan=False)
    device = _model_device(model)
    training = capture_training_state(optimizer, scheduler, None, early_stopping)
    if device == "cuda":
        training["cuda_rng"] = torch.cuda.get_rng_state_all()
    state = {
        "schema": "streamed-epoch-state-v1",
        "arm": model.arm,
        "device": device,
        "completed_epoch": completed_epoch,
        "batches_per_epoch": batches_per_epoch,
        "hashes": dict(hashes),
        "model": model.state_dict(),
        "training": training,
        "sampler": sampler_state,
        "selection_state": selection_state,
        "journal": {
            "event_id": event_id,
            "logical_step": journal.position.logical_step,
            "cumulative_exposure": journal.position.cumulative_exposure,
        },
    }
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def restore_epoch_state(
    path: Path, *, model, optimizer, scheduler, sampler, early_stopping,
    journal: BatchJournal, expected_hashes: dict[str, str], expected_batches_per_epoch: int,
    attempt_id: str,
) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(state, dict) or state.get("schema") != "streamed-epoch-state-v1"
            or state.get("arm") != model.arm or state.get("device") != _model_device(model)
            or state.get("hashes") != expected_hashes
            or state.get("batches_per_epoch") != expected_batches_per_epoch):
        raise ValueError("Checkpoint model or configuration is incompatible.")
    if not isinstance(state.get("completed_epoch"), int) or state["completed_epoch"] < 0:
        raise ValueError("Checkpoint has no completed epoch.")
    sampler_state = state["sampler"]
    current_sampler = sampler.state_dict()
    if any(sampler_state.get(key) != current_sampler.get(key)
           for key in ("schema", "split", "source_manifest_hash")):
        raise ValueError("Checkpoint sampler or source manifest is incompatible.")
    _validate_scheduler(scheduler, optimizer)
    saved_scheduler = state["training"]["scheduler"]
    if (not isinstance(saved_scheduler, dict) or saved_scheduler.get("mode") != "min"
            or saved_scheduler.get("factor") != 0.5 or saved_scheduler.get("patience") != 6
            or saved_scheduler.get("min_lrs") != scheduler.min_lrs):
        raise ValueError("Checkpoint scheduler is incompatible.")
    saved_model = state["model"]
    current_model = model.state_dict()
    if (saved_model.keys() != current_model.keys()
            or any(saved_model[key].shape != current_model[key].shape
                   or saved_model[key].dtype != current_model[key].dtype for key in current_model)):
        raise ValueError("Checkpoint model state is incompatible.")
    if state["device"] == "cuda" and (
        "cuda_rng" not in state["training"]
        or len(state["training"]["cuda_rng"]) != torch.cuda.device_count()
    ):
        raise ValueError("Checkpoint CUDA RNG state is incompatible.")
    watermark = state["journal"]
    journal.reconcile(
        watermark=watermark["event_id"], attempt_id=attempt_id,
        expected_step=watermark["logical_step"],
        expected_exposure=watermark["cumulative_exposure"],
    )
    model.load_state_dict(state["model"], strict=True)
    sampler.load_state_dict(sampler_state)
    restore_training_state(state["training"], optimizer, scheduler, None, early_stopping)
    if state["device"] == "cuda":
        torch.cuda.set_rng_state_all(state["training"]["cuda_rng"])
    return {
        "completed_epoch": state["completed_epoch"],
        "selection_state": state["selection_state"],
        "journal": watermark,
    }
