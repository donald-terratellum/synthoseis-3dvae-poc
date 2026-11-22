"""Checkpoint-bound, validation-only retrieval scoring for the opt-in experiment."""

from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch

from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_checkpoint import _validate_hashes
from scripts.streamed_overlap_integrity import (
    _digest_array, _window_record, load_manifest, open_validation_sources, verify_inputs,
)
from scripts.streamed_overlap_journal import BatchJournal, STEP_FIELDS
from scripts.streamed_overlap_manifest import _window
from scripts.streamed_overlap_metrics import _eligible_queries, calculate_rankings, summarize_rankings
from scripts.streamed_overlap_model import build_optimizer, fresh_e1_model
from scripts.streamed_overlap_targets import _read_bounded
from scripts.streamed_overlap_yield import SEISMIC_KEY
from scripts.train import streamed_inference_context
from src.tokenizer.core.model_adapter import choose_torch_device
from src.tokenizer.core.preprocess import preprocess_for_token
from src.tokenizer.core.similarity import cosine_similarity


def _checkpoint_state(path: Path) -> tuple[dict, str]:
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
        handle.seek(0)
        state = torch.load(handle, map_location="cpu", weights_only=True)
        handle.seek(0)
        after = hashlib.sha256()
        while chunk := handle.read(1024 * 1024):
            after.update(chunk)
    if digest.digest() != after.digest():
        raise ValueError("Checkpoint changed during loading.")
    if not isinstance(state, dict) or state.get("schema") != "streamed-epoch-state-v1":
        raise ValueError("A streamed epoch-state checkpoint is required.")
    return state, digest.hexdigest()


def _identity(state: dict, digest: str, journal: BatchJournal, manifest_hash: str) -> dict:
    if journal.lock_fd == -1 or journal.fields != STEP_FIELDS or journal._read():
        raise ValueError("A locked, complete step journal is required for checkpoint verification.")
    epoch, batches, arm = state.get("completed_epoch"), state.get("batches_per_epoch"), state.get("arm")
    watermark, hashes = state.get("journal"), state.get("hashes")
    if (arm not in {"S1", "S2", "S3"}
            or isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1
            or isinstance(batches, bool) or not isinstance(batches, int) or batches < 1
            or not isinstance(watermark, dict) or not isinstance(hashes, dict)
            or hashes.get("validation_manifest") != manifest_hash):
        raise ValueError("Checkpoint must contain a completed epoch for this validation manifest.")
    _validate_hashes(hashes)
    step = epoch * batches
    if watermark.get("logical_step") != step or watermark.get("cumulative_exposure") != 12 * step:
        raise ValueError("Checkpoint epoch and journal trajectory disagree.")
    matching = [row for row in journal.rows if row["event_id"] == watermark.get("event_id")]
    stale = {event for record in journal._records() for event in record.get("uncommitted_event_ids", [])}
    if len(matching) != 1 or matching[0]["event_id"] in stale:
        raise ValueError("Checkpoint watermark is missing or was discarded by recovery.")
    row = matching[0]
    if (row["arm"] != arm or row["logical_step"] != str(step)
            or row["cumulative_exposure"] != str(12 * step) or row["epoch"] != str(epoch)
            or row["batch_index"] != str(batches) or not row["seed"].isdigit()
            or not row["run_id"] or not row["attempt_id"]):
        raise ValueError("Checkpoint identity disagrees with its journal watermark.")
    sampler = state.get("sampler")
    if (not isinstance(sampler, dict) or sampler.get("split") != "train"
            or sampler.get("batches_drawn") != step
            or sampler.get("source_manifest_hash") != hashes.get("source")):
        raise ValueError("Checkpoint sampler identity disagrees with the completed trajectory.")
    return {
        "arm": arm, "run_id": row["run_id"], "attempt_id": row["attempt_id"],
        "experiment_seed": int(row["seed"]), "epoch": epoch, "step": step,
        "checkpoint_sha256": digest, "journal_event_id": row["event_id"],
        "validation_manifest_sha256": manifest_hash, "config_sha256": hashes["config"],
        "source_sha256": hashes["source"], "target_sha256": hashes["target"],
    }


def _require_trained_coverage(state: dict, model, journal: BatchJournal, identity: dict) -> None:
    if identity["arm"] != "S3" or model.coverage_head is None:
        raise ValueError("Directional scoring requires a trained S3 coverage head.")
    training = state.get("training")
    optimizer = training.get("optimizer") if isinstance(training, dict) else None
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("param_groups"), list):
        raise ValueError("Directional scoring requires saved coverage optimizer state.")
    groups = [group for group in optimizer["param_groups"] if group.get("name") == "coverage"]
    expected = list(model.coverage_head.parameters())
    slots = optimizer.get("state")
    if (len(groups) != 1 or not isinstance(slots, dict)
            or not isinstance(groups[0].get("params"), list)
            or len(groups[0]["params"]) != len(expected)
            or len(set(groups[0]["params"])) != len(expected)):
        raise ValueError("Coverage optimizer state does not match the enabled head.")
    has_nonzero_gradient = False
    for parameter_id, parameter in zip(groups[0]["params"], expected):
        slot = slots.get(parameter_id)
        if not isinstance(slot, dict):
            raise ValueError("Coverage head has not received an optimizer update.")
        step = slot.get("step")
        if (not isinstance(step, torch.Tensor) or step.numel() != 1
                or not math.isfinite(float(step.item())) or not 1 <= step.item() <= identity["step"]
                or not float(step.item()).is_integer()
                or any(not isinstance(slot.get(name), torch.Tensor)
                       or slot[name].shape != parameter.shape or not torch.isfinite(slot[name]).all()
                       for name in ("exp_avg", "exp_avg_sq"))):
            raise ValueError("Coverage head has not received a valid optimizer update.")
        if torch.any(slot["exp_avg_sq"] < 0):
            raise ValueError("Coverage optimizer variance is invalid.")
        has_nonzero_gradient |= bool(torch.any(slot["exp_avg_sq"] > 0).item())
    if not has_nonzero_gradient:
        raise ValueError("Coverage head has no recorded nonzero gradients.")
    cutoff = next(index for index, row in enumerate(journal.rows)
                  if row["event_id"] == identity["journal_event_id"])
    stale = {event for record in journal._records() for event in record.get("uncommitted_event_ids", [])}
    valid_rows = [
        row for row in journal.rows[:cutoff + 1]
        if row["event_id"] not in stale and row["valid_coverage"].isdigit()
        and int(row["valid_coverage"]) > 0 and row["coverage_loss"]
    ]
    if not valid_rows or any(not math.isfinite(float(row["coverage_loss"]))
                             or float(row["coverage_loss"]) < 0 for row in valid_rows):
        raise ValueError("No committed valid coverage update is recorded for the head.")


def _read_input(window_data: dict, groups: dict, fingerprints: dict, patch_shape: tuple[int, ...]) -> np.ndarray:
    window = _window(window_data)
    if window.shape != patch_shape or window.source_id not in groups:
        raise ValueError("Validation input window is unsupported or has an unknown source.")
    source = groups[window.source_id]
    array = source[SEISMIC_KEY]
    if len(array.shape) != 3 or any(end > size for end, size in zip(window.end, array.shape)):
        raise ValueError("Validation seismic window is outside native source support.")
    original = _read_bounded(array, window.slices)
    record = fingerprints.get(content_hash(_window_record(window)))
    if not isinstance(record, dict) or record.get("arrays", {}).get(SEISMIC_KEY) != _digest_array(original):
        raise ValueError("Validation seismic window changed after fingerprint verification.")
    return preprocess_for_token(original)


def evaluate_checkpoint(
    checkpoint_path: Path, journal: BatchJournal, manifest_path: Path, source_root: Path, *,
    device: str, inference_seed: int, encode_batch_size: int, max_wall_seconds: float,
    random_seed: int, random_repeats: int, bootstrap_seed: int, bootstrap_repeats: int,
    score_pair_head: bool,
) -> dict:
    """Score only a completed checkpoint against its verified frozen validation inputs."""
    if (isinstance(encode_batch_size, bool) or not isinstance(encode_batch_size, int) or encode_batch_size < 1
            or isinstance(max_wall_seconds, bool) or not isinstance(max_wall_seconds, (int, float))
            or not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0
            or not isinstance(score_pair_head, bool)):
        raise ValueError("Explicit positive evaluation batch size and wall-time limit are required.")
    if (any(isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (inference_seed, random_seed, bootstrap_seed))
            or inference_seed >= 2**32
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 2
                   for value in (random_repeats, bootstrap_repeats))):
        raise ValueError("Explicit valid inference, random and bootstrap configuration is required.")
    started = time.monotonic()
    manifest = load_manifest(manifest_path)
    queries = _eligible_queries(manifest)
    state, checkpoint_hash = _checkpoint_state(checkpoint_path)
    identity = _identity(state, checkpoint_hash, journal, manifest["manifest_sha256"])
    model = fresh_e1_model(identity["arm"], model_seed=0, head_seed=0)
    model.load_state_dict(state["model"], strict=True)
    if score_pair_head:
        _require_trained_coverage(state, model, journal, identity)
    # Validate that the checkpoint optimizer uses this architecture's parameter groups.
    expected_groups = build_optimizer(model).state_dict()["param_groups"]
    saved_optimizer = state.get("training", {}).get("optimizer", {})
    saved_groups = saved_optimizer.get("param_groups", [])
    if ([group.get("name") for group in saved_groups] != [group["name"] for group in expected_groups]
            or any(len(saved["params"]) != len(expected["params"])
                   for saved, expected in zip(saved_groups, expected_groups))):
        raise ValueError("Checkpoint optimizer groups do not match the model architecture.")
    actual_device = choose_torch_device(device)
    model.to(actual_device)
    groups = open_validation_sources(source_root.resolve(strict=True), manifest)
    verification = verify_inputs(manifest, groups)
    fingerprints = manifest["input_integrity"]["fingerprints"]["windows"]
    scores = {"mu_cosine": {}, "pair_head": {}}
    with streamed_inference_context(model, actual_device, inference_seed):
        for query in queries:
            windows = [query["window"], *(candidate["window"] for candidate in query["candidates"])]
            latents = []
            for offset in range(0, len(windows), encode_batch_size):
                if time.monotonic() - started > max_wall_seconds:
                    raise TimeoutError("Evaluation wall-time limit exceeded; no partial report is valid.")
                cubes = np.stack([
                    _read_input(window, groups, fingerprints, model.vae.patch_shape)
                    for window in windows[offset:offset + encode_batch_size]
                ])
                inputs = torch.from_numpy(cubes[:, None]).to(actual_device)
                mu, _ = model.vae.encoder(inputs)
                if mu.ndim != 2 or mu.shape[0] != len(cubes) or not torch.isfinite(mu).all():
                    raise ValueError("Encoder produced invalid validation mu vectors.")
                latents.append(mu)
            mu = torch.cat(latents)
            mu_a, mu_b = mu[:1].expand(len(mu) - 1, -1), mu[1:]
            values = mu.cpu().numpy()
            cosine = [cosine_similarity(values[0], candidate) for candidate in values[1:]]
            candidates = query["candidates"]
            scores["mu_cosine"][query["id"]] = dict(zip((item["id"] for item in candidates), cosine))
            if score_pair_head:
                head = model.coverage_head
                if head is None:
                    raise ValueError("Trained coverage head is unavailable.")
                prediction = head(mu_a, mu_b).cpu().tolist()
                scores["pair_head"][query["id"]] = dict(zip((item["id"] for item in candidates), prediction))
    reports = {}
    for scorer in ("mu_cosine", "pair_head") if score_pair_head else ("mu_cosine",):
        ranking = calculate_rankings(
            manifest, scores[scorer], scorer=scorer, arm=identity["arm"],
            experiment_seed=identity["experiment_seed"], random_seed=random_seed,
            random_repeats=random_repeats,
        )
        reports[scorer] = {
            "summary": summarize_rankings(ranking, bootstrap_seed=bootstrap_seed, bootstrap_repeats=bootstrap_repeats),
            "rows": [asdict(row) for row in ranking.rows],
        }
    if time.monotonic() - started > max_wall_seconds:
        raise TimeoutError("Evaluation wall-time limit exceeded; no partial report is valid.")
    return {
        "schema": "streamed-retrieval-evaluation-v1", "checkpoint": identity,
        "input_verification": verification, "scorers": reports, "inference_seed": inference_seed,
        "encode_batch_size": encode_batch_size, "max_wall_seconds": max_wall_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "limitation": "Retrieval metrics do not establish reconstruction guardrails or adoption.",
    }
