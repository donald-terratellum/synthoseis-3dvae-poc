"""Checkpoint-bound, validation-only retrieval scoring for the opt-in experiment."""

from dataclasses import asdict
import hashlib
import math
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
import torch

from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_checkpoint import _validate_hashes
from scripts.streamed_overlap_integrity import (
    _digest_array, _window_record, load_manifest, open_validation_sources, verify_inputs,
)
from scripts.streamed_overlap_journal import BatchJournal, STEP_FIELDS, _sync_directory
from scripts.streamed_overlap_manifest import _window
from scripts.streamed_overlap_metrics import (
    _eligible_queries, calculate_rankings, event_coverage_diagnostics, linear_ndcg_at_10,
    physical_overlap_diagnostics, summarize_rankings,
)
from scripts.streamed_overlap_model import build_optimizer, fresh_e1_model
from scripts.streamed_overlap_targets import _read_bounded
from scripts.streamed_overlap_yield import SEISMIC_KEY
from scripts.train import build_checkpoint_payload, compute_vae_losses, streamed_inference_context
from src.tokenizer.core.model_adapter import choose_torch_device
from src.tokenizer.core.preprocess import normalize_cube_by_std, preprocess_for_token
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


def _require_trained_head(state: dict, head, journal: BatchJournal, identity: dict, name: str) -> None:
    training = state.get("training")
    optimizer = training.get("optimizer") if isinstance(training, dict) else None
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("param_groups"), list):
        raise ValueError(f"{name.capitalize()} scoring requires saved optimizer state.")
    groups = [group for group in optimizer["param_groups"] if group.get("name") == name]
    expected = list(head.parameters())
    slots = optimizer.get("state")
    if (len(groups) != 1 or not isinstance(slots, dict)
            or not isinstance(groups[0].get("params"), list)
            or len(groups[0]["params"]) != len(expected)
            or len(set(groups[0]["params"])) != len(expected)):
        raise ValueError(f"{name.capitalize()} optimizer state does not match the enabled head.")
    has_nonzero_gradient = False
    for parameter_id, parameter in zip(groups[0]["params"], expected):
        slot = slots.get(parameter_id)
        if not isinstance(slot, dict):
            raise ValueError(f"{name.capitalize()} head has not received an optimizer update.")
        step = slot.get("step")
        if (not isinstance(step, torch.Tensor) or step.numel() != 1
                or not math.isfinite(float(step.item())) or not 1 <= step.item() <= identity["step"]
                or not float(step.item()).is_integer()
                or any(not isinstance(slot.get(name), torch.Tensor)
                       or slot[name].shape != parameter.shape or not torch.isfinite(slot[name]).all()
                       for name in ("exp_avg", "exp_avg_sq"))):
            raise ValueError(f"{name.capitalize()} head has not received a valid optimizer update.")
        if torch.any(slot["exp_avg_sq"] < 0):
            raise ValueError(f"{name.capitalize()} optimizer variance is invalid.")
        has_nonzero_gradient |= bool(torch.any(slot["exp_avg_sq"] > 0).item())
    if not has_nonzero_gradient:
        raise ValueError(f"{name.capitalize()} head has no recorded nonzero gradients.")
    cutoff = next(index for index, row in enumerate(journal.rows)
                  if row["event_id"] == identity["journal_event_id"])
    stale = {event for record in journal._records() for event in record.get("uncommitted_event_ids", [])}
    valid_rows = [
        row for row in journal.rows[:cutoff + 1]
        if row["event_id"] not in stale and row[f"{name}_loss"]
        and (name != "coverage" or (row["valid_coverage"].isdigit() and int(row["valid_coverage"]) > 0))
    ]
    if not valid_rows or any(not math.isfinite(float(row[f"{name}_loss"]))
                             or float(row[f"{name}_loss"]) < 0 for row in valid_rows):
        raise ValueError(f"No committed valid {name} update is recorded for the head.")


def _require_trained_coverage(state: dict, model, journal: BatchJournal, identity: dict) -> None:
    if identity["arm"] != "S3" or model.coverage_head is None:
        raise ValueError("Directional scoring requires a trained S3 coverage head.")
    _require_trained_head(state, model.coverage_head, journal, identity, "coverage")


def _require_trained_overlap(state: dict, model, journal: BatchJournal, identity: dict) -> None:
    if identity["arm"] not in {"S2", "S3"} or model.overlap_head is None:
        raise ValueError("Physical overlap scoring requires an enabled trained head.")
    _require_trained_head(state, model.overlap_head, journal, identity, "overlap")


def _read_original(window_data: dict, groups: dict, fingerprints: dict, patch_shape: tuple[int, ...]) -> np.ndarray:
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
    return original


def _query_scores(model, query, groups, fingerprints, device, encode_batch_size, deadline, score_pair_head,
                  score_overlap=False):
    windows = [query["window"], *(candidate["window"] for candidate in query["candidates"])]
    latents = []
    originals = []
    for offset in range(0, len(windows), encode_batch_size):
        if time.monotonic() > deadline:
            raise TimeoutError("Evaluation wall-time limit exceeded; no partial report is valid.")
        batch_originals = [
            _read_original(window, groups, fingerprints, model.vae.patch_shape)
            for window in windows[offset:offset + encode_batch_size]
        ]
        if score_overlap:
            originals.extend(batch_originals)
        cubes = np.stack([preprocess_for_token(cube) for cube in batch_originals])
        inputs = torch.from_numpy(cubes[:, None]).to(device)
        mu, _ = model.vae.encoder(inputs)
        if mu.ndim != 2 or mu.shape[0] != len(cubes) or not torch.isfinite(mu).all():
            raise ValueError("Encoder produced invalid validation mu vectors.")
        latents.append(mu)
    mu = torch.cat(latents)
    values = mu.cpu().numpy()
    cosine = [cosine_similarity(values[0], candidate) for candidate in values[1:]]
    pair_scores = None
    if score_pair_head:
        head = model.coverage_head
        if head is None:
            raise ValueError("Trained coverage head is unavailable.")
        pair_scores = head(mu[:1].expand(len(mu) - 1, -1), mu[1:]).cpu().tolist()
    overlap_scores = waveform = None
    if score_overlap:
        head = model.overlap_head
        if head is None:
            raise ValueError("Trained overlap head is unavailable.")
        overlap_scores = head(mu[:1].expand(len(mu) - 1, -1), mu[1:]).cpu().tolist()
        vectors = [np.asarray(cube, dtype=np.float64).ravel() for cube in originals]
        centered = [vector - vector.mean() for vector in vectors]
        norms = [np.linalg.norm(vector) for vector in centered]
        if any(not math.isfinite(float(norm)) or norm == 0 for norm in norms):
            raise ValueError("Waveform baseline requires finite nonconstant original windows.")
        waveform = [float((np.clip(np.dot(centered[0], candidate) / (norms[0] * norm), -1, 1) + 1) / 2)
                    for candidate, norm in zip(centered[1:], norms[1:])]
    return cosine, pair_scores, overlap_scores, waveform


class StreamedEpochValidator:
    """Fixed validation-only reconstruction and mu-cosine callback for the epoch helper."""

    def __init__(
        self, manifest_path: Path, source_root: Path, *, manifest_sha256: str,
        reconstruction_query_ids: tuple[str, ...], reconstruction_seed: int,
        encode_batch_size: int, reconstruction_batch_size: int, max_wall_seconds: float,
        lpips_loss_fn,
    ):
        if (not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64
                or any(char not in "0123456789abcdef" for char in manifest_sha256)
                or not isinstance(reconstruction_query_ids, tuple) or not reconstruction_query_ids
                or any(not isinstance(value, str) or not value for value in reconstruction_query_ids)
                or len(set(reconstruction_query_ids)) != len(reconstruction_query_ids)
                or isinstance(reconstruction_seed, bool) or not isinstance(reconstruction_seed, int)
                or not 0 <= reconstruction_seed < 2**32
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 1
                       for value in (encode_batch_size, reconstruction_batch_size))
                or isinstance(max_wall_seconds, bool) or not isinstance(max_wall_seconds, (int, float))
                or not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0
                or not callable(lpips_loss_fn)):
            raise ValueError("Fixed validation identities, batch sizes, seed and wall-time limit are required.")
        self.manifest_path = manifest_path
        self.source_root = source_root
        self.manifest_sha256 = manifest_sha256
        self.reconstruction_query_ids = reconstruction_query_ids
        self.reconstruction_seed = reconstruction_seed
        self.encode_batch_size = encode_batch_size
        self.reconstruction_batch_size = reconstruction_batch_size
        self.max_wall_seconds = float(max_wall_seconds)
        self.lpips_loss_fn = lpips_loss_fn
        self.config_sha256 = self.configuration_hash()

    def configuration_hash(self) -> str:
        return content_hash({
            "schema": "streamed-epoch-validation-v1", "manifest_sha256": self.manifest_sha256,
            "reconstruction_query_ids": self.reconstruction_query_ids, "reconstruction_seed": self.reconstruction_seed,
            "encode_batch_size": self.encode_batch_size, "reconstruction_batch_size": self.reconstruction_batch_size,
            "max_wall_seconds": self.max_wall_seconds,
        })

    def __call__(self, model) -> dict:
        if self.configuration_hash() != self.config_sha256:
            raise ValueError("Epoch validation configuration changed after it was frozen.")
        if model.training or torch.is_grad_enabled():
            raise ValueError("Epoch validation requires eval mode and inference-only execution.")
        if isinstance(self.lpips_loss_fn, torch.nn.Module) and self.lpips_loss_fn.training:
            raise ValueError("Epoch validation LPIPS module must be in eval mode.")
        started = time.monotonic()
        deadline = started + self.max_wall_seconds
        manifest = load_manifest(self.manifest_path)
        if manifest["manifest_sha256"] != self.manifest_sha256:
            raise ValueError("Epoch validation manifest does not match the frozen configuration.")
        queries = _eligible_queries(manifest)
        by_id = {query["id"]: query for query in queries}
        if any(query_id not in by_id for query_id in self.reconstruction_query_ids):
            raise ValueError("Every reconstruction query must belong to the frozen eligible manifest.")
        groups = open_validation_sources(self.source_root.resolve(strict=True), manifest)
        verify_inputs(manifest, groups)
        fingerprints = manifest["input_integrity"]["fingerprints"]["windows"]
        devices = {parameter.device for parameter in model.parameters()}
        if len(devices) != 1:
            raise ValueError("Epoch validation requires a model on one device.")
        device = next(iter(devices))
        rankings = []
        for query in queries:
            cosine, _, _, _ = _query_scores(
                model, query, groups, fingerprints, device, self.encode_batch_size, deadline, False,
            )
            candidates = query["candidates"]
            rankings.append(linear_ndcg_at_10(
                [candidate["target"]["coverage"] for candidate in candidates],
                cosine, [candidate["id"] for candidate in candidates],
            ))
        generator = torch.Generator(device="cpu").manual_seed(self.reconstruction_seed)
        noise = torch.randn((len(self.reconstruction_query_ids), model.vae.latent_dim), generator=generator)
        sums = {"reconstruction_mae": 0.0, "lpips_loss": 0.0, "kl": 0.0}
        for start in range(0, len(self.reconstruction_query_ids), self.reconstruction_batch_size):
            if time.monotonic() > deadline:
                raise TimeoutError("Epoch validation wall-time limit exceeded; no partial report is valid.")
            ids = self.reconstruction_query_ids[start:start + self.reconstruction_batch_size]
            originals = [
                _read_original(by_id[query_id]["window"], groups, fingerprints, model.vae.patch_shape)
                for query_id in ids
            ]
            inputs = torch.from_numpy(np.stack([preprocess_for_token(cube) for cube in originals])[:, None]).to(device)
            targets = torch.from_numpy(np.stack([normalize_cube_by_std(cube) for cube in originals])[:, None]).to(device)
            mu, logvar = model.vae.encoder(inputs)
            if (not torch.isfinite(mu).all() or not torch.isfinite(logvar).all()
                    or mu.shape != (len(ids), model.vae.latent_dim) or logvar.shape != mu.shape):
                raise ValueError("Epoch validation encoder produced invalid latent parameters.")
            latent = mu + (0.5 * logvar).exp() * noise[start:start + len(ids)].to(device)
            reconstruction, _ = model.vae.decoder(latent)
            if reconstruction.shape != targets.shape or not torch.isfinite(reconstruction).all():
                raise ValueError("Epoch validation reconstruction is non-finite or has the wrong shape.")
            _, mae, kl, lpips, _ = compute_vae_losses(
                reconstruction, targets, mu, logvar, kl_weight=0.001,
                rec_loss_fn=torch.nn.functional.l1_loss, lpips_loss_fn=self.lpips_loss_fn, lpips_weight=0.1,
            )
            for name, value in (("reconstruction_mae", mae), ("lpips_loss", lpips), ("kl", kl)):
                scalar = float(value.item())
                if not math.isfinite(scalar) or scalar < 0:
                    raise ValueError(f"Epoch validation {name} must be finite and nonnegative.")
                sums[name] += len(ids) * scalar
        if time.monotonic() > deadline:
            raise TimeoutError("Epoch validation wall-time limit exceeded; no partial report is valid.")
        return {
            "split": "validation", "manifest_sha256": self.manifest_sha256,
            "epoch_validation_sha256": self.config_sha256,
            **{name: value / len(self.reconstruction_query_ids) for name, value in sums.items()},
            "metrics": {"mu_cosine_ndcg10": float(np.mean(rankings))},
        }


def export_tokenizer_checkpoint(
    checkpoint_path: Path, journal: BatchJournal, manifest_path: Path, output_path: Path,
) -> dict:
    """Export a completed streamed VAE for explicit tokenizer mu-mode evaluation."""
    manifest = load_manifest(manifest_path)
    _eligible_queries(manifest)
    state, checkpoint_hash = _checkpoint_state(checkpoint_path)
    identity = _identity(state, checkpoint_hash, journal, manifest["manifest_sha256"])
    model = fresh_e1_model(identity["arm"], model_seed=0, head_seed=0)
    model.load_state_dict(state["model"], strict=True)
    payload = build_checkpoint_payload(model.vae, epoch=identity["epoch"])
    payload["streamed_origin"] = identity
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        digest = hashlib.sha256()
        with temporary.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        with temporary.open("rb") as source, output_path.open("xb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
        _sync_directory(output_path.parent)
        written = hashlib.sha256()
        with output_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                written.update(chunk)
        if written.digest() != digest.digest():
            raise ValueError("Exported tokenizer checkpoint differs from its source bytes.")
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "checkpoint": identity, "tokenizer_checkpoint_sha256": digest.hexdigest(),
        "embedding_mode": "mu", "latent_mode": "vae",
    }


def _validate_evaluation_parameters(
    *, inference_seed: int, encode_batch_size: int, max_wall_seconds: float,
    random_seed: int, random_repeats: int, bootstrap_seed: int, bootstrap_repeats: int,
) -> None:
    if (isinstance(encode_batch_size, bool) or not isinstance(encode_batch_size, int) or encode_batch_size < 1
            or isinstance(max_wall_seconds, bool) or not isinstance(max_wall_seconds, (int, float))
            or not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0):
        raise ValueError("Explicit positive evaluation batch size and wall-time limit are required.")
    if (any(isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (inference_seed, random_seed, bootstrap_seed))
            or inference_seed >= 2**32
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 2
                   for value in (random_repeats, bootstrap_repeats))):
        raise ValueError("Explicit valid inference, random and bootstrap configuration is required.")


def evaluate_checkpoint(
    checkpoint_path: Path, journal: BatchJournal, manifest_path: Path, source_root: Path, *,
    device: str, inference_seed: int, encode_batch_size: int, max_wall_seconds: float,
    random_seed: int, random_repeats: int, bootstrap_seed: int, bootstrap_repeats: int,
    score_pair_head: bool, score_overlap: bool = False,
) -> dict:
    """Score only a completed checkpoint against its verified frozen validation inputs."""
    if not isinstance(score_pair_head, bool) or not isinstance(score_overlap, bool):
        raise ValueError("Explicit positive evaluation batch size and wall-time limit are required.")
    _validate_evaluation_parameters(
        inference_seed=inference_seed, encode_batch_size=encode_batch_size, max_wall_seconds=max_wall_seconds,
        random_seed=random_seed, random_repeats=random_repeats,
        bootstrap_seed=bootstrap_seed, bootstrap_repeats=bootstrap_repeats,
    )
    started = time.monotonic()
    manifest = load_manifest(manifest_path)
    queries = _eligible_queries(manifest)
    state, checkpoint_hash = _checkpoint_state(checkpoint_path)
    identity = _identity(state, checkpoint_hash, journal, manifest["manifest_sha256"])
    model = fresh_e1_model(identity["arm"], model_seed=0, head_seed=0)
    model.load_state_dict(state["model"], strict=True)
    if score_pair_head:
        _require_trained_coverage(state, model, journal, identity)
    if score_overlap:
        _require_trained_overlap(state, model, journal, identity)
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
    scores = {"mu_cosine": {}, "pair_head": {}, "overlap": {}, "waveform": {}}
    with streamed_inference_context(model, actual_device, inference_seed):
        for query in queries:
            cosine, prediction, overlap, waveform = _query_scores(
                model, query, groups, fingerprints, actual_device, encode_batch_size,
                started + max_wall_seconds, score_pair_head, score_overlap,
            )
            candidates = query["candidates"]
            scores["mu_cosine"][query["id"]] = dict(zip((item["id"] for item in candidates), cosine))
            if score_pair_head:
                if prediction is None:
                    raise ValueError("Directional coverage scores are unavailable.")
                scores["pair_head"][query["id"]] = dict(zip((item["id"] for item in candidates), prediction))
            if score_overlap:
                if overlap is None or waveform is None:
                    raise ValueError("Physical overlap or waveform scores are unavailable.")
                scores["overlap"][query["id"]] = dict(zip((item["id"] for item in candidates), overlap))
                scores["waveform"][query["id"]] = dict(zip((item["id"] for item in candidates), waveform))
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
        if scorer == "pair_head":
            reports[scorer]["event_coverage_diagnostics"] = event_coverage_diagnostics(manifest, scores[scorer])
    overlap_report = None
    if score_overlap:
        overlap_report = physical_overlap_diagnostics(
            manifest, scores["overlap"], constant=0.0, waveform_predictions=scores["waveform"],
        )
        overlap_report["waveform_protocol"] = (
            "Mean-centered Pearson correlation of original same-index XYZ windows, mapped from [-1, 1] to [0, 1]."
        )
    elapsed = time.monotonic() - started
    if elapsed > max_wall_seconds:
        raise TimeoutError("Evaluation wall-time limit exceeded; no partial report is valid.")
    result = {
        "schema": "streamed-retrieval-evaluation-v1", "checkpoint": identity,
        "input_verification": verification, "scorers": reports, "inference_seed": inference_seed,
        "encode_batch_size": encode_batch_size, "max_wall_seconds": max_wall_seconds,
        "elapsed_seconds": elapsed,
        "limitation": "Retrieval metrics do not establish reconstruction guardrails or adoption.",
    }
    if overlap_report is not None:
        result["physical_overlap_diagnostics"] = overlap_report
    return result
