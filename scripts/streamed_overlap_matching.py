"""Verify recorded matched-arm initialization and input streams, without launching training."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from scripts.p7_control import capture_training_state
from scripts.streamed_overlap_audit import HOLDOUT_NAMES, PATCH_SHAPE, content_hash
from scripts.streamed_overlap_checkpoint import _validate_hashes
from scripts.streamed_overlap_evaluation import _checkpoint_state, _identity
from scripts.streamed_overlap_journal import BatchJournal, STEP_FIELDS, _publish_exclusive
from scripts.streamed_overlap_model import build_optimizer, fresh_e1_model
from scripts.train import EarlyStoppingState, streamed_inference_context


@dataclass(frozen=True)
class ArmEvidence:
    initial_checkpoint: Path
    final_checkpoint: Path
    journal: BatchJournal


def matching_config_hash(config: dict) -> str:
    required = {
        "schema", "run_id", "experiment_seed", "model_seed", "head_seed", "sampler_seed",
        "monitor_seed", "validation_seed", "final_epoch", "batches_per_epoch", "sparse_probability",
    }
    if (not isinstance(config, dict) or set(config) != required
            or config["schema"] != "streamed-matching-config-v1"
            or not isinstance(config["run_id"], str) or not config["run_id"]
            or any(character in config["run_id"] for character in "\r\n")
            or any(isinstance(config[key], bool) or not isinstance(config[key], int)
                   or not 0 <= config[key] < 2**32
                   for key in ("experiment_seed", "model_seed", "head_seed", "sampler_seed",
                               "monitor_seed", "validation_seed"))
            or any(isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1
                   for key in ("final_epoch", "batches_per_epoch"))
            or isinstance(config["sparse_probability"], bool)
            or not isinstance(config["sparse_probability"], (int, float))
            or not math.isfinite(config["sparse_probability"]) or not 0 <= config["sparse_probability"] <= 1):
        raise ValueError("An explicit valid matching configuration is required.")
    return content_hash(config)


def _permitted_file(path: Path) -> None:
    if ("fake_data_validation_v2" in path.parts or "fake_data_validation_v2" in path.resolve().parts
            or path.is_symlink() or not path.is_file()):
        raise ValueError(f"Evidence must be a regular file outside validation-v2: {path}")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _recovery_files(journal: BatchJournal) -> dict[Path, str]:
    directory = journal.recovery_dir
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Recovery records must be in a regular run-owned directory.")
    result = {}
    for path in directory.glob("*.json"):
        _permitted_file(path)
        result[path] = _digest(path)
    return result


def _rngs_equal(left: dict, right: dict, device: str) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    tensor_keys = ("torch_rng", "mps_rng") if device == "mps" else ("torch_rng",)
    return (
        all(isinstance(left.get(key), torch.Tensor) and isinstance(right.get(key), torch.Tensor)
            and torch.equal(left[key], right[key]) for key in tensor_keys)
        and left.get("python_rng") is not None and left.get("python_rng") == right.get("python_rng")
        and left.get("numpy_rng") is not None and left.get("numpy_rng") == right.get("numpy_rng")
    )


def _initial_state(state: dict, arm: str, final: dict, model_seed: int, head_seed: int, sampler_seed: int) -> None:
    if (state.get("arm") != arm or state.get("device") not in {"cpu", "mps"}
            or state.get("device") != final.get("device")
            or type(state.get("completed_epoch")) is not int or state["completed_epoch"] != 0
            or state.get("batches_per_epoch") != final.get("batches_per_epoch")
            or state.get("hashes") != final.get("hashes")
            or state.get("journal") != {"event_id": None, "logical_step": 0, "cumulative_exposure": 0}
            or state.get("selection_state") != {}):
        raise ValueError(f"{arm} initial checkpoint is not bound to its final trajectory.")
    hashes = state["hashes"]
    _validate_hashes(hashes)
    streams = np.random.SeedSequence(sampler_seed).spawn(2)
    expected_sampler = {
        "schema": "streamed-pair-sampler-v1", "split": "train",
        "source_manifest_hash": hashes["source"], "batches_drawn": 0,
        "pair_rng": np.random.default_rng(streams[0]).bit_generator.state,
        "augmentation_rng": np.random.default_rng(streams[1]).bit_generator.state,
    }
    if state.get("sampler") != expected_sampler:
        raise ValueError(f"{arm} initial sampler does not match the specified stream seed.")
    training = state.get("training")
    if (not isinstance(training, dict) or not isinstance(training.get("optimizer"), dict)
            or training["optimizer"].get("state") != {}):
        raise ValueError(f"{arm} initial optimizer must have no updates.")
    actual = state.get("model")
    expected = fresh_e1_model(arm, model_seed=model_seed, head_seed=head_seed).state_dict()
    if (not isinstance(actual, dict) or actual.keys() != expected.keys()
            or any(not isinstance(actual[key], torch.Tensor)
                   or actual[key].dtype != expected[key].dtype
                   or actual[key].shape != expected[key].shape
                   or not torch.equal(actual[key], expected[key]) for key in expected)):
        raise ValueError(f"{arm} initial model is not the specified fresh E1 initialization.")


def verify_arm_initial_checkpoint(
    initial_checkpoint: Path, checkpoint_path: Path, *, arm: str, matching_config: dict, expected_hashes: dict,
) -> str:
    """Verify the retained fresh checkpoint before an arm restores or advances."""
    config_hash = matching_config_hash(matching_config)
    if arm not in {"S1", "S2", "S3"} or expected_hashes.get("matching") != config_hash:
        raise ValueError("Initial verification requires the bound arm and matching configuration.")
    for path in (initial_checkpoint, checkpoint_path):
        _permitted_file(path)
    if initial_checkpoint.samefile(checkpoint_path):
        raise ValueError("The initial checkpoint must be retained separately from last.pt.")
    initial, initial_hash = _checkpoint_state(initial_checkpoint)
    current, current_hash = _checkpoint_state(checkpoint_path)
    if (current.get("arm") != arm or current.get("hashes") != expected_hashes
            or type(current.get("completed_epoch")) is not int or current["completed_epoch"] < 0
            or current.get("batches_per_epoch") != matching_config["batches_per_epoch"]):
        raise ValueError("Current checkpoint does not match the arm configuration.")
    _initial_state(
        initial, arm, current, matching_config["model_seed"], matching_config["head_seed"],
        matching_config["sampler_seed"],
    )
    if current.get("completed_epoch") == 0 and initial_hash != current_hash:
        raise ValueError("An epoch-zero last checkpoint must match the retained initial checkpoint.")
    for path, digest in ((initial_checkpoint, initial_hash), (checkpoint_path, current_hash)):
        _permitted_file(path)
        if _digest(path) != digest:
            raise ValueError("Initial or current checkpoint changed during verification.")
    return initial_hash


def verify_suite_initialization(
    arms: dict[str, ArmEvidence], *, matching_config: dict, expected_hashes: dict,
) -> dict:
    """Verify three fresh, unadvanced arms without starting or authorizing training."""
    config_hash = matching_config_hash(matching_config)
    _validate_hashes(expected_hashes)
    if expected_hashes.get("matching") != config_hash:
        raise ValueError("Suite initialization must bind the matching configuration.")
    matching_config = dict(matching_config)
    expected_hashes = dict(expected_hashes)
    if matching_config_hash(matching_config) != config_hash:
        raise ValueError("Matching configuration changed during verification.")
    if not isinstance(arms, dict) or set(arms) != {"S1", "S2", "S3"}:
        raise ValueError("Suite initialization requires S1, S2 and S3 evidence.")

    def check_journal(journal: BatchJournal) -> None:
        if (journal.lock_fd == -1 or journal.fields != STEP_FIELDS or journal.position is None
                or journal.attempt_id is None or journal.position.logical_step != 0
                or journal.position.cumulative_exposure != 0 or journal.rows or journal._read()):
            raise ValueError("Each arm requires an empty, reconciled, locked step journal.")

    identities = {}
    initial_training = None
    initial_device = None
    roots = set()
    files = []
    recoveries = []
    for arm in ("S1", "S2", "S3"):
        evidence = arms[arm]
        if not isinstance(evidence, ArmEvidence) or not isinstance(evidence.journal, BatchJournal):
            raise ValueError("Each arm requires checkpoint and locked journal evidence.")
        journal = evidence.journal
        root = journal.path.parent
        if ("fake_data_validation_v2" in root.parts or root.is_symlink()
                or "fake_data_validation_v2" in root.resolve().parts or not root.is_dir()
                or evidence.initial_checkpoint.parent != root or evidence.final_checkpoint.parent != root
                or evidence.initial_checkpoint.name != "initial.pt" or evidence.final_checkpoint.name != "last.pt"
                or root.resolve() in roots):
            raise ValueError("Each arm requires its own regular journal and checkpoint directory.")
        roots.add(root.resolve())
        _permitted_file(journal.path)
        recovery = _recovery_files(journal)
        check_journal(journal)
        journal_hash = _digest(journal.path)
        recoveries.append((journal, recovery))
        initial_hash = verify_arm_initial_checkpoint(
            evidence.initial_checkpoint, evidence.final_checkpoint, arm=arm,
            matching_config=matching_config, expected_hashes=expected_hashes,
        )
        state, state_hash = _checkpoint_state(evidence.initial_checkpoint)
        last, last_hash = _checkpoint_state(evidence.final_checkpoint)
        if (state_hash != initial_hash or last_hash != initial_hash
                or type(last.get("completed_epoch")) is not int or last["completed_epoch"] != 0):
            raise ValueError("Each arm must retain identical epoch-zero checkpoints.")

        device = state["device"]
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is required to verify an MPS initialization.")
        model = fresh_e1_model(arm, matching_config["model_seed"], matching_config["head_seed"]).to(device)
        optimizer = build_optimizer(model)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=6, min_lr=1e-5,
        )
        stopping = EarlyStoppingState(float("inf"), 0)
        with streamed_inference_context(model, device, matching_config["experiment_seed"]):
            expected = capture_training_state(optimizer, scheduler, None, stopping)
        training = state["training"]
        if (training.get("optimizer") != expected["optimizer"]
                or training.get("scheduler") != expected["scheduler"]
                or training.get("early_stopping") != expected["early_stopping"]
                or not _rngs_equal(training, expected, device)):
            raise ValueError("Initial optimizer, scheduler, stopping or seeded RNG state is incompatible.")
        if initial_training is not None and (
            device != initial_device or not _rngs_equal(training, initial_training, device)
        ):
            raise ValueError("Suite arms have different initial devices or RNG states.")
        initial_training, initial_device = training, device
        identities[arm] = initial_hash
        files.extend(((evidence.initial_checkpoint, initial_hash),
                      (evidence.final_checkpoint, last_hash), (journal.path, journal_hash)))

    for path, digest in files:
        _permitted_file(path)
        if _digest(path) != digest:
            raise ValueError("Suite initialization evidence changed during verification.")
    for journal, recovery in recoveries:
        if _recovery_files(journal) != recovery:
            raise ValueError("Suite journal recovery records changed during verification.")
        _permitted_file(journal.path)
        check_journal(journal)
    if matching_config_hash(matching_config) != config_hash:
        raise ValueError("Matching configuration changed during verification.")
    report = {
        "schema": "streamed-suite-initialization-v1",
        "status": "MATCHED_INITIALIZATION_ONLY",
        "matching_config_sha256": config_hash,
        "initial_checkpoint_sha256": identities,
        "device": initial_device,
        "limitation": (
            "Fresh initialization only; does not verify subsequent streams, source data, "
            "the full suite configuration or authorize training."
        ),
    }
    report["sha256"] = content_hash(report)
    return report


def verify_saved_suite_initialization(
    path: Path, initial_checkpoints: dict[str, Path], *, matching_config: dict,
    expected_hashes: dict, expected_evidence_sha256: str,
) -> dict:
    """Verify pinned initialization evidence and retained snapshots without requiring epoch-zero last files."""
    config_hash = matching_config_hash(matching_config)
    _validate_hashes(expected_hashes)
    if (expected_hashes.get("matching") != config_hash
            or not isinstance(expected_evidence_sha256, str) or len(expected_evidence_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_evidence_sha256)
            or not isinstance(initial_checkpoints, dict) or set(initial_checkpoints) != {"S1", "S2", "S3"}):
        raise ValueError("Pinned evidence, matching configuration and three initial checkpoints are required.")
    _permitted_file(path)
    raw = path.read_bytes()
    file_hash = hashlib.sha256(raw).hexdigest()
    record = json.loads(raw)
    if (not isinstance(record, dict)
            or set(record) != {"schema", "matching_config", "checkpoint_hashes", "initialization", "sha256"}
            or record["schema"] != "streamed-suite-initialization-evidence-v1"
            or record["sha256"] != expected_evidence_sha256
            or record["sha256"] != content_hash({key: value for key, value in record.items() if key != "sha256"})
            or record["matching_config"] != matching_config or record["checkpoint_hashes"] != expected_hashes):
        raise ValueError("Saved suite initialization evidence is invalid or does not match the pinned configuration.")
    report = record["initialization"]
    if (not isinstance(report, dict) or report.get("schema") != "streamed-suite-initialization-v1"
            or report.get("status") != "MATCHED_INITIALIZATION_ONLY"
            or report.get("sha256") != content_hash({key: value for key, value in report.items() if key != "sha256"})
            or report.get("matching_config_sha256") != config_hash
            or not isinstance(report.get("initial_checkpoint_sha256"), dict)
            or set(report["initial_checkpoint_sha256"]) != {"S1", "S2", "S3"}):
        raise ValueError("Saved suite initialization report is invalid.")
    root = path.parent.resolve(strict=True)
    roots = set()
    snapshots = []
    for arm in ("S1", "S2", "S3"):
        initial = initial_checkpoints[arm]
        _permitted_file(initial)
        if (initial.name != "initial.pt" or initial.parent.is_symlink()
                or initial.parent.resolve().parent != root or initial.parent.resolve() in roots):
            raise ValueError("Retained initial checkpoints must be in separate suite arm directories.")
        roots.add(initial.parent.resolve())
        state, digest = _checkpoint_state(initial)
        if (digest != report["initial_checkpoint_sha256"][arm]
                or state.get("hashes") != expected_hashes
                or state.get("batches_per_epoch") != matching_config["batches_per_epoch"]
                or state.get("device") != report.get("device")):
            raise ValueError("Retained initial checkpoint does not match saved suite evidence.")
        _initial_state(
            state, arm, state, matching_config["model_seed"], matching_config["head_seed"],
            matching_config["sampler_seed"],
        )
        snapshots.append((initial, digest))
    for initial, digest in snapshots:
        _permitted_file(initial)
        if _digest(initial) != digest:
            raise ValueError("Retained initial checkpoint changed during verification.")
    _permitted_file(path)
    if _digest(path) != file_hash:
        raise ValueError("Suite initialization evidence changed during verification.")
    return record


def save_suite_initialization(
    suite_dir: Path, arms: dict[str, ArmEvidence], *, matching_config: dict,
    expected_hashes: dict, source_root: Path,
) -> dict:
    """Persist or reverify fresh-suite evidence without overwriting existing files."""
    if ("fake_data_validation_v2" in suite_dir.parts or "fake_data_validation_v2" in source_root.parts
            or suite_dir.is_symlink()):
        raise ValueError("Suite initialization requires directories outside validation-v2.")
    root = suite_dir.resolve(strict=True)
    source = source_root.resolve(strict=True)
    if ("fake_data_validation_v2" in root.parts or "fake_data_validation_v2" in source.parts
            or not root.is_dir() or not source.is_dir()
            or root.is_relative_to(source) or source.is_relative_to(root)):
        raise ValueError("Suite and source directories must be separate and outside validation-v2.")
    report = verify_suite_initialization(arms, matching_config=matching_config, expected_hashes=expected_hashes)
    checkpoints = {arm: arms[arm].initial_checkpoint for arm in ("S1", "S2", "S3")}
    if any(path.parent.resolve().parent != root for path in checkpoints.values()):
        raise ValueError("Each arm directory must be directly inside the suite directory.")
    record = {
        "schema": "streamed-suite-initialization-evidence-v1",
        "matching_config": dict(matching_config),
        "checkpoint_hashes": dict(expected_hashes),
        "initialization": report,
    }
    record["sha256"] = content_hash(record)
    path = suite_dir / "initialization_evidence.json"
    if not path.exists() and not path.is_symlink():
        payload = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        def write(handle):
            handle.write(payload)

        _publish_exclusive(path, write)
    saved = verify_saved_suite_initialization(
        path, checkpoints, matching_config=matching_config, expected_hashes=expected_hashes,
        expected_evidence_sha256=record["sha256"],
    )
    if saved != record or verify_suite_initialization(
        arms, matching_config=matching_config, expected_hashes=expected_hashes,
    ) != report:
        raise ValueError("Suite initialization evidence changed during publication.")
    return {"path": path, "evidence_sha256": record["sha256"]}


def _valid_pair(pair: object, category: str) -> bool:
    if (not isinstance(pair, dict) or set(pair) != {"category", "a", "b", "augmentation_seed_a", "augmentation_seed_b"}
            or pair["category"] != category):
        return False
    for key in ("augmentation_seed_a", "augmentation_seed_b"):
        value = pair[key]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
            return False
    for key in ("a", "b"):
        window = pair[key]
        if not isinstance(window, dict) or set(window) != {"source_id", "origin", "shape"}:
            return False
        source, origin = window["source_id"], window["origin"]
        if (not isinstance(source, str) or not source.startswith("train/")
                or any(part in HOLDOUT_NAMES or part in {"", ".", "..", "fake_data_validation_v2"}
                       for part in source.split("/"))
                or not isinstance(origin, list) or len(origin) != 3
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in origin)
                or not isinstance(window["shape"], list)
                or any(type(value) is not int for value in window["shape"])
                or window["shape"] != list(PATCH_SHAPE)):
            return False
    return True


def _valid_augmentation(pair: object) -> bool:
    def finite(value: object) -> bool:
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)

    if not isinstance(pair, list) or len(pair) != 2:
        return False
    for view in pair:
        if (not isinstance(view, dict) or set(view) != {"input_mode", "transform", "noise", "retained_fraction"}
                or view["input_mode"] not in {"extrema", "sparse"}
                or not isinstance(view["transform"], dict)
                or set(view["transform"]) != {"angle_degrees", "scale", "flip_x", "flip_y", "swap_xy"}
                or (view["noise"] is not None and not isinstance(view["noise"], dict))):
            return False
        transform, noise, retained = view["transform"], view["noise"], view["retained_fraction"]
        if (not finite(retained) or not 0 <= retained <= 1
                or not finite(transform["angle_degrees"])
                or not isinstance(transform["scale"], list) or len(transform["scale"]) != 3
                or any(not finite(value) or value <= 0 for value in transform["scale"])
                or any(not isinstance(transform[key], bool) for key in ("flip_x", "flip_y", "swap_xy"))):
            return False
        if noise is not None and (
            set(noise) != {"snr_db", "pre_noise_rms", "added_noise_rms", "filter_irlen"}
            or not finite(noise["snr_db"]) or not 12 <= noise["snr_db"] <= 60
            or any(not finite(noise[key]) or noise[key] <= 0 for key in ("pre_noise_rms", "added_noise_rms"))
            or type(noise["filter_irlen"]) is not int or noise["filter_irlen"] < 1
        ):
            return False
    return True


def _recorded_stream(evidence: ArmEvidence, identity: dict, batches_per_epoch: int) -> tuple[list[str], list[tuple[Path, str]]]:
    journal = evidence.journal
    cutoff = next(index for index, row in enumerate(journal.rows)
                  if row["event_id"] == identity["journal_event_id"])
    stale = {event for record in journal._records() for event in record.get("uncommitted_event_ids", [])}
    rows = [row for row in journal.rows[:cutoff + 1] if row["event_id"] not in stale]
    if len(rows) != identity["step"]:
        raise ValueError("Committed journal stream has missing or duplicate steps.")
    stream, files = [], []
    root = journal.path.parent
    provenance_dir = root / "provenance"
    if provenance_dir.is_symlink() or not provenance_dir.is_dir():
        raise ValueError("Provenance must be in a regular run-owned directory.")
    for step, row in enumerate(rows, 1):
        if (row["logical_step"] != str(step) or row["cumulative_exposure"] != str(12 * step)
                or row["completed_pairs"] != "12" or row["reconstructed_a"] != "12"
                or row["epoch"] != str((step - 1) // batches_per_epoch + 1)
                or row["batch_index"] != str((step - 1) % batches_per_epoch + 1)
                or (row["run_id"], row["arm"], row["seed"])
                != (identity["run_id"], identity["arm"], str(identity["experiment_seed"]))):
            raise ValueError("Committed journal stream has an invalid trajectory.")
        relative = Path(row["provenance"])
        if relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] != "provenance":
            raise ValueError("Journal provenance must name a run-owned record.")
        path = root / relative
        _permitted_file(path)
        before = _digest(path)
        record = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(record, dict) or set(record) != {"schema", "event_id", "pairs", "augmentation"}
                or record["schema"] != "streamed-step-provenance-v1"
                or record["event_id"] != row["event_id"]
                or not isinstance(record["pairs"], list) or len(record["pairs"]) != 12
                or not isinstance(record["augmentation"], list) or len(record["augmentation"]) != 12
                or any(not _valid_pair(pair, category) for pair, category in zip(
                    record["pairs"], ["same_overlap"] * 6 + ["same_disjoint"] * 3 + ["cross_source"] * 3
                ))
                or any(not _valid_augmentation(pair) for pair in record["augmentation"])
                or _digest(path) != before):
            raise ValueError("Journal provenance record is invalid or changed.")
        stream.append(content_hash({"pairs": record["pairs"], "augmentation": record["augmentation"]}))
        files.append((path, before))
    return stream, files


def verify_matched_arm_records(
    reference: ArmEvidence, candidate: ArmEvidence, *, matching_config: dict,
) -> dict:
    """Compare fresh initial states and committed records; does not establish adoption readiness."""
    config_hash = matching_config_hash(matching_config)
    matching_config = dict(matching_config)
    if matching_config_hash(matching_config) != config_hash:
        raise ValueError("Matching configuration changed during verification.")
    model_seed, head_seed, sampler_seed = (
        matching_config[key] for key in ("model_seed", "head_seed", "sampler_seed")
    )
    loaded = []
    evidence_files = []
    recovery_snapshots = []
    for item, arm in ((reference, "S1"), (candidate, None)):
        if not isinstance(item, ArmEvidence):
            raise ValueError("Two explicit arm evidence sets are required.")
        for path in (item.initial_checkpoint, item.final_checkpoint):
            _permitted_file(path)
        _permitted_file(item.journal.path)
        journal_hash = _digest(item.journal.path)
        recovery_snapshots.append((item.journal, _recovery_files(item.journal)))
        initial, initial_hash = _checkpoint_state(item.initial_checkpoint)
        final, final_hash = _checkpoint_state(item.final_checkpoint)
        actual_arm = final.get("arm")
        if (not isinstance(actual_arm, str)
                or (arm == "S1" and actual_arm != "S1")
                or (arm is None and actual_arm not in {"S2", "S3"})):
            raise ValueError("Reference must be S1 and candidate must be S2 or S3.")
        identity = _identity(final, final_hash, item.journal, final.get("hashes", {}).get("validation_manifest"))
        if (final["hashes"].get("matching") != config_hash
                or identity["run_id"] != matching_config["run_id"]
                or identity["experiment_seed"] != matching_config["experiment_seed"]
                or identity["epoch"] != matching_config["final_epoch"]
                or final["batches_per_epoch"] != matching_config["batches_per_epoch"]):
            raise ValueError("Arm checkpoint does not match the frozen matching configuration.")
        if (item.journal.position is None or item.journal.position.logical_step != identity["step"]
                or item.journal.position.cumulative_exposure != 12 * identity["step"]):
            raise ValueError("Arm evidence must use the reconciled final checkpoint.")
        _initial_state(initial, actual_arm, final, model_seed, head_seed, sampler_seed)
        stream, files = _recorded_stream(item, identity, final["batches_per_epoch"])
        loaded.append((initial, final, identity, stream, initial_hash, final_hash))
        evidence_files.extend((
            (item.initial_checkpoint, initial_hash), (item.final_checkpoint, final_hash),
            (item.journal.path, journal_hash), *files,
        ))
    left, right = loaded
    if (left[0]["hashes"] != right[0]["hashes"]
            or left[0]["batches_per_epoch"] != right[0]["batches_per_epoch"]
            or any(left[2][key] != right[2][key] for key in (
                "run_id", "experiment_seed", "epoch", "step", "source_sha256", "target_sha256",
                "validation_manifest_sha256",
            ))
            or left[1]["sampler"] != right[1]["sampler"]
            or not _rngs_equal(left[0]["training"], right[0]["training"], left[0]["device"])
            or not _rngs_equal(left[1]["training"], right[1]["training"], left[1]["device"])
            or left[0]["device"] != right[0]["device"]):
        raise ValueError("Matched arms have different configuration, exposure or RNG states.")
    if left[3] != right[3]:
        raise ValueError("Matched arms have different committed pair or augmentation records.")
    for path, digest in evidence_files:
        _permitted_file(path)
        if _digest(path) != digest:
            raise ValueError("Matched-arm evidence changed during verification.")
    if any(_recovery_files(journal) != snapshot for journal, snapshot in recovery_snapshots):
        raise ValueError("Journal recovery records changed during verification.")
    result = {
        "schema": "streamed-matched-arm-records-v2", "status": "MATCHED_RECORDED_STREAMS_ONLY",
        "reference": left[2], "candidate": right[2],
        "initial_checkpoint_sha256": {"S1": left[4], right[2]["arm"]: right[4]},
        "stream_sha256": content_hash(left[3]), "steps": left[2]["step"],
        "matching_config_sha256": config_hash,
        "model_seed": model_seed, "head_seed": head_seed, "sampler_seed": sampler_seed,
        "limitation": (
            "Recorded streams and RNG states only; records do not prove source data or actual model inputs "
            "were unchanged, nor establish evaluation, historical provenance or adoption readiness."
        ),
    }
    result["sha256"] = content_hash(result)
    return result
