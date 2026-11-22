"""Disposable real-data MPS epoch/recovery smoke; not a production training runner."""

import argparse
import json
import math
import os
from pathlib import Path
import resource
import time
from unittest.mock import patch

import torch
from torch.utils.tensorboard import SummaryWriter

from scripts.streamed_overlap_audit import REGENERATING_DATASET_NAME, content_hash
from scripts.streamed_overlap_checkpoint import restore_epoch_state, save_epoch_state
from scripts.streamed_overlap_evaluation import StreamedEpochValidator
from scripts.streamed_overlap_integrity import load_manifest
from scripts.streamed_overlap_journal import BatchJournal, STEP_FIELDS, _sync_directory
from scripts.streamed_overlap_metrics import _eligible_queries
from scripts.streamed_overlap_model import build_optimizer, fresh_e1_model
from scripts.streamed_overlap_sampler import StreamedPairSampler
from scripts.streamed_overlap_selection import _digest, select_checkpoints
from scripts.streamed_overlap_smoke import _validate_config
from scripts.train import (
    EarlyStoppingState, SliceLPIPSLoss, initialize_streamed_epoch_metrics,
    streamed_monitor_hash, train_streamed_epoch,
)


class SmokeInterruption(RuntimeError):
    pass


def _config(config: dict, output: Path) -> tuple[Path, Path, dict]:
    base_keys = {"source_root", "seed", "sparse_probability", "mps_memory_fraction", "max_wall_seconds"}
    extra_keys = {
        "validation_manifest", "validation_manifest_sha256", "reconstruction_query_ids",
        "encode_batch_size", "reconstruction_batch_size",
    }
    if not isinstance(config, dict) or set(config) != base_keys | extra_keys:
        raise ValueError("Integration smoke requires exactly the documented explicit settings.")
    root = _validate_config({key: config[key] for key in base_keys}, output)
    forbidden = Path("/Volumes/CrucialX9/fake_data_validation_v2")
    resolved_output = output.resolve()
    if (REGENERATING_DATASET_NAME in resolved_output.parts
            or resolved_output.is_relative_to(forbidden) or forbidden.is_relative_to(resolved_output)):
        raise ValueError("Integration smoke output must be outside validation-v2.")
    if (not isinstance(config["validation_manifest"], str)
            or not config["validation_manifest"]
            or REGENERATING_DATASET_NAME in Path(config["validation_manifest"]).parts):
        raise ValueError("A permitted primary validation manifest is required.")
    path = Path(config["validation_manifest"]).resolve(strict=True)
    if REGENERATING_DATASET_NAME in path.parts or path.is_relative_to(root):
        raise ValueError("Validation manifest must be outside source data and validation-v2.")
    manifest = load_manifest(path)
    if manifest.get("manifest_sha256") != config["validation_manifest_sha256"]:
        raise ValueError("Primary validation manifest hash does not match the explicit setting.")
    ids = config["reconstruction_query_ids"]
    if (not isinstance(ids, list) or not ids
            or any(not isinstance(value, str) or not value for value in ids)
            or len(set(ids)) != len(ids)
            or any(isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1
                   for key in ("encode_batch_size", "reconstruction_batch_size"))):
        raise ValueError("Explicit unique reconstruction queries and positive batch sizes are required.")
    if not set(ids) <= {query["id"] for query in _eligible_queries(manifest)}:
        raise ValueError("Reconstruction queries must belong to the frozen eligible manifest.")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("The integration smoke must not enable MPS CPU fallback.")
    return root, path, manifest


def run_integration_smoke(config: dict, output: Path) -> dict:
    root, manifest_path, manifest = _config(config, output)
    if not torch.backends.mps.is_available():
        raise RuntimeError("The integration smoke requires MPS.")
    started = time.monotonic()
    deadline = started + config["max_wall_seconds"]
    print("Smoke: loading fixed real-data train and validation monitors.", flush=True)
    torch.mps.set_per_process_memory_fraction(config["mps_memory_fraction"])
    torch.manual_seed(config["seed"])
    torch.mps.manual_seed(config["seed"])
    samplers = {
        split: StreamedPairSampler(root, split, config["seed"])
        for split in ("train", "validation")
    }
    if samplers["train"].source_manifest_hash != samplers["validation"].source_manifest_hash:
        raise ValueError("Train and validation source inventories differ.")
    monitor_batches = {
        split: sampler.load_batch(sampler.next_specs(), sparse_probability=config["sparse_probability"])
        for split, sampler in samplers.items()
    }
    sampler = StreamedPairSampler(root, "train", config["seed"])
    model = fresh_e1_model("S3", model_seed=config["seed"], head_seed=(config["seed"] + 1) % 2**32).to("mps")
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=6, min_lr=1e-5)
    lpips = SliceLPIPSLoss().to("mps").eval()
    stop = EarlyStoppingState(float("inf"), 0)
    validator = StreamedEpochValidator(
        manifest_path, root, manifest_sha256=config["validation_manifest_sha256"],
        reconstruction_query_ids=tuple(config["reconstruction_query_ids"]),
        reconstruction_seed=config["seed"], encode_batch_size=config["encode_batch_size"],
        reconstruction_batch_size=config["reconstruction_batch_size"],
        max_wall_seconds=config["max_wall_seconds"], lpips_loss_fn=lpips,
    )
    hashes = {
        "config": content_hash(config), "source": sampler.source_manifest_hash,
        "target": manifest["target_implementation_sha256"],
        "validation_manifest": manifest["manifest_sha256"],
        "monitor": streamed_monitor_hash(monitor_batches), "epoch_validation": validator.config_sha256,
    }
    journal_path = output / "batch_metrics.csv"
    epoch_csv = output / "training_metrics.csv"
    checkpoint = output / "last.pt"
    initialize_streamed_epoch_metrics(epoch_csv)
    journal = BatchJournal(journal_path, STEP_FIELDS, create=True)
    try:
        journal.reconcile(watermark=None, attempt_id="smoke-first", expected_step=0, expected_exposure=0)
        save_epoch_state(
            checkpoint, model=model, optimizer=optimizer, scheduler=scheduler, sampler=sampler,
            early_stopping=stop, journal=journal, hashes=hashes, completed_epoch=0,
            batches_per_epoch=1, selection_state={},
        )
        with SummaryWriter(log_dir=str(output / "events")) as writer:
            def epoch():
                return train_streamed_epoch(
                    model, optimizer, scheduler, sampler, journal, lpips, "mps", writer,
                    run_id="integration-smoke", seed=config["seed"], epoch=1, batches_per_epoch=1,
                    sparse_probability=config["sparse_probability"], monitor_batches=monitor_batches,
                    monitor_seed=config["seed"], validation_seed=config["seed"],
                    validate_epoch=validator, epoch_csv=epoch_csv, checkpoint_path=checkpoint,
                    hashes=hashes, early_stopping=stop, selection_state={},
                )

            print("Smoke: running one disposable optimizer step and injecting interruption.", flush=True)
            with patch("scripts.train.monitor_streamed_pairs", side_effect=SmokeInterruption("injected")):
                try:
                    epoch()
                except SmokeInterruption:
                    pass
                else:
                    raise ValueError("The injected smoke interruption did not occur.")
            if journal.position is not None or len(journal.rows) != 1:
                raise ValueError("The interrupted smoke did not preserve exactly one uncommitted row.")
            stale = dict(journal.rows[0])
            restored = restore_epoch_state(
                checkpoint, model=model, optimizer=optimizer, scheduler=scheduler, sampler=sampler,
                early_stopping=stop, journal=journal, expected_hashes=hashes,
                expected_batches_per_epoch=1, attempt_id="smoke-replay",
            )
            if restored["completed_epoch"] != 0:
                raise ValueError("The initial smoke checkpoint was not restored.")
            print("Smoke: replaying the step and running frozen primary validation.", flush=True)
            epoch()
        replay = journal.rows[-1]
        if (len(journal.rows) != 2 or replay["replay_of_event"] != stale["event_id"]
                or replay["cumulative_exposure"] != "12"
                or not math.isclose(float(replay["total_loss"]), float(stale["total_loss"]),
                                    rel_tol=1e-6, abs_tol=1e-6)):
            raise ValueError("The recovered MPS step did not reproduce the interrupted trajectory.")
        print("Smoke: verifying reconstruction checkpoint selection.", flush=True)
        selection = select_checkpoints(checkpoint, journal, manifest_path, retrieval_report=None)
        if selection["best_reconstruction"] is None or selection["best_retrieval"] is not None:
            raise ValueError("Smoke selection must publish reconstruction only.")
        checkpoint_hash = _digest(checkpoint)
        if _digest(output / "best_reconstruction.pt") != checkpoint_hash:
            raise ValueError("The selected reconstruction checkpoint differs from the completed smoke epoch.")
    finally:
        journal.close()
    torch.mps.synchronize()
    if time.monotonic() > deadline:
        raise TimeoutError("Integration smoke wall-time limit exceeded; no success report is valid.")
    return {
        "status": "DISPOSABLE_INTEGRATION_SMOKE_PASSED", "optimizer_steps": 2,
        "committed_smoke_steps": 1, "pairs_per_step": 12, "replayed": True,
        "validation_manifest_sha256": manifest["manifest_sha256"],
        "source_manifest_sha256": sampler.source_manifest_hash,
        "checkpoint_sha256": checkpoint_hash, "validation_config_sha256": validator.config_sha256,
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        "limitation": (
            "Disposable smoke only: no production training, trained S1 reference, pair-head retrieval "
            "scoring, tokenizer CLI search or launch approval."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    _config(config, args.output)
    args.output.mkdir()
    _sync_directory(args.output.parent)
    report = run_integration_smoke(config, args.output)
    with (args.output / "integration_smoke_report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _sync_directory(args.output)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
