"""Evaluation-only child for a future logged suite runner."""

import argparse
import json
import math
import os
from pathlib import Path
from typing import BinaryIO

import torch

from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_epoch_runner import (
    _bind_suite_inputs, _read_pinned_suite_record, run_suite_from_record,
)
from scripts.streamed_overlap_journal import _publish_exclusive


def evaluate(
    record_path: Path, suite_dir: Path, *, config_sha256: str, attempt_id: str, mps_memory_fraction: float,
) -> dict:
    """Evaluate completed arms only; logging and remaining resource checks are external."""
    if (not isinstance(attempt_id, str) or not attempt_id or any(c in attempt_id for c in "\r\n")
            or isinstance(mps_memory_fraction, bool)
            or not isinstance(mps_memory_fraction, (int, float))
            or not math.isfinite(mps_memory_fraction) or not 0 < mps_memory_fraction <= 0.5):
        raise ValueError("An explicit attempt ID and MPS memory fraction in (0, 0.5] are required.")
    record, _, _ = _read_pinned_suite_record(record_path, suite_dir, config_sha256)
    if (suite_dir.is_symlink() or not suite_dir.is_dir()
            or "fake_data_validation_v2" in suite_dir.parts
            or "fake_data_validation_v2" in suite_dir.resolve().parts):
        raise ValueError("Evaluation requires an existing permitted suite directory.")
    _bind_suite_inputs(suite_dir, record, resume=True)
    if not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("Evaluation requires MPS without CPU fallback.")
    torch.mps.set_per_process_memory_fraction(mps_memory_fraction)
    results = run_suite_from_record(
        record_path, suite_dir, expected_config_sha256=config_sha256,
        resume=True, attempt_id=attempt_id, evaluate_only=True,
    )
    final_checkpoints = {}
    for arm in ("S1", "S2", "S3"):
        checkpoint = results[arm]["last_epoch"]["checkpoint"]
        if checkpoint.get("arm") != arm or checkpoint.get("config_sha256") != config_sha256:
            raise ValueError("Completed evaluation returned a mismatched checkpoint identity.")
        final_checkpoints[arm] = checkpoint
    hashes = {}
    for key in ("mu_cosine_comparisons", "pair_head_metric_checks"):
        report = results["S3"][key]
        if report.get("sha256") != content_hash({k: v for k, v in report.items() if k != "sha256"}):
            raise ValueError(f"Completed evaluation returned an invalid {key} report.")
        hashes[key] = report["sha256"]
    _read_pinned_suite_record(record_path, suite_dir, config_sha256)
    _bind_suite_inputs(suite_dir, record, resume=True)
    return {
        "schema": "streamed-evaluation-child-result-v1", "status": "REPORTS_COMPUTED_NOT_ADOPTION",
        "suite_config_sha256": config_sha256, "suite_dir": str(suite_dir.resolve()), "attempt_id": attempt_id,
        "mps_memory_fraction": mps_memory_fraction, "final_checkpoints": final_checkpoints,
        "report_sha256": hashes,
        "limitation": "Evaluation only; resource readiness, durable log capture and adoption remain separate.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-record", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--mps-memory-fraction", type=float, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if (not args.receipt.is_absolute() or args.receipt.exists() or args.receipt.is_symlink()
            or args.receipt.parent.is_symlink() or not args.receipt.parent.is_dir()):
        raise ValueError("Evaluation receipt requires a new absolute file in a regular directory.")
    result = evaluate(
        args.suite_record, args.suite_dir, config_sha256=args.config_sha256,
        attempt_id=args.attempt_id, mps_memory_fraction=args.mps_memory_fraction,
    )
    data = (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()

    def write(handle: BinaryIO) -> None:
        handle.write(data)

    _publish_exclusive(args.receipt, write)
    print(data.decode(), end="", flush=True)


if __name__ == "__main__":
    main()
