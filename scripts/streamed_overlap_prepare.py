"""Prepare a suite-input record without starting training or evaluation."""

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import BinaryIO

import torch

from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_epoch_runner import (
    HistoricalReconstruction, _reference_hash, _validate_formal_config, _validate_reconstruction_config,
    _validate_suite_inputs, prepare_epoch_inputs, suite_input_record, suite_workload,
)
from scripts.streamed_overlap_guardrails import _sha256
from scripts.streamed_overlap_journal import _publish_exclusive
from scripts.streamed_overlap_matching import _permitted_file, matching_config_hash


def _read_object(path: Path) -> dict:
    _permitted_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def prepare(config: dict, output: Path, *, mps_memory_fraction: float, max_wall_seconds: float) -> dict:
    keys = {
        "suite_dir", "source_root", "validation_manifest", "source_sha256", "target_sha256",
        "validation_manifest_sha256", "matching_config", "epoch_validation", "formal_config", "guardrail_config",
        "search_manifest", "diagnostic_manifest", "guardrail_manifest", "historical",
    }
    if not isinstance(config, dict) or set(config) != keys:
        raise ValueError("Suite preparation requires all documented settings and artifact paths.")
    if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in (mps_memory_fraction, max_wall_seconds))
            or not 0 < mps_memory_fraction <= 0.5 or max_wall_seconds <= 0):
        raise ValueError("Explicit positive preparation time and MPS memory limits are required.")
    path_keys = ("suite_dir", "source_root", "validation_manifest", "search_manifest",
                 "diagnostic_manifest", "guardrail_manifest")
    if any(not isinstance(config[key], str) or not config[key] or not Path(config[key]).is_absolute()
           for key in path_keys):
        raise ValueError("Suite preparation requires absolute artifact paths.")
    root = Path(config["source_root"]).resolve(strict=True)
    suite = Path(config["suite_dir"])
    destination = suite.resolve()
    result_path = output.resolve()
    forbidden = Path("/Volumes/CrucialX9/fake_data_validation_v2")
    if (not root.is_dir()
            or any("fake_data_validation_v2" in path.parts for path in (
                Path(config["source_root"]), suite, output, root, destination, result_path,
            ))
            or any(path.is_relative_to(forbidden) or forbidden.is_relative_to(path)
                   for path in (root, destination, result_path))
            or destination.is_relative_to(root) or root.is_relative_to(destination)
            or result_path.is_relative_to(root) or result_path.is_relative_to(destination)):
        raise ValueError("Sources, suite and preparation output must be separate and outside validation-v2.")
    if suite.exists() or suite.is_symlink() or output.exists() or output.is_symlink():
        raise FileExistsError("Refusing an existing suite destination or preparation output.")
    for key in ("validation_manifest", "search_manifest", "diagnostic_manifest", "guardrail_manifest"):
        path = Path(config[key])
        _permitted_file(path)
        if path.resolve().is_relative_to(root):
            raise ValueError("Frozen manifests must be outside source data.")
    historical = config["historical"]
    reference_keys = {"report_path", "report_sha256", "checkpoint_path", "checkpoint_sha256", "checkpoint_label"}
    if (not isinstance(historical, dict) or set(historical) != reference_keys
            or any(not isinstance(historical[key], str) or not historical[key]
                   or not Path(historical[key]).is_absolute() for key in ("report_path", "checkpoint_path"))):
        raise ValueError("An explicit historical reference and absolute paths are required.")
    reference = HistoricalReconstruction(
        Path(historical["report_path"]), historical["report_sha256"],
        Path(historical["checkpoint_path"]), historical["checkpoint_sha256"], historical["checkpoint_label"],
    )
    reference_hash = _reference_hash(reference)
    search = _read_object(Path(config["search_manifest"]))
    diagnostic = _read_object(Path(config["diagnostic_manifest"]))
    guardrail = _read_object(Path(config["guardrail_manifest"]))
    if (any(not _sha256(config[key]) for key in ("source_sha256", "target_sha256", "validation_manifest_sha256"))
            or search.get("validation_manifest_sha256") != config["validation_manifest_sha256"]
            or diagnostic.get("validation_manifest_sha256") != config["validation_manifest_sha256"]):
        raise ValueError("Source, target and evaluation manifests must bind explicit matching hashes.")
    matching = config["matching_config"]
    formal = config["formal_config"]
    guardrail_config = config["guardrail_config"]
    _validate_formal_config(search, diagnostic, formal)
    _validate_reconstruction_config(guardrail, guardrail_config, formal)
    workload = suite_workload(matching, search)
    if formal["final_epoch"] != matching["final_epoch"]:
        raise ValueError("Evaluation and suite budget must use the same final epoch.")
    hashes = {
        "source": config["source_sha256"], "target": config["target_sha256"],
        "validation_manifest": config["validation_manifest_sha256"], "matching": matching_config_hash(matching),
        "formal_evaluation": content_hash(formal), "reconstruction_guardrail": content_hash(guardrail_config),
        "reconstruction_reference": reference_hash,
    }
    if not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("Suite preparation requires MPS without CPU fallback.")
    started = time.monotonic()
    torch.mps.set_per_process_memory_fraction(mps_memory_fraction)
    batches, validator, lpips, observed = prepare_epoch_inputs(
        manifest_path=Path(config["validation_manifest"]), source_root=root, hashes=hashes,
        guardrail_manifest=guardrail, matching_config=matching, epoch_validation=config["epoch_validation"],
    )
    hashes.update(observed)
    record = suite_input_record(
        suite, validate_epoch=validator, manifest_path=Path(config["validation_manifest"]), source_root=root,
        search_manifest=search, diagnostic_manifest=diagnostic, formal_config=formal, guardrail_manifest=guardrail,
        guardrail_config=guardrail_config, historical=reference, matching_config=matching, hashes=hashes,
    )
    hashes["config"] = content_hash(record)
    _validate_suite_inputs(
        monitor_batches=batches, validate_epoch=validator, lpips_loss_fn=lpips,
        manifest_path=Path(config["validation_manifest"]), source_root=root, search_manifest=search,
        diagnostic_manifest=diagnostic, formal_config=formal, guardrail_manifest=guardrail,
        guardrail_config=guardrail_config, historical=reference, matching_config=matching, hashes=hashes,
    )
    elapsed = time.monotonic() - started
    if elapsed > max_wall_seconds:
        raise TimeoutError("Suite preparation time limit exceeded; no input record was written.")
    data = (json.dumps(record, indent=2, allow_nan=False) + "\n").encode()

    def write(handle: BinaryIO) -> None:
        handle.write(data)

    _publish_exclusive(output, write)
    return {
        "status": "SUITE_INPUTS_PREPARED_NOT_AUTHORIZED", "config_sha256": hashes["config"],
        "output": str(result_path), "workload": workload,
        "preparation": {"mps_memory_fraction": mps_memory_fraction, "max_wall_seconds": max_wall_seconds,
                        "elapsed_seconds": elapsed},
        "limitation": (
            "No optimizer step, full validation preflight, historical measurement provenance verification, "
            "launch resource check or approval."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mps-memory-fraction", type=float, required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    args = parser.parse_args()
    result = prepare(
        _read_object(args.config), args.output, mps_memory_fraction=args.mps_memory_fraction,
        max_wall_seconds=args.max_wall_seconds,
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
