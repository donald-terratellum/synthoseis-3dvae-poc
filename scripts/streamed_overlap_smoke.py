"""No-optimizer MPS smoke for the streamed sampler, joint losses and monitors."""

import argparse
import json
import math
import os
from pathlib import Path
import resource
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from scripts.streamed_overlap_audit import REGENERATING_DATASET_NAME
from scripts.streamed_overlap_model import fresh_e1_model, joint_loss
from scripts.streamed_overlap_journal import _sync_directory
from scripts.streamed_overlap_sampler import StreamedPairSampler
from scripts.train import SliceLPIPSLoss, monitor_streamed_pairs, streamed_monitor_hash


def _validate_config(config: dict, output: Path) -> Path:
    if (not isinstance(config, dict)
            or set(config) != {"source_root", "seed", "sparse_probability", "mps_memory_fraction", "max_wall_seconds"}
            or not isinstance(config["source_root"], str) or not config["source_root"]
            or isinstance(config["seed"], bool) or not isinstance(config["seed"], int)
            or not 0 <= config["seed"] < 2**32
            or any(isinstance(config[key], bool) or not isinstance(config[key], (int, float))
                   or not math.isfinite(config[key])
                   for key in ("sparse_probability", "mps_memory_fraction", "max_wall_seconds"))
            or not 0 <= config["sparse_probability"] <= 1
            or not 0 < config["mps_memory_fraction"] <= 0.5
            or config["max_wall_seconds"] <= 0):
        raise ValueError("Smoke requires explicit source, seed, sparse probability, MPS limit and wall-time limit.")
    root = Path(config["source_root"]).resolve(strict=True)
    forbidden = Path("/Volumes/CrucialX9/fake_data_validation_v2")
    if (REGENERATING_DATASET_NAME in root.parts or root.is_relative_to(forbidden)
            or forbidden.is_relative_to(root)):
        raise ValueError("The regenerating validation-v2 dataset is not permitted.")
    if output.resolve().is_relative_to(root):
        raise ValueError("Smoke output must be outside the source root.")
    return root


def run_smoke(config: dict, output: Path) -> dict:
    root = _validate_config(config, output)
    if not torch.backends.mps.is_available():
        raise RuntimeError("The smoke requires MPS; CPU fallback is not permitted.")
    started = time.monotonic()
    deadline = started + config["max_wall_seconds"]
    torch.mps.set_per_process_memory_fraction(config["mps_memory_fraction"])
    torch.manual_seed(config["seed"])
    torch.mps.manual_seed(config["seed"])
    samplers = {split: StreamedPairSampler(root, split, config["seed"]) for split in ("train", "validation")}
    if samplers["train"].source_manifest_hash != samplers["validation"].source_manifest_hash:
        raise ValueError("Train and validation source inventories differ.")
    batches = {}
    for split, sampler in samplers.items():
        if time.monotonic() > deadline:
            raise TimeoutError("Smoke wall-time limit exceeded; no partial report is valid.")
        initial = sampler.state_dict()
        specs = sampler.next_specs()
        sampler.load_state_dict(initial)
        if sampler.next_specs() != specs:
            raise ValueError(f"{split} sampler did not reproduce its pair specifications.")
        batches[split] = sampler.load_batch(specs, sparse_probability=config["sparse_probability"])
    model = fresh_e1_model("S3", model_seed=config["seed"], head_seed=(config["seed"] + 1) % 2**32).to("mps")
    lpips = SliceLPIPSLoss().to("mps").eval()
    batch = batches["train"]
    result = model(torch.as_tensor(batch.input_a, device="mps"), torch.as_tensor(batch.input_b, device="mps"))
    losses = joint_loss(
        result, torch.as_tensor(batch.dense_a, device="mps"), lpips,
        torch.as_tensor(batch.overlap, device="mps"), torch.as_tensor(batch.coverage, device="mps"),
        torch.as_tensor(batch.coverage_valid, device="mps"),
    )
    if losses.valid_coverage == 0:
        raise ValueError("Smoke training pairs have no valid coverage labels.")
    losses.total.backward()
    torch.mps.synchronize()
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
           for parameter in model.parameters()):
        raise ValueError("Smoke produced non-finite model gradients.")
    heads = {"encoder": model.vae.encoder, "overlap": model.overlap_head, "coverage": model.coverage_head}
    for name, module in heads.items():
        if module is None:
            raise ValueError(f"Smoke requires the S3 {name} module.")
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        if (not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients)
                or not any(torch.any(gradient != 0).item() for gradient in gradients)):
            raise ValueError(f"Smoke requires finite nonzero {name} gradients.")
    model.zero_grad(set_to_none=True)
    cpu_rng = torch.get_rng_state().clone()
    mps_rng = torch.mps.get_rng_state().clone()
    with SummaryWriter(log_dir=str(output / "events")) as writer:
        monitors, monitor_seconds = monitor_streamed_pairs(
            model, lpips, "mps", batches, writer, seed=config["seed"], step=1,
        )
    torch.mps.synchronize()
    if (not torch.equal(cpu_rng, torch.get_rng_state())
            or not torch.equal(mps_rng, torch.mps.get_rng_state()) or not model.training):
        raise ValueError("Smoke monitor did not restore RNG and model training state.")
    if any(not isinstance(values["valid_coverage"], int) or values["valid_coverage"] < 1
           for values in monitors.values()):
        raise ValueError("Smoke monitors require valid coverage in both splits.")
    if time.monotonic() > deadline:
        raise TimeoutError("Smoke wall-time limit exceeded; no partial report is valid.")
    return {
        "status": "NO_OPTIMIZER_SMOKE_PASSED", "source_manifest_sha256": samplers["train"].source_manifest_hash,
        "pairs": {split: len(batch.provenance) for split, batch in batches.items()},
        "valid_coverage": {split: values["valid_coverage"] for split, values in monitors.items()},
        "monitor_sha256": streamed_monitor_hash(batches), "monitor_seconds": monitor_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        "optimizer_step": False,
        "limitation": "No optimizer, epoch, checkpoint recovery, retrieval scoring or training readiness was tested.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not isinstance(config, dict):
        raise ValueError("Smoke configuration must be a JSON object.")
    _validate_config(config, args.output)
    args.output.mkdir()
    _sync_directory(args.output.parent)
    result = run_smoke(config, args.output)
    with (args.output / "smoke_report.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _sync_directory(args.output)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
