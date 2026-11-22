"""Full-store, post-selection reconstruction measurements for frozen patch stores."""

import hashlib
import importlib.metadata
import inspect
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch
import zarr

from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_evaluation import _checkpoint_state, _identity
from scripts.streamed_overlap_guardrails import _manifest
from scripts.streamed_overlap_integrity import _digest_array, load_manifest
from scripts.streamed_overlap_model import fresh_e1_model
from scripts.streamed_overlap_targets import _read_bounded
from scripts.train import SliceLPIPSLoss, compute_vae_losses, streamed_inference_context
from src.augmentations import keep_trace_extrema_only
from src.model import VAE3D
from src.tokenizer.core.model_adapter import VaeLatentAdapter, choose_torch_device
from src.tokenizer.core.preprocess import preprocess_for_token


PATCH_SHAPE = (32, 32, 64)


def _runtime() -> dict:
    files = {
        "producer": Path(__file__), "training_losses": Path(inspect.getfile(compute_vae_losses)),
        "model": Path(inspect.getfile(VAE3D)), "tokenizer_adapter": Path(inspect.getfile(VaeLatentAdapter)),
        "tokenizer_preprocess": Path(inspect.getfile(preprocess_for_token)),
        "extrema": Path(inspect.getfile(keep_trace_extrema_only)),
        **{
            name: Path(__file__).with_name(filename)
            for name, filename in (
                ("streamed_model", "streamed_overlap_model.py"),
                ("checkpoint_identity", "streamed_overlap_evaluation.py"),
                ("input_fingerprint", "streamed_overlap_integrity.py"),
                ("bounded_reader", "streamed_overlap_targets.py"),
                ("manifest_hash", "streamed_overlap_audit.py"),
            )
        },
    }
    return {
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "versions": {
            "torch": str(torch.__version__), "numpy": np.__version__, "zarr": zarr.__version__,
            "lpips": importlib.metadata.version("lpips"), "torchvision": importlib.metadata.version("torchvision"),
        },
        "implementation_sha256": {name: _file_hash(path) for name, path in files.items()},
    }


def lpips_fingerprint(loss: SliceLPIPSLoss) -> str:
    if type(loss) is not SliceLPIPSLoss or any(module.training for module in loss.modules()):
        raise ValueError("An evaluation-mode SliceLPIPSLoss is required.")
    if any(parameter.requires_grad for parameter in loss.parameters()):
        raise ValueError("LPIPS parameters must be frozen.")
    state = {
        name: _digest_array(value.detach().cpu().numpy())
        for name, value in loss.state_dict().items()
    }
    return content_hash({
        "schema": "slice-lpips-state-v1", "min_spatial_size": loss.min_spatial_size,
        "eps": loss.eps, "implementation_sha256": hashlib.sha256(
            inspect.getsource(type(loss)).encode()
        ).hexdigest(), "state": state,
    })


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _patch_store(path: Path):
    resolved = path.resolve(strict=True)
    if "fake_data_validation_v2" in resolved.parts:
        raise ValueError("The regenerating validation-v2 dataset is not permitted.")
    group = zarr.open_group(str(resolved), mode="r")
    if group.attrs.get("scaling_mode") != "divide_by_std" or "patches" not in group:
        raise ValueError("Guardrail stores require baked divide_by_std patches.")
    patches = group["patches"]
    if not isinstance(patches, zarr.Array):
        raise ValueError("Guardrail patches must be a Zarr array.")
    if len(patches.shape) != 4 or tuple(patches.shape[1:]) != PATCH_SHAPE or patches.shape[0] < 1:
        raise ValueError("Guardrail stores require nonempty 32x32x64 patch arrays.")
    if not np.issubdtype(patches.dtype, np.number) or np.issubdtype(patches.dtype, np.complexfloating):
        raise ValueError("Guardrail patches must have a real numeric dtype.")
    return resolved, patches


def _patch(patches, index: int) -> tuple[np.ndarray, np.ndarray]:
    target = np.ascontiguousarray(
        _read_bounded(patches, (index, slice(None), slice(None), slice(None))), dtype=np.float32,
    )
    if target.shape != PATCH_SHAPE or not np.isfinite(target).all():
        raise ValueError("Guardrail targets must be finite 32x32x64 patches.")
    return preprocess_for_token(target), target


def _records(domain: str, patches, deadline: float) -> list[dict]:
    records = []
    for index in range(patches.shape[0]):
        if time.monotonic() > deadline:
            raise TimeoutError("Guardrail wall-time limit exceeded; no partial report is valid.")
        inputs, target = _patch(patches, index)
        records.append({
            "id": f"{domain}:{index}", "input_sha256": _digest_array(inputs),
            "target_sha256": _digest_array(target),
        })
    return records


def freeze_reconstruction_manifest(
    synthetic_store: Path, real_store: Path, *, seed: int, device: str,
    lpips_loss_fn: SliceLPIPSLoss, max_wall_seconds: float,
) -> dict:
    """Fingerprint every baked-scaled patch; this does not approve the datasets."""
    if (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32
            or isinstance(max_wall_seconds, bool) or not isinstance(max_wall_seconds, (int, float))
            or not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0):
        raise ValueError("A valid sampling seed and positive wall-time limit are required.")
    deadline = time.monotonic() + max_wall_seconds
    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError("An explicit device is required.")
    actual_device = choose_torch_device(device)
    if {parameter.device.type for parameter in lpips_loss_fn.parameters()} != {actual_device.type}:
        raise ValueError("LPIPS must be on the explicitly requested device.")
    lpips_hash = lpips_fingerprint(lpips_loss_fn)
    datasets, stores = {}, {}
    for domain, path in (("synthetic", synthetic_store), ("real", real_store)):
        resolved, patches = _patch_store(path)
        stores[domain] = {"path": str(resolved), "patch_count": int(patches.shape[0])}
        datasets[domain] = _records(domain, patches, deadline)
    if stores["synthetic"]["path"] == stores["real"]["path"]:
        raise ValueError("Synthetic and real guardrail stores must be distinct.")
    manifest = {
        "schema": "streamed-reconstruction-guardrail-v1", "usage": "post_selection_guardrail",
        "protocol": {
            "input_mode": "tokenizer", "latent_mode": "sampled",
            "sampling_method": "normal_reparameterization", "seed": seed,
            "patch_shape": list(PATCH_SHAPE), "lpips_sha256": lpips_hash,
            "device": device, "runtime": _runtime(),
        },
        "stores": stores, "datasets": datasets,
    }
    manifest["manifest_sha256"] = content_hash(manifest)
    _verify_stores(manifest, deadline)
    if lpips_fingerprint(lpips_loss_fn) != lpips_hash:
        raise ValueError("LPIPS state changed while freezing the manifest.")
    if manifest["protocol"]["runtime"] != _runtime():
        raise ValueError("Reconstruction implementation changed while freezing the manifest.")
    if time.monotonic() > deadline:
        raise TimeoutError("Guardrail wall-time limit exceeded; no partial manifest is valid.")
    return manifest


def _verify_stores(manifest: dict, deadline: float) -> None:
    stores = manifest.get("stores")
    if not isinstance(stores, dict) or set(stores) != {"synthetic", "real"}:
        raise ValueError("Frozen full-store identities are required.")
    paths = set()
    for domain in ("synthetic", "real"):
        entry = stores[domain]
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("Invalid frozen patch-store identity.")
        resolved, patches = _patch_store(Path(entry["path"]))
        if (str(resolved) != entry["path"] or entry.get("patch_count") != patches.shape[0]
                or _records(domain, patches, deadline) != manifest["datasets"][domain]):
            raise ValueError(f"{domain} guardrail store differs from its frozen full coverage.")
        paths.add(resolved)
    if len(paths) != 2:
        raise ValueError("Synthetic and real guardrail stores must be distinct.")
    if time.monotonic() > deadline:
        raise TimeoutError("Guardrail wall-time limit exceeded; no partial report is valid.")


def measure_reconstruction(
    manifest: dict, checkpoint_path: Path, *, arm: str, device: str,
    lpips_loss_fn: SliceLPIPSLoss, max_wall_seconds: float,
    journal=None, validation_manifest_path: Path | None = None,
    historical_checkpoint_sha256: str | None = None, historical_checkpoint_label: str | None = None,
) -> dict:
    """Measure a verified checkpoint on every frozen patch; never makes an adoption decision."""
    started = time.monotonic()
    _manifest(manifest)
    if (device not in {"cpu", "mps", "cuda"} or isinstance(max_wall_seconds, bool)
            or not isinstance(max_wall_seconds, (int, float))
            or not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0):
        raise ValueError("An explicit device and positive wall-time limit are required.")
    if manifest["protocol"].get("device") != device or manifest["protocol"].get("runtime") != _runtime():
        raise ValueError("Device or reconstruction runtime differs from the frozen protocol.")
    deadline = started + max_wall_seconds
    lpips_hash = lpips_fingerprint(lpips_loss_fn)
    if lpips_hash != manifest["protocol"]["lpips_sha256"]:
        raise ValueError("LPIPS state differs from the frozen reconstruction protocol.")
    actual_device = choose_torch_device(device)
    if {parameter.device.type for parameter in lpips_loss_fn.parameters()} != {actual_device.type}:
        raise ValueError("LPIPS must be on the explicitly requested device.")
    if arm == "P0b":
        if (journal is not None or validation_manifest_path is not None
                or not isinstance(historical_checkpoint_label, str) or not historical_checkpoint_label
                or not isinstance(historical_checkpoint_sha256, str)
                or _file_hash(checkpoint_path) != historical_checkpoint_sha256):
            raise ValueError("Historical reference requires its explicit checkpoint hash and label.")
        with torch.random.fork_rng(devices=[]):
            model = VaeLatentAdapter(checkpoint_path, device=device).model
        checkpoint = {
            "arm": arm, "checkpoint_sha256": historical_checkpoint_sha256,
            "checkpoint_label": historical_checkpoint_label,
        }
    elif arm in {"S1", "S2", "S3"}:
        if (journal is None or validation_manifest_path is None
                or historical_checkpoint_sha256 is not None or historical_checkpoint_label is not None):
            raise ValueError("Streamed checkpoints require their locked journal and validation manifest.")
        validation = load_manifest(validation_manifest_path)
        state, digest = _checkpoint_state(checkpoint_path)
        checkpoint = _identity(state, digest, journal, validation["manifest_sha256"])
        if checkpoint["arm"] != arm:
            raise ValueError("Requested arm differs from the verified checkpoint.")
        wrapper = fresh_e1_model(arm, model_seed=0, head_seed=0)
        wrapper.load_state_dict(state["model"], strict=True)
        model = wrapper.vae.to(actual_device)
    else:
        raise ValueError("Unknown reconstruction checkpoint arm.")
    if tuple(model.patch_shape) != PATCH_SHAPE:
        raise ValueError("Checkpoint patch shape does not match the guardrail protocol.")
    if {parameter.device for parameter in model.parameters()} != {
        parameter.device for parameter in lpips_loss_fn.parameters()
    }:
        raise ValueError("Checkpoint and LPIPS parameters must use the same device.")
    _verify_stores(manifest, deadline)
    measurements = {}
    with streamed_inference_context(model, actual_device, manifest["protocol"]["seed"]):
        for domain in ("synthetic", "real"):
            _, patches = _patch_store(Path(manifest["stores"][domain]["path"]))
            generator = torch.Generator(device="cpu").manual_seed(manifest["protocol"]["seed"])
            records = []
            for index, expected in enumerate(manifest["datasets"][domain]):
                if time.monotonic() > deadline:
                    raise TimeoutError("Guardrail wall-time limit exceeded; no partial report is valid.")
                inputs, target = _patch(patches, index)
                if (_digest_array(inputs) != expected["input_sha256"]
                        or _digest_array(target) != expected["target_sha256"]):
                    raise ValueError(f"{domain} guardrail patch changed during measurement.")
                x = torch.from_numpy(inputs[None, None]).to(actual_device)
                y = torch.from_numpy(target[None, None]).to(actual_device)
                mu, logvar = model.encoder(x)
                if (mu.shape != (1, model.latent_dim) or logvar.shape != mu.shape
                        or not torch.isfinite(mu).all() or not torch.isfinite(logvar).all()):
                    raise ValueError("Guardrail encoder produced invalid latent parameters.")
                noise = torch.randn((1, model.latent_dim), generator=generator).to(actual_device)
                prediction, _ = model.decoder(mu + (0.5 * logvar).exp() * noise)
                if prediction.shape != y.shape or not torch.isfinite(prediction).all():
                    raise ValueError("Guardrail decoder produced an invalid reconstruction.")
                if domain == "synthetic":
                    _, mae, kl, lpips, _ = compute_vae_losses(
                        prediction, y, mu, logvar, 0.001,
                        rec_loss_fn=torch.nn.functional.l1_loss, lpips_loss_fn=lpips_loss_fn,
                        lpips_weight=0.1,
                    )
                    values = {"mae": float(mae.item()), "kl": float(kl.item()), "lpips": float(lpips.item())}
                else:
                    values = {"mae": float(torch.nn.functional.l1_loss(prediction, y).item())}
                if any(not math.isfinite(value) or value < 0 for value in values.values()):
                    raise ValueError("Guardrail reconstruction metrics must be finite and nonnegative.")
                records.append({**expected, **values})
            measurements[domain] = records
    _verify_stores(manifest, deadline)
    if _file_hash(checkpoint_path) != checkpoint["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during reconstruction measurement.")
    if lpips_fingerprint(lpips_loss_fn) != lpips_hash:
        raise ValueError("LPIPS state changed during reconstruction measurement.")
    if manifest["protocol"]["runtime"] != _runtime():
        raise ValueError("Reconstruction implementation changed during measurement.")
    if time.monotonic() > deadline:
        raise TimeoutError("Guardrail wall-time limit exceeded; no partial report is valid.")
    return {
        "manifest_sha256": manifest["manifest_sha256"], "protocol": manifest["protocol"],
        "checkpoint": checkpoint, "datasets": measurements,
        "measurement": {
            "device": device, "elapsed_seconds": time.monotonic() - started,
            "coverage": "all frozen patches in both stores",
        },
        "limitation": "Dataset/reference approval, source provenance and adoption are not established.",
    }
