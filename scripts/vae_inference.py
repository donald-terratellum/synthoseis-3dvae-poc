import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import time

import numpy as np
import zarr

from src.tokenizer.core.io_zarr import (
    compute_padding,
    normalize_patch_size,
    remove_padding,
    resolve_chunk_shape,
)
from src.tokenizer.core.model_adapter import VaeLatentAdapter
from src.tokenizer.core.search_engine import run_reconstruction_on_padded_volume


def _resolve_input_array_and_key(zarr_path: Path, key: str | None):
    root = zarr.open(str(zarr_path), mode="r")

    if hasattr(root, "shape"):
        return root, "data"

    if key:
        if key not in root:
            raise ValueError(f"Requested input key '{key}' not found in {zarr_path}")
        return root[key], key

    for candidate in ("seismic", "volume", "data"):
        if candidate in root:
            return root[candidate], candidate

    array_keys = list(root.array_keys())
    if not array_keys:
        raise ValueError(f"No arrays found in zarr path: {zarr_path}")
    return root[array_keys[0]], array_keys[0]


def _normalize_for_reconstruction(cube: np.ndarray, std_eps: float = 1e-6) -> tuple[np.ndarray, float]:
    arr = np.asarray(cube, dtype=np.float32)
    std = float(arr.std())
    if std < std_eps:
        std = std_eps
    return np.ascontiguousarray(arr / std, dtype=np.float32), std


def _denormalize_reconstruction(recon_cube: np.ndarray, std: float) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(recon_cube, dtype=np.float32) * float(std), dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full VAE reconstruction inference from seismic zarr to output zarr"
    )
    parser.add_argument("--input-zarr", type=Path, required=True, help="Input seismic zarr store path")
    parser.add_argument("--model", type=Path, required=True, help="Trained VAE checkpoint .pt path")
    parser.add_argument("--output-zarr", type=Path, default=None, help="Output zarr store path (default: input store)")
    parser.add_argument("--input-key", type=str, default=None, help="Input array key in zarr store")
    parser.add_argument(
        "--output-key",
        type=str,
        default=None,
        help="Output array key in zarr store (default: <input-key>.vae-inference)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs="+",
        default=None,
        help="Patch size: one value for cubic or three values X Y Z (default: checkpoint patch_shape)",
    )
    parser.add_argument("--stride", type=int, default=16, help="Sliding window stride")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for reconstruction inference")
    parser.add_argument("--base-pad", type=int, default=16, help="Base symmetric zero-pad before inference")
    parser.add_argument("--device", type=str, default="auto", help="Inference device: auto|cpu|cuda|mps")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output array key if present")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()

    adapter = VaeLatentAdapter(checkpoint_path=args.model, device=args.device)
    patch_shape = normalize_patch_size(args.patch_size) if args.patch_size is not None else tuple(adapter.patch_shape)

    input_arr, resolved_input_key = _resolve_input_array_and_key(args.input_zarr, args.input_key)
    input_data = np.asarray(input_arr, dtype=np.float32)
    if input_data.ndim != 3:
        raise ValueError(f"Input array must be 3D, got shape {input_data.shape}")

    if tuple(patch_shape) != tuple(adapter.patch_shape):
        raise ValueError(
            f"patch_size {tuple(patch_shape)} must match checkpoint patch_shape {tuple(adapter.patch_shape)}"
        )

    output_zarr = args.output_zarr if args.output_zarr is not None else args.input_zarr
    output_key = args.output_key if args.output_key else f"{resolved_input_key}.vae-inference"

    padding = compute_padding(input_data.shape, patch_size=patch_shape, base_pad=int(args.base_pad))
    padded = np.pad(input_data, pad_width=padding, mode="constant", constant_values=0.0).astype(np.float32, copy=False)

    def on_progress(completed: int, total: int, eta_seconds: float) -> None:
        print(f"progress windows={completed}/{total} eta_s={eta_seconds:.1f}")

    recon_padded = run_reconstruction_on_padded_volume(
        padded_volume=padded,
        patch_size=patch_shape,
        stride=int(args.stride),
        preprocess_fn=_normalize_for_reconstruction,
        reconstruct_batch_fn=adapter.reconstruct_batch,
        postprocess_fn=_denormalize_reconstruction,
        batch_size=int(args.batch_size),
        progress_callback=on_progress,
    )
    recon = remove_padding(recon_padded, padding)

    if recon.shape != input_data.shape:
        raise RuntimeError(f"reconstruction shape mismatch: got {recon.shape}, expected {input_data.shape}")

    output_root = zarr.open(str(output_zarr), mode="a")
    if output_key in output_root:
        if not args.overwrite:
            raise ValueError(
                f"Output key '{output_key}' already exists in {output_zarr}. Use --overwrite to replace it."
            )
        del output_root[output_key]

    input_chunks = getattr(input_arr, "chunks", None)
    if input_chunks is not None and len(input_chunks) == 3:
        output_chunks = resolve_chunk_shape(recon.shape, input_chunks)
    else:
        output_chunks = resolve_chunk_shape(recon.shape, (16, 16, -1))

    output_root.create_array(output_key, data=recon, chunks=output_chunks)

    elapsed = max(1e-6, time.time() - started)
    print(
        "vae-inference completed: "
        f"input_zarr={args.input_zarr} input_key={resolved_input_key} "
        f"model={args.model} output_zarr={output_zarr} output_key={output_key} "
        f"shape={recon.shape} elapsed_s={elapsed:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
