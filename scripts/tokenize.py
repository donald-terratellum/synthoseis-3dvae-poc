import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import numpy as np
import time
import zarr

from src.tokenizer.config.defaults import RuntimeConfig
from src.tokenizer.core.io_zarr import (
    extract_centered_cube,
    load_volume_numpy,
    prepare_temp_search_volume,
    remove_padding,
    resolve_chunk_shape,
)
from src.tokenizer.core.metrics import write_benchmark_report
from src.tokenizer.core.model_adapter import VaeLatentAdapter, cube_to_latent_128
from src.tokenizer.core.jobs import compute_total_windows_from_padded_shape
from src.tokenizer.core.preprocess import preprocess_for_token
from src.tokenizer.core.search_engine import run_similarity_search_on_padded_volume
from src.tokenizer.core.token_picker import build_preprocessed_token_cube


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seismic tokenizer application entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_token = subparsers.add_parser("build-token", help="Build a latent token from a selected cube")
    build_token.add_argument("--source", type=Path, required=True, help="Input source seismic zarr path")
    build_token.add_argument("--key", type=str, default=None, help="Optional zarr array key")
    build_token.add_argument("--patch-size", type=int, default=32, help="Cube patch size")
    build_token.add_argument("--model-path", type=Path, default=None, help="Optional model checkpoint path")
    build_token.add_argument("--device", type=str, default="auto", help="Inference device: auto|cpu|cuda|mps")
    build_token.add_argument("--latent-mode", type=str, choices=("vae", "pooled"), default="vae")
    build_token.add_argument("--x", type=int, required=True, help="Token center x index")
    build_token.add_argument("--y", type=int, required=True, help="Token center y index")
    build_token.add_argument("--z", type=int, required=True, help="Token center z index")

    search_volume = subparsers.add_parser("search-volume", help="Run full-volume latent search")
    search_volume.add_argument("--source", type=Path, required=True, help="Source volume used for token creation")
    search_volume.add_argument("--search", type=Path, required=True, help="Search volume path")
    search_volume.add_argument("--output", type=Path, required=True, help="Output similarity volume path")
    search_volume.add_argument("--source-key", type=str, default=None, help="Optional zarr array key for source volume")
    search_volume.add_argument("--search-key", type=str, default=None, help="Optional zarr array key for search volume")
    search_volume.add_argument("--temp-zarr", type=Path, default=None, help="Optional temp zarr path override")
    search_volume.add_argument("--patch-size", type=int, default=32, help="Search window patch size")
    search_volume.add_argument("--stride", type=int, default=16, help="Search window stride")
    search_volume.add_argument("--batch-size", type=int, default=32, help="Batch size for latent inference")
    search_volume.add_argument("--model-path", type=Path, default=None, help="Optional model checkpoint path")
    search_volume.add_argument("--device", type=str, default="auto", help="Inference device: auto|cpu|cuda|mps")
    search_volume.add_argument("--latent-mode", type=str, choices=("vae", "pooled"), default="vae")
    search_volume.add_argument("--x", type=int, default=None, help="Optional token center x in source volume")
    search_volume.add_argument("--y", type=int, default=None, help="Optional token center y in source volume")
    search_volume.add_argument("--z", type=int, default=None, help="Optional token center z in source volume")
    search_volume.add_argument("--benchmark-json", type=Path, default=None, help="Optional benchmark report JSON path")

    ui = subparsers.add_parser("ui", help="Launch desktop UI")
    ui.add_argument("--source", type=Path, help="Optional source volume path to load at startup")
    return parser


def cmd_build_token(args: argparse.Namespace) -> int:
    runtime = RuntimeConfig()
    runtime.validate()
    volume = load_volume_numpy(args.source, key=args.key)
    token_cube = build_preprocessed_token_cube(
        volume,
        center_xyz=(args.x, args.y, args.z),
        patch_size=args.patch_size,
    )
    if args.latent_mode == "vae":
        model_path = args.model_path if args.model_path is not None else runtime.model_path
        adapter = VaeLatentAdapter(checkpoint_path=model_path, device=args.device)
        token_latent = adapter.encode_cube(token_cube)
    else:
        token_latent = cube_to_latent_128(token_cube)
    non_zero = int((token_cube != 0.0).sum())
    print(
        "build-token: "
        f"source={args.source} center=({args.x},{args.y},{args.z}) "
        f"cube_shape={token_cube.shape} non_zero={non_zero} latent_shape={token_latent.shape} "
        f"latent_mode={args.latent_mode} model={runtime.model_path}"
    )
    return 0


def cmd_search_volume(args: argparse.Namespace) -> int:
    runtime = RuntimeConfig()
    runtime.validate()
    started = time.time()

    if args.patch_size != 32:
        raise ValueError("patch_size currently must be 32 for latent adapter compatibility")

    source_volume = load_volume_numpy(args.source, key=args.source_key)
    search_volume = load_volume_numpy(args.search, key=args.search_key)
    temp_zarr = args.temp_zarr if args.temp_zarr is not None else runtime.temp_input_zarr
    temp_zarr.parent.mkdir(parents=True, exist_ok=True)

    cx = int(args.x) if args.x is not None else int(source_volume.shape[0] // 2)
    cy = int(args.y) if args.y is not None else int(source_volume.shape[1] // 2)
    cz = int(args.z) if args.z is not None else int(source_volume.shape[2] // 2)

    token_cube = extract_centered_cube(source_volume, x=cx, y=cy, z=cz, patch_size=args.patch_size)
    token_prep = preprocess_for_token(token_cube)
    if args.latent_mode == "vae":
        model_path = args.model_path if args.model_path is not None else runtime.model_path
        adapter = VaeLatentAdapter(checkpoint_path=model_path, device=args.device)
        token_latent = adapter.encode_cube(token_prep)
        latent_batch_fn = adapter.encode_batch
    else:
        token_latent = cube_to_latent_128(token_prep)
        latent_batch_fn = lambda cubes: np.stack([cube_to_latent_128(c) for c in cubes], axis=0).astype(np.float32)

    padded, padding, temp_chunks = prepare_temp_search_volume(
        source_volume=search_volume,
        out_zarr_path=temp_zarr,
        patch_size=args.patch_size,
        base_pad=16,
        chunk_spec=(16, 16, -1),
    )

    total_windows = compute_total_windows_from_padded_shape(
        padded_shape=padded.shape,
        patch_size=args.patch_size,
        stride=args.stride,
    )

    similarity_padded = run_similarity_search_on_padded_volume(
        padded_volume=padded,
        token_latent=token_latent,
        patch_size=args.patch_size,
        stride=args.stride,
        preprocess_fn=preprocess_for_token,
        latent_batch_fn=latent_batch_fn,
        batch_size=args.batch_size,
    )
    similarity = remove_padding(similarity_padded, padding)

    if similarity.shape != search_volume.shape:
        raise RuntimeError(
            f"output shape mismatch: got {similarity.shape}, expected {search_volume.shape}"
        )

    output_chunks = resolve_chunk_shape(search_volume.shape, (16, 16, -1))
    output_root = zarr.open(str(args.output), mode="w")
    output_root.create_array(
        "data",
        data=similarity,
        chunks=output_chunks,
    )

    elapsed = max(1e-6, time.time() - started)
    windows_per_sec = float(total_windows / elapsed)
    if args.benchmark_json is not None:
        write_benchmark_report(
            args.benchmark_json,
            {
                "status": "completed",
                "elapsed_s": float(elapsed),
                "windows_per_sec": float(windows_per_sec),
                "total_windows": int(total_windows),
                "patch_size": int(args.patch_size),
                "stride": int(args.stride),
                "batch_size": int(args.batch_size),
                "latent_mode": str(args.latent_mode),
                "device": str(args.device),
                "source_shape": list(source_volume.shape),
                "search_shape": list(search_volume.shape),
                "padded_shape": list(padded.shape),
                "output_shape": list(similarity.shape),
                "output_min": float(similarity.min()),
                "output_max": float(similarity.max()),
                "output_mean": float(similarity.mean()),
                "output_std": float(similarity.std()),
            },
        )

    print(
        "search-volume completed: "
        f"source={args.source} search={args.search} temp={temp_zarr} output={args.output} "
        f"padding={padding} padded_shape={padded.shape} temp_chunks={temp_chunks} total_windows={total_windows} "
        f"output_min={float(similarity.min()):.6f} output_max={float(similarity.max()):.6f}"
    )
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    try:
        from PySide6.QtWidgets import QApplication

        from src.tokenizer.ui.controller import TokenizerController
        from src.tokenizer.ui.main_window import MainWindow
    except ImportError:
        print("ui launch failed: PySide6 is not installed")
        return 2

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    controller = TokenizerController(window)

    if args.source:
        try:
            controller.on_source_load_requested(str(args.source))
        except Exception as exc:
            window.status_label.setText(f"Failed loading source: {exc}")

    window.show()
    return int(app.exec())


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build-token":
        return cmd_build_token(args)
    if args.command == "search-volume":
        return cmd_search_volume(args)
    if args.command == "ui":
        return cmd_ui(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
