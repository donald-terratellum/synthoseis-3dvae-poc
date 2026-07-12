import time
from typing import Any, Callable, Optional, Sequence

import numpy as np

from src.tokenizer.core.preprocess import preprocess_for_token
from src.tokenizer.core.similarity import compute_similarity, hann_window_3d


def iterate_window_starts(axis_len: int, patch_size: int, stride: int):
    if axis_len < patch_size:
        raise ValueError("axis_len must be >= patch_size")
    if (axis_len - patch_size) % stride != 0:
        raise ValueError("axis_len must align to patch_size/stride grid")
    for start in range(0, axis_len - patch_size + 1, stride):
        yield start


def run_similarity_search_on_padded_volume(
    padded_volume: np.ndarray,
    token_latent: np.ndarray,
    patch_size: int | Sequence[int] = 32,
    stride: int = 16,
    preprocess_fn: Callable[[np.ndarray], np.ndarray] = preprocess_for_token,
    latent_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    latent_batch_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    similarity_mode: str = "cosine",
    batch_size: int = 32,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    if padded_volume.ndim != 3:
        raise ValueError("padded_volume must be 3D")
    if latent_fn is None:
        if latent_batch_fn is None:
            raise ValueError("latent_fn or latent_batch_fn must be provided")
    single_latent_fn = latent_fn
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    if isinstance(patch_size, int):
        sx, sy, sz = int(patch_size), int(patch_size), int(patch_size)
    else:
        dims = tuple(int(v) for v in patch_size)
        if len(dims) != 3:
            raise ValueError("patch_size must contain 3 values")
        sx, sy, sz = dims

    nx, ny, nz = padded_volume.shape
    out_sum = np.zeros_like(padded_volume, dtype=np.float32)
    out_wgt = np.zeros_like(padded_volume, dtype=np.float32)
    taper = hann_window_3d((sx, sy, sz))

    x_starts = list(iterate_window_starts(nx, sx, stride))
    y_starts = list(iterate_window_starts(ny, sy, stride))
    z_starts = list(iterate_window_starts(nz, sz, stride))
    total_windows = int(len(x_starts) * len(y_starts) * len(z_starts))
    completed_windows = 0
    started = time.time()

    batch_cubes: list[np.ndarray] = []
    batch_coords: list[tuple[int, int, int]] = []

    def flush_batch() -> None:
        nonlocal completed_windows
        if not batch_cubes:
            return

        cubes_np = np.stack(batch_cubes, axis=0).astype(np.float32, copy=False)
        if latent_batch_fn is not None:
            latents = latent_batch_fn(cubes_np)
        else:
            if single_latent_fn is None:
                raise ValueError("latent_fn or latent_batch_fn must be provided")
            latents = np.stack([single_latent_fn(c) for c in cubes_np], axis=0).astype(np.float32, copy=False)

        for (xs, ys, zs), latent in zip(batch_coords, latents):
            xe = xs + sx
            ye = ys + sy
            ze = zs + sz
            sim = compute_similarity(latent, token_latent, mode=similarity_mode)
            out_sum[xs:xe, ys:ye, zs:ze] += sim * taper
            out_wgt[xs:xe, ys:ye, zs:ze] += taper
            completed_windows += 1

        if progress_callback is not None:
            elapsed = max(1e-6, time.time() - started)
            rate = completed_windows / elapsed
            remaining = max(0, total_windows - completed_windows)
            eta = remaining / max(rate, 1e-6)
            progress_callback(completed_windows, total_windows, float(eta))

        batch_cubes.clear()
        batch_coords.clear()

    for xs in x_starts:
        xe = xs + sx
        for ys in y_starts:
            ye = ys + sy
            for zs in z_starts:
                if should_cancel is not None and should_cancel():
                    flush_batch()
                    return (out_sum / np.clip(out_wgt, 1e-8, None)).astype(np.float32, copy=False)
                ze = zs + sz

                cube = padded_volume[xs:xe, ys:ye, zs:ze]
                prep = preprocess_fn(cube)
                batch_cubes.append(prep)
                batch_coords.append((xs, ys, zs))
                if len(batch_cubes) >= batch_size:
                    flush_batch()

    flush_batch()

    out = out_sum / np.clip(out_wgt, 1e-8, None)
    return out.astype(np.float32, copy=False)


def run_reconstruction_on_padded_volume(
    padded_volume: np.ndarray,
    patch_size: int | Sequence[int] = 32,
    stride: int = 16,
    preprocess_fn: Callable[[np.ndarray], np.ndarray | tuple[np.ndarray, Any]] = preprocess_for_token,
    reconstruct_batch_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    postprocess_fn: Callable[[np.ndarray, Any], np.ndarray] | None = None,
    batch_size: int = 32,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    if padded_volume.ndim != 3:
        raise ValueError("padded_volume must be 3D")
    if reconstruct_batch_fn is None:
        raise ValueError("reconstruct_batch_fn must be provided")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    if isinstance(patch_size, int):
        sx, sy, sz = int(patch_size), int(patch_size), int(patch_size)
    else:
        dims = tuple(int(v) for v in patch_size)
        if len(dims) != 3:
            raise ValueError("patch_size must contain 3 values")
        sx, sy, sz = dims

    nx, ny, nz = padded_volume.shape
    out_sum = np.zeros_like(padded_volume, dtype=np.float32)
    out_wgt = np.zeros_like(padded_volume, dtype=np.float32)
    taper = hann_window_3d((sx, sy, sz))

    x_starts = list(iterate_window_starts(nx, sx, stride))
    y_starts = list(iterate_window_starts(ny, sy, stride))
    z_starts = list(iterate_window_starts(nz, sz, stride))
    total_windows = int(len(x_starts) * len(y_starts) * len(z_starts))
    completed_windows = 0
    started = time.time()

    batch_cubes: list[np.ndarray] = []
    batch_meta: list[Any] = []
    batch_coords: list[tuple[int, int, int]] = []

    def flush_batch() -> None:
        nonlocal completed_windows
        if not batch_cubes:
            return

        cubes_np = np.stack(batch_cubes, axis=0).astype(np.float32, copy=False)
        recons = reconstruct_batch_fn(cubes_np)
        recons = np.asarray(recons, dtype=np.float32)
        if recons.shape != cubes_np.shape:
            raise ValueError(f"reconstruct_batch_fn must return shape {cubes_np.shape}, got {recons.shape}")

        for (xs, ys, zs), recon, meta in zip(batch_coords, recons, batch_meta):
            xe = xs + sx
            ye = ys + sy
            ze = zs + sz

            recon_patch = recon
            if postprocess_fn is not None:
                recon_patch = np.asarray(postprocess_fn(recon_patch, meta), dtype=np.float32)
            if recon_patch.shape != (sx, sy, sz):
                raise ValueError(
                    f"postprocessed reconstruction patch must match {(sx, sy, sz)}, got {recon_patch.shape}"
                )

            out_sum[xs:xe, ys:ye, zs:ze] += recon_patch * taper
            out_wgt[xs:xe, ys:ye, zs:ze] += taper
            completed_windows += 1

        if progress_callback is not None:
            elapsed = max(1e-6, time.time() - started)
            rate = completed_windows / elapsed
            remaining = max(0, total_windows - completed_windows)
            eta = remaining / max(rate, 1e-6)
            progress_callback(completed_windows, total_windows, float(eta))

        batch_cubes.clear()
        batch_meta.clear()
        batch_coords.clear()

    for xs in x_starts:
        xe = xs + sx
        for ys in y_starts:
            ye = ys + sy
            for zs in z_starts:
                if should_cancel is not None and should_cancel():
                    flush_batch()
                    return (out_sum / np.clip(out_wgt, 1e-8, None)).astype(np.float32, copy=False)
                ze = zs + sz

                cube = padded_volume[xs:xe, ys:ye, zs:ze]
                prepped = preprocess_fn(cube)
                if isinstance(prepped, tuple):
                    prep_cube = np.asarray(prepped[0], dtype=np.float32)
                    prep_meta = prepped[1]
                else:
                    prep_cube = np.asarray(prepped, dtype=np.float32)
                    prep_meta = None

                if prep_cube.shape != (sx, sy, sz):
                    raise ValueError(f"preprocess_fn must return shape {(sx, sy, sz)}, got {prep_cube.shape}")

                batch_cubes.append(prep_cube)
                batch_meta.append(prep_meta)
                batch_coords.append((xs, ys, zs))
                if len(batch_cubes) >= batch_size:
                    flush_batch()

    flush_batch()

    out = out_sum / np.clip(out_wgt, 1e-8, None)
    return out.astype(np.float32, copy=False)
