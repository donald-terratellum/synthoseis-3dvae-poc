import time
from typing import Callable, Optional

import numpy as np

from src.tokenizer.core.preprocess import preprocess_for_token
from src.tokenizer.core.similarity import cosine_similarity, hann_window_3d


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
    patch_size: int = 32,
    stride: int = 16,
    preprocess_fn: Callable[[np.ndarray], np.ndarray] = preprocess_for_token,
    latent_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    latent_batch_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    batch_size: int = 32,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    if padded_volume.ndim != 3:
        raise ValueError("padded_volume must be 3D")
    if latent_fn is None:
        if latent_batch_fn is None:
            raise ValueError("latent_fn or latent_batch_fn must be provided")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    nx, ny, nz = padded_volume.shape
    out_sum = np.zeros_like(padded_volume, dtype=np.float32)
    out_wgt = np.zeros_like(padded_volume, dtype=np.float32)
    taper = hann_window_3d(patch_size)

    x_starts = list(iterate_window_starts(nx, patch_size, stride))
    y_starts = list(iterate_window_starts(ny, patch_size, stride))
    z_starts = list(iterate_window_starts(nz, patch_size, stride))
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
            latents = np.stack([latent_fn(c) for c in cubes_np], axis=0).astype(np.float32, copy=False)

        for (xs, ys, zs), latent in zip(batch_coords, latents):
            xe = xs + patch_size
            ye = ys + patch_size
            ze = zs + patch_size
            sim = cosine_similarity(latent, token_latent)
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
        xe = xs + patch_size
        for ys in y_starts:
            ye = ys + patch_size
            for zs in z_starts:
                if should_cancel is not None and should_cancel():
                    flush_batch()
                    return (out_sum / np.clip(out_wgt, 1e-8, None)).astype(np.float32, copy=False)
                ze = zs + patch_size

                cube = padded_volume[xs:xe, ys:ye, zs:ze]
                prep = preprocess_fn(cube)
                batch_cubes.append(prep)
                batch_coords.append((xs, ys, zs))
                if len(batch_cubes) >= batch_size:
                    flush_batch()

    flush_batch()

    out = out_sum / np.clip(out_wgt, 1e-8, None)
    return out.astype(np.float32, copy=False)
