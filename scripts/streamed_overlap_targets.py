"""Streaming seismic-event index helpers for the overlap experiment."""

from collections.abc import Iterator

import numpy as np
import zarr
from scipy.ndimage import uniform_filter1d


def _smooth_extrema_input(cube: np.ndarray) -> np.ndarray:
    filtered = uniform_filter1d(cube, size=5, axis=2, mode="nearest")
    return 0.2 * cube + 0.8 * filtered


def local_extrema_indices(cube: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return combined strict Z-axis peak/trough indices in a 3D cube."""
    values = np.asarray(cube)
    if values.ndim != 3:
        raise ValueError("Local extrema require a 3D XYZ array.")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError("Local extrema input must contain real numeric amplitudes.")
    values = values.astype(np.result_type(values.dtype, np.float32), copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Local extrema input contains non-finite amplitudes.")
    if values.shape[2] < 3:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty.copy(), empty.copy()

    smoothed = _smooth_extrema_input(values)
    center = smoothed[:, :, 1:-1]
    shallower = smoothed[:, :, :-2]
    deeper = smoothed[:, :, 2:]
    peak_indices = np.where((center > shallower) & (center > deeper))
    trough_indices = np.where((center < shallower) & (center < deeper))
    return (
        np.concatenate((peak_indices[0], trough_indices[0])),
        np.concatenate((peak_indices[1], trough_indices[1])),
        np.concatenate((peak_indices[2] + 1, trough_indices[2] + 1)),
    )


def iter_zarr_local_extrema(
    array: zarr.Array, xy_chunk_shape: tuple[int, int] = (50, 50)
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield source-coordinate extrema from full-depth XY chunks of a Zarr array."""
    if len(array.shape) != 3:
        raise ValueError("Streaming local extrema requires a 3D XYZ Zarr array.")
    if len(xy_chunk_shape) != 2 or any(size < 1 for size in xy_chunk_shape):
        raise ValueError("XY chunk dimensions must be positive.")

    nx, ny, _ = array.shape
    chunk_x, chunk_y = xy_chunk_shape
    for x0 in range(0, nx, chunk_x):
        x1 = min(x0 + chunk_x, nx)
        for y0 in range(0, ny, chunk_y):
            y1 = min(y0 + chunk_y, ny)
            cube = np.asarray(array[x0:x1, y0:y1, :])
            i, j, k = local_extrema_indices(cube)
            yield i + x0, j + y0, k
