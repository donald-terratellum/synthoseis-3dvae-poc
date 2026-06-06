from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import zarr


def normalize_patch_size(patch_size: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(patch_size, int):
        dims = (int(patch_size), int(patch_size), int(patch_size))
    else:
        values = tuple(int(v) for v in patch_size)
        if len(values) == 1:
            dims = (values[0], values[0], values[0])
        elif len(values) == 3:
            dims = values
        else:
            raise ValueError("patch_size expects either 1 value or 3 values")
    if any(v <= 0 for v in dims):
        raise ValueError("patch_size values must be > 0")
    return dims


def open_volume_array(zarr_path: Path, key: Optional[str] = None):
    root = zarr.open(str(zarr_path), mode="r")

    if hasattr(root, "shape"):
        return root

    if key:
        return root[key]

    for candidate in ("seismic", "volume", "data"):
        if candidate in root:
            return root[candidate]

    array_keys = list(root.array_keys())
    if not array_keys:
        raise ValueError(f"No arrays found in zarr path: {zarr_path}")
    return root[array_keys[0]]


def load_volume_numpy(zarr_path: Path, key: Optional[str] = None) -> np.ndarray:
    arr = open_volume_array(zarr_path, key=key)
    return np.asarray(arr, dtype=np.float32)


def extract_centered_cube(
    volume: np.ndarray,
    x: int,
    y: int,
    z: int,
    patch_size: int | Sequence[int] = 32,
) -> np.ndarray:
    sx, sy, sz = normalize_patch_size(patch_size)
    if volume.ndim != 3:
        raise ValueError("volume must be a 3D array")

    hx, hy, hz = sx // 2, sy // 2, sz // 2
    x0, x1 = x - hx, x - hx + sx
    y0, y1 = y - hy, y - hy + sy
    z0, z1 = z - hz, z - hz + sz

    xs0, xs1 = max(0, x0), min(volume.shape[0], x1)
    ys0, ys1 = max(0, y0), min(volume.shape[1], y1)
    zs0, zs1 = max(0, z0), min(volume.shape[2], z1)

    out = np.zeros((sx, sy, sz), dtype=np.float32)

    ox0, oy0, oz0 = xs0 - x0, ys0 - y0, zs0 - z0
    ox1, oy1, oz1 = ox0 + (xs1 - xs0), oy0 + (ys1 - ys0), oz0 + (zs1 - zs0)

    if xs1 > xs0 and ys1 > ys0 and zs1 > zs0:
        out[ox0:ox1, oy0:oy1, oz0:oz1] = volume[xs0:xs1, ys0:ys1, zs0:zs1]

    return out


def compute_axis_padding(axis_len: int, patch_size: int = 32, base_pad: int = 16) -> Tuple[int, int]:
    if axis_len <= 0:
        raise ValueError("axis_len must be > 0")
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if base_pad < 0:
        raise ValueError("base_pad must be >= 0")

    padded = axis_len + (2 * base_pad)
    remainder = padded % patch_size
    extra_tail = 0 if remainder == 0 else patch_size - remainder
    return base_pad, base_pad + extra_tail


def compute_padding(volume_shape: Sequence[int], patch_size: int = 32, base_pad: int = 16) -> Tuple[Tuple[int, int], ...]:
    if len(volume_shape) != 3:
        raise ValueError("volume_shape must have 3 axes")
    patch_shape = normalize_patch_size(patch_size)
    return tuple(
        compute_axis_padding(int(axis), patch_size=int(patch_axis), base_pad=base_pad)
        for axis, patch_axis in zip(volume_shape, patch_shape)
    )


def resolve_chunk_shape(volume_shape: Sequence[int], chunk_spec: Sequence[int]) -> Tuple[int, ...]:
    if len(volume_shape) != len(chunk_spec):
        raise ValueError("volume_shape and chunk_spec lengths must match")

    resolved = []
    for axis_len, chunk in zip(volume_shape, chunk_spec):
        if chunk == -1:
            resolved.append(int(axis_len))
        elif chunk <= 0:
            raise ValueError("chunk sizes must be positive or -1")
        else:
            resolved.append(int(chunk))
    return tuple(resolved)


def prepare_temp_search_volume(
    source_volume: np.ndarray,
    out_zarr_path: Path,
    patch_size: int | Sequence[int] = 32,
    base_pad: int = 16,
    chunk_spec: Sequence[int] = (16, 16, -1),
) -> tuple[np.ndarray, tuple[tuple[int, int], ...], tuple[int, ...]]:
    if source_volume.ndim != 3:
        raise ValueError("source_volume must be 3D")

    padding = compute_padding(source_volume.shape, patch_size=patch_size, base_pad=base_pad)
    padded = np.pad(source_volume, pad_width=padding, mode="constant", constant_values=0.0).astype(np.float32)
    chunks = resolve_chunk_shape(padded.shape, chunk_spec)

    root = zarr.open(str(out_zarr_path), mode="w")
    root.create_array("data", data=padded, chunks=chunks)

    return padded, padding, chunks


def remove_padding(padded_volume: np.ndarray, padding: Sequence[Tuple[int, int]]) -> np.ndarray:
    if padded_volume.ndim != 3:
        raise ValueError("padded_volume must be 3D")
    if len(padding) != 3:
        raise ValueError("padding must contain 3 axis tuples")

    xs = slice(int(padding[0][0]), padded_volume.shape[0] - int(padding[0][1]))
    ys = slice(int(padding[1][0]), padded_volume.shape[1] - int(padding[1][1]))
    zs = slice(int(padding[2][0]), padded_volume.shape[2] - int(padding[2][1]))
    return np.asarray(padded_volume[xs, ys, zs], dtype=np.float32)
