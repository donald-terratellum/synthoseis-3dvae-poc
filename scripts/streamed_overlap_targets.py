"""Versioned, unaugmented canonical targets for the streamed experiment."""

from collections.abc import Iterator
from dataclasses import dataclass
import math

import numpy as np
import zarr
from scipy.ndimage import uniform_filter1d

from scripts.streamed_overlap_audit import canonical_overlap


TARGET_CONTRACT_VERSION = "observable-event-coverage-v1"
LABEL_DEPTH_LIMIT = 1500
LABEL_PADDING_END = 1510
MAX_READ_BYTES = 256 * 1024**2


@dataclass(frozen=True)
class Window:
    source_id: str
    origin: tuple[int, int, int]
    shape: tuple[int, int, int]

    def __post_init__(self):
        if not self.source_id or len(self.origin) != 3 or len(self.shape) != 3:
            raise ValueError("A window requires a source ID and XYZ origin and shape.")
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (*self.origin, *self.shape)):
            raise ValueError("Window coordinates and dimensions must be integers.")
        if any(v < 0 for v in self.origin) or any(v <= 0 for v in self.shape):
            raise ValueError("Window origin must be nonnegative and shape positive.")

    @property
    def end(self) -> tuple[int, int, int]:
        return tuple(start + size for start, size in zip(self.origin, self.shape))

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return tuple(slice(start, end) for start, end in zip(self.origin, self.end))


@dataclass(frozen=True)
class PairTarget:
    overlap: float | None
    coverage: float | None
    unavailable_reason: str | None = None
    event_numerator: int | None = None
    event_denominator: int | None = None
    fluid_a: tuple[float, float, float] | None = None
    fluid_b: tuple[float, float, float] | None = None


def _supported(array, window: Window, *, labels: bool) -> bool:
    shape = getattr(array, "shape", ())
    return (
        len(shape) == 3
        and all(end <= size for end, size in zip(window.end, shape))
        and (
            not labels
            or window.end[2] <= LABEL_DEPTH_LIMIT
            or window.origin[2] >= LABEL_PADDING_END
        )
    )


def _fluid_fractions(age: np.ndarray, oil: np.ndarray, gas: np.ndarray,
                     low: float, high: float) -> tuple[float, float, float] | None:
    selected = (age >= low) & (age <= high)
    count = int(selected.sum())
    if not count:
        return None
    oil_mask = oil[selected] > 0.5
    gas_mask = gas[selected] > 0.5
    both = int(np.count_nonzero(oil_mask & gas_mask))
    oil_fraction = (int(oil_mask.sum()) - 0.5 * both) / count
    gas_fraction = (int(gas_mask.sum()) - 0.5 * both) / count
    return oil_fraction, gas_fraction, 1.0 - oil_fraction - gas_fraction


def _fluid_factor(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return 1.0 - 0.5 * sum(abs(x - y) for x, y in zip(a, b))


def _read_bounded(array, selection: tuple) -> np.ndarray:
    if len(selection) != len(array.shape):
        raise ValueError("A target read must select every array axis.")
    spans = []
    for selector, size in zip(selection, array.shape):
        if isinstance(selector, int):
            spans.append((selector, selector + 1))
        else:
            start, stop, step = selector.indices(size)
            if step != 1:
                raise ValueError("Only contiguous reads are supported.")
            spans.append((start, stop))
    requested = math.prod(stop - start for start, stop in spans) * array.dtype.itemsize
    chunks = getattr(array, "chunks", None)
    if chunks is None:
        decoded = requested
    else:
        chunk_count = math.prod(
            (stop - 1) // chunk - start // chunk + 1
            for (start, stop), chunk in zip(spans, chunks)
        )
        decoded = chunk_count * math.prod(chunks) * array.dtype.itemsize
    if max(requested, decoded) > MAX_READ_BYTES:
        raise ValueError(f"Target read exceeds {MAX_READ_BYTES} byte decoded/requested limit.")
    return np.asarray(array[selection])


def pair_target(
    source_a,
    source_b,
    window_a: Window,
    window_b: Window,
    seismic_key: str = "seismicCubes_cumsum_fullstack",
    xy_chunk_shape: tuple[int, int] = (50, 50),
) -> PairTarget:
    """Compute labels from bounded Zarr slices before augmentation.

    Source IDs must uniquely identify the underlying store. The caller must enforce
    that invariant; different-source pairs are zero-label negatives.
    """
    if window_a.shape != window_b.shape:
        raise ValueError("Physical overlap requires equal canonical window shapes.")
    if len(xy_chunk_shape) != 2 or any(size < 1 for size in xy_chunk_shape):
        raise ValueError("XY chunk dimensions must be positive.")
    same = window_a.source_id == window_b.source_id
    overlap = canonical_overlap(window_a.origin, window_b.origin, window_a.shape, same)

    keys = (seismic_key, "geologic_age_faulted", "oil_closures", "gas_closures")
    for label, source, window in (("a", source_a, window_a), ("b", source_b, window_b)):
        for key in keys:
            if key not in source:
                return PairTarget(None if key == seismic_key else overlap, None,
                                  f"{label}:missing:{key}")
            if not _supported(source[key], window, labels=key != seismic_key):
                return PairTarget(None, None, f"{label}:unsupported:{key}")
            dtype = source[key].dtype
            if (
                not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_))
                or np.issubdtype(dtype, np.complexfloating)
            ):
                return PairTarget(None if key == seismic_key else overlap, None,
                                  f"{label}:unusable_dtype:{key}")
    if not same or overlap == 0:
        return PairTarget(overlap, 0.0)

    start = tuple(max(a, b) for a, b in zip(window_a.origin, window_b.origin))
    end = tuple(min(a, b) for a, b in zip(window_a.end, window_b.end))
    age = source_a["geologic_age_faulted"]
    edge_slices = (slice(start[0], end[0]), slice(start[1], end[1]))
    shallow = _read_bounded(age, (*edge_slices, start[2]))
    deep = _read_bounded(age, (*edge_slices, end[2] - 1))
    if not np.isfinite(shallow).all() or not np.isfinite(deep).all():
        return PairTarget(overlap, None, "nonfinite:shared_age_bounds")
    low, high = sorted((float(shallow.mean(dtype=np.float64)), float(deep.mean(dtype=np.float64))))

    fractions = []
    for label, source, window in (("a", source_a, window_a), ("b", source_b, window_b)):
        arrays = [_read_bounded(source[key], window.slices) for key in keys[1:]]
        if any(not np.isfinite(array).all() for array in arrays):
            return PairTarget(overlap, None, f"{label}:nonfinite:labels")
        result = _fluid_fractions(*arrays, low, high)
        if result is None:
            return PairTarget(overlap, None, f"{label}:empty_age_support")
        fractions.append(result)

    numerator = denominator = 0
    seismic = source_a[seismic_key]
    for x0 in range(start[0], end[0], xy_chunk_shape[0]):
        for y0 in range(start[1], end[1], xy_chunk_shape[1]):
            cube = _read_bounded(seismic, (
                slice(x0, min(x0 + xy_chunk_shape[0], end[0])),
                slice(y0, min(y0 + xy_chunk_shape[1], end[1])),
                slice(None),
            ))
            if not np.isfinite(cube).all():
                return PairTarget(overlap, None, "a:nonfinite:seismic")
            _, _, z = local_extrema_indices(cube)
            denominator += int(np.count_nonzero((z >= window_a.origin[2]) & (z < window_a.end[2])))
            numerator += int(np.count_nonzero((z >= start[2]) & (z < end[2])))
    if denominator == 0:
        return PairTarget(overlap, None, "a:zero_event_denominator")
    correction = _fluid_factor(fractions[0], fractions[1])
    return PairTarget(overlap, numerator / denominator * correction, None,
                      numerator, denominator, fractions[0], fractions[1])


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
            cube = _read_bounded(array, (slice(x0, x1), slice(y0, y1), slice(None)))
            i, j, k = local_extrema_indices(cube)
            yield i + x0, j + y0, k
