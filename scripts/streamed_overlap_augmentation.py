"""Opt-in streamed view augmentation with bounded real geometric context."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import affine_transform
from scipy.signal import butter, filtfilt, tf2zpk

from scripts.streamed_overlap_targets import MAX_READ_BYTES, Window, _read_bounded
from src.augmentations import apply_input_random_sparse_keep, keep_trace_extrema_only
from src.tokenizer.core.preprocess import normalize_cube_by_std


@dataclass(frozen=True)
class ViewTransform:
    angle_degrees: float
    scale: tuple[float, float, float]
    flip_x: bool
    flip_y: bool
    swap_xy: bool


def sample_transform(rng: np.random.Generator) -> ViewTransform:
    scales = rng.uniform(0.9, 1.1, size=3)
    return ViewTransform(
        float(rng.uniform(0, 360)),
        (float(scales[0]), float(scales[1]), float(scales[2])),
        bool(rng.random() < 0.5),
        bool(rng.random() < 0.5),
        bool(rng.random() < 0.5),
    )


def spatial_view(array, window: Window, transform: ViewTransform) -> np.ndarray:
    if len(array.shape) != 3:
        raise ValueError("Seismic array must use three XYZ axes.")
    if window.shape[0] != window.shape[1] and transform.swap_xy:
        raise ValueError("X/Y swap requires equal horizontal patch dimensions.")
    if not math.isfinite(transform.angle_degrees) or any(
        not math.isfinite(scale) or scale <= 0 for scale in transform.scale
    ):
        raise ValueError("Rotation and scale must be finite and scale must be positive.")
    angle = math.radians(transform.angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    matrix = rotation @ np.diag(1.0 / np.asarray(transform.scale))
    center_out = (np.asarray(window.shape, dtype=np.float64) - 1) / 2
    center_source = np.asarray(window.origin, dtype=np.float64) + center_out
    extent = np.abs(matrix) @ center_out
    lower = np.floor(center_source - extent).astype(int)
    upper = np.ceil(center_source + extent).astype(int) + 1
    if any(start < 0 or end > limit for start, end, limit in zip(lower, upper, array.shape)):
        raise ValueError("unsupported:geometric_context")
    slices = tuple(slice(int(start), int(end)) for start, end in zip(lower, upper))
    context = _read_bounded(array, slices)
    if not np.isfinite(context).all():
        raise ValueError("nonfinite:geometric_context")
    offset = center_source - lower - matrix @ center_out
    affine = np.eye(4)
    affine[:3, :3] = matrix
    affine[:3, 3] = offset
    view = affine_transform(
        context, affine, output_shape=window.shape,
        order=1, mode="constant", cval=np.nan, prefilter=False,
    )
    if not np.isfinite(view).all():
        raise ValueError("nonfinite:transformed_view")
    if transform.flip_x:
        view = np.flip(view, axis=0)
    if transform.flip_y:
        view = np.flip(view, axis=1)
    if transform.swap_xy:
        view = np.swapaxes(view, 0, 1)
    return np.ascontiguousarray(view, dtype=np.float32)


def _filter_coefficients(low_hz: float, high_hz: float, spacing_ms: float, order: int):
    if not all(math.isfinite(value) for value in (low_hz, high_hz, spacing_ms)):
        raise ValueError("Noise filter settings must be finite.")
    if spacing_ms <= 0 or not 0 < low_hz < high_hz < 500 / spacing_ms:
        raise ValueError("Noise band and sample spacing must satisfy Nyquist.")
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("Noise filter order must be a positive integer.")
    nyquist = 500 / spacing_ms
    coefficients = butter(order, [low_hz / nyquist, high_hz / nyquist],
                          btype="bandpass", output="ba")
    if not isinstance(coefficients, tuple) or len(coefficients) != 2:
        raise RuntimeError("Butterworth filter did not return numerator and denominator.")
    b, a = coefficients
    _, poles, _ = tf2zpk(b, a)
    pole_radius = float(np.max(np.abs(poles)))
    if not math.isfinite(pole_radius) or not 0 < pole_radius < 1:
        raise ValueError("invalid:unstable_noise_filter")
    irlen = int(math.ceil(math.log(1e-9) / math.log(pole_radius)))
    return b, a, irlen


def bandlimited_noise(
    signal: np.ndarray,
    rng: np.random.Generator,
    *,
    low_hz: float,
    high_hz: float,
    spacing_ms: float,
    order: int,
    snr_db: float,
) -> tuple[np.ndarray, dict]:
    if signal.ndim != 3 or not np.isfinite(signal).all():
        raise ValueError("Noise input must be a finite 3D XYZ array.")
    if not math.isfinite(snr_db) or not 12 <= snr_db <= 60:
        raise ValueError("Added noise SNR must be between 12 and 60 dB.")
    rms = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))
    if not math.isfinite(rms) or rms <= 0:
        raise ValueError("invalid:zero_or_nonfinite_pre_noise_rms")
    b, a, irlen = _filter_coefficients(low_hz, high_hz, spacing_ms, order)
    shape = (*signal.shape[:2], signal.shape[2] + 2 * irlen)
    if math.prod(shape) * np.dtype(np.float64).itemsize > MAX_READ_BYTES:
        raise ValueError("Noise context exceeds bounded allocation limit.")
    white = rng.standard_normal(shape)
    filtered = filtfilt(b, a, white, axis=2, method="gust", irlen=irlen)
    noise = filtered[:, :, irlen:irlen + signal.shape[2]]
    noise_rms = float(np.sqrt(np.mean(np.square(noise))))
    if not math.isfinite(noise_rms) or noise_rms <= 0:
        raise ValueError("invalid:zero_or_nonfinite_filtered_noise_rms")
    target_rms = rms * 10 ** (-snr_db / 20)
    scaled = noise * (target_rms / noise_rms)
    return np.asarray(signal + scaled, dtype=np.float32), {
        "snr_db": snr_db, "pre_noise_rms": rms,
        "added_noise_rms": target_rms, "filter_irlen": irlen,
    }


def augment_view(
    array,
    window: Window,
    rng: np.random.Generator,
    metadata: dict,
    *,
    input_mode: str = "extrema",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return sparse encoder input and normalized dense reconstruction target."""
    if input_mode not in {"extrema", "sparse"}:
        raise ValueError("input_mode must be extrema or sparse.")
    low, high = (float(value) for value in metadata["bandlimits_hz"])
    spacing = float(metadata["sample_spacing_ms_from_digi"])
    order = metadata["filter_order"]
    _filter_coefficients(low, high, spacing, order)
    transform = sample_transform(rng)
    view = spatial_view(array, window, transform)
    noise_stats = None
    if rng.random() < 0.5:
        view, noise_stats = bandlimited_noise(
            view, rng, low_hz=low, high_hz=high,
            spacing_ms=spacing, order=order, snr_db=float(rng.uniform(12, 60)),
        )
    dense = normalize_cube_by_std(view)
    if not np.isfinite(dense).all():
        raise ValueError("nonfinite:normalized_view")
    if input_mode == "extrema":
        sparse = keep_trace_extrema_only(dense)
    else:
        sparse = apply_input_random_sparse_keep(dense, rng=rng)
    return np.ascontiguousarray(sparse), np.ascontiguousarray(dense), {
        "transform": transform,
        "noise": noise_stats,
        "retained_fraction": float(np.count_nonzero(sparse) / sparse.size),
    }
