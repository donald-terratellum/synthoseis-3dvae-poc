import numpy as np

from src.augmentations import keep_trace_extrema_only


DEFAULT_STD_EPS = 1e-6


def normalize_cube_by_std(cube: np.ndarray, std_eps: float = DEFAULT_STD_EPS) -> np.ndarray:
    arr = np.asarray(cube, dtype=np.float32)
    std = float(arr.std())
    if std < std_eps:
        std = std_eps
    return (arr / std).astype(np.float32, copy=False)


def preprocess_for_token(cube: np.ndarray, std_eps: float = DEFAULT_STD_EPS) -> np.ndarray:
    normalized = normalize_cube_by_std(cube, std_eps=std_eps)
    extrema = keep_trace_extrema_only(normalized)
    return np.ascontiguousarray(extrema, dtype=np.float32)
