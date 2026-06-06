import numpy as np
from typing import Sequence


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    av = np.asarray(a, dtype=np.float32).reshape(-1)
    bv = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom < eps:
        return 0.0
    return float(np.dot(av, bv) / denom)


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float32).reshape(-1)
    bv = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.dot(av, bv))


def compute_similarity(
    a: np.ndarray,
    b: np.ndarray,
    mode: str = "cosine",
    eps: float = 1e-8,
) -> float:
    mode_norm = str(mode).strip().lower()
    if mode_norm == "cosine":
        return cosine_similarity(a, b, eps=eps)
    if mode_norm == "dot":
        return dot_product(a, b)
    raise ValueError(f"unsupported similarity mode: {mode}")


def hann_window_3d(size: int | Sequence[int] = 32) -> np.ndarray:
    if isinstance(size, int):
        sx, sy, sz = int(size), int(size), int(size)
    else:
        dims = tuple(int(v) for v in size)
        if len(dims) != 3:
            raise ValueError("size must contain 3 values")
        sx, sy, sz = dims
    if sx <= 0 or sy <= 0 or sz <= 0:
        raise ValueError("size must be > 0 on all axes")

    def hann_1d(n_size: int) -> np.ndarray:
        if n_size == 1:
            return np.ones((1,), dtype=np.float32)
        n = np.arange(n_size, dtype=np.float32)
        return 0.5 - 0.5 * np.cos((2.0 * np.pi * n) / float(n_size))

    wx = hann_1d(sx)
    wy = hann_1d(sy)
    wz = hann_1d(sz)
    h3d = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
    return h3d.astype(np.float32, copy=False)
