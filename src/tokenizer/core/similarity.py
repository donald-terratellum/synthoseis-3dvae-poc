import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    av = np.asarray(a, dtype=np.float32).reshape(-1)
    bv = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom < eps:
        return 0.0
    return float(np.dot(av, bv) / denom)


def hann_window_3d(size: int = 32) -> np.ndarray:
    if size <= 0:
        raise ValueError("size must be > 0")
    if size == 1:
        return np.ones((1, 1, 1), dtype=np.float32)
    n = np.arange(size, dtype=np.float32)
    w = 0.5 - 0.5 * np.cos((2.0 * np.pi * n) / float(size))
    h3d = w[:, None, None] * w[None, :, None] * w[None, None, :]
    return h3d.astype(np.float32, copy=False)
