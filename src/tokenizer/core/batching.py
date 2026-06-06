from typing import Callable

import numpy as np


def is_retryable_infer_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error" in msg or "mps" in msg


def adaptive_batch_map(
    items: np.ndarray,
    infer_fn: Callable[[np.ndarray], np.ndarray],
    initial_batch_size: int,
) -> np.ndarray:
    arr = np.asarray(items)
    if arr.ndim < 1:
        raise ValueError("items must have a batch axis")
    if initial_batch_size <= 0:
        raise ValueError("initial_batch_size must be > 0")

    n = int(arr.shape[0])
    if n == 0:
        return np.empty((0,), dtype=np.float32)

    out_parts = []
    idx = 0
    current_bs = min(initial_batch_size, n)

    while idx < n:
        end = min(n, idx + current_bs)
        chunk = arr[idx:end]
        try:
            out = infer_fn(chunk)
            out_parts.append(np.asarray(out))
            idx = end
        except Exception as exc:  # pragma: no cover - error type depends on backend runtime
            if current_bs <= 1 or not is_retryable_infer_error(exc):
                raise
            current_bs = max(1, current_bs // 2)

    return np.concatenate(out_parts, axis=0)
