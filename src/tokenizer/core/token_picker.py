from typing import Tuple

import numpy as np

from .io_zarr import extract_centered_cube
from .preprocess import preprocess_for_token


def build_preprocessed_token_cube(
    volume: np.ndarray,
    center_xyz: Tuple[int, int, int],
    patch_size: int = 32,
) -> np.ndarray:
    x, y, z = center_xyz
    cube = extract_centered_cube(volume, x=x, y=y, z=z, patch_size=patch_size)
    return preprocess_for_token(cube)
