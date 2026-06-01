import unittest

import numpy as np

from src.tokenizer.core.io_zarr import extract_centered_cube
from src.tokenizer.core.token_picker import build_preprocessed_token_cube


class TokenPickerTests(unittest.TestCase):
    def test_extract_centered_cube_pads_at_edges(self):
        volume = np.ones((3, 3, 3), dtype=np.float32)
        cube = extract_centered_cube(volume, x=0, y=0, z=0, patch_size=4)

        self.assertEqual(cube.shape, (4, 4, 4))
        self.assertEqual(float(cube.sum()), 8.0)
        self.assertEqual(int(np.count_nonzero(cube)), 8)

    def test_build_preprocessed_token_cube_has_expected_shape_and_dtype(self):
        rng = np.random.default_rng(7)
        volume = rng.normal(size=(8, 8, 8)).astype(np.float32)
        token_cube = build_preprocessed_token_cube(volume, center_xyz=(3, 3, 3), patch_size=4)

        self.assertEqual(token_cube.shape, (4, 4, 4))
        self.assertEqual(token_cube.dtype, np.float32)
        self.assertTrue(token_cube.flags["C_CONTIGUOUS"])
        self.assertTrue(np.isfinite(token_cube).all())


if __name__ == "__main__":
    unittest.main()
