import unittest
import numpy as np

from src.augmentations import apply_vertical_warp_to_cube, sample_vertical_warp_target_indices


class VerticalWarpTests(unittest.TestCase):
    def test_sampled_vertical_warp_indices_obey_constraints(self):
        nz = 32
        for _ in range(512):
            target = sample_vertical_warp_target_indices(nz)
            increments = np.diff(target)

            self.assertEqual(target.shape, (nz,))
            self.assertTrue(np.isclose(float(target[0]), 0.0, atol=1e-6))
            self.assertTrue(np.isclose(float(target[-1]), float(nz - 1), atol=1e-6))
            self.assertTrue(np.all(increments > 0.0))
            self.assertGreaterEqual(float(increments.min()), 0.5 - 1e-6)
            self.assertLessEqual(float(increments.max()), 2.0 + 1e-6)
            self.assertTrue(np.isclose(float(increments.mean()), 1.0, atol=1e-6))

    def test_apply_vertical_warp_to_cube_preserves_shape_and_endpoints(self):
        nz = 32
        source = np.arange(nz, dtype=np.float32)
        cube = np.broadcast_to(source, (3, 2, nz)).copy()

        current = np.arange(nz, dtype=np.float32)
        control_src = np.array([-5.0, 0.0, 15.5, 31.0, 36.0], dtype=np.float32)
        control_dst = np.array([-5.0, 0.0, 13.0, 31.0, 36.0], dtype=np.float32)
        target = np.interp(current, control_src, control_dst).astype(np.float32)

        warped = apply_vertical_warp_to_cube(cube, target)

        self.assertEqual(warped.shape, cube.shape)
        self.assertTrue(np.isclose(float(warped[0, 0, 0]), 0.0, atol=1e-6))
        self.assertTrue(np.isclose(float(warped[0, 0, -1]), 31.0, atol=1e-6))
        self.assertFalse(np.isclose(float(warped[0, 0, 15]), float(cube[0, 0, 15]), atol=1e-3))


if __name__ == '__main__':
    unittest.main()
