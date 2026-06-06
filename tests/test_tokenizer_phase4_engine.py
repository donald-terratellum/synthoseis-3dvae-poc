import unittest

import numpy as np

from src.tokenizer.core.model_adapter import cube_to_latent_128
from src.tokenizer.core.preprocess import preprocess_for_token
from src.tokenizer.core.search_engine import iterate_window_starts, run_similarity_search_on_padded_volume
from src.tokenizer.core.similarity import cosine_similarity, hann_window_3d


class Phase4EngineTests(unittest.TestCase):
    def test_cosine_similarity_expected_values(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        c = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        z = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.assertAlmostEqual(cosine_similarity(a, b), 1.0, places=6)
        self.assertAlmostEqual(cosine_similarity(a, c), -1.0, places=6)
        self.assertAlmostEqual(cosine_similarity(a, z), 0.0, places=6)

    def test_hann_window_shape_and_range(self):
        h = hann_window_3d(32)
        self.assertEqual(h.shape, (32, 32, 32))
        self.assertEqual(h.dtype, np.float32)
        self.assertGreaterEqual(float(h.min()), 0.0)
        self.assertLessEqual(float(h.max()), 1.0)

    def test_hann_window_overlap_add_is_constant_in_full_overlap_region(self):
        patch_size = 32
        stride = 16
        shape = (64, 64, 64)
        taper = hann_window_3d(patch_size)
        out_sum = np.zeros(shape, dtype=np.float32)
        ones_cube = np.ones((patch_size, patch_size, patch_size), dtype=np.float32)

        for xs in iterate_window_starts(shape[0], patch_size, stride):
            xe = xs + patch_size
            for ys in iterate_window_starts(shape[1], patch_size, stride):
                ye = ys + patch_size
                for zs in iterate_window_starts(shape[2], patch_size, stride):
                    ze = zs + patch_size
                    out_sum[xs:xe, ys:ye, zs:ze] += ones_cube * taper

        full_overlap_region = out_sum[16:48, 16:48, 16:48]
        self.assertTrue(np.isfinite(out_sum).all())
        self.assertGreater(float(full_overlap_region.size), 0.0)
        self.assertLess(
            float(full_overlap_region.max() - full_overlap_region.min()),
            1e-5,
        )

    def test_deterministic_overlap_search_constant_volume_yields_zero_output(self):
        # Constant input -> extrema-only preprocessing yields zeros -> zero latent vectors -> zero cosine.
        padded = np.ones((64, 64, 64), dtype=np.float32)
        token_latent = np.zeros((128,), dtype=np.float32)

        out = run_similarity_search_on_padded_volume(
            padded_volume=padded,
            token_latent=token_latent,
            patch_size=32,
            stride=16,
            preprocess_fn=preprocess_for_token,
            latent_fn=cube_to_latent_128,
        )

        self.assertEqual(out.shape, padded.shape)
        self.assertTrue(np.isfinite(out).all())
        self.assertAlmostEqual(float(out.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(out.max()), 0.0, places=6)
        self.assertAlmostEqual(float(out.min()), 0.0, places=6)

    def test_overlap_search_batched_latent_path(self):
        padded = np.ones((64, 64, 64), dtype=np.float32)
        token_latent = np.zeros((128,), dtype=np.float32)

        def batch_latent(cubes: np.ndarray) -> np.ndarray:
            return np.stack([cube_to_latent_128(c) for c in cubes], axis=0).astype(np.float32)

        out = run_similarity_search_on_padded_volume(
            padded_volume=padded,
            token_latent=token_latent,
            patch_size=32,
            stride=16,
            preprocess_fn=preprocess_for_token,
            latent_batch_fn=batch_latent,
            batch_size=4,
        )

        self.assertEqual(out.shape, padded.shape)
        self.assertTrue(np.isfinite(out).all())

    def test_overlap_search_similarity_mode_switch_changes_values(self):
        padded = np.ones((2, 2, 2), dtype=np.float32)
        token_latent = np.array([2.0], dtype=np.float32)

        def identity_preprocess(cube: np.ndarray) -> np.ndarray:
            return cube

        def latent_scalar(cube: np.ndarray) -> np.ndarray:
            return np.array([float(cube.reshape(-1)[0])], dtype=np.float32)

        out_cos = run_similarity_search_on_padded_volume(
            padded_volume=padded,
            token_latent=token_latent,
            patch_size=1,
            stride=1,
            preprocess_fn=identity_preprocess,
            latent_fn=latent_scalar,
            similarity_mode="cosine",
        )
        out_dot = run_similarity_search_on_padded_volume(
            padded_volume=padded,
            token_latent=token_latent,
            patch_size=1,
            stride=1,
            preprocess_fn=identity_preprocess,
            latent_fn=latent_scalar,
            similarity_mode="dot",
        )

        self.assertTrue(np.allclose(out_cos, 1.0, atol=1e-6))
        self.assertTrue(np.allclose(out_dot, 2.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
