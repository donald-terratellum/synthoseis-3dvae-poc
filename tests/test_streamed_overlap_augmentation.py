import unittest
from unittest.mock import patch

import numpy as np

from scripts.streamed_overlap_augmentation import (
    ViewTransform,
    augment_view,
    bandlimited_noise,
    spatial_view,
)
from scripts.streamed_overlap_targets import Window
from src.tokenizer.core.preprocess import normalize_cube_by_std


class StreamedAugmentationTests(unittest.TestCase):
    def test_identity_and_discrete_transforms_use_real_context(self):
        source = np.arange(12 * 12 * 12, dtype=np.float32).reshape(12, 12, 12)
        window = Window("s", (3, 3, 3), (4, 4, 4))
        identity = ViewTransform(0, (1, 1, 1), False, False, False)
        expected = source[3:7, 3:7, 3:7]
        np.testing.assert_array_equal(spatial_view(source, window, identity), expected)
        flipped = ViewTransform(0, (1, 1, 1), True, False, True)
        np.testing.assert_array_equal(
            spatial_view(source, window, flipped), np.swapaxes(np.flip(expected, axis=0), 0, 1)
        )
        edge = Window("s", (0, 0, 0), (4, 4, 4))
        rotation = ViewTransform(45, (1, 1, 1), False, False, False)
        with self.assertRaisesRegex(ValueError, "unsupported:geometric_context"):
            spatial_view(source, edge, rotation)

    def test_bandlimited_noise_has_requested_snr_and_depth_band(self):
        signal = np.ones((3, 3, 1024), dtype=np.float32)
        result, stats = bandlimited_noise(
            signal, np.random.default_rng(42),
            low_hz=5, high_hz=45, spacing_ms=4, order=4, snr_db=20,
        )
        noise = result - signal
        self.assertAlmostEqual(float(np.sqrt(np.mean(noise ** 2))), 0.1, places=5)
        self.assertEqual(stats["snr_db"], 20)
        spectrum = np.abs(np.fft.rfft(noise, axis=2)) ** 2
        frequencies = np.fft.rfftfreq(signal.shape[2], d=0.004)
        in_band = (frequencies >= 5) & (frequencies <= 45)
        self.assertGreater(float(spectrum[:, :, in_band].sum() / spectrum.sum()), 0.85)
        with self.assertRaisesRegex(ValueError, "pre_noise_rms"):
            bandlimited_noise(np.zeros_like(signal), np.random.default_rng(1),
                              low_hz=5, high_hz=45, spacing_ms=4, order=4, snr_db=20)
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            bandlimited_noise(signal, np.random.default_rng(1),
                              low_hz=5, high_hz=125, spacing_ms=4, order=4, snr_db=20)

    def test_sparse_encoder_input_does_not_modify_dense_target_or_global_rng(self):
        source = np.arange(12 * 12 * 12, dtype=np.float32).reshape(12, 12, 12) + 1
        window = Window("s", (3, 3, 3), (4, 4, 4))
        metadata = {"bandlimits_hz": [5, 45], "sample_spacing_ms_from_digi": 4,
                    "filter_order": 4, "source_noise_sn_db": -1000}
        identity = ViewTransform(0, (1, 1, 1), False, False, False)
        np.random.seed(123)
        before = np.random.get_state()
        with patch("scripts.streamed_overlap_augmentation.sample_transform", return_value=identity):
            sparse, dense, info = augment_view(
                source, window, np.random.default_rng(42), metadata, input_mode="sparse"
            )
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        np.testing.assert_array_equal(dense, normalize_cube_by_std(source[window.slices]))
        self.assertTrue(np.all((sparse == 0) | (sparse == dense)))
        self.assertGreater(np.count_nonzero(sparse), 0)
        self.assertLess(np.count_nonzero(sparse), dense.size)
        self.assertIsNone(info["noise"])

    def test_invalid_filter_metadata_is_rejected_even_without_noise(self):
        source = np.ones((12, 12, 12), dtype=np.float32)
        window = Window("s", (3, 3, 3), (4, 4, 4))
        metadata = {"bandlimits_hz": [5, 125], "sample_spacing_ms_from_digi": 4,
                    "filter_order": 4}
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            augment_view(source, window, np.random.default_rng(42), metadata)

    def test_dense_target_includes_noise_and_source_snr_is_not_reused(self):
        source = np.arange(12 * 12 * 12, dtype=np.float32).reshape(12, 12, 12) + 1
        window = Window("s", (3, 3, 3), (4, 4, 4))
        metadata = {"bandlimits_hz": [5, 45], "sample_spacing_ms_from_digi": 4,
                    "filter_order": 4, "source_noise_sn_db": -1000}
        identity = ViewTransform(0, (1, 1, 1), False, False, False)
        with patch("scripts.streamed_overlap_augmentation.sample_transform", return_value=identity):
            sparse_a, dense_a, stats = augment_view(
                source, window, np.random.default_rng(2), metadata
            )
            sparse_b, dense_b, _ = augment_view(
                source, window, np.random.default_rng(2),
                {**metadata, "source_noise_sn_db": 1000}
            )
        self.assertIsNotNone(stats["noise"])
        self.assertGreaterEqual(stats["noise"]["snr_db"], 12)
        self.assertLessEqual(stats["noise"]["snr_db"], 60)
        np.testing.assert_array_equal(sparse_a, sparse_b)
        np.testing.assert_array_equal(dense_a, dense_b)
        self.assertFalse(np.array_equal(dense_a, normalize_cube_by_std(source[window.slices])))


if __name__ == "__main__":
    unittest.main()
