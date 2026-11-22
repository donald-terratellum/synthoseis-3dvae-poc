import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import zarr

from scripts.streamed_overlap_targets import (
    Window,
    _fluid_factor,
    _fluid_fractions,
    iter_zarr_local_extrema,
    local_extrema_indices,
    pair_target,
    _smooth_extrema_input,
)


class LocalExtremaTests(unittest.TestCase):
    def test_strict_peaks_and_troughs_use_full_z_indices(self):
        cube = np.zeros((2, 1, 7), dtype=np.float32)
        cube[0, 0] = [3, 1, 4, 0, 2, 0, 5]
        cube[1, 0] = [0, 2, 2, 0, 1, 0, 0]

        i, j, k = local_extrema_indices(cube)
        self.assertEqual(set(zip(i, j, k)), {
            (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4), (0, 0, 5),
            (1, 0, 2),
        })

    def test_extrema_input_uses_weighted_five_sample_average_with_nearest_edges(self):
        cube = np.array([[[0, 0, 4, 0, 0, 0, 5, 0, 0]]], dtype=np.float32)

        smoothed = _smooth_extrema_input(cube)

        np.testing.assert_allclose(
            smoothed[0, 0],
            [0.64, 0.64, 1.44, 0.64, 1.44, 0.8, 1.8, 0.8, 0.8],
            rtol=0,
            atol=1e-6,
        )
        i, j, k = local_extrema_indices(cube)
        self.assertEqual(set(k.tolist()), {2, 3, 4, 5, 6})

    def test_nonfinite_or_non_3d_input_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "3D"):
            local_extrema_indices(np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "real numeric"):
            local_extrema_indices(np.array([[[1 + 1j, 0, 0]]]))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            local_extrema_indices(np.array([[[0.0, np.nan, 0.0]]]))

    def test_short_depth_returns_empty_index_arrays(self):
        indices = local_extrema_indices(np.zeros((2, 3, 2)))
        self.assertEqual([values.size for values in indices], [0, 0, 0])

    def test_zarr_iterator_streams_xy_chunks_and_restores_global_coordinates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "seismic.zarr"
            group = zarr.open_group(str(path), mode="w")
            cube = np.zeros((3, 4, 7), dtype=np.float32)
            cube[2, 3] = [3, 1, 4, 0, 2, 0, 5]
            array = group.create_array("seismic", data=cube, chunks=(2, 2, 7))

            chunks = list(iter_zarr_local_extrema(array, xy_chunk_shape=(2, 2)))

        actual = set()
        for i, j, k in chunks:
            actual.update(zip(i.tolist(), j.tolist(), k.tolist()))
        self.assertEqual(actual, {
            (2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 4), (2, 3, 5),
        })

    def test_decoded_chunk_limit_is_enforced_before_read(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            group = zarr.open_group(str(Path(temp_dir) / "seismic.zarr"), mode="w")
            array = group.create_array("seismic", data=np.zeros((2, 2, 7)), chunks=(2, 2, 7))
            with patch("scripts.streamed_overlap_targets.MAX_READ_BYTES", 100):
                with self.assertRaisesRegex(ValueError, "decoded/requested limit"):
                    list(iter_zarr_local_extrema(array, xy_chunk_shape=(1, 1)))


class PairTargetTests(unittest.TestCase):
    def setUp(self):
        self.source = {
            "seismicCubes_cumsum_fullstack": np.array(
                [[[3, 1, 4, 0, 2, 0, 5]]], dtype=np.float32
            ),
            "geologic_age_faulted": np.ones((1, 1, 9), dtype=np.float32),
            "oil_closures": np.zeros((1, 1, 9), dtype=np.uint8),
            "gas_closures": np.zeros((1, 1, 9), dtype=np.uint8),
        }
        self.a = Window("s", (0, 0, 0), (1, 1, 5))
        self.b = Window("s", (0, 0, 1), (1, 1, 5))

    def test_directional_events_detected_on_full_traces_not_crop_edges(self):
        ab = pair_target(self.source, self.source, self.a, self.b)
        ba = pair_target(self.source, self.source, self.b, self.a)
        self.assertAlmostEqual(ab.overlap, 0.8)
        self.assertEqual((ab.event_numerator, ab.event_denominator), (4, 4))
        self.assertEqual((ba.event_numerator, ba.event_denominator), (4, 5))
        self.assertEqual(ab.coverage, 1.0)
        self.assertEqual(ba.coverage, 0.8)

    def test_fraction_denominator_and_overlapping_closures(self):
        age = np.ones((1, 1, 10))
        oil = np.zeros_like(age)
        gas = np.zeros_like(age)
        oil[0, 0, 0] = 1
        self.assertEqual(_fluid_fractions(age, oil, gas, 1, 1), (0.1, 0, 0.9))
        gas[0, 0, 0] = 1
        np.testing.assert_allclose(_fluid_fractions(age, oil, gas, 1, 1), (0.05, 0.05, 0.9))
        self.assertEqual(_fluid_factor((1, 0, 0), (0, 1, 0)), 0)
        self.assertAlmostEqual(_fluid_factor((0.1, 0, 0.9), (0, 0, 1)), 0.9)

    def test_window_fluid_correction_and_age_support(self):
        self.source["oil_closures"][0, 0, 0] = 1
        result = pair_target(self.source, self.source, self.a, self.b)
        self.assertAlmostEqual(result.coverage, 0.8)
        self.source["geologic_age_faulted"][0, 0, 0] = 2
        result = pair_target(self.source, self.source, self.a, self.b)
        self.assertAlmostEqual(result.coverage, 1.0)

    def test_zero_negatives_and_unavailable_support(self):
        other = Window("other", self.b.origin, self.b.shape)
        self.assertEqual(pair_target(self.source, self.source, self.a, other).coverage, 0)
        disjoint = Window("s", (0, 0, 5), (1, 1, 2))
        first = Window("s", (0, 0, 0), (1, 1, 2))
        self.assertEqual(pair_target(self.source, self.source, first, disjoint).coverage, 0)
        unsupported = Window("s", (0, 0, 4), (1, 1, 5))
        self.assertIn("unsupported", pair_target(self.source, self.source, self.a, unsupported).unavailable_reason)
        del self.source["oil_closures"]
        self.assertIn("missing", pair_target(self.source, self.source, self.a, self.b).unavailable_reason)

    def test_invalid_nonfinite_empty_and_zero_denominator(self):
        self.source["seismicCubes_cumsum_fullstack"][:] = 0
        self.assertEqual(pair_target(self.source, self.source, self.a, self.b).unavailable_reason,
                         "a:zero_event_denominator")
        self.source["geologic_age_faulted"][0, 0, 1] = np.nan
        self.assertEqual(pair_target(self.source, self.source, self.a, self.b).unavailable_reason,
                         "a:nonfinite:labels")
        source = {key: np.repeat(value, 2, axis=0) for key, value in self.source.items()}
        source["geologic_age_faulted"][0] = 0
        source["geologic_age_faulted"][1] = 2
        a = Window("s", (0, 0, 0), (2, 1, 5))
        b = Window("s", (0, 0, 1), (2, 1, 5))
        self.assertEqual(pair_target(source, source, a, b).unavailable_reason,
                         "a:empty_age_support")

    def test_nonfinite_negative_is_unavailable_not_zero(self):
        other = Window("other", self.b.origin, self.b.shape)
        other_source = {key: value.copy() for key, value in self.source.items()}
        other_source["gas_closures"] = other_source["gas_closures"].astype(float)
        other_source["gas_closures"][0, 0, 1] = np.nan
        result = pair_target(self.source, other_source, self.a, other)
        self.assertIsNone(result.coverage)
        self.assertEqual(result.unavailable_reason, "b:nonfinite:labels")

    def test_excludes_known_padded_label_tail(self):
        source = {
            "seismicCubes_cumsum_fullstack": np.zeros((1, 1, 1510)),
            "geologic_age_faulted": np.zeros((1, 1, 1510)),
            "oil_closures": np.zeros((1, 1, 1510)),
            "gas_closures": np.zeros((1, 1, 1510)),
        }
        padded = Window("s", (0, 0, 1499), (1, 1, 2))
        result = pair_target(source, source, padded, padded)
        self.assertIsNone(result.overlap)
        self.assertIn("unsupported:geologic_age_faulted", result.unavailable_reason)
        larger = {key: np.pad(value, ((0, 0), (0, 0), (0, 10))) for key, value in source.items()}
        after_padding = Window("s", (0, 0, 1510), (1, 1, 2))
        self.assertEqual(pair_target(larger, larger, after_padding, after_padding).unavailable_reason,
                         "a:zero_event_denominator")


if __name__ == "__main__":
    unittest.main()
