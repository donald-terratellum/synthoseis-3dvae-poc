import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from scripts.streamed_overlap_targets import (
    iter_zarr_local_extrema,
    local_extrema_indices,
)


class LocalExtremaTests(unittest.TestCase):
    def test_strict_peaks_and_troughs_use_full_z_indices(self):
        cube = np.zeros((2, 1, 7), dtype=np.float32)
        cube[0, 0] = [3, 1, 4, 0, 2, 0, 5]
        cube[1, 0] = [0, 2, 2, 0, 1, 0, 0]

        i, j, k = local_extrema_indices(cube)
        self.assertEqual(set(zip(i, j, k)), {
            (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4), (0, 0, 5),
            (1, 0, 3), (1, 0, 4),
        })

    def test_nonfinite_or_non_3d_input_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "3D"):
            local_extrema_indices(np.zeros((2, 3)))
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


if __name__ == "__main__":
    unittest.main()
