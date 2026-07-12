import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from src.tokenizer.core.io_zarr import compute_padding, remove_padding
from src.tokenizer.core.search_engine import run_reconstruction_on_padded_volume


class VaeInferenceAlignmentTests(unittest.TestCase):
    def test_output_alignment_offset_peak_is_zero(self):
        rng = np.random.default_rng(1234)
        input_volume = rng.normal(loc=0.0, scale=1.0, size=(128, 128, 192)).astype(np.float32)

        patch_shape = (32, 32, 32)
        padding = compute_padding(input_volume.shape, patch_size=patch_shape, base_pad=16)
        padded_input = np.pad(input_volume, pad_width=padding, mode="constant", constant_values=0.0).astype(np.float32)

        def preprocess_identity(cube: np.ndarray):
            return np.asarray(cube, dtype=np.float32)

        def reconstruct_identity(batch_cubes: np.ndarray) -> np.ndarray:
            return np.asarray(batch_cubes, dtype=np.float32)

        padded_output = run_reconstruction_on_padded_volume(
            padded_volume=padded_input,
            patch_size=patch_shape,
            stride=16,
            preprocess_fn=preprocess_identity,
            reconstruct_batch_fn=reconstruct_identity,
            batch_size=8,
        )
        output_volume = remove_padding(padded_output, padding)

        self.assertEqual(output_volume.shape, input_volume.shape)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "output.zarr"
            root = zarr.open(str(out_path), mode="w")
            root.create_array("data", data=output_volume, chunks=(16, 16, 64))

            out_arr = np.asarray(root["data"], dtype=np.float32)

            x = int(rng.integers(32, input_volume.shape[0] - 32 - 32))
            y = int(rng.integers(32, input_volume.shape[1] - 32 - 32))
            z = int(rng.integers(32, input_volume.shape[2] - 64 - 32))

            out_cube = out_arr[x : x + 32, y : y + 32, z : z + 64]
            self.assertEqual(out_cube.shape, (32, 32, 64))

            best_corr = -2.0
            best_offset = None

            for dx in range(-32, 33, 1):
                for dy in range(-32, 33, 1):
                    for dz in range(-32, 33, 1):
                        x0 = x + dx
                        y0 = y + dy
                        z0 = z + dz
                        x1 = x0 + 32
                        y1 = y0 + 32
                        z1 = z0 + 64
                        if x0 < 0 or y0 < 0 or z0 < 0:
                            continue
                        if x1 > input_volume.shape[0] or y1 > input_volume.shape[1] or z1 > input_volume.shape[2]:
                            continue

                        in_cube = input_volume[x0:x1, y0:y1, z0:z1]
                        corr = float(np.corrcoef(in_cube.reshape(-1), out_cube.reshape(-1))[0, 1])
                        if np.isnan(corr):
                            continue
                        if corr > best_corr:
                            best_corr = corr
                            best_offset = (dx, dy, dz)

            self.assertIsNotNone(best_offset)
            self.assertEqual(best_offset, (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
