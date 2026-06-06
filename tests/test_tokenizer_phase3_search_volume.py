import subprocess
import sys
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import zarr

from src.tokenizer.core.io_zarr import compute_padding, resolve_chunk_shape
from src.tokenizer.core.jobs import compute_total_windows_from_padded_shape


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZE_SCRIPT = REPO_ROOT / "scripts" / "tokenize.py"


class SearchVolumeIntegrationTests(unittest.TestCase):
    def test_search_volume_prepares_temp_zarr_with_expected_shape_and_chunks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            source_path = tmp / "source.zarr"
            search_path = tmp / "search.zarr"
            temp_path = tmp / "temp_input.zarr"
            output_path = tmp / "output.zarr"
            bench_path = tmp / "bench.json"

            source = zarr.open(str(source_path), mode="w")
            source.create_array("data", data=np.ones((7, 9, 13), dtype=np.float32))

            search_data = np.ones((7, 9, 13), dtype=np.float32)
            search = zarr.open(str(search_path), mode="w")
            search.create_array("data", data=search_data)

            result = subprocess.run(
                [
                    sys.executable,
                    str(TOKENIZE_SCRIPT),
                    "search-volume",
                    "--source",
                    str(source_path),
                    "--search",
                    str(search_path),
                    "--output",
                    str(output_path),
                    "--temp-zarr",
                    str(temp_path),
                    "--latent-mode",
                    "pooled",
                    "--benchmark-json",
                    str(bench_path),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)

            padding = compute_padding(search_data.shape, patch_size=32, base_pad=16)
            expected_padded_shape = tuple(
                dim + low + high for dim, (low, high) in zip(search_data.shape, padding)
            )
            expected_temp_chunks = resolve_chunk_shape(expected_padded_shape, (16, 16, -1))
            expected_output_chunks = resolve_chunk_shape(search_data.shape, (16, 16, -1))

            temp_root = zarr.open(str(temp_path), mode="r")
            self.assertEqual(tuple(temp_root["data"].shape), expected_padded_shape)
            self.assertEqual(tuple(temp_root["data"].chunks), expected_temp_chunks)

            out_root = zarr.open(str(output_path), mode="r")
            self.assertEqual(tuple(out_root["data"].shape), tuple(search_data.shape))
            self.assertEqual(tuple(out_root["data"].chunks), expected_output_chunks)

            self.assertTrue(bench_path.exists())
            bench = json.loads(bench_path.read_text(encoding="utf-8"))
            self.assertEqual(bench["status"], "completed")
            expected_windows = compute_total_windows_from_padded_shape(expected_padded_shape, patch_size=32, stride=16)
            self.assertEqual(bench["total_windows"], expected_windows)
            self.assertIn("windows_per_sec", bench)
            self.assertIn("elapsed_s", bench)


if __name__ == "__main__":
    unittest.main()
