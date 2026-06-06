import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZE_SCRIPT = REPO_ROOT / "scripts" / "tokenize.py"


class TokenizerRegressionTests(unittest.TestCase):
    def test_repeated_search_volume_outputs_are_identical(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            source_path = tmp / "source.zarr"
            search_path = tmp / "search.zarr"
            out1 = tmp / "out1.zarr"
            out2 = tmp / "out2.zarr"

            data = np.arange(7 * 9 * 13, dtype=np.float32).reshape(7, 9, 13)
            source = zarr.open(str(source_path), mode="w")
            source.create_array("data", data=data)

            search = zarr.open(str(search_path), mode="w")
            search.create_array("data", data=data[::-1].copy())

            base_cmd = [
                sys.executable,
                str(TOKENIZE_SCRIPT),
                "search-volume",
                "--source",
                str(source_path),
                "--search",
                str(search_path),
                "--latent-mode",
                "pooled",
                "--x",
                "3",
                "--y",
                "4",
                "--z",
                "6",
                "--batch-size",
                "8",
            ]

            r1 = subprocess.run(
                [*base_cmd, "--output", str(out1)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(r1.returncode, 0, msg=r1.stderr)

            r2 = subprocess.run(
                [*base_cmd, "--output", str(out2)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(r2.returncode, 0, msg=r2.stderr)

            a = np.asarray(zarr.open(str(out1), mode="r")["data"], dtype=np.float32)
            b = np.asarray(zarr.open(str(out2), mode="r")["data"], dtype=np.float32)
            self.assertEqual(a.shape, b.shape)
            self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
