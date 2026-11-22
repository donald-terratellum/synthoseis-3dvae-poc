import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import zarr

from scripts.streamed_overlap_audit import LABEL_KEYS, Source
from scripts.streamed_overlap_yield import SEISMIC_KEY, run_preflight


class YieldPreflightTests(unittest.TestCase):
    def test_train_validation_yield_is_deterministic_and_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for split in ("train", "validation"):
                for number in range(2):
                    path = root / split / str(number) / "model_data.zarr"
                    group = zarr.open_group(str(path), mode="w")
                    signal = np.broadcast_to(
                        np.sin(np.arange(128, dtype=np.float32)), (32, 32, 128)
                    ).copy()
                    group.create_array(SEISMIC_KEY, data=signal, chunks=(32, 32, 128))
                    for key in LABEL_KEYS:
                        group.create_array(key, data=np.ones_like(signal), chunks=(32, 32, 128))
                    sources.append(Source(path, split))
            before = {path: path.stat().st_mtime_ns for path in root.rglob("*")}
            with patch("scripts.streamed_overlap_yield.discover_sources", return_value=sources), \
                 patch("scripts.streamed_overlap_yield.inspect_source", return_value={"issues": []}):
                first = run_preflight(root, seed=7, samples_per_source=1)
                second = run_preflight(root, seed=7, samples_per_source=1)
            self.assertEqual(first["pairs"], second["pairs"])
            self.assertEqual(first["summary"]["train"]["pairs_by_category"], {
                "same_zero_shift": 2, "same_z16": 2,
                "same_disjoint": 2, "cross_source": 2,
            })
            self.assertEqual(first["summary"]["validation"]["unavailable_reasons"], {})
            self.assertEqual(first["status"], "LOCAL_TARGET_VALIDATION_PENDING")
            self.assertEqual(before, {path: path.stat().st_mtime_ns for path in root.rglob("*")})


if __name__ == "__main__":
    unittest.main()
