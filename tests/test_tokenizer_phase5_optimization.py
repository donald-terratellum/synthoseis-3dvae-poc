import time
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from src.tokenizer.core.batching import adaptive_batch_map
from src.tokenizer.core.jobs import SearchJobRunner, cleanup_output_artifact


class Phase5OptimizationTests(unittest.TestCase):
    def test_adaptive_batch_map_retries_with_smaller_batches(self):
        calls = []

        def fake_infer(batch: np.ndarray) -> np.ndarray:
            calls.append(int(batch.shape[0]))
            if batch.shape[0] > 2:
                raise RuntimeError("out of memory")
            return np.sum(batch, axis=1, keepdims=True).astype(np.float32)

        items = np.arange(12, dtype=np.float32).reshape(6, 2)
        out = adaptive_batch_map(items, fake_infer, initial_batch_size=6)

        self.assertEqual(out.shape, (6, 1))
        self.assertIn(6, calls)
        self.assertTrue(any(size <= 2 for size in calls))

    def test_search_job_runner_repeated_runs_are_stable(self):
        for _ in range(2):
            runner = SearchJobRunner(total_windows=24)
            runner.start()

            deadline = time.time() + 5.0
            saw_done = False
            while time.time() < deadline and not saw_done:
                for event in runner.poll_events():
                    if event.kind == "done":
                        saw_done = True
                        break
                time.sleep(0.01)

            runner.join(timeout=1.0)
            self.assertTrue(saw_done)

    def test_randomized_cancel_timing_is_stable(self):
        rng = random.Random(11)
        for _ in range(3):
            runner = SearchJobRunner(total_windows=300)
            runner.start()
            time.sleep(rng.uniform(0.01, 0.08))
            runner.cancel()

            deadline = time.time() + 5.0
            done_message = None
            while time.time() < deadline and done_message is None:
                for event in runner.poll_events():
                    if event.kind == "done":
                        done_message = event.message
                        break
                time.sleep(0.01)

            runner.join(timeout=1.0)
            self.assertEqual(done_message, "canceled")

    def test_cleanup_output_artifact_removes_existing_zarr(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "sim.zarr"
            # Pre-create output to ensure recovery cleanup semantics are exercised.
            root = zarr.open(str(out_path), mode="w")
            root.create_array("data", data=np.ones((4, 4, 4), dtype=np.float32))
            self.assertTrue(out_path.exists())

            cleanup_output_artifact(str(out_path))
            self.assertFalse(out_path.exists())


if __name__ == "__main__":
    unittest.main()
