import time
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from src.tokenizer.core.io_zarr import compute_axis_padding, compute_padding, resolve_chunk_shape
from src.tokenizer.core.jobs import (
    SearchExecutionSpec,
    SearchJobRunner,
    compute_total_windows_from_padded_shape,
    estimate_total_windows_for_source_shape,
)


class Phase3CoreTests(unittest.TestCase):
    def test_axis_padding_divisible_by_patch(self):
        low, high = compute_axis_padding(axis_len=95, patch_size=32, base_pad=16)
        padded_len = 95 + low + high
        self.assertEqual(low, 16)
        self.assertEqual(padded_len % 32, 0)

    def test_compute_padding_three_axes(self):
        padding = compute_padding((64, 95, 101), patch_size=32, base_pad=16)
        self.assertEqual(len(padding), 3)
        for axis_len, (low, high) in zip((64, 95, 101), padding):
            self.assertGreaterEqual(low, 16)
            self.assertGreaterEqual(high, 16)
            self.assertEqual((axis_len + low + high) % 32, 0)

    def test_chunk_mapping_resolves_negative_one(self):
        chunks = resolve_chunk_shape((128, 128, 96), (16, 16, -1))
        self.assertEqual(chunks, (16, 16, 96))

    def test_total_windows_from_padded_shape(self):
        total = compute_total_windows_from_padded_shape((96, 96, 96), patch_size=32, stride=16)
        self.assertEqual(total, 125)

    def test_estimate_total_windows_for_source_shape(self):
        total = estimate_total_windows_for_source_shape((64, 64, 64), patch_size=32, stride=16, base_pad=16)
        self.assertEqual(total, 125)

    def test_background_job_lifecycle_completes(self):
        runner = SearchJobRunner(total_windows=24)
        runner.start()

        deadline = time.time() + 5.0
        saw_progress = False
        saw_done = False
        while time.time() < deadline and not saw_done:
            for event in runner.poll_events():
                if event.kind == "progress":
                    saw_progress = True
                if event.kind == "done":
                    saw_done = True
            time.sleep(0.01)

        runner.join(timeout=1.0)
        self.assertTrue(saw_progress)
        self.assertTrue(saw_done)

    def test_background_job_can_cancel(self):
        runner = SearchJobRunner(total_windows=400)
        runner.start()
        time.sleep(0.05)
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

    def test_background_job_execution_spec_emits_artifact(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "sim.zarr"
            spec = SearchExecutionSpec(
                search_volume=np.ones((16, 16, 16), dtype=np.float32),
                token_latent=np.zeros((128,), dtype=np.float32),
                patch_size=32,
                stride=16,
                batch_size=4,
                output_zarr_path=str(out_path),
                latent_mode="pooled",
            )
            runner = SearchJobRunner(execution_spec=spec)
            runner.start()

            deadline = time.time() + 8.0
            saw_progress = False
            saw_done = False
            artifact_path = None
            saw_metrics = False
            while time.time() < deadline and not saw_done:
                for event in runner.poll_events():
                    if event.kind == "progress":
                        saw_progress = True
                    if event.kind == "status" and event.message.startswith("metrics:"):
                        saw_metrics = True
                    if event.kind == "artifact":
                        artifact_path = event.artifact_path
                    if event.kind == "done":
                        saw_done = True
                time.sleep(0.01)

            runner.join(timeout=1.0)
            self.assertTrue(saw_progress)
            self.assertTrue(saw_metrics)
            self.assertTrue(saw_done)
            self.assertEqual(artifact_path, str(out_path))

            root = zarr.open(str(out_path), mode="r")
            self.assertEqual(tuple(root["data"].shape), (16, 16, 16))


if __name__ == "__main__":
    unittest.main()
