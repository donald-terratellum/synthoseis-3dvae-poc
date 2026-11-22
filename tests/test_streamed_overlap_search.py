import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import zarr

from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_integrity import _digest_array, _window_record
from scripts.streamed_overlap_manifest import _window
from scripts.streamed_overlap_search import (
    PATCH_SHAPE, _search_results, freeze_search_manifest, run_bounded_searches,
)
from scripts.streamed_overlap_yield import SEISMIC_KEY
from src.tokenizer.core.io_zarr import compute_padding
from tests.test_streamed_overlap_metrics import frozen_fixture


class BoundedSearchTests(unittest.TestCase):
    def setUp(self):
        self.manifest = frozen_fixture()
        self.query = self.manifest["queries"][0]
        self.query["window"].update({"origin": [0, 0, 0], "shape": list(PATCH_SHAPE)})
        self.window = _window(self.query["window"])
        self.data = np.arange(48 * 48 * 80, dtype=np.float32).reshape(48, 48, 80)
        self.groups = {self.window.source_id: {SEISMIC_KEY: self.data}}
        record = _window_record(self.window)
        self.manifest["input_integrity"]["fingerprints"]["windows"][content_hash(record)] = {
            "window": record, "arrays": {SEISMIC_KEY: _digest_array(self.data[self.window.slices])},
        }
        self.manifest["manifest_sha256"] = content_hash({
            k: v for k, v in self.manifest.items() if k != "manifest_sha256"
        })
        self.plans = [{
            "query_id": self.query["id"], "region_origin": [0, 0, 0], "region_shape": list(PATCH_SHAPE),
        }]
        self.settings = {
            "stride": 16, "batch_size": 4, "max_region_bytes": 1_000_000, "max_windows": 100,
            "max_wall_seconds": 30, "device": "cpu", "mps_memory_fraction": 0.5,
        }

    def freeze(self):
        return freeze_search_manifest(self.manifest, self.groups, self.plans, **self.settings)

    def test_freeze_binds_region_and_rejects_unsafe_plans(self):
        frozen = self.freeze()
        self.assertEqual(frozen["searches"][0]["region_sha256"],
                         _digest_array(self.data[self.window.slices]))
        for changes in ({"max_region_bytes": 4}, {"max_windows": 1}, {"stride": 13}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                freeze_search_manifest(self.manifest, self.groups, self.plans, **{**self.settings, **changes})
        bad = [{**self.plans[0], "region_shape": list(self.data.shape)}]
        with self.assertRaisesRegex(ValueError, "whole-volume"):
            freeze_search_manifest(self.manifest, self.groups, bad, **self.settings)
        self.data[0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "frozen validation"):
            self.freeze()

    def write_results(self, scores: Path, output: Path):
        padding = compute_padding(PATCH_SHAPE, patch_size=PATCH_SHAPE, base_pad=16)
        padded = tuple(size + lo + hi for size, (lo, hi) in zip(PATCH_SHAPE, padding))
        rows = []
        for x in range(0, padded[0] - PATCH_SHAPE[0] + 1, 16):
            for y in range(0, padded[1] - PATCH_SHAPE[1] + 1, 16):
                for z in range(0, padded[2] - PATCH_SHAPE[2] + 1, 16):
                    origin = [x - 16, y - 16, z - 16]
                    contained = all(0 <= value and value + width <= dim
                                    for value, width, dim in zip(origin, PATCH_SHAPE, PATCH_SHAPE))
                    rows.append({"origin": origin, "score": 0.9 if contained else 0.1,
                                 "fully_contained": contained})
        scores.write_text("".join(json.dumps(row) + "\n" for row in rows))
        values = np.zeros(PATCH_SHAPE, dtype=np.float32)
        values[1, 2, 3] = 1
        zarr.open_group(str(output), mode="w").create_array("data", data=values)

    def test_separates_window_rank_from_hann_peak_and_rejects_missing_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scores, output = root / "scores", root / "output.zarr"
            self.write_results(scores, output)
            result = _search_results(scores, output, self.window, self.window, 16)
            self.assertEqual(result["query_window_rank"], 1)
            self.assertEqual(result["best_window_center_error_voxels"], 0)
            self.assertEqual(result["hann_voxel_peak"], [1, 2, 3])
            self.assertGreater(result["hann_voxel_peak_error_voxels"], 0)
            scores.write_text("\n".join(scores.read_text().splitlines()[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "all windows"):
                _search_results(scores, output, self.window, self.window, 16)

    def test_runner_uses_exact_vae_mu_cli_and_rejects_changed_region(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            frozen = self.freeze()
            commands = []

            def fake_run(command, **kwargs):
                commands.append(command)
                self.assertTrue(kwargs["check"])
                self.assertIn("--latent-mode", command)
                self.assertEqual(command[command.index("--latent-mode") + 1], "vae")
                self.assertEqual(command[command.index("--embedding-mode") + 1], "mu")
                self.assertEqual(command[command.index("--similarity-mode") + 1], "cosine")
                self.write_results(Path(command[command.index("--window-scores-jsonl") + 1]),
                                   Path(command[command.index("--output") + 1]))

            with patch("scripts.streamed_overlap_search.verify_inputs",
                       return_value={"status": "VALIDATION_INPUTS_VERIFIED"}), \
                 patch("scripts.streamed_overlap_search.export_tokenizer_checkpoint",
                       return_value={"checkpoint": {
                           "arm": "S1", "validation_manifest_sha256": self.manifest["manifest_sha256"],
                       }}), \
                 patch("scripts.streamed_overlap_search.subprocess.run", side_effect=fake_run):
                result = run_bounded_searches(
                    root / "checkpoint", None, root / "manifest", source, self.manifest,
                    self.groups, frozen, root / "output",
                )
                self.assertEqual(len(commands), 1)
                self.assertEqual(result["searches"][0]["query_window_rank"], 1)
                self.data[0, 0, 0] += 1
                with self.assertRaisesRegex(ValueError, "frozen validation"):
                    run_bounded_searches(
                        root / "checkpoint", None, root / "manifest", source, self.manifest,
                        self.groups, frozen, root / "other",
                    )
                altered = copy.deepcopy(frozen)
                altered["stride"] = 8
                with self.assertRaisesRegex(ValueError, "frozen identity"):
                    run_bounded_searches(
                        root / "checkpoint", None, root / "manifest", source, self.manifest,
                        self.groups, altered, root / "third",
                    )

    def test_mps_subprocess_watermarks_are_compatible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            settings = {**self.settings, "device": "mps"}
            frozen = freeze_search_manifest(self.manifest, self.groups, self.plans, **settings)

            def fake_run(command, **kwargs):
                environment = kwargs["env"]
                self.assertEqual(float(environment["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]), 0.5)
                self.assertLess(float(environment["PYTORCH_MPS_LOW_WATERMARK_RATIO"]), 0.5)
                self.assertEqual(environment["PYTORCH_ENABLE_MPS_FALLBACK"], "0")
                self.write_results(Path(command[command.index("--window-scores-jsonl") + 1]),
                                   Path(command[command.index("--output") + 1]))

            with patch("scripts.streamed_overlap_search.verify_inputs",
                       return_value={"status": "VALIDATION_INPUTS_VERIFIED"}), \
                 patch("scripts.streamed_overlap_search.export_tokenizer_checkpoint",
                       return_value={"checkpoint": {
                           "arm": "S1", "validation_manifest_sha256": self.manifest["manifest_sha256"],
                       }}), \
                 patch("scripts.streamed_overlap_search.subprocess.run", side_effect=fake_run):
                run_bounded_searches(
                    root / "checkpoint", None, root / "manifest", source, self.manifest,
                    self.groups, frozen, root / "output",
                )

    def test_child_failure_identifies_the_log_and_never_returns_a_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            frozen = self.freeze()
            with patch("scripts.streamed_overlap_search.verify_inputs",
                       return_value={"status": "VALIDATION_INPUTS_VERIFIED"}), \
                 patch("scripts.streamed_overlap_search.export_tokenizer_checkpoint",
                       return_value={"checkpoint": {
                           "arm": "S1", "validation_manifest_sha256": self.manifest["manifest_sha256"],
                       }}), \
                 patch("scripts.streamed_overlap_search.subprocess.run",
                       side_effect=subprocess.CalledProcessError(1, ["tokenizer"])):
                with self.assertRaisesRegex(RuntimeError, "tokenizer.log"):
                    run_bounded_searches(
                        root / "checkpoint", None, root / "manifest", source, self.manifest,
                        self.groups, frozen, root / "output",
                    )


if __name__ == "__main__":
    unittest.main()
