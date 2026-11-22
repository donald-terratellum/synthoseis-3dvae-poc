import copy
import unittest

import numpy as np

from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_diagnostics import bind_diagnostics, diagnostic_report, verify_diagnostics
from scripts.streamed_overlap_integrity import _window_record
from scripts.streamed_overlap_manifest import _window
from tests.test_streamed_overlap_metrics import frozen_fixture, predictions


class ValidationDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.manifest = frozen_fixture()
        sources = set()
        for query in self.manifest["queries"]:
            for record in (query["window"], *(item["window"] for item in query["candidates"])):
                record.update({"origin": [0, 0, 0], "shape": [2, 2, 5]})
                sources.add(record["source_id"])
        self.manifest["manifest_sha256"] = content_hash({
            key: value for key, value in self.manifest.items() if key != "manifest_sha256"
        })
        self.groups = {}
        for source in sources:
            lithology = np.zeros((2, 2, 5), dtype=np.float32)
            lithology[:, :, 1:3] = 1
            self.groups[source] = {
                "faulted_net_to_gross": np.full((2, 2, 5), 0.4, dtype=np.float32),
                "faulted_lithology": lithology,
                "faults/faulted_channel_labels": np.zeros_like(lithology),
            }

    def bind(self):
        return bind_diagnostics(self.manifest, self.groups, thin_max_voxels=2, max_wall_seconds=30)

    def test_complete_thin_runs_and_edge_censoring_are_distinct(self):
        source_id = self.manifest["queries"][0]["window"]["source_id"]
        record = self.groups[source_id]["faulted_lithology"]
        record[0, 0] = [1, 1, 0, 0, 0]
        bound = self.bind()
        key = content_hash(_window_record(_window(self.manifest["queries"][0]["window"])))
        result = bound["windows"][key]
        self.assertEqual(result["complete_thin_sand_runs"], 3)
        self.assertEqual(result["window_edge_censored_sand_runs"], 1)
        self.assertAlmostEqual(result["mean_ntg_over_rock"], 0.4)
        self.assertFalse(result["no_corrected_sand"])

    def test_missing_padded_and_water_only_windows_have_explicit_status(self):
        source_id = self.manifest["queries"][0]["window"]["source_id"]
        source = self.groups[source_id]
        source["faulted_lithology"].fill(-1)
        bound = self.bind()
        key = content_hash(_window_record(_window(self.manifest["queries"][0]["window"])))
        result = bound["windows"][key]
        self.assertTrue(result["no_corrected_sand"])
        self.assertIsNone(result["mean_ntg_over_rock"])
        self.assertEqual(result["ntg_unavailable_reason"], "no rock voxels in window")
        del source["faults/faulted_channel_labels"]
        missing = self.bind()["windows"][key]
        self.assertEqual(missing["reason"], "missing:faults/faulted_channel_labels")
        altered = copy.deepcopy(self.manifest)
        for query in altered["queries"]:
            if query["window"]["source_id"] == source_id:
                query["window"]["origin"][2] = 1499
        altered["manifest_sha256"] = content_hash({
            k: v for k, v in altered.items() if k != "manifest_sha256"
        })
        padded = bind_diagnostics(altered, self.groups, thin_max_voxels=2, max_wall_seconds=30)
        key = content_hash(_window_record(_window(altered["queries"][0]["window"])))
        self.assertEqual(padded["windows"][key]["reason"], "unsupported:faulted_net_to_gross")

    def test_reporting_rechecks_fingerprints_and_reports_pair_differences(self):
        bound = self.bind()
        report = diagnostic_report(
            self.manifest, self.groups, bound, predictions(self.manifest), max_wall_seconds=30,
        )
        self.assertEqual(len(report["pairs"]), 2000)
        self.assertAlmostEqual(report["pairs"][0]["ntg_difference"], 0)
        self.assertTrue(report["pairs"][0]["thin_run_either"])
        self.assertEqual(report["cohorts"]["thin_run_either"]["true"]["count"], 2000)
        self.assertEqual(report["cohorts"]["no_corrected_sand_either"]["true"]["count"], 0)
        source_id = self.manifest["queries"][0]["window"]["source_id"]
        self.groups[source_id]["faulted_net_to_gross"][0, 0, 0] = 0.8
        with self.assertRaisesRegex(ValueError, "fingerprints"):
            verify_diagnostics(self.manifest, self.groups, bound, max_wall_seconds=30)
        with self.assertRaisesRegex(ValueError, "fingerprints"):
            diagnostic_report(self.manifest, self.groups, bound, predictions(self.manifest), max_wall_seconds=30)


if __name__ == "__main__":
    unittest.main()
