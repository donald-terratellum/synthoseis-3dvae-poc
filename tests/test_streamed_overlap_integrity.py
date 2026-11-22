import copy
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_integrity import (
    bind_manifest, input_fingerprints, verify_inputs, verify_validation_preflight,
)
from scripts.streamed_overlap_targets import Window
from scripts.streamed_overlap_yield import SEISMIC_KEY


class ValidationIntegrityTests(unittest.TestCase):
    def setUp(self):
        seismic = np.arange(48, dtype=np.float32).reshape(2, 2, 12)
        age = np.broadcast_to(np.arange(12, dtype=np.float32), seismic.shape).copy()
        self.source = {
            SEISMIC_KEY: seismic,
            "geologic_age_faulted": age,
            "oil_closures": np.zeros_like(seismic),
            "gas_closures": np.zeros_like(seismic),
        }
        self.groups = {"validation/s": self.source}
        a = asdict(Window("validation/s", (0, 0, 2), (2, 2, 5)))
        b = asdict(Window("validation/s", (0, 0, 3), (2, 2, 5)))
        self.plan = [{
            "id": "q", "cohort": "shallower", "window": a,
            "candidates": [{"category": "same_overlap", "window": b}],
        }]
        self.labeled = copy.deepcopy(self.plan)
        self.labeled[0]["eligible"] = True
        self.labeled[0]["candidates"][0].update({
            "id": 0, "target": {"coverage": 0.5},
        })
        self.eligibility = {"eligible_queries": 1}
        self.manifest = {
            "manifest_sha256": "reference",
            "candidate_plan_sha256": content_hash(self.plan),
            "queries": self.labeled,
            "eligibility": self.eligibility,
        }

    def bind(self):
        with patch("scripts.streamed_overlap_integrity.label_candidates",
                   return_value=(self.labeled, self.eligibility)):
            return bind_manifest(self.manifest, self.groups)

    def test_bind_preserves_plan_and_verifies_recorded_inputs(self):
        bound = self.bind()
        self.assertEqual(bound["queries"], self.labeled)
        self.assertEqual(bound["input_integrity"]["reference_manifest_sha256"], "reference")
        result = verify_inputs(bound, self.groups)
        self.assertEqual(result["status"], "VALIDATION_INPUTS_VERIFIED")
        self.assertEqual(result["verified_windows"], 2)
        self.assertEqual(result["verified_full_depth_event_regions"], 1)

    def test_changed_label_or_full_depth_trace_fails_verification(self):
        bound = self.bind()
        self.source["oil_closures"][0, 0, 3] = 1
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch: windows"):
            verify_inputs(bound, self.groups)
        self.source["oil_closures"][0, 0, 3] = 0
        self.source[SEISMIC_KEY][0, 0, 10] += 1
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch: event_traces"):
            verify_inputs(bound, self.groups)

    def test_preflight_checks_frozen_identity_and_validation_content(self):
        bound = self.bind()
        bound["target_implementation_sha256"] = "a" * 64
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "manifest.json"
            path.write_text("{}")
            def verify(**overrides):
                settings = {
                    "expected_manifest_sha256": bound["manifest_sha256"],
                    "expected_target_sha256": "a" * 64,
                }
                return verify_validation_preflight(path, root, **{**settings, **overrides})

            with patch("scripts.streamed_overlap_integrity.load_manifest", return_value=bound), \
                 patch("scripts.streamed_overlap_integrity.open_validation_sources", return_value=self.groups) as opened:
                report = verify()
                self.assertEqual(report["status"], "VALIDATION_INPUTS_VERIFIED")
                self.assertEqual(report["verified_windows"], 2)
                self.assertEqual(opened.call_count, 2)
                opened.reset_mock()
                with self.assertRaisesRegex(ValueError, "manifest or target implementation"):
                    verify(expected_target_sha256="b" * 64)
                opened.assert_not_called()
                self.source[SEISMIC_KEY][0, 0, 10] += 1
                with self.assertRaisesRegex(ValueError, "fingerprint mismatch: event_traces"):
                    verify()
                self.source[SEISMIC_KEY][0, 0, 10] -= 1
                changed = {**bound, "manifest_sha256": "changed"}
                with patch("scripts.streamed_overlap_integrity.load_manifest", side_effect=[bound, changed]):
                    with self.assertRaisesRegex(ValueError, "manifest changed during preflight"):
                        verify()

    def test_preflight_refuses_validation_v2_before_loading_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "manifest.json"
            path.write_text("{}")
            with patch("scripts.streamed_overlap_integrity.load_manifest") as load:
                with self.assertRaisesRegex(ValueError, "permitted primary"):
                    verify_validation_preflight(
                        path, root / "fake_data_validation_v2",
                        expected_manifest_sha256="a" * 64, expected_target_sha256="b" * 64,
                    )
                load.assert_not_called()

    def test_unrelated_source_region_is_not_claimed_verified(self):
        original = input_fingerprints(self.plan, self.groups)
        other = copy.deepcopy(self.source)
        other[SEISMIC_KEY] = np.pad(other[SEISMIC_KEY], ((0, 1), (0, 0), (0, 0)))
        other[SEISMIC_KEY][2, 0, 0] = 1234
        self.assertEqual(original, input_fingerprints(self.plan, {"validation/s": other}))

    def test_changed_relevance_cannot_be_bound(self):
        altered = copy.deepcopy(self.labeled)
        altered[0]["candidates"][0]["target"]["coverage"] = 0.6
        with patch("scripts.streamed_overlap_integrity.label_candidates",
                   return_value=(altered, self.eligibility)):
            with self.assertRaisesRegex(ValueError, "target labels or eligibility"):
                bind_manifest(self.manifest, self.groups)

    def test_json_round_trip_lists_match_in_memory_tuples(self):
        self.labeled[0]["candidates"][0]["target"]["fluid_a"] = [0.1, 0.2, 0.7]
        runtime = copy.deepcopy(self.labeled)
        runtime[0]["candidates"][0]["target"]["fluid_a"] = (0.1, 0.2, 0.7)
        with patch("scripts.streamed_overlap_integrity.label_candidates",
                   return_value=(runtime, self.eligibility)):
            bound = bind_manifest(self.manifest, self.groups)
        self.assertIn("input_integrity", bound)


if __name__ == "__main__":
    unittest.main()
