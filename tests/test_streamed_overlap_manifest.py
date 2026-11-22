import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.streamed_overlap_audit import LABEL_KEYS, Source
from scripts.streamed_overlap_manifest import label_candidates, plan_candidates
from scripts.streamed_overlap_targets import PairTarget
from scripts.streamed_overlap_yield import SEISMIC_KEY


class ValidationManifestTests(unittest.TestCase):
    def setUp(self):
        self.sources = [
            Source(Path(f"/fixture/validation/source-{i}/model_data.zarr"), "validation")
            for i in range(5)
        ]
        volume = np.broadcast_to(np.float32(0), (100, 100, 256))
        self.groups = {
            source.identifier: {key: volume for key in (SEISMIC_KEY, *LABEL_KEYS)}
            for source in self.sources
        }

    def test_candidate_plan_is_model_independent_deterministic_and_stratified(self):
        queries = plan_candidates(self.sources, self.groups, seed=41, queries_per_cohort_source=2)
        self.assertEqual(queries, plan_candidates(self.sources, self.groups, 41, 2))
        self.assertEqual(len(queries), 20)
        self.assertEqual({query["cohort"] for query in queries}, {"shallower", "deeper"})
        for query in queries:
            categories = [candidate["category"] for candidate in query["candidates"]]
            self.assertEqual(categories.count("same_overlap"), 3)
            self.assertEqual(categories.count("same_disjoint"), 3)
            self.assertEqual(categories.count("cross_source"), 4)
        self.assertTrue(any(query["candidates"][0]["category"] != "same_overlap" for query in queries))

    def test_known_positive_and_all_candidate_validity_determine_floor(self):
        queries = plan_candidates(self.sources, self.groups, seed=41, queries_per_cohort_source=20)

        def target(_source_a, _source_b, a, b):
            positive = a.source_id == b.source_id and b.origin[2] - a.origin[2] < 64
            return PairTarget(0.5 if positive else 0.0, 0.5 if positive else 0.0)

        with patch("scripts.streamed_overlap_manifest.pair_target", side_effect=target):
            labeled, summary = label_candidates(queries, self.groups)
        self.assertTrue(summary["eligibility_floor_passed"])
        self.assertEqual(summary["eligible_queries"], 200)
        self.assertTrue(all(query["eligible"] for query in labeled))

        def invalid(_source_a, _source_b, a, b):
            if a.source_id != b.source_id:
                return PairTarget(0.0, None, "b:nonfinite:labels")
            return target(_source_a, _source_b, a, b)

        with patch("scripts.streamed_overlap_manifest.pair_target", side_effect=invalid):
            labeled, summary = label_candidates(queries[:1], self.groups)
        self.assertFalse(labeled[0]["eligible"])
        self.assertFalse(summary["eligibility_floor_passed"])
        self.assertEqual(summary["ineligibility_reasons"]["b:nonfinite:labels"], 4)
        self.assertTrue(any(
            candidate["target"]["coverage"] is None for candidate in labeled[0]["candidates"]
        ))


if __name__ == "__main__":
    unittest.main()
