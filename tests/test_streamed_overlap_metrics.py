import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_epoch_runner import SuiteInitialization, _suite_cosine_comparisons
from scripts.streamed_overlap_metrics import (
    calculate_rankings,
    compare_matched_cosine,
    event_coverage_diagnostics,
    linear_ndcg_at_10,
    paired_source_bootstrap,
    physical_overlap_diagnostics,
    retrieval_metric_gate,
    summarize_rankings,
)


def frozen_fixture():
    queries = []
    for cohort in ("shallower", "deeper"):
        for source in range(5):
            source_id = f"validation/source-{source}"
            for index in range(20):
                candidates = [
                    {
                        "id": candidate,
                        "category": "same_overlap" if candidate < 3 else (
                            "same_disjoint" if candidate < 6 else "cross_source"
                        ),
                        "window": {"source_id": source_id if candidate < 6 else "validation/other"},
                        "target": {"coverage": (1.0, 0.5, 0.2)[candidate] if candidate < 3 else 0.0},
                    }
                    for candidate in range(10)
                ]
                queries.append({
                    "id": f"{cohort}-{source}-{index}", "cohort": cohort,
                    "window": {"source_id": source_id}, "candidates": candidates, "eligible": True,
                })
    manifest = {
        "schema": "streamed-retrieval-validation-v2",
        "input_integrity": {
            "algorithm": "sha256-array-dtype-shape-c-order-v1",
            "fingerprints": {"windows": {"fixture": "unused"}, "event_traces": {"fixture": "unused"}},
        },
        "queries": queries,
        "eligibility": {
            "eligible_queries": 200,
            "cohorts": {
                cohort: {"eligible_queries": 100, "eligible_sources": 5}
                for cohort in ("shallower", "deeper")
            },
            "eligibility_floor_passed": True,
        },
    }
    manifest["manifest_sha256"] = content_hash(manifest)
    return manifest


def predictions(manifest, reverse=False):
    return {
        query["id"]: {
            candidate["id"]: (1 - candidate["target"]["coverage"] if reverse else candidate["target"]["coverage"])
            for candidate in query["candidates"]
        }
        for query in manifest["queries"] if query["eligible"]
    }


def make_report(manifest, *, arm="S3", scorer="pair_head", experiment_seed=1, reverse=False):
    return calculate_rankings(
        manifest, predictions(manifest, reverse), scorer=scorer, arm=arm,
        experiment_seed=experiment_seed, random_seed=17, random_repeats=5,
    )


class RankingMetricTests(unittest.TestCase):
    def test_overlap_diagnostics_report_categories_bins_and_matched_baselines(self):
        manifest = frozen_fixture()
        for query in manifest["queries"]:
            for candidate in query["candidates"]:
                candidate["target"]["overlap"] = (0.25, 0.5, 1.0)[candidate["id"]] \
                    if candidate["category"] == "same_overlap" else 0.0
        manifest["manifest_sha256"] = content_hash({
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        })
        overlap = {
            query["id"]: {candidate["id"]: candidate["target"]["overlap"]
                          for candidate in query["candidates"]}
            for query in manifest["queries"]
        }
        waveform = {
            query["id"]: {candidate["id"]: 0.5 for candidate in query["candidates"]}
            for query in manifest["queries"]
        }
        result = physical_overlap_diagnostics(manifest, overlap, constant=0.0, waveform_predictions=waveform)
        self.assertEqual(result["overlap_bins"]["0-0.25"]["count"], 1400)
        self.assertEqual(result["overlap_bins"]["0.25-0.5"]["count"], 200)
        self.assertEqual(result["overlap_bins"]["0.5-0.75"]["count"], 200)
        self.assertEqual(result["overlap_bins"]["0.75-1"]["count"], 200)
        self.assertEqual(result["categories"]["same_overlap"]["metrics"]["prediction"]["mae"], 0)
        self.assertAlmostEqual(result["categories"]["same_overlap"]["metrics"]["constant"]["bias"], -7 / 12)
        self.assertEqual(result["categories"]["cross_source"]["metrics"]["waveform"]["rmse"], 0.5)
        waveform[manifest["queries"][0]["id"]].pop(0)
        with self.assertRaisesRegex(ValueError, "candidate IDs"):
            physical_overlap_diagnostics(manifest, overlap, constant=0.0, waveform_predictions=waveform)
        waveform[manifest["queries"][0]["id"]][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite values"):
            physical_overlap_diagnostics(manifest, overlap, constant=0.0, waveform_predictions=waveform)
        waveform[manifest["queries"][0]["id"]][0] = 0.5
        candidate = manifest["queries"][0]["candidates"][3]
        candidate["target"]["overlap"] = 0.1
        manifest["manifest_sha256"] = content_hash({
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        })
        with self.assertRaisesRegex(ValueError, "zero physical overlap"):
            physical_overlap_diagnostics(manifest, overlap, constant=0.0, waveform_predictions=waveform)

    def test_event_diagnostics_preserve_directional_errors_and_fluid_availability(self):
        manifest = frozen_fixture()
        target = manifest["queries"][0]["candidates"][0]["target"]
        target.update({"fluid_a": [0.2, 0.0, 0.8], "fluid_b": [0.0, 0.1, 0.9]})
        manifest["manifest_sha256"] = content_hash({
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        })
        scores = predictions(manifest)
        first = manifest["queries"][0]
        scores[first["id"]][0] = 0.8
        result = event_coverage_diagnostics(manifest, scores)
        row = result["pairs"][0]
        self.assertAlmostEqual(row["error"], -0.2)
        self.assertAlmostEqual(row["fluid_mismatch"], 0.2)
        self.assertIsNone(row["fluid_mismatch_unavailable_reason"])
        self.assertIsNone(result["pairs"][1]["fluid_mismatch"])
        self.assertIsNotNone(result["pairs"][1]["fluid_mismatch_unavailable_reason"])
        overlap = result["categories"]["same_overlap"]
        self.assertEqual(overlap["count"], 600)
        self.assertEqual(overlap["fluid_mismatch_available"], 1)
        self.assertAlmostEqual(overlap["mae"], 0.2 / 600)
        self.assertAlmostEqual(overlap["bias"], -0.2 / 600)
        target["fluid_a"] = [0.2, 0.0, 0.2]
        manifest["manifest_sha256"] = content_hash({
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        })
        with self.assertRaisesRegex(ValueError, "fluid fractions"):
            event_coverage_diagnostics(manifest, scores)

    def test_linear_gain_and_candidate_id_ties(self):
        labels = [1.0, 0.5] + [0.0] * 8
        scores = [0.0] * 10
        ids = [9, 0] + list(range(1, 9))
        actual = linear_ndcg_at_10(labels, scores, ids)
        expected = (0.5 + 1.0 / np.log2(11)) / (1.0 + 0.5 / np.log2(3))
        self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(linear_ndcg_at_10(labels, labels, ids), 1.0)
        with self.assertRaises(ValueError):
            linear_ndcg_at_10([0.0] * 10, scores, ids)

    def test_frozen_queries_random_baseline_and_pair_cohorts(self):
        manifest = frozen_fixture()
        report = make_report(manifest)
        self.assertEqual(report, make_report(manifest))
        result = summarize_rankings(report, bootstrap_seed=33, bootstrap_repeats=50)
        self.assertEqual(result["eligible_queries"], 200)
        self.assertEqual(result["ndcg_at_10"], 1.0)
        self.assertEqual(result["pair_cohorts"]["different_source"]["candidates"], 800)
        self.assertEqual(result["pair_cohorts"]["different_source"]["mean_relevance"], 0.0)
        self.assertGreater(result["paired_source_bootstrap_95"][0], 0)
        self.assertEqual(retrieval_metric_gate(
            report, bootstrap_seed=33, bootstrap_repeats=50, replication=None
        )["status"], "PENDING")
        repeated = make_report(manifest, experiment_seed=2)
        self.assertEqual(retrieval_metric_gate(
            report, bootstrap_seed=33, bootstrap_repeats=50, replication=repeated
        )["status"], "PASS_METRICS_ONLY")
        with self.assertRaisesRegex(ValueError, "independent"):
            retrieval_metric_gate(report, bootstrap_seed=33, bootstrap_repeats=50, replication=report)
        bad = make_report(manifest, experiment_seed=2, reverse=True)
        self.assertEqual(retrieval_metric_gate(
            report, bootstrap_seed=33, bootstrap_repeats=50, replication=bad
        )["status"], "FAIL")

    def test_manifest_and_predictions_fail_closed(self):
        manifest = frozen_fixture()
        supplied = predictions(manifest)
        supplied.pop(next(iter(supplied)))
        with self.assertRaisesRegex(ValueError, "exactly"):
            calculate_rankings(manifest, supplied, scorer="pair_head", arm="S3",
                               experiment_seed=1, random_seed=2, random_repeats=2)
        with self.assertRaisesRegex(ValueError, "Only S3"):
            make_report(manifest, arm="S1")
        altered = copy.deepcopy(manifest)
        altered["queries"][0]["candidates"][0]["target"]["coverage"] = None
        altered["queries"][0]["candidates"][0]["target"]["unavailable_reason"] = "fixture:unavailable"
        altered["manifest_sha256"] = content_hash({
            key: value for key, value in altered.items() if key != "manifest_sha256"
        })
        with self.assertRaisesRegex(ValueError, "eligibility disagrees"):
            make_report(altered)
        altered = copy.deepcopy(manifest)
        altered["queries"].pop()
        altered["eligibility"]["eligible_queries"] = 199
        altered["eligibility"]["cohorts"]["deeper"]["eligible_queries"] = 99
        altered["eligibility"]["eligibility_floor_passed"] = False
        altered["manifest_sha256"] = content_hash({
            key: value for key, value in altered.items() if key != "manifest_sha256"
        })
        with self.assertRaisesRegex(ValueError, "eligibility floor"):
            make_report(altered)
        altered = copy.deepcopy(manifest)
        altered["queries"][0]["candidates"][0]["window"]["source_id"] = "validation/../train"
        altered["manifest_sha256"] = content_hash({
            key: value for key, value in altered.items() if key != "manifest_sha256"
        })
        with self.assertRaisesRegex(ValueError, "candidate identity"):
            make_report(altered)
        report = make_report(manifest)
        with self.assertRaisesRegex(ValueError, "at least 200"):
            summarize_rankings(replace(report, rows=report.rows[:-1]), bootstrap_seed=1, bootstrap_repeats=2)

    def test_source_bootstrap_resamples_whole_clusters(self):
        report = make_report(frozen_fixture())
        differences = [1.0 if row.source_id == "validation/source-0" else 0.0 for row in report.rows]
        low, high = paired_source_bootstrap(report.rows, differences, seed=7, repeats=1000)
        self.assertEqual(low, 0.0)
        self.assertGreaterEqual(high, 0.6)

    def test_depth_cohort_regression_threshold(self):
        report = make_report(frozen_fixture())
        boundary = replace(report, rows=tuple(
            replace(row, random_ndcg=0.25, ndcg=0.23 if row.cohort == "shallower" else 0.31)
            for row in report.rows
        ))
        repeated = replace(boundary, experiment_seed=2)
        gate = retrieval_metric_gate(boundary, bootstrap_seed=3, bootstrap_repeats=30, replication=repeated)
        self.assertTrue(gate["checks"]["no_depth_cohort_regression_over_0_02"])
        below = replace(boundary, rows=tuple(
            replace(row, ndcg=0.229) if row.cohort == "shallower" else row for row in boundary.rows
        ))
        gate = retrieval_metric_gate(below, bootstrap_seed=3, bootstrap_repeats=30,
                                     replication=replace(below, experiment_seed=2))
        self.assertFalse(gate["checks"]["no_depth_cohort_regression_over_0_02"])
        self.assertEqual(gate["status"], "FAIL")

    def test_cosine_is_compared_separately_with_matched_s1(self):
        manifest = frozen_fixture()
        candidate = make_report(manifest, scorer="mu_cosine")
        baseline = make_report(manifest, arm="S1", scorer="mu_cosine", reverse=True)
        comparison = compare_matched_cosine(candidate, baseline, bootstrap_seed=4, bootstrap_repeats=20)
        self.assertGreater(comparison["mean_ndcg_gain_over_s1"], 0)
        different_seed = make_report(manifest, arm="S1", scorer="mu_cosine", experiment_seed=2)
        with self.assertRaisesRegex(ValueError, "matched-seed"):
            compare_matched_cosine(candidate, different_seed, bootstrap_seed=4, bootstrap_repeats=20)
        s2 = make_report(manifest, arm="S2", scorer="mu_cosine", reverse=True)
        comparison = compare_matched_cosine(candidate, s2, bootstrap_seed=4, bootstrap_repeats=20)
        self.assertGreater(comparison["mean_ndcg_gain_over_s2"], 0)

    def test_suite_reports_selected_cosine_comparisons_and_detects_changed_report(self):
        manifest = frozen_fixture()
        config = {"random_seed": 17, "random_repeats": 5, "bootstrap_seed": 4, "bootstrap_repeats": 20}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = {arm: root / arm / "initial.pt" for arm in ("S1", "S2", "S3")}
            suite = SuiteInitialization(root / "initialization_evidence.json", "unused", initial)
            results = {}
            for arm in ("S1", "S2", "S3"):
                identity = {
                    "arm": arm, "run_id": "suite", "experiment_seed": 1, "epoch": 2, "step": 2,
                    "checkpoint_sha256": arm, "config_sha256": "config", "source_sha256": "source",
                    "target_sha256": "target", "validation_manifest_sha256": manifest["manifest_sha256"],
                }
                results[arm] = {"selected_evaluations": {}}
                kinds = (("final", "reconstruction", "retrieval") if arm == "S3"
                         else ("final", "reconstruction"))
                for kind in kinds:
                    ranking = make_report(manifest, arm=arm, scorer="mu_cosine", reverse=arm != "S2")
                    report = {
                        "schema": "streamed-retrieval-evaluation-v1", "checkpoint": identity,
                        "input_verification": {
                            "status": "VALIDATION_INPUTS_VERIFIED", "manifest_sha256": manifest["manifest_sha256"],
                            "verified_windows": 1, "verified_full_depth_event_regions": 1,
                        },
                        "scorers": {"mu_cosine": {
                            "summary": summarize_rankings(ranking, bootstrap_seed=4, bootstrap_repeats=20),
                            "rows": [asdict(row) for row in ranking.rows],
                        }},
                    }
                    directory = root / arm / "evaluation"
                    if kind != "final":
                        directory = directory / f"selected-{kind}"
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory / "retrieval.json"
                    path.write_text(json.dumps(report), encoding="utf-8")
                    completion = {
                        "checkpoint": identity,
                        "config_sha256": (content_hash(config) if kind == "final" else content_hash({
                            "formal_evaluation": config, "selection_kind": kind,
                        })),
                        "artifacts": {"retrieval": {
                            "path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }},
                    }
                    completion["sha256"] = content_hash(completion)
                    if kind == "final":
                        results[arm]["last_epoch"] = completion
                    else:
                        results[arm]["selected_evaluations"][kind] = completion
                if arm != "S1":
                    matched = {
                        "status": "MATCHED_RECORDED_STREAMS_ONLY",
                        "reference": results["S1"]["last_epoch"]["checkpoint"], "candidate": identity,
                    }
                    matched["sha256"] = content_hash(matched)
                    results[arm]["matched_streams"] = {"record": matched}
            result = _suite_cosine_comparisons(results, suite, manifest, config)
            self.assertGreater(result["comparisons"]["S2_reconstruction_vs_S1_reconstruction"][
                "mean_ndcg_gain_over_s1"], 0)
            self.assertLess(result["comparisons"]["S3_reconstruction_vs_S2_reconstruction"][
                "mean_ndcg_gain_over_s2"], 0)
            self.assertLess(result["comparisons"]["S3_final_vs_S2_final"]["mean_ndcg_gain_over_s2"], 0)
            self.assertEqual(_suite_cosine_comparisons(results, suite, manifest, config), result)
            matched = results["S2"]["matched_streams"]["record"]
            original = matched["candidate"]
            matched["candidate"] = {**original, "step": 3}
            matched["sha256"] = content_hash({k: v for k, v in matched.items() if k != "sha256"})
            with self.assertRaisesRegex(ValueError, "matched arm records"):
                _suite_cosine_comparisons(results, suite, manifest, config)
            matched["candidate"] = original
            matched["sha256"] = content_hash({k: v for k, v in matched.items() if k != "sha256"})
            changed = root / "S3" / "evaluation" / "selected-retrieval" / "retrieval.json"
            changed.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing, changed or mismatched"):
                _suite_cosine_comparisons(results, suite, manifest, config)


if __name__ == "__main__":
    unittest.main()
