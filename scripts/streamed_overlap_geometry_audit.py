"""Explore frozen validation retrieval using oracle geometry, without loading a model."""

import argparse
import json
import math
from numbers import Real
from pathlib import Path

from scripts.streamed_overlap_metrics import _eligible_queries, calculate_rankings, summarize_rankings


def audit_geometry(
    manifest: dict, *, expected_manifest_sha256: str, random_seed: int, random_repeats: int,
    bootstrap_seed: int, bootstrap_repeats: int,
) -> dict:
    if manifest.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("Validation manifest does not match the expected hash.")
    queries = _eligible_queries(manifest)
    predictions = {}
    categories = {}
    positive = 0
    positive_zero_overlap = 0
    for query in queries:
        scores = {}
        for candidate in query["candidates"]:
            overlap = candidate["target"].get("overlap")
            if (isinstance(overlap, bool) or not isinstance(overlap, Real)
                    or not math.isfinite(overlap) or not 0 <= overlap <= 1):
                raise ValueError("Every eligible candidate needs a finite overlap in [0, 1].")
            if candidate["category"] != "same_overlap" and overlap != 0:
                raise ValueError("Disjoint and cross-source candidates must have zero overlap.")
            scores[candidate["id"]] = float(overlap)
            category = candidate["category"]
            categories[category] = categories.get(category, 0) + 1
            if candidate["target"]["coverage"] > 0:
                positive += 1
                positive_zero_overlap += int(overlap == 0)
        predictions[query["id"]] = scores
    # The ranking machinery consumes numeric scores; no encoder is invoked.
    ranking = calculate_rankings(
        manifest, predictions, scorer="mu_cosine", arm="S1", experiment_seed=0,
        random_seed=random_seed, random_repeats=random_repeats,
    )
    summary = summarize_rankings(ranking, bootstrap_seed=bootstrap_seed, bootstrap_repeats=bootstrap_repeats)
    return {
        "schema": "streamed-validation-geometry-audit-v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "eligible_queries": len(queries),
        "query_sources": len({query["window"]["source_id"] for query in queries}),
        "candidate_categories": categories,
        "positive_coverage_pairs": positive,
        "positive_coverage_pairs_with_zero_overlap": positive_zero_overlap,
        "overlap_oracle_ndcg_at_10": summary["ndcg_at_10"],
        "random_ndcg_at_10": summary["random_ndcg_at_10"],
        "gain_over_random": summary["gain"],
        "paired_source_bootstrap_95": summary["paired_source_bootstrap_95"],
        "cohorts": {
            key: {name: cohort[name] for name in ("queries", "sources", "ndcg_at_10", "random_ndcg_at_10")}
            for key, cohort in summary["cohorts"].items()
        },
        "random_seed": random_seed, "random_repeats": random_repeats,
        "bootstrap_seed": bootstrap_seed, "bootstrap_repeats": bootstrap_repeats,
        "limitation": (
            "Exploratory validation oracle uses ground-truth overlap, not encoder predictions or a deployable "
            "baseline. It does not verify source data, establish geological understanding, or authorize training."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument("--random-repeats", type=int, required=True)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, required=True)
    args = parser.parse_args()
    result = audit_geometry(
        json.loads(args.manifest.read_text(encoding="utf-8")), expected_manifest_sha256=args.manifest_sha256,
        random_seed=args.random_seed, random_repeats=args.random_repeats,
        bootstrap_seed=args.bootstrap_seed, bootstrap_repeats=args.bootstrap_repeats,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
