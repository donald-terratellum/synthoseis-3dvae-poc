"""Model-independent ranking calculations for the frozen validation contract.

These functions consume already-produced numeric scores. They do not load a model,
verify source fingerprints, establish that a head was trained, or select checkpoints.
"""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_manifest import (
    COHORTS,
    MIN_COHORT_QUERIES,
    MIN_COHORT_SOURCES,
    MIN_QUERIES,
)


@dataclass(frozen=True)
class QueryRanking:
    query_id: str
    source_id: str
    cohort: str
    ndcg: float
    random_ndcg: float
    same_source_count: int
    same_source_score_sum: float
    same_source_relevance_sum: float
    different_source_count: int
    different_source_score_sum: float
    different_source_relevance_sum: float


@dataclass(frozen=True)
class RankingReport:
    manifest_sha256: str
    scorer: str
    arm: str
    experiment_seed: int
    random_seed: int
    random_repeats: int
    rows: tuple[QueryRanking, ...]


def linear_ndcg_at_10(relevance, scores, candidate_ids) -> float:
    """Linear-gain NDCG@10, using ascending candidate ID to break score ties."""
    relevance = np.asarray(relevance, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    ids = np.asarray(candidate_ids)
    if (relevance.ndim != 1 or len(relevance) < 10 or scores.shape != relevance.shape
            or ids.shape != relevance.shape or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0)
            or len(set(ids.tolist())) != len(ids)
            or not np.isfinite(scores).all() or not np.isfinite(relevance).all()
            or np.any((relevance < 0) | (relevance > 1)) or not np.any(relevance > 0)):
        raise ValueError("NDCG requires ten known candidates, unique integer IDs and positive relevance.")
    discount = np.log2(np.arange(2, 12, dtype=np.float64))
    order = np.lexsort((ids, -scores))[:10]
    ideal = np.sort(relevance)[::-1][:10]
    return float(np.sum(relevance[order] / discount) / np.sum(ideal / discount))


def _eligible_queries(manifest: dict) -> list[dict]:
    integrity = manifest.get("input_integrity")
    if (manifest.get("schema") != "streamed-retrieval-validation-v2"
            or not isinstance(integrity, dict)
            or integrity.get("algorithm") != "sha256-array-dtype-shape-c-order-v1"
            or not isinstance(integrity.get("fingerprints"), dict)
            or not isinstance(integrity["fingerprints"].get("windows"), dict)
            or not integrity["fingerprints"]["windows"]
            or not isinstance(integrity["fingerprints"].get("event_traces"), dict)
            or not integrity["fingerprints"]["event_traces"]):
        raise ValueError("A source-bound frozen validation manifest is required.")
    expected_hash = manifest.get("manifest_sha256")
    if not isinstance(expected_hash, str) or expected_hash != content_hash({
        key: value for key, value in manifest.items() if key not in {"manifest_sha256", "measurement"}
    }):
        raise ValueError("Validation manifest content hash is invalid.")
    queries = manifest.get("queries")
    if not isinstance(queries, list):
        raise ValueError("Validation manifest queries are missing.")
    eligible = []
    seen = set()
    cohort_sources = {cohort: set() for cohort in COHORTS}
    cohort_counts = {cohort: 0 for cohort in COHORTS}
    for query in queries:
        query_id = query.get("id")
        cohort = query.get("cohort")
        source_id = query.get("window", {}).get("source_id")
        candidates = query.get("candidates")
        if (not isinstance(query_id, str) or not query_id or query_id in seen
                or cohort not in COHORTS or not isinstance(source_id, str)
                or not _validation_source(source_id) or not isinstance(candidates, list)
                or not isinstance(query.get("eligible"), bool)):
            raise ValueError("Invalid validation query identity, source or cohort.")
        seen.add(query_id)
        ids = set()
        relevance = []
        for candidate in candidates:
            candidate_id = candidate.get("id")
            source = candidate.get("window", {}).get("source_id")
            category = candidate.get("category")
            value = candidate.get("target", {}).get("coverage")
            if (isinstance(candidate_id, bool) or not isinstance(candidate_id, int)
                    or candidate_id < 0 or candidate_id in ids or not isinstance(source, str)
                    or not _validation_source(source)
                    or category not in {"same_overlap", "same_disjoint", "cross_source"}
                    or (category == "cross_source") != (source != source_id)):
                raise ValueError("Invalid candidate identity, source or category.")
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) or not 0 <= float(value) <= 1
            ):
                raise ValueError("Candidate relevance must be known in [0, 1] or unavailable.")
            if value is None and not candidate.get("target", {}).get("unavailable_reason"):
                raise ValueError("Unavailable candidate relevance requires an explicit reason.")
            if value is not None and category in {"cross_source", "same_disjoint"} and value != 0:
                raise ValueError("Disjoint and cross-source pairs must have zero event relevance.")
            ids.add(candidate_id)
            relevance.append(value)
        is_eligible = len(candidates) >= 10 and all(value is not None for value in relevance) and any(
            value is not None and value > 0 for value in relevance
        )
        if query["eligible"] != is_eligible:
            raise ValueError("Manifest eligibility disagrees with frozen candidate labels.")
        if is_eligible:
            eligible.append(query)
            cohort_counts[cohort] += 1
            cohort_sources[cohort].add(source_id)
    declared = manifest.get("eligibility")
    if (not isinstance(declared, dict) or declared.get("eligible_queries") != len(eligible)
            or not isinstance(declared.get("cohorts"), dict)):
        raise ValueError("Manifest eligibility summary disagrees with queries.")
    for cohort in COHORTS:
        summary = declared["cohorts"].get(cohort)
        if (not isinstance(summary, dict) or summary.get("eligible_queries") != cohort_counts[cohort]
                or summary.get("eligible_sources") != len(cohort_sources[cohort])):
            raise ValueError("Manifest cohort eligibility summary disagrees with queries.")
    passes = len(eligible) >= MIN_QUERIES and all(
        cohort_counts[cohort] >= MIN_COHORT_QUERIES
        and len(cohort_sources[cohort]) >= MIN_COHORT_SOURCES
        for cohort in COHORTS
    )
    if not passes or declared.get("eligibility_floor_passed") is not True:
        raise ValueError("Frozen validation eligibility floor is not met.")
    return eligible


def _validation_source(source_id: str) -> bool:
    parts = source_id.split("/")
    return len(parts) == 2 and parts[0] == "validation" and parts[1] not in {"", ".", ".."}


def calculate_rankings(
    manifest: dict, predictions: dict[str, dict[int, float]], *,
    scorer: str, arm: str, experiment_seed: int, random_seed: int, random_repeats: int,
) -> RankingReport:
    """Calculate rankings from provided scores, without invoking a model."""
    if scorer not in {"pair_head", "mu_cosine"} or arm not in {"S1", "S2", "S3"}:
        raise ValueError("Unknown scorer or experiment arm.")
    if scorer == "pair_head" and arm != "S3":
        raise ValueError("Only S3 has an enabled directional semantic head.")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in (experiment_seed, random_seed)) or (
        isinstance(random_repeats, bool) or not isinstance(random_repeats, int) or random_repeats < 2
    ):
        raise ValueError("Nonnegative seeds and at least two random repeats are required.")
    queries = _eligible_queries(manifest)
    if set(predictions) != {query["id"] for query in queries}:
        raise ValueError("Predictions must cover exactly the frozen eligible queries.")
    rng = np.random.default_rng(random_seed)
    rows = []
    for query in queries:
        candidates = query["candidates"]
        ids = [candidate["id"] for candidate in candidates]
        supplied = predictions[query["id"]]
        if (not isinstance(supplied, dict)
                or any(isinstance(key, bool) or not isinstance(key, int) for key in supplied)
                or set(supplied) != set(ids)):
            raise ValueError(f"Predictions do not match frozen candidates for {query['id']}.")
        scores = [supplied[candidate_id] for candidate_id in ids]
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
               for value in scores):
            raise ValueError("Every candidate score must be a finite scalar.")
        if scorer == "pair_head" and any(not 0 <= value <= 1 for value in scores):
            raise ValueError("Directional head scores must be in [0, 1].")
        relevance = np.asarray([candidate["target"]["coverage"] for candidate in candidates], dtype=np.float64)
        ndcg = linear_ndcg_at_10(relevance, scores, ids)
        # A uniformly random permutation gives each ordering equal probability.
        random_values = [
            linear_ndcg_at_10(relevance, rng.permutation(len(ids)), ids)
            for _ in range(random_repeats)
        ]
        same = [index for index, candidate in enumerate(candidates) if candidate["category"] != "cross_source"]
        different = [index for index, candidate in enumerate(candidates) if candidate["category"] == "cross_source"]
        rows.append(QueryRanking(
            query["id"], query["window"]["source_id"], query["cohort"], ndcg,
            float(np.mean(random_values)),
            len(same), sum(scores[index] for index in same), float(relevance[same].sum()),
            len(different), sum(scores[index] for index in different), float(relevance[different].sum()),
        ))
    return RankingReport(manifest["manifest_sha256"], scorer, arm, experiment_seed,
                         random_seed, random_repeats, tuple(rows))


def event_coverage_diagnostics(manifest: dict, predictions: dict[str, dict[int, float]]) -> dict:
    """Report frozen-pair coverage errors and the recorded fluid-fraction mismatch."""
    queries = _eligible_queries(manifest)
    if set(predictions) != {query["id"] for query in queries}:
        raise ValueError("Coverage predictions must cover exactly the frozen eligible queries.")
    rows = []
    for query in queries:
        candidates = query["candidates"]
        supplied = predictions[query["id"]]
        if (not isinstance(supplied, dict) or set(supplied) != {candidate["id"] for candidate in candidates}
                or any(isinstance(key, bool) or not isinstance(key, int) for key in supplied)):
            raise ValueError("Coverage predictions must match the frozen candidate IDs.")
        for candidate in candidates:
            target = candidate["target"]
            prediction = supplied[candidate["id"]]
            if (isinstance(prediction, bool) or not isinstance(prediction, Real)
                    or not math.isfinite(prediction) or not 0 <= prediction <= 1):
                raise ValueError("Coverage predictions must be finite values in [0, 1].")
            fluid_a, fluid_b = target.get("fluid_a"), target.get("fluid_b")
            mismatch = None
            if fluid_a is not None or fluid_b is not None:
                fractions: list[list[float]] = []
                for fluid in (fluid_a, fluid_b):
                    if (not isinstance(fluid, (list, tuple)) or len(fluid) != 3
                            or any(isinstance(value, bool) or not isinstance(value, Real)
                                   or not math.isfinite(value) or not -1e-12 <= float(value) <= 1 + 1e-12
                                   for value in fluid)
                            or not math.isclose(sum(fluid), 1.0, rel_tol=0, abs_tol=1e-9)):
                        raise ValueError("Recorded fluid fractions must be valid matched triples.")
                    fractions.append([float(value) for value in fluid])
                mismatch = 0.5 * sum(abs(a - b) for a, b in zip(*fractions))
            error = float(prediction) - float(target["coverage"])
            rows.append({
                "query_id": query["id"], "candidate_id": candidate["id"],
                "source_id": query["window"]["source_id"], "cohort": query["cohort"],
                "category": candidate["category"], "target": float(target["coverage"]),
                "prediction": float(prediction), "error": error, "fluid_mismatch": mismatch,
                "fluid_mismatch_unavailable_reason": None if mismatch is not None
                else "fluid fractions not recorded for this pair",
            })
    categories = {}
    for category in ("same_overlap", "same_disjoint", "cross_source"):
        selected = [row for row in rows if row["category"] == category]
        if not selected:
            categories[category] = {"count": 0, "unavailable_reason": "no pairs in eligible queries"}
            continue
        errors = np.asarray([row["error"] for row in selected], dtype=np.float64)
        categories[category] = {
            "count": len(selected), "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors**2))), "bias": float(np.mean(errors)),
            "fluid_mismatch_available": sum(row["fluid_mismatch"] is not None for row in selected),
        }
    return {
        "manifest_sha256": manifest["manifest_sha256"], "pairs": rows, "categories": categories,
        "limitations": [
            "Fluid fractions are the approved coarse closure proxy, not verified per-layer fluid labels.",
            "NTG, thin-layer and no-reservoir diagnostics require separately fingerprinted inputs.",
            "Pair errors describe correlated validation pairs, not independent geological realizations.",
        ],
    }


def physical_overlap_diagnostics(
    manifest: dict, predictions: dict[str, dict[int, float]], *,
    constant: float, waveform_predictions: dict[str, dict[int, float]],
) -> dict:
    """Compare numeric overlap predictions against fixed constant and waveform baselines."""
    queries = _eligible_queries(manifest)
    if (isinstance(constant, bool) or not isinstance(constant, Real)
            or not math.isfinite(constant) or not 0 <= float(constant) <= 1):
        raise ValueError("The fixed constant overlap baseline must be finite and in [0, 1].")
    expected_queries = {query["id"] for query in queries}
    if set(predictions) != expected_queries or set(waveform_predictions) != expected_queries:
        raise ValueError("Overlap predictions and waveform baseline must cover exactly the eligible queries.")
    rows = []
    for query in queries:
        ids = {candidate["id"] for candidate in query["candidates"]}
        supplied = (predictions[query["id"]], waveform_predictions[query["id"]])
        if any(not isinstance(values, dict) or set(values) != ids
               or any(isinstance(key, bool) or not isinstance(key, int) for key in values)
               for values in supplied):
            raise ValueError("Overlap predictions and waveform baseline must match the candidate IDs.")
        for candidate in query["candidates"]:
            target = candidate["target"].get("overlap")
            value = supplied[0][candidate["id"]]
            waveform = supplied[1][candidate["id"]]
            if any(isinstance(number, bool) or not isinstance(number, Real)
                   or not math.isfinite(number) or not 0 <= float(number) <= 1
                   for number in (target, value, waveform)):
                raise ValueError("Overlap targets and predictions must be known finite values in [0, 1].")
            if candidate["category"] in {"same_disjoint", "cross_source"} and target != 0:
                raise ValueError("Disjoint and cross-source pairs must have zero physical overlap.")
            rows.append({
                "query_id": query["id"], "candidate_id": candidate["id"],
                "source_id": query["window"]["source_id"], "cohort": query["cohort"],
                "category": candidate["category"], "target": float(target),
                "prediction": float(value), "constant": float(constant), "waveform": float(waveform),
            })

    def summarize(selected):
        if not selected:
            return {"count": 0, "unavailable_reason": "no pairs in eligible queries"}
        metrics = {}
        for name in ("prediction", "constant", "waveform"):
            errors = np.asarray([row[name] - row["target"] for row in selected], dtype=np.float64)
            metrics[name] = {
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors**2))), "bias": float(np.mean(errors)),
            }
        return {"count": len(selected), "metrics": metrics}

    categories = {
        category: summarize([row for row in rows if row["category"] == category])
        for category in ("same_overlap", "same_disjoint", "cross_source")
    }
    edges = (0.0, 0.25, 0.5, 0.75, 1.0)
    bins = {
        f"{low:g}-{high:g}": summarize([
            row for row in rows if low <= row["target"] < high
            or (high == 1.0 and row["target"] == 1.0)
        ])
        for low, high in zip(edges, edges[1:])
    }
    return {
        "manifest_sha256": manifest["manifest_sha256"], "pairs": rows, "categories": categories,
        "overlap_bins": bins, "constant_baseline": float(constant),
        "bin_rule": "[low, high), except the final bin includes 1",
        "limitation": "Numeric predictions only; scorer identity, waveform protocol and head training require verification.",
    }


def paired_source_bootstrap(rows: tuple[QueryRanking, ...], differences, *, seed: int, repeats: int) -> tuple[float, float]:
    if (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            or isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 2):
        raise ValueError("A nonnegative seed and at least two bootstrap repeats are required.")
    values = np.asarray(differences, dtype=np.float64)
    if values.shape != (len(rows),) or not np.isfinite(values).all() or not rows:
        raise ValueError("Bootstrap requires a finite paired difference for every query.")
    sources = sorted({row.source_id for row in rows})
    if len(sources) < MIN_COHORT_SOURCES:
        raise ValueError("Source-cluster bootstrap requires at least five sources.")
    by_source = {
        source: values[[index for index, row in enumerate(rows) if row.source_id == source]]
        for source in sources
    }
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        chosen = rng.integers(0, len(sources), size=len(sources))
        groups = [by_source[sources[int(position)]] for position in chosen]
        samples[index] = sum(float(group.sum()) for group in groups) / sum(len(group) for group in groups)
    low, high = np.percentile(samples, (2.5, 97.5))
    return float(low), float(high)


def summarize_rankings(report: RankingReport, *, bootstrap_seed: int, bootstrap_repeats: int) -> dict:
    rows = report.rows
    if (len(rows) < MIN_QUERIES or len({row.query_id for row in rows}) != len(rows)
            or any(not math.isfinite(value) or not 0 <= value <= 1
                   for row in rows for value in (row.ndcg, row.random_ndcg))):
        raise ValueError("Ranking report requires at least 200 unique queries with valid NDCG values.")
    gain = np.asarray([row.ndcg - row.random_ndcg for row in rows], dtype=np.float64)
    interval = paired_source_bootstrap(rows, gain, seed=bootstrap_seed, repeats=bootstrap_repeats)
    cohorts = {}
    for cohort in COHORTS:
        selected = [row for row in rows if row.cohort == cohort]
        if len(selected) < MIN_COHORT_QUERIES or len({row.source_id for row in selected}) < MIN_COHORT_SOURCES:
            raise ValueError("Ranking report no longer satisfies cohort eligibility.")
        cohorts[cohort] = {
            "queries": len(selected), "sources": len({row.source_id for row in selected}),
            "ndcg_at_10": float(np.mean([row.ndcg for row in selected])),
            "random_ndcg_at_10": float(np.mean([row.random_ndcg for row in selected])),
            "gain": float(np.mean([row.ndcg - row.random_ndcg for row in selected])),
        }
    pair_cohorts = {}
    for label, prefix in (("same_source", "same_source"), ("different_source", "different_source")):
        count = sum(getattr(row, f"{prefix}_count") for row in rows)
        if count == 0:
            raise ValueError(f"No {label} candidates in frozen eligible queries.")
        pair_cohorts[label] = {
            "candidates": count,
            "mean_score": sum(getattr(row, f"{prefix}_score_sum") for row in rows) / count,
            "mean_relevance": sum(getattr(row, f"{prefix}_relevance_sum") for row in rows) / count,
        }
    return {
        "manifest_sha256": report.manifest_sha256, "scorer": report.scorer, "arm": report.arm,
        "experiment_seed": report.experiment_seed, "eligible_queries": len(rows),
        "ndcg_at_10": float(np.mean([row.ndcg for row in rows])),
        "random_ndcg_at_10": float(np.mean([row.random_ndcg for row in rows])),
        "gain": float(np.mean(gain)), "paired_source_bootstrap_95": list(interval),
        "random_seed": report.random_seed, "random_repeats": report.random_repeats,
        "bootstrap_seed": bootstrap_seed, "bootstrap_repeats": bootstrap_repeats,
        "cohorts": cohorts, "pair_cohorts": pair_cohorts,
    }


def _matched_rows(first: RankingReport, second: RankingReport) -> tuple[QueryRanking, ...]:
    if first.manifest_sha256 != second.manifest_sha256:
        raise ValueError("Comparison requires the same frozen manifest.")
    other = {row.query_id: row for row in second.rows}
    if (len(other) != len(second.rows) or len(other) != len(first.rows)
            or len({row.query_id for row in first.rows}) != len(first.rows)):
        raise ValueError("Comparison query identities differ.")
    matched = []
    for row in first.rows:
        second_row = other.get(row.query_id)
        if second_row is None or (row.source_id, row.cohort) != (second_row.source_id, second_row.cohort):
            raise ValueError("Comparison query identities or cohorts differ.")
        matched.append(second_row)
    return tuple(matched)


def compare_matched_cosine(
    candidate: RankingReport, baseline: RankingReport, *, bootstrap_seed: int, bootstrap_repeats: int,
) -> dict:
    if (candidate.scorer != "mu_cosine" or baseline.scorer != "mu_cosine"
            or (candidate.arm, baseline.arm) not in {("S2", "S1"), ("S3", "S1"), ("S3", "S2")}
            or candidate.experiment_seed != baseline.experiment_seed):
        raise ValueError("Cosine comparison requires a matched-seed baseline and later experimental arm.")
    matched = _matched_rows(candidate, baseline)
    differences = [row.ndcg - reference.ndcg for row, reference in zip(candidate.rows, matched)]
    interval = paired_source_bootstrap(candidate.rows, differences, seed=bootstrap_seed, repeats=bootstrap_repeats)
    return {
        f"mean_ndcg_gain_over_{baseline.arm.lower()}": float(np.mean(differences)),
        "paired_source_bootstrap_95": list(interval),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_repeats": bootstrap_repeats,
    }


def matched_cosine_cohort_diagnostics(
    candidate: RankingReport, baseline: RankingReport, *, bootstrap_seed: int, bootstrap_repeats: int,
) -> dict:
    """Explore depth-cohort encoder differences without changing the retrieval gate."""
    overall = compare_matched_cosine(
        candidate, baseline, bootstrap_seed=bootstrap_seed, bootstrap_repeats=bootstrap_repeats,
    )
    matched = _matched_rows(candidate, baseline)
    cohorts = {}
    for cohort in COHORTS:
        pairs = [(row, reference) for row, reference in zip(candidate.rows, matched) if row.cohort == cohort]
        rows = tuple(row for row, _ in pairs)
        if len(rows) < MIN_COHORT_QUERIES or len({row.source_id for row in rows}) < MIN_COHORT_SOURCES:
            raise ValueError("Matched cosine diagnostics require the frozen cohort eligibility floors.")
        differences = [row.ndcg - reference.ndcg for row, reference in pairs]
        cohorts[cohort] = {
            "queries": len(rows),
            "sources": len({row.source_id for row in rows}),
            "mean_ndcg_difference": float(np.mean(differences)),
            "paired_source_bootstrap_95": list(paired_source_bootstrap(
                rows, differences, seed=bootstrap_seed, repeats=bootstrap_repeats,
            )),
        }
    return {
        "overall": overall,
        "cohorts": cohorts,
        "limitation": (
            "Exploratory query-weighted depth cohorts; intervals are not multiplicity-adjusted. "
            "Event-coverage proxy only, not a geological-understanding or adoption test."
        ),
    }


def retrieval_metric_gate(
    primary: RankingReport, *, bootstrap_seed: int, bootstrap_repeats: int,
    replication: RankingReport | None,
) -> dict:
    """Assess retrieval metrics only; this is not an adoption or checkpoint-selection decision."""
    if primary.scorer != "pair_head" or primary.arm != "S3":
        raise ValueError("The primary retrieval metric requires the S3 directional head.")
    summary = summarize_rankings(primary, bootstrap_seed=bootstrap_seed, bootstrap_repeats=bootstrap_repeats)
    checks = {
        "gain_at_least_0_01": summary["gain"] >= 0.01,
        "positive_paired_lower_95": summary["paired_source_bootstrap_95"][0] > 0,
        "no_depth_cohort_regression_over_0_02": all(
            cohort["gain"] >= -0.02 for cohort in summary["cohorts"].values()
        ),
    }
    if replication is None:
        checks["independent_seed_repeats_gain_direction"] = None
    else:
        if (replication.scorer != "pair_head" or replication.arm != "S3"
                or replication.experiment_seed == primary.experiment_seed
                or replication.random_seed != primary.random_seed
                or replication.random_repeats != primary.random_repeats):
            raise ValueError("Replication requires an independent S3 seed and the same random-baseline configuration.")
        _matched_rows(primary, replication)
        repeated = summarize_rankings(replication, bootstrap_seed=bootstrap_seed, bootstrap_repeats=bootstrap_repeats)
        checks["independent_seed_repeats_gain_direction"] = repeated["gain"] > 0
    status = "FAIL" if False in checks.values() else "PENDING" if None in checks.values() else "PASS_METRICS_ONLY"
    return {
        "status": status, "checks": checks, "primary": summary,
        "limitation": (
            "Numeric metrics only: input fingerprint verification, trained-head and checkpoint identity, "
            "reconstruction guardrails, and independent execution provenance are not established here."
        ),
    }
