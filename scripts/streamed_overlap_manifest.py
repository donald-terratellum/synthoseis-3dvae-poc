"""Construct a fixed, model-independent validation retrieval candidate manifest."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import resource
import time

import numpy as np
import scipy
import zarr

from scripts.streamed_overlap_audit import (
    LABEL_KEYS,
    PATCH_SHAPE,
    canonical_overlap,
    content_hash,
    discover_sources,
    inspect_source,
)
from scripts.streamed_overlap_targets import (
    LABEL_DEPTH_LIMIT,
    TARGET_CONTRACT_VERSION,
    Window,
    pair_target,
)
from scripts.streamed_overlap_yield import SEISMIC_KEY, _origin


COHORTS = ("shallower", "deeper")
POSITIVE_OFFSETS = (8, 16, 32)
DISJOINT_OFFSETS = (64, 96, 128)
MIN_QUERIES = 200
MIN_COHORT_QUERIES = 20
MIN_COHORT_SOURCES = 5


def plan_candidates(sources, groups, seed: int, queries_per_cohort_source: int) -> list[dict]:
    """Choose all queries and candidates before any relevance labels are inspected."""
    if queries_per_cohort_source < 1:
        raise ValueError("queries_per_cohort_source must be positive.")
    if len(sources) < MIN_COHORT_SOURCES:
        raise ValueError("At least five validation sources are required.")
    if len({source.identifier for source in sources}) != len(sources):
        raise ValueError("Validation source identifiers must be unique.")
    rng = np.random.default_rng(seed)
    queries = []
    seen = set()
    for source_index, source in enumerate(sources):
        group = groups[source.identifier]
        # _origin checks every required array and ensures full same-index support.
        probe = _origin(group, rng, depth_margin=max(DISJOINT_OFFSETS))
        del probe
        depth_limit = min(
            min(group[key].shape[2] for key in (SEISMIC_KEY, *LABEL_KEYS)),
            LABEL_DEPTH_LIMIT,
        )
        max_depth = depth_limit - PATCH_SHAPE[2] - max(DISJOINT_OFFSETS)
        if max_depth < 1:
            raise ValueError(f"Insufficient depth for both query cohorts: {source.identifier}")
        middle = (max_depth + 1) // 2
        for cohort in COHORTS:
            z_min, z_max = (0, middle - 1) if cohort == COHORTS[0] else (middle, max_depth)
            for sample in range(queries_per_cohort_source):
                origin = _origin(group, rng, depth_margin=max(DISJOINT_OFFSETS))
                origin = (origin[0], origin[1], int(rng.integers(z_min, z_max + 1)))
                if (source.identifier, origin) in seen:
                    raise ValueError("Duplicate query origin; choose a different seed or smaller count.")
                seen.add((source.identifier, origin))
                a = Window(source.identifier, origin, PATCH_SHAPE)
                candidates = []
                for category, offsets in (
                    ("same_overlap", POSITIVE_OFFSETS), ("same_disjoint", DISJOINT_OFFSETS)
                ):
                    for offset in offsets:
                        b = Window(source.identifier, (origin[0], origin[1], origin[2] + offset), PATCH_SHAPE)
                        candidates.append({"category": category, "window": asdict(b)})
                for delta in range(1, 5):
                    other = sources[(source_index + delta) % len(sources)]
                    b = Window(other.identifier, _origin(groups[other.identifier], rng), PATCH_SHAPE)
                    candidates.append({"category": "cross_source", "window": asdict(b)})
                rng.shuffle(candidates)
                queries.append({
                    "id": f"{source.identifier}:{cohort}:{sample}",
                    "cohort": cohort,
                    "window": asdict(a),
                    "candidates": candidates,
                })
    return queries


def _window(data: dict) -> Window:
    return Window(data["source_id"], tuple(data["origin"]), tuple(data["shape"]))


def label_candidates(queries: list[dict], groups: dict) -> tuple[list[dict], dict]:
    counts = {cohort: {"eligible_queries": 0, "source_ids": set()} for cohort in COHORTS}
    reasons = Counter()
    labeled = []
    for query in queries:
        a = _window(query["window"])
        candidates = []
        for index, candidate in enumerate(query["candidates"]):
            b = _window(candidate["window"])
            target = pair_target(groups[a.source_id], groups[b.source_id], a, b)
            candidates.append({**candidate, "id": index, "target": asdict(target)})
        unavailable = [candidate for candidate in candidates if candidate["target"]["coverage"] is None]
        positive = any(
            candidate["target"]["coverage"] is not None
            and candidate["target"]["coverage"] > 0
            for candidate in candidates
        )
        eligible = len(candidates) >= 10 and not unavailable and positive
        if eligible:
            counts[query["cohort"]]["eligible_queries"] += 1
            counts[query["cohort"]]["source_ids"].add(a.source_id)
        else:
            if unavailable:
                reasons.update(candidate["target"]["unavailable_reason"] for candidate in unavailable)
            if not positive:
                reasons["no_known_positive"] += 1
            if len(candidates) < 10:
                reasons["fewer_than_ten_candidates"] += 1
        labeled.append({**query, "candidates": candidates, "eligible": eligible})
    summary = {
        cohort: {
            "eligible_queries": counts[cohort]["eligible_queries"],
            "eligible_sources": len(counts[cohort]["source_ids"]),
        }
        for cohort in COHORTS
    }
    total = sum(summary[cohort]["eligible_queries"] for cohort in COHORTS)
    passes = total >= MIN_QUERIES and all(
        values["eligible_queries"] >= MIN_COHORT_QUERIES
        and values["eligible_sources"] >= MIN_COHORT_SOURCES
        for values in summary.values()
    )
    return labeled, {
        "eligible_queries": total,
        "cohorts": summary,
        "ineligibility_reasons": dict(reasons),
        "eligibility_floor_passed": passes,
    }


def build_manifest(root: Path, seed: int, queries_per_cohort_source: int) -> dict:
    started = time.monotonic()
    sources = [source for source in discover_sources(root) if source.split == "validation"]
    inventory = [inspect_source(source, SEISMIC_KEY) for source in sources]
    for source, metadata in zip(sources, inventory):
        if metadata["issues"]:
            raise ValueError(f"Source inventory issues for {source.identifier}: {metadata['issues']}")
    groups = {source.identifier: zarr.open_group(str(source.path), mode="r") for source in sources}
    queries = plan_candidates(sources, groups, seed, queries_per_cohort_source)
    plan_hash = content_hash(queries)
    labeled, eligibility = label_candidates(queries, groups)
    code = Path(__file__).with_name("streamed_overlap_targets.py").read_bytes()
    manifest = {
        "schema": "streamed-retrieval-validation-v1",
        "status": "LOCAL_TARGET_VALIDATION_PENDING",
        "contract_version": TARGET_CONTRACT_VERSION,
        "target_implementation_sha256": hashlib.sha256(code).hexdigest(),
        "overlap_implementation_sha256": hashlib.sha256(
            inspect.getsource(canonical_overlap).encode()
        ).hexdigest(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "seed": seed,
        "queries_per_cohort_source": queries_per_cohort_source,
        "cohort_definition": "Query origin depth in the shallower/deeper half of the supported range, reserving 128 samples for disjoint candidates.",
        "candidate_rule": "Three same-source Z offsets 8/16/32, three disjoint Z offsets 64/96/128, and four other validation sources. Ties use candidate id ascending.",
        "eligibility_rule": "At least ten fixed candidates, every relevance known, and at least one strictly positive relevance.",
        "source_metadata_sha256": content_hash(inventory),
        "source_metadata": inventory,
        "candidate_plan_sha256": plan_hash,
        "eligibility": eligibility,
        "queries": labeled,
        "limitations": [
            "Closure fractions are an operational proxy; source generation-time worktrees are unknown.",
            "Source metadata hashes do not establish full source-content immutability; verify input content before model evaluation.",
            "Eligibility does not certify MPS readiness, reconstruction guardrails, or authorize training.",
        ],
    }
    manifest["manifest_sha256"] = content_hash(manifest)
    manifest["measurement"] = {
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--queries-per-cohort-source", required=True, type=int)
    args = parser.parse_args()
    root = args.source_root.resolve(strict=True)
    if args.output.resolve().is_relative_to(root):
        raise ValueError("Manifest output must be outside the source root.")
    with args.output.open("x") as handle:
        manifest = build_manifest(root, args.seed, args.queries_per_cohort_source)
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        key: manifest[key] for key in ("status", "manifest_sha256", "eligibility", "measurement")
    }, indent=2))
    if not manifest["eligibility"]["eligibility_floor_passed"]:
        raise SystemExit("Validation retrieval eligibility floor not met.")


if __name__ == "__main__":
    main()
