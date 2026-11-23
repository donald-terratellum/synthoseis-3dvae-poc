"""Sample primary-validation geology labels for exploratory evaluation feasibility."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import numpy as np

from scripts.streamed_fluid_labels import lithology_masks
from scripts.streamed_overlap_integrity import load_manifest, open_validation_sources
from scripts.streamed_overlap_manifest import _window
from scripts.streamed_overlap_metrics import _eligible_queries
from scripts.streamed_overlap_targets import _read_bounded, _supported


KEYS = ("faulted_lithology", "faults/faulted_channel_labels")


def _features(window, groups: dict) -> dict:
    if window.source_id not in groups:
        raise ValueError(f"Validation source missing for {window.source_id}.")
    group = groups[window.source_id]
    arrays = []
    for key in KEYS:
        if key not in group or not _supported(group[key], window, labels=True):
            raise ValueError(f"Missing or unsupported {key} in {window.source_id}.")
        values = _read_bounded(group[key], window.slices)
        if (not (np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_))
                or np.iscomplexobj(values)
                or not np.isfinite(values).all()):
            raise ValueError(f"Invalid {key} values in {window.source_id}.")
        arrays.append(values)
    masks = lithology_masks(*arrays)
    rock_count = int(np.count_nonzero(masks["rock"]))
    return {
        "rock_voxels": rock_count,
        "corrected_sand_voxels": int(np.count_nonzero(masks["sand"])),
        "channel_voxels": int(np.count_nonzero(masks["channel"])),
        "corrected_sand_fraction_of_rock": (
            float(np.count_nonzero(masks["sand"]) / rock_count) if rock_count else None
        ),
    }


def sample_plan(manifest: dict) -> list[dict]:
    """Choose one query per source before inspecting any geological label."""
    by_source = {}
    for query in _eligible_queries(manifest):
        by_source.setdefault(query["window"]["source_id"], []).append(query)
    selected = []
    for index, source in enumerate(sorted(by_source)):
        cohort = ("shallower", "deeper")[index % 2]
        candidates = sorted((q for q in by_source[source] if q["cohort"] == cohort), key=lambda q: q["id"])
        if not candidates:
            raise ValueError(f"Pilot query unavailable for {source} in {cohort}.")
        query = candidates[0]
        cross = [candidate for candidate in query["candidates"] if candidate["category"] == "cross_source"]
        if not cross:
            raise ValueError(f"Pilot query {query['id']} has no cross-source candidates.")
        selected.append({"query": query, "candidates": cross})
    return selected


def summarize_sample(manifest: dict, groups: dict, *, max_wall_seconds: float) -> dict:
    if (isinstance(max_wall_seconds, bool) or not isinstance(max_wall_seconds, (int, float))
            or not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0):
        raise ValueError("A positive finite wall limit is required.")
    plan = sample_plan(manifest)
    started = time.monotonic()
    samples = []
    paired_differences = []
    query_windows_with_rock = 0
    queries_with_usable_pair = 0
    channel_sources = set()
    for item in plan:
        windows = [item["query"]["window"], *(candidate["window"] for candidate in item["candidates"])]
        features = []
        for value in windows:
            if time.monotonic() - started > max_wall_seconds:
                raise TimeoutError("Semantic feasibility wall limit exceeded; no partial report is valid.")
            feature = _features(_window(value), groups)
            features.append(feature)
            if feature["channel_voxels"]:
                channel_sources.add(value["source_id"])
        samples.extend(features)
        a = features[0]["corrected_sand_fraction_of_rock"]
        query_windows_with_rock += int(a is not None)
        usable = 0
        for candidate in features[1:]:
            b = candidate["corrected_sand_fraction_of_rock"]
            if a is not None and b is not None:
                paired_differences.append(abs(a - b))
                usable += 1
        queries_with_usable_pair += int(usable > 0)
    if time.monotonic() - started > max_wall_seconds:
        raise TimeoutError("Semantic feasibility wall limit exceeded; no partial report is valid.")
    fractions = [sample["corrected_sand_fraction_of_rock"] for sample in samples
                 if sample["corrected_sand_fraction_of_rock"] is not None]
    if not fractions:
        raise ValueError("No sampled window contains rock.")
    return {
        "schema": "streamed-semantic-feasibility-sample-v1",
        "validation_manifest_sha256": manifest["manifest_sha256"],
        "selection_rule": (
            "One first-by-id eligible query per sorted source; alternate shallower/deeper by source order; "
            "include all its frozen cross-source candidates. Selection precedes label inspection."
        ),
        "query_sources": len(plan),
        "queries_by_cohort": dict(Counter(item["query"]["cohort"] for item in plan)),
        "cross_source_pairs": sum(len(item["candidates"]) for item in plan),
        "sampled_windows": len(samples),
        "windows_with_rock": len(fractions),
        "query_windows_with_rock": query_windows_with_rock,
        "query_sources_with_usable_pair": queries_with_usable_pair,
        "windows_with_channel_voxels": sum(sample["channel_voxels"] > 0 for sample in samples),
        "sources_with_channel_voxels": len(channel_sources),
        "corrected_sand_fraction_of_rock": {
            "min": min(fractions), "median": float(np.median(fractions)), "max": max(fractions),
        },
        "cross_source_absolute_sand_fraction_difference": {
            "available_pairs": len(paired_differences),
            "min": min(paired_differences) if paired_differences else None,
            "median": float(np.median(paired_differences)) if paired_differences else None,
            "max": max(paired_differences) if paired_differences else None,
        },
        "elapsed_seconds": time.monotonic() - started,
        "limitation": (
            "Exploratory nonrepresentative frozen-validation sample, not a held-out test or approved target. "
            "Repeated candidates are correlated; labels and source arrays are not independently validated or "
            "fingerprint-verified by this sample. Channel presence and sand fraction are not complete geology."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    args = parser.parse_args()
    if ("fake_data_validation_v2" in args.source_root.parts
            or "fake_data_validation_v2" in args.source_root.resolve().parts
            or "fake_data_validation_v2" in args.manifest.parts
            or "fake_data_validation_v2" in args.manifest.resolve().parts):
        raise ValueError("The regenerating validation-v2 dataset is not permitted.")
    manifest = load_manifest(args.manifest)
    if manifest["manifest_sha256"] != args.manifest_sha256:
        raise ValueError("Validation manifest does not match the expected hash.")
    groups = open_validation_sources(args.source_root, manifest)
    print(json.dumps(summarize_sample(manifest, groups, max_wall_seconds=args.max_wall_seconds),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
