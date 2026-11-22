"""Bind and verify hashes of the validation data used by a fixed manifest."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
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
    canonical_overlap,
    content_hash,
    discover_sources,
    inspect_source,
)
from scripts.streamed_overlap_manifest import _window, label_candidates
from scripts.streamed_overlap_targets import Window, _read_bounded
from scripts.streamed_overlap_yield import SEISMIC_KEY


INPUT_KEYS = (SEISMIC_KEY, "geologic_age_faulted", "oil_closures", "gas_closures")


def _digest_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    if not np.isfinite(contiguous).all():
        raise ValueError("Non-finite source data cannot be fingerprinted as valid.")
    digest = hashlib.sha256()
    header = json.dumps({
        "shape": list(contiguous.shape), "dtype": contiguous.dtype.str,
    }, sort_keys=True).encode()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _window_record(window: Window) -> dict:
    return {"source_id": window.source_id, "origin": list(window.origin), "shape": list(window.shape)}


def _event_region(a: Window, b: Window) -> dict | None:
    if a.source_id != b.source_id:
        return None
    start = [max(x, y) for x, y in zip(a.origin, b.origin)]
    end = [min(x, y) for x, y in zip(a.end, b.end)]
    if any(x >= y for x, y in zip(start, end)):
        return None
    return {"source_id": a.source_id, "xy_start": start[:2], "xy_end": end[:2]}


def input_fingerprints(queries: list[dict], groups: dict) -> dict:
    """Hash canonical input windows and overlapping full-depth seismic traces."""
    windows = {}
    regions = {}
    for query in queries:
        a = _window(query["window"])
        for data in [query["window"], *(candidate["window"] for candidate in query["candidates"])]:
            window = _window(data)
            record = _window_record(window)
            windows[content_hash(record)] = record
        for candidate in query["candidates"]:
            region = _event_region(a, _window(candidate["window"]))
            if region is not None:
                regions[content_hash(region)] = region

    window_hashes = {}
    for key in sorted(windows):
        record = windows[key]
        window = _window(record)
        if window.source_id not in groups:
            raise ValueError(f"Unknown validation source in manifest: {window.source_id}")
        source = groups[window.source_id]
        array_hashes = {}
        for array_key in INPUT_KEYS:
            if array_key not in source:
                raise ValueError(f"Missing fingerprint array {array_key} in {window.source_id}")
            array = source[array_key]
            if len(array.shape) != 3 or any(end > size for end, size in zip(window.end, array.shape)):
                raise ValueError(f"Unsupported fingerprint window {window.source_id}/{array_key}")
            array_hashes[array_key] = _digest_array(_read_bounded(array, window.slices))
        window_hashes[key] = {"window": record, "arrays": array_hashes}

    event_hashes = {}
    for key in sorted(regions):
        record = regions[key]
        array = groups[record["source_id"]][SEISMIC_KEY]
        x0, y0 = record["xy_start"]
        x1, y1 = record["xy_end"]
        pieces = []
        for x in range(x0, x1, 50):
            for y in range(y0, y1, 50):
                slices = (slice(x, min(x + 50, x1)), slice(y, min(y + 50, y1)), slice(None))
                pieces.append({
                    "origin_xy": [x, y],
                    "sha256": _digest_array(_read_bounded(array, slices)),
                })
        event_hashes[key] = {"region": record, "chunks": pieces}
    return {"windows": window_hashes, "event_traces": event_hashes}


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    expected = manifest.get("manifest_sha256")
    actual = content_hash({
        key: value for key, value in manifest.items()
        if key not in {"manifest_sha256", "measurement"}
    })
    if not isinstance(expected, str) or expected != actual:
        raise ValueError("Manifest content hash is invalid.")
    target_path = Path(__file__).with_name("streamed_overlap_targets.py")
    if manifest.get("target_implementation_sha256") != hashlib.sha256(target_path.read_bytes()).hexdigest():
        raise ValueError("Target implementation differs from the frozen manifest.")
    overlap_hash = hashlib.sha256(inspect.getsource(canonical_overlap).encode()).hexdigest()
    if manifest.get("overlap_implementation_sha256") != overlap_hash:
        raise ValueError("Overlap implementation differs from the frozen manifest.")
    if manifest.get("numpy_version") != np.__version__ or manifest.get("scipy_version") != scipy.__version__:
        raise ValueError("NumPy/SciPy versions differ from the frozen manifest.")
    return manifest


def open_validation_sources(root: Path, manifest: dict) -> dict:
    sources = [source for source in discover_sources(root) if source.split == "validation"]
    if len({source.identifier for source in sources}) != len(sources):
        raise ValueError("Validation source identifiers are not unique.")
    inventory = [inspect_source(source, SEISMIC_KEY) for source in sources]
    if content_hash(inventory) != manifest.get("source_metadata_sha256"):
        raise ValueError("Validation source inventory differs from the frozen manifest.")
    return {source.identifier: zarr.open_group(str(source.path), mode="r") for source in sources}


def _plans(queries: list[dict]) -> list[dict]:
    return [{
        "id": query["id"],
        "cohort": query["cohort"],
        "window": query["window"],
        "candidates": [
            {"category": candidate["category"], "window": candidate["window"]}
            for candidate in query["candidates"]
        ],
    } for query in queries]


def bind_manifest(manifest: dict, groups: dict) -> dict:
    if "input_integrity" in manifest:
        raise ValueError("Manifest already has input fingerprints.")
    plans = _plans(manifest["queries"])
    if content_hash(plans) != manifest["candidate_plan_sha256"]:
        raise ValueError("Candidate plan does not match the frozen plan hash.")
    labeled, eligibility = label_candidates(plans, groups)
    if (content_hash(labeled) != content_hash(manifest["queries"])
            or eligibility != manifest["eligibility"]):
        raise ValueError("Current target labels or eligibility differ from the frozen manifest.")
    fingerprints = input_fingerprints(plans, groups)
    bound = deepcopy(manifest)
    bound.pop("measurement", None)
    bound.pop("manifest_sha256")
    bound["schema"] = "streamed-retrieval-validation-v2"
    bound["input_integrity"] = {
        "algorithm": "sha256-array-dtype-shape-c-order-v1",
        "reference_manifest_sha256": manifest["manifest_sha256"],
        "fingerprints": fingerprints,
    }
    bound["limitations"] = [
        "Closure fractions are an operational proxy; source generation-time worktrees are unknown.",
        "Fingerprints cover canonical seismic, age, oil/gas windows and full-depth event traces, not whole sources or other diagnostic labels.",
        "Eligibility and input fingerprints do not certify MPS readiness, reconstruction guardrails, or authorize training.",
    ]
    bound["manifest_sha256"] = content_hash(bound)
    return bound


def verify_inputs(manifest: dict, groups: dict) -> dict:
    integrity = manifest.get("input_integrity")
    if not isinstance(integrity, dict) or integrity.get("algorithm") != "sha256-array-dtype-shape-c-order-v1":
        raise ValueError("Manifest has no supported input fingerprints.")
    expected = integrity["fingerprints"]
    actual = input_fingerprints(_plans(manifest["queries"]), groups)
    if actual != expected:
        for kind in ("windows", "event_traces"):
            expected_items, actual_items = expected[kind], actual[kind]
            for key in sorted(set(expected_items) | set(actual_items)):
                if expected_items.get(key) != actual_items.get(key):
                    raise ValueError(f"Validation input fingerprint mismatch: {kind}/{key}")
        raise ValueError("Validation input fingerprints differ.")
    return {
        "status": "VALIDATION_INPUTS_VERIFIED",
        "manifest_sha256": manifest["manifest_sha256"],
        "verified_windows": len(actual["windows"]),
        "verified_full_depth_event_regions": len(actual["event_traces"]),
        "limitation": "Only recorded input windows and event traces are verified; complete source cubes and other diagnostic labels are not.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("bind", "verify"))
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.source_root.resolve(strict=True)
    if args.output.resolve().is_relative_to(root):
        raise ValueError("Integrity output must be outside the source root.")
    started = time.monotonic()
    with args.output.open("x") as handle:
        manifest = load_manifest(args.manifest)
        groups = open_validation_sources(root, manifest)
        result = bind_manifest(manifest, groups) if args.mode == "bind" else verify_inputs(manifest, groups)
        result["measurement"] = {
            "elapsed_seconds": time.monotonic() - started,
            "max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    if args.mode == "verify":
        print(json.dumps(result, indent=2))
    else:
        fingerprints = result["input_integrity"]["fingerprints"]
        print(json.dumps({
            "status": "INPUT_FINGERPRINTS_BOUND_NOT_YET_VERIFIED",
            "manifest_sha256": result["manifest_sha256"],
            "windows": len(fingerprints["windows"]),
            "event_regions": len(fingerprints["event_traces"]),
            "measurement": result["measurement"],
        }, indent=2))


if __name__ == "__main__":
    main()
