"""Fingerprint and report validation-only NTG and corrected-sand proxies."""

import hashlib
import math
from pathlib import Path
import time

import numpy as np

from scripts.streamed_fluid_labels import lithology_masks
from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_integrity import _digest_array, _window_record
from scripts.streamed_overlap_manifest import _window
from scripts.streamed_overlap_metrics import _eligible_queries, event_coverage_diagnostics
from scripts.streamed_overlap_targets import _read_bounded, _supported


KEYS = ("faulted_net_to_gross", "faulted_lithology", "faults/faulted_channel_labels")


def _implementation_hashes() -> dict[str, str]:
    directory = Path(__file__).parent
    names = (
        "streamed_overlap_diagnostics.py", "streamed_fluid_labels.py",
        "streamed_overlap_targets.py", "streamed_overlap_integrity.py",
    )
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}


def _thin_runs(sand: np.ndarray, maximum: int) -> tuple[int, int]:
    padded = np.pad(sand, ((0, 0), (0, 0), (1, 1)), constant_values=False)
    starts = np.argwhere(~padded[:, :, :-1] & padded[:, :, 1:])
    ends = np.argwhere(padded[:, :, :-1] & ~padded[:, :, 1:])
    if len(starts) != len(ends) or not np.array_equal(starts[:, :2], ends[:, :2]):
        raise ValueError("Corrected-sand run boundaries are inconsistent.")
    lengths = ends[:, 2] - starts[:, 2]
    complete = (starts[:, 2] > 0) & (ends[:, 2] < sand.shape[2])
    return int(np.count_nonzero(complete & (lengths <= maximum))), int(np.count_nonzero(~complete))


def _window_diagnostic(window, groups: dict, thin_max_voxels: int) -> dict:
    record = _window_record(window)
    if window.source_id not in groups:
        return {"window": record, "status": "UNAVAILABLE", "reason": "missing:source"}
    group = groups[window.source_id]
    arrays = {}
    for key in KEYS:
        if key not in group:
            return {"window": record, "status": "UNAVAILABLE", "reason": f"missing:{key}"}
        array = group[key]
        if not _supported(array, window, labels=True):
            return {"window": record, "status": "UNAVAILABLE", "reason": f"unsupported:{key}"}
        dtype = np.dtype(array.dtype)
        if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)) or np.issubdtype(
            dtype, np.complexfloating
        ):
            return {"window": record, "status": "UNAVAILABLE", "reason": f"unusable_dtype:{key}"}
        values = _read_bounded(array, window.slices)
        if not np.isfinite(values).all():
            return {"window": record, "status": "UNAVAILABLE", "reason": f"nonfinite:{key}"}
        arrays[key] = values
    ntg = arrays[KEYS[0]]
    if np.any((ntg < 0) | (ntg > 1)):
        return {"window": record, "status": "UNAVAILABLE", "reason": "out_of_range:faulted_net_to_gross"}
    masks = lithology_masks(arrays[KEYS[1]], arrays[KEYS[2]])
    sand, rock = masks["sand"], masks["rock"]
    rock_count, sand_count = int(np.count_nonzero(rock)), int(np.count_nonzero(sand))
    thin_count, censored = _thin_runs(sand, thin_max_voxels)
    return {
        "window": record, "status": "AVAILABLE",
        "array_sha256": {key: _digest_array(arrays[key]) for key in KEYS},
        "rock_voxels": rock_count, "corrected_sand_voxels": sand_count,
        "mean_ntg_over_rock": float(np.mean(ntg[rock], dtype=np.float64)) if rock_count else None,
        "ntg_unavailable_reason": None if rock_count else "no rock voxels in window",
        "no_corrected_sand": sand_count == 0,
        "complete_thin_sand_runs": thin_count, "window_edge_censored_sand_runs": censored,
    }


def bind_diagnostics(manifest: dict, groups: dict, *, thin_max_voxels: int, max_wall_seconds: float) -> dict:
    """Freeze extra labels and a predeclared thin-run proxy for the eligible cohort."""
    if (isinstance(thin_max_voxels, bool) or not isinstance(thin_max_voxels, int) or thin_max_voxels < 1
            or isinstance(max_wall_seconds, bool) or not isinstance(max_wall_seconds, (int, float))
            or not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0):
        raise ValueError("A positive predeclared thin-run threshold and wall limit are required.")
    queries = _eligible_queries(manifest)
    windows = {}
    for query in queries:
        for value in (query["window"], *(candidate["window"] for candidate in query["candidates"])):
            window = _window(value)
            windows[content_hash(_window_record(window))] = window
    deadline = time.monotonic() + max_wall_seconds
    records = {}
    for key in sorted(windows):
        if time.monotonic() > deadline:
            raise TimeoutError("Diagnostic binding wall-time limit exceeded; no partial manifest is valid.")
        records[key] = _window_diagnostic(windows[key], groups, thin_max_voxels)
    if time.monotonic() > deadline:
        raise TimeoutError("Diagnostic binding wall-time limit exceeded; no partial manifest is valid.")
    result = {
        "schema": "streamed-validation-diagnostics-v1",
        "validation_manifest_sha256": manifest["manifest_sha256"],
        "thin_max_voxels": thin_max_voxels, "implementation_sha256": _implementation_hashes(),
        "windows": records,
        "limitations": [
            "NTG is the stored faulted net-to-gross mean over corrected rock, not an independently validated property.",
            "Thin runs are complete within-window corrected-sand runs, not identified geological layers.",
            "Window-edge sand runs are censored; no corrected sand does not establish geological reservoir absence.",
        ],
    }
    result["diagnostic_manifest_sha256"] = content_hash(result)
    return result


def verify_diagnostics(manifest: dict, groups: dict, bound: dict, *, max_wall_seconds: float) -> dict:
    if (not isinstance(bound, dict) or bound.get("schema") != "streamed-validation-diagnostics-v1"
            or bound.get("validation_manifest_sha256") != manifest.get("manifest_sha256")
            or bound.get("diagnostic_manifest_sha256") != content_hash({
                key: value for key, value in bound.items() if key != "diagnostic_manifest_sha256"
            })
            or bound.get("implementation_sha256") != _implementation_hashes()):
        raise ValueError("Diagnostic manifest or implementation differs from its frozen identity.")
    current = bind_diagnostics(
        manifest, groups, thin_max_voxels=bound["thin_max_voxels"], max_wall_seconds=max_wall_seconds,
    )
    if current != bound:
        raise ValueError("Validation diagnostic inputs differ from their frozen fingerprints.")
    return current


def diagnostic_report(
    manifest: dict, groups: dict, bound: dict, predictions: dict, *, max_wall_seconds: float,
) -> dict:
    verified = verify_diagnostics(manifest, groups, bound, max_wall_seconds=max_wall_seconds)
    coverage = event_coverage_diagnostics(manifest, predictions)
    queries = {query["id"]: query for query in _eligible_queries(manifest)}
    rows = []
    for pair in coverage["pairs"]:
        query = queries[pair["query_id"]]
        candidate = next(item for item in query["candidates"] if item["id"] == pair["candidate_id"])
        windows = [_window(query["window"]), _window(candidate["window"])]
        records = [verified["windows"][content_hash(_window_record(window))] for window in windows]
        unavailable = [record["reason"] for record in records if record["status"] != "AVAILABLE"]
        row = dict(pair)
        if unavailable:
            row.update({"diagnostic_status": "UNAVAILABLE", "diagnostic_unavailable_reasons": unavailable,
                        "ntg_difference": None, "ntg_unavailable_reason": "; ".join(unavailable),
                        "thin_run_either": None, "no_corrected_sand_either": None})
        else:
            a, b = records
            mean_a, mean_b = a["mean_ntg_over_rock"], b["mean_ntg_over_rock"]
            row.update({
                "diagnostic_status": "AVAILABLE", "diagnostic_unavailable_reasons": [],
                "ntg_difference": None if mean_a is None or mean_b is None else abs(mean_a - mean_b),
                "ntg_unavailable_reason": "one or both windows contain no rock"
                if mean_a is None or mean_b is None else None,
                "thin_run_either": bool(a["complete_thin_sand_runs"] or b["complete_thin_sand_runs"]),
                "no_corrected_sand_either": a["no_corrected_sand"] or b["no_corrected_sand"],
                "a_window_edge_censored_runs": a["window_edge_censored_sand_runs"],
                "b_window_edge_censored_runs": b["window_edge_censored_sand_runs"],
            })
        rows.append(row)
    cohorts = {}
    for field in ("thin_run_either", "no_corrected_sand_either"):
        cohorts[field] = {}
        for value in (False, True):
            selected = [row for row in rows if row[field] is value]
            if not selected:
                cohorts[field][str(value).lower()] = {
                    "count": 0, "unavailable_reason": "no available pairs in this diagnostic cohort",
                }
                continue
            errors = np.asarray([row["error"] for row in selected], dtype=np.float64)
            ntg_differences = [row["ntg_difference"] for row in selected if row["ntg_difference"] is not None]
            cohorts[field][str(value).lower()] = {
                "count": len(selected), "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors**2))), "bias": float(np.mean(errors)),
                "ntg_difference_count": len(ntg_differences),
                "mean_absolute_ntg_difference": float(np.mean(ntg_differences)) if ntg_differences else None,
            }
    return {
        "schema": "streamed-validation-diagnostic-report-v1",
        "validation_manifest_sha256": manifest["manifest_sha256"],
        "diagnostic_manifest_sha256": verified["diagnostic_manifest_sha256"],
        "pairs": rows, "cohorts": cohorts,
        "limitations": [
            *verified["limitations"],
            "Predictions are supplied by the caller; this report does not verify checkpoint provenance.",
            "Correlated validation pairs are not independent geological realizations.",
        ],
    }
