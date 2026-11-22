"""Freeze and execute bounded, primary-validation tokenizer searches."""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import zarr

from scripts.streamed_overlap_audit import content_hash
from scripts.streamed_overlap_evaluation import export_tokenizer_checkpoint
from scripts.streamed_overlap_integrity import _digest_array, _window_record, verify_inputs
from scripts.streamed_overlap_manifest import _window
from scripts.streamed_overlap_metrics import _eligible_queries
from scripts.streamed_overlap_targets import MAX_READ_BYTES, Window, _read_bounded, _supported
from scripts.streamed_overlap_yield import SEISMIC_KEY
from src.tokenizer.core.io_zarr import compute_padding
from src.tokenizer.core.jobs import compute_total_windows_from_padded_shape


PATCH_SHAPE = (32, 32, 64)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__), root / "scripts/tokenize.py", root / "src/tokenizer/core/search_engine.py",
        root / "src/tokenizer/core/io_zarr.py", root / "src/tokenizer/core/preprocess.py",
        root / "src/tokenizer/core/model_adapter.py", root / "src/tokenizer/core/similarity.py",
        root / "src/tokenizer/core/jobs.py", root / "src/tokenizer/config/defaults.py", root / "src/model.py",
    )
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def freeze_search_manifest(
    manifest: dict, groups: dict, plans: list[dict], *, stride: int, batch_size: int,
    max_region_bytes: int, max_windows: int, max_wall_seconds: float,
    device: str, mps_memory_fraction: float,
) -> dict:
    """Bind explicit source regions and resource limits before model results are examined."""
    queries = {query["id"]: query for query in _eligible_queries(manifest)}
    if (not isinstance(plans, list) or not plans
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                   for v in (stride, batch_size, max_region_bytes, max_windows))
            or max_region_bytes > MAX_READ_BYTES
            or isinstance(max_wall_seconds, bool) or not isinstance(max_wall_seconds, (int, float))
            or not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0
            or device not in {"cpu", "mps"}
            or isinstance(mps_memory_fraction, bool) or not isinstance(mps_memory_fraction, (int, float))
            or not math.isfinite(mps_memory_fraction) or not 0 < mps_memory_fraction <= 0.5):
        raise ValueError("Explicit bounded search plans and valid resource limits are required.")
    records = []
    seen = set()
    fingerprints = manifest["input_integrity"]["fingerprints"]["windows"]
    for plan in plans:
        if not isinstance(plan, dict) or set(plan) != {"query_id", "region_origin", "region_shape"}:
            raise ValueError("Each bounded plan requires a query ID and XYZ region origin and shape.")
        query_id = plan["query_id"]
        if not isinstance(query_id, str) or query_id not in queries or query_id in seen:
            raise ValueError("Bounded searches require unique eligible query IDs.")
        seen.add(query_id)
        query = _window(queries[query_id]["window"])
        if query.shape != PATCH_SHAPE:
            raise ValueError("Bounded tokenizer searches require the canonical patch shape.")
        try:
            region = Window(query.source_id, tuple(plan["region_origin"]), tuple(plan["region_shape"]))
        except (TypeError, KeyError) as error:
            raise ValueError("Region coordinates must be explicit XYZ integer sequences.") from error
        if (any(start > q_start or end < q_end for start, end, q_start, q_end in
                zip(region.origin, region.end, query.origin, query.end))
                or query.source_id not in groups or SEISMIC_KEY not in groups[query.source_id]):
            raise ValueError("The bounded region must contain its primary-validation query.")
        array = groups[query.source_id][SEISMIC_KEY]
        if (not _supported(array, region, labels=False) or tuple(region.shape) == tuple(array.shape)
                or math.prod(region.shape) * np.dtype(array.dtype).itemsize > max_region_bytes
                or math.prod(region.shape) * np.dtype(np.float32).itemsize > max_region_bytes):
            raise ValueError("Search region is unsupported, whole-volume, or exceeds its byte limit.")
        padded = tuple(size + low + high for size, (low, high) in
                       zip(region.shape, compute_padding(region.shape, patch_size=PATCH_SHAPE, base_pad=16)))
        if any((size - patch) % stride for size, patch in zip(padded, PATCH_SHAPE)):
            raise ValueError("Padded bounded region must align with the configured tokenizer stride.")
        if compute_total_windows_from_padded_shape(padded, patch_size=PATCH_SHAPE, stride=stride) > max_windows:
            raise ValueError("Bounded region exceeds the explicit search-window limit.")
        if any(not any(0 <= start - low and start - low + patch <= size
                       for start in range(0, padded_size - patch + 1, stride))
               for size, patch, padded_size, (low, _) in zip(
                   region.shape, PATCH_SHAPE, padded,
                   compute_padding(region.shape, patch_size=PATCH_SHAPE, base_pad=16))):
            raise ValueError("Bounded region has no fully contained tokenizer search window.")
        query_record = _window_record(query)
        bound = fingerprints.get(content_hash(query_record))
        if not isinstance(bound, dict) or bound.get("window") != query_record:
            raise ValueError("Query is missing its frozen source fingerprint.")
        if _digest_array(_read_bounded(array, query.slices)) != bound["arrays"][SEISMIC_KEY]:
            raise ValueError("Query seismic differs from the frozen validation input.")
        values = _read_bounded(array, region.slices)
        records.append({
            "query_id": query_id, "query": query_record, "region": _window_record(region),
            "region_sha256": _digest_array(values),
        })
    result = {
        "schema": "streamed-bounded-tokenizer-search-v1",
        "validation_manifest_sha256": manifest["manifest_sha256"],
        "patch_shape": list(PATCH_SHAPE), "stride": stride, "batch_size": batch_size,
        "max_region_bytes": max_region_bytes, "max_windows": max_windows,
        "max_wall_seconds": float(max_wall_seconds),
        "device": device, "mps_memory_fraction": float(mps_memory_fraction),
        "implementation_sha256": _implementation_hashes(), "searches": records,
    }
    result["search_manifest_sha256"] = content_hash(result)
    return result


def _search_results(scores_path: Path, output_path: Path, region: Window, query: Window,
                    stride: int) -> dict:
    padding = compute_padding(region.shape, patch_size=PATCH_SHAPE, base_pad=16)
    padded = tuple(size + low + high for size, (low, high) in zip(region.shape, padding))
    expected_count = compute_total_windows_from_padded_shape(padded, patch_size=PATCH_SHAPE, stride=stride)
    seen = set()
    contained_windows = []
    with scores_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            origin, score, contained = row.get("origin"), row.get("score"), row.get("fully_contained")
            if (not isinstance(origin, list) or len(origin) != 3
                    or any(isinstance(v, bool) or not isinstance(v, int) for v in origin)
                    or isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score) or not isinstance(contained, bool)):
                raise ValueError("Invalid tokenizer search-window score record.")
            start = tuple(value + pad[0] for value, pad in zip(origin, padding))
            if (start in seen or any(value < 0 or value + patch > size or value % stride
                                     for value, patch, size in zip(start, PATCH_SHAPE, padded))):
                raise ValueError("Tokenizer search-window coordinates are duplicated or invalid.")
            seen.add(start)
            expected_contained = all(0 <= value and value + patch <= size
                                     for value, patch, size in zip(origin, PATCH_SHAPE, region.shape))
            if contained != expected_contained:
                raise ValueError("Tokenizer window containment is inconsistent.")
            if contained:
                contained_windows.append((-float(score), tuple(origin)))
    if len(seen) != expected_count or not contained_windows:
        raise ValueError("Tokenizer search did not produce all windows or a fully contained window.")
    contained_windows.sort()
    best = contained_windows[0]
    output = np.asarray(zarr.open_group(str(output_path), mode="r")["data"])
    if tuple(output.shape) != region.shape or not np.isfinite(output).all():
        raise ValueError("Tokenizer voxel similarity map is invalid.")
    peak = tuple(int(value) for value in np.unravel_index(int(np.argmax(output)), output.shape))
    query_center = tuple(start - base + size // 2 for start, base, size in
                         zip(query.origin, region.origin, PATCH_SHAPE))
    query_origin = tuple(start - base for start, base in zip(query.origin, region.origin))
    query_rank = next((index for index, (_, origin) in enumerate(contained_windows, 1)
                       if origin == query_origin), None)
    best_center = tuple(value + size // 2 for value, size in zip(best[1], PATCH_SHAPE))
    return {
        "window_count": len(seen), "contained_window_count": len(contained_windows),
        "window_scores_sha256": _file_digest(scores_path), "hann_similarity_sha256": _digest_array(output),
        "query_window_rank": query_rank,
        "query_window_rank_unavailable_reason": None if query_rank is not None
        else "query origin is not on the configured tokenizer search grid",
        "best_window_origin": list(best[1]), "best_window_score": -best[0],
        "best_window_center_error_voxels": math.dist(best_center, query_center),
        "hann_voxel_peak": list(peak), "hann_voxel_peak_score": float(output[peak]),
        "hann_voxel_peak_error_voxels": math.dist(peak, query_center),
    }


def verify_search_manifest(manifest: dict, groups: dict, search_manifest: dict) -> None:
    """Recheck frozen search settings, implementation and source-region fingerprints."""
    if (not isinstance(search_manifest, dict)
            or search_manifest.get("schema") != "streamed-bounded-tokenizer-search-v1"
            or search_manifest.get("search_manifest_sha256") != content_hash({
                key: value for key, value in search_manifest.items() if key != "search_manifest_sha256"
            })
            or search_manifest.get("validation_manifest_sha256") != manifest.get("manifest_sha256")
            or search_manifest.get("implementation_sha256") != _implementation_hashes()):
        raise ValueError("Bounded search manifest or implementation differs from its frozen identity.")
    plans = [{"query_id": record["query_id"], "region_origin": record["region"]["origin"],
              "region_shape": record["region"]["shape"]} for record in search_manifest["searches"]]
    keys = ("stride", "batch_size", "max_region_bytes", "max_windows", "max_wall_seconds",
            "device", "mps_memory_fraction")
    current = freeze_search_manifest(manifest, groups, plans, **{key: search_manifest[key] for key in keys})
    if current != search_manifest:
        raise ValueError("Bounded source regions differ from the frozen search manifest.")


def run_bounded_searches(
    checkpoint_path: Path, journal, manifest_path: Path, source_root: Path, manifest: dict, groups: dict,
    search_manifest: dict, output: Path,
) -> dict:
    """Run the exact tokenizer CLI on verified frozen regions of an evaluated checkpoint."""
    started = time.monotonic()
    root = source_root.resolve(strict=True)
    forbidden = Path("/Volumes/CrucialX9/fake_data_validation_v2")
    destination = output.resolve()
    if (destination.is_relative_to(root) or forbidden.is_relative_to(destination)
            or destination.is_relative_to(forbidden) or "fake_data_validation_v2" in destination.parts
            or "fake_data_validation_v2" in root.parts or forbidden.is_relative_to(root)):
        raise ValueError("Bounded searches must exclude validation-v2 and write outside the source root.")
    verify_search_manifest(manifest, groups, search_manifest)
    if search_manifest["device"] == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("MPS CPU fallback is not permitted for bounded searches.")
    verification = verify_inputs(manifest, groups)
    deadline = started + search_manifest["max_wall_seconds"]
    if time.monotonic() > deadline:
        raise TimeoutError("Bounded search wall-time limit exceeded; no partial report is valid.")
    output.mkdir()
    export = export_tokenizer_checkpoint(checkpoint_path, journal, manifest_path, output / "tokenizer.pt")
    if export["checkpoint"]["validation_manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Tokenizer checkpoint and bounded searches use different validation manifests.")
    results = []
    for index, record in enumerate(search_manifest["searches"]):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Bounded search wall-time limit exceeded; no partial report is valid.")
        region, query = _window(record["region"]), _window(record["query"])
        values = _read_bounded(groups[region.source_id][SEISMIC_KEY], region.slices)
        if _digest_array(values) != record["region_sha256"]:
            raise ValueError("Bounded search region changed before CLI execution.")
        folder = output / f"search-{index:04d}"
        folder.mkdir()
        bounded = folder / "bounded.zarr"
        zarr.open_group(str(bounded), mode="w").create_array("data", data=values)
        scores, result_map = folder / "scores.jsonl", folder / "similarity.zarr"
        center = [start - base + size // 2 for start, base, size in
                  zip(query.origin, region.origin, PATCH_SHAPE)]
        command = [
            sys.executable, "-u", str(Path(__file__).with_name("tokenize.py")), "search-volume",
            "--source", str(bounded), "--search", str(bounded), "--output", str(result_map),
            "--temp-zarr", str(folder / "temp.zarr"), "--model-path", str(output / "tokenizer.pt"),
            "--device", search_manifest["device"], "--latent-mode", "vae", "--embedding-mode", "mu",
            "--similarity-mode", "cosine", "--patch-size", *(str(size) for size in PATCH_SHAPE),
            "--stride", str(search_manifest["stride"]), "--batch-size", str(search_manifest["batch_size"]),
            "--x", str(center[0]), "--y", str(center[1]), "--z", str(center[2]),
            "--window-scores-jsonl", str(scores),
        ]
        env = os.environ.copy()
        if search_manifest["device"] == "mps":
            env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(search_manifest["mps_memory_fraction"])
            env["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = str(0.8 * search_manifest["mps_memory_fraction"])
            env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        log_path = folder / "tokenizer.log"
        with log_path.open("x", encoding="utf-8") as log:
            try:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True,
                               timeout=remaining)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f"Tokenizer search failed; see {log_path}") from error
            except subprocess.TimeoutExpired as error:
                raise TimeoutError(f"Tokenizer search timed out; see {log_path}") from error
        results.append({
            "query_id": record["query_id"], "region_sha256": record["region_sha256"],
            **_search_results(scores, result_map, region, query, search_manifest["stride"]),
        })
    if time.monotonic() > deadline:
        raise TimeoutError("Bounded search wall-time limit exceeded; no partial report is valid.")
    return {
        "schema": "streamed-bounded-tokenizer-search-report-v1",
        "search_manifest_sha256": search_manifest["search_manifest_sha256"],
        "checkpoint": export, "input_verification": verification, "searches": results,
        "elapsed_seconds": time.monotonic() - started,
        "limitation": "Bounded tokenizer search is not pair-head retrieval or an adoption decision.",
    }
