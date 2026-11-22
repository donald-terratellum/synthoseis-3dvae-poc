"""Read-only, bounded source inspection; semantic targets remain unapproved."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import time
from typing import Literal

import numpy as np
import zarr

from scripts.sample_patches import (
    DEFAULT_SEISMIC_KEY,
    compute_lithology_fractions,
    has_temp_folder_sibling,
)
from scripts.streamed_fluid_labels import (
    FLUID_KEYS,
    VERIFIED_SEISMIC_FILES,
    inspect_fluid_fill,
    lithology_masks,
)


HOLDOUT_NAMES = frozenset(("validation", "test", "test_synthoseis"))
LABEL_KEYS = (
    "geologic_age_faulted",
    "faulted_lithology",
    "gas_closures",
    "oil_closures",
    "brine_closures",
    "faults/faulted_channel_labels",
)
CONTEXT_KEYS = ("fault_segments_id", "onlap_segments")
PATCH_SHAPE = (32, 32, 64)
PENDING = "LOCAL_TARGET_VALIDATION_PENDING"
Split = Literal["train", "validation"]


def canonical_overlap(
    origin_a: tuple[int, int, int],
    origin_b: tuple[int, int, int],
    shape: tuple[int, int, int],
    same_source: bool,
) -> float:
    if any(len(values) != 3 for values in (origin_a, origin_b, shape)):
        raise ValueError("Origins and shape must have three XYZ coordinates.")
    if any(value <= 0 for value in shape) or any(
        value < 0 for value in (*origin_a, *origin_b)
    ):
        raise ValueError("Shape must be positive and origins nonnegative.")
    if not same_source:
        return 0.0
    return math.prod(
        max(0.0, 1.0 - abs(a - b) / size)
        for a, b, size in zip(origin_a, origin_b, shape)
    )


def content_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Source:
    path: Path
    split: Split

    @property
    def identifier(self) -> str:
        return f"{self.split}/{self.path.parent.name}"


def discover_sources(root: Path) -> list[Source]:
    root = root.resolve(strict=True)
    if (not root.is_dir() or any(part in HOLDOUT_NAMES for part in root.parts)
            or root.name.endswith(".zarr")):
        raise ValueError("source_root must be the parent training-source directory.")
    sources: list[Source] = []
    seen: set[Path] = set()
    forbidden = [(root / name).resolve() for name in HOLDOUT_NAMES]

    def fail_walk(error: OSError) -> None:
        raise error

    for split in ("train", "validation"):
        base = root if split == "train" else root / "validation"
        if base.is_symlink():
            raise ValueError(f"Symlinked split root requires review: {base}")
        if not base.is_dir():
            raise FileNotFoundError(f"Missing {split} source root: {base}")
        base = base.resolve(strict=True)
        if not base.is_relative_to(root):
            raise ValueError(f"Split root escapes source_root: {base}")
        for current, directories, _ in os.walk(base, followlinks=False, onerror=fail_walk):
            for name in list(directories):
                path = Path(current) / name
                if name.startswith(".") or name in HOLDOUT_NAMES:
                    directories.remove(name)
                    continue
                if path.is_symlink():
                    raise ValueError(f"Symlinked source directory requires review: {path}")
                if name.endswith(".zarr"):
                    directories.remove(name)
                    if name != "model_data.zarr":
                        continue
                    resolved = path.resolve(strict=True)
                    if not resolved.is_relative_to(base):
                        raise ValueError(f"Source escapes its split root: {path}")
                    if split == "train" and any(
                        resolved.is_relative_to(holdout) for holdout in forbidden
                    ):
                        raise ValueError(f"Training source aliases a holdout: {path}")
                    if has_temp_folder_sibling(path):
                        print(f"EXCLUDED incomplete source with temp sibling: {path}", flush=True)
                        continue
                    if resolved in seen:
                        raise ValueError(f"Duplicate/overlapping source split: {path}")
                    seen.add(resolved)
                    sources.append(Source(resolved, split))
    for split in ("train", "validation"):
        if not any(source.split == split for source in sources):
            raise ValueError(f"No sources discovered for {split}.")
    return sorted(sources, key=lambda source: source.identifier)


def parameter_metadata(directory: Path) -> dict:
    paths = sorted(directory.glob("model_parameters_*.txt"))
    if len(paths) != 1:
        raise ValueError(f"Expected one model_parameters_*.txt in {directory}; got {len(paths)}")
    path = paths[0]
    text = path.read_text()
    entries = re.findall(r"^bandpass_bandlimits:\s*(.+)$", text, flags=re.MULTILINE)
    if len(entries) != 1:
        raise ValueError(f"Expected one bandpass_bandlimits entry in {path}")
    parts = entries[0].split(",")
    if len(parts) != 2:
        raise ValueError(f"Expected two bandpass limits in {path}")
    low, high = map(float, parts)
    stamp = path.stem.removeprefix("model_parameters_")
    config_path = directory / f"model_config_{stamp}.json"
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Model config must be an object: {config_path}")
    spacing = float(config["digi"])
    order = config["bandwidth_ord"]
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError(f"Invalid bandwidth_ord in {config_path}")
    low_range = config["bandwidth_low"]
    high_range = config["bandwidth_high"]
    source_snr_range = config["signal_to_noise_ratio_db"]
    if not all(isinstance(values, list) and len(values) == 2 for values in (low_range, high_range)):
        raise ValueError(f"Invalid bandwidth range in {config_path}")
    if not isinstance(source_snr_range, list) or len(source_snr_range) != 3:
        raise ValueError(f"Invalid signal_to_noise_ratio_db in {config_path}")
    low_range = [float(value) for value in low_range]
    high_range = [float(value) for value in high_range]
    source_snr_range = [float(value) for value in source_snr_range]
    sn_entries = re.findall(r"^sn_db:\s*(.+)$", text, flags=re.MULTILINE)
    if len(sn_entries) != 1:
        raise ValueError(f"Expected one sn_db entry in {path}")
    source_snr = float(sn_entries[0])
    all_numeric_values = (
        low,
        high,
        spacing,
        source_snr,
        *low_range,
        *high_range,
        *source_snr_range,
    )
    if not all(math.isfinite(value) for value in all_numeric_values):
        raise ValueError(f"Nonfinite band/sample spacing in {path}")
    if spacing <= 0 or not 0 < low < high < 500.0 / spacing:
        raise ValueError(f"Band limits/sample spacing violate Nyquist in {path}")
    nyquist = 500.0 / spacing
    if (
        not 0 < low_range[0] <= low_range[1] < nyquist
        or not 0 < high_range[0] <= high_range[1] < nyquist
    ):
        raise ValueError(f"Invalid bandwidth range in {config_path}")
    if not source_snr_range[0] <= source_snr_range[1] <= source_snr_range[2]:
        raise ValueError(f"Invalid signal_to_noise_ratio_db in {config_path}")
    rounding_tolerance = 0.0051
    if not (
        low_range[0] - rounding_tolerance <= low <= low_range[1] + rounding_tolerance
        and high_range[0] - rounding_tolerance <= high <= high_range[1] + rounding_tolerance
    ):
        raise ValueError(f"Recorded band limits fall outside configured ranges in {path}")
    sha = re.findall(r"^SHA:\s*(\S+)", text, flags=re.MULTILINE)
    if len(sha) != 1:
        raise ValueError(f"Expected one generator SHA in {path}")
    return {
        "parameters_path": str(path),
        "parameters_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "parameters_mtime_ns": path.stat().st_mtime_ns,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "config_mtime_ns": config_path.stat().st_mtime_ns,
        "generator_sha": sha[0],
        "generator_provenance_status": (
            "Recorded commit SHA only; generation-time dirty-tree diff and command are not stored."
        ),
        "generator_seeds": config.get("seeds"),
        "bandlimits_hz": [low, high],
        "sample_spacing_ms_from_digi": spacing,
        "nyquist_hz_if_ms": 500.0 / spacing,
        "filter_order": order,
        "configured_bandwidth_low_range_hz": low_range,
        "configured_bandwidth_high_range_hz": high_range,
        "source_noise_sn_db": source_snr,
        "configured_source_noise_snr_db": source_snr_range,
        "source_noise_snr_within_configured_range": (
            source_snr_range[0] <= source_snr <= source_snr_range[2]
        ),
        "configured_cube_shape": config.get("cube_shape"),
        "pad_samples": config.get("pad_samples"),
        "infill_factor": config.get("infill_factor"),
        "include_channels": config.get("include_channels"),
        "channel_ntg_threshold": config.get("channel_ntg_threshold"),
        "interpretation_status": (
            "Actual cutoffs, JSON ranges, order and sample spacing are recorded; "
            "per-source generation worktree/command provenance is not attested."
        ),
    }


def array_metadata(array: zarr.Array) -> dict:
    return {
        "shape": list(array.shape),
        "chunks": list(array.chunks),
        "dtype": str(array.dtype),
        "chunk_uncompressed_bytes": math.prod(array.chunks) * array.dtype.itemsize,
    }


def inventory_arrays(group: zarr.Group, prefix: str = "") -> dict[str, dict]:
    inventory = {}
    for name in sorted(group.keys()):
        key = f"{prefix}{name}"
        value = group[name]
        if isinstance(value, zarr.Array):
            inventory[key] = array_metadata(value)
        elif isinstance(value, zarr.Group):
            inventory.update(inventory_arrays(value, f"{key}/"))
        else:
            raise ValueError(f"Unsupported Zarr member at {key}")
    return inventory


def require_array(group: zarr.Group, key: str) -> zarr.Array:
    value = group[key]
    if not isinstance(value, zarr.Array):
        raise ValueError(f"Expected an array, not a group, at {key}")
    return value


def inspect_source(source: Source, seismic_key: str) -> dict:
    group = zarr.open_group(str(source.path), mode="r")
    all_arrays = inventory_arrays(group)
    arrays = {}
    issues = []
    for key in (seismic_key, *LABEL_KEYS, *CONTEXT_KEYS):
        if key not in group:
            issues.append(f"Missing array: {key}")
            continue
        array = require_array(group, key)
        if len(array.shape) != 3:
            raise ValueError(f"Expected 3D source array: {source.path}/{key}")
        arrays[key] = {**array_metadata(array), "attributes": dict(array.attrs)}
    if seismic_key not in arrays:
        raise ValueError(f"Missing seismic array {seismic_key}: {source.path}")
    try:
        parameters = parameter_metadata(source.path.parent)
    except (ValueError, KeyError, FileNotFoundError) as error:
        parameters = None
        issues.append(f"Parameter metadata invalid: {error}")
    warnings = []
    if parameters and not parameters["source_noise_snr_within_configured_range"]:
        warnings.append(
            "Recorded source sn_db falls outside the JSON signal_to_noise_ratio_db range."
        )
    return {
        "source_id": source.identifier,
        "path": str(source.path),
        "split": source.split,
        "store_directory_mtime_ns": source.path.stat().st_mtime_ns,
        "source_directory_mtime_ns": source.path.parent.stat().st_mtime_ns,
        "all_arrays": all_arrays,
        "arrays": arrays,
        "parameters": parameters,
        "semantic_arrays_complete": all(key in arrays for key in LABEL_KEYS),
        "warnings": warnings,
        "issues": issues,
        "axes_status": (
            "XYZ per the generator convention; saved-source generation worktrees "
            "are not independently attested."
        ),
    }


@dataclass
class ReadBudget:
    max_region_voxels: int = 65_536
    max_decoded_bytes_per_read: int = 256 * 1024**2
    max_total_requested_voxels: int = 20_000_000
    requested_voxels: int = 0
    requested_bytes: int = 0
    estimated_decoded_bytes: int = 0
    largest_estimated_decode: int = 0
    reads: int = 0

    def read(
        self, array: zarr.Array, origin: tuple[int, int, int], shape: tuple[int, int, int]
    ) -> np.ndarray:
        if len(origin) != 3 or len(shape) != 3 or len(array.shape) != 3:
            raise ValueError("Bounded reads require three XYZ coordinates.")
        if any(start < 0 or size < 1 or start + size > limit
               for start, size, limit in zip(origin, shape, array.shape)):
            raise ValueError(f"Read exceeds array bounds: origin={origin}, shape={shape}")
        count = math.prod(shape)
        if count > self.max_region_voxels:
            raise ValueError(f"Read exceeds bounded region budget: {count} voxels")
        if self.requested_voxels + count > self.max_total_requested_voxels:
            raise ValueError("Audit exceeds its total requested-voxel budget.")
        chunks = math.prod(
            (start + size - 1) // chunk - start // chunk + 1
            for start, size, chunk in zip(origin, shape, array.chunks)
        )
        decoded = chunks * math.prod(array.chunks) * array.dtype.itemsize
        if decoded > self.max_decoded_bytes_per_read:
            raise ValueError(f"Read exceeds decoded-chunk budget: {decoded} bytes")
        region = tuple(slice(start, start + size) for start, size in zip(origin, shape))
        values = np.asarray(array[region])
        if tuple(values.shape) != shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid shape/nonfinite source region: {array.path}, {region}")
        self.requested_voxels += count
        self.requested_bytes += values.nbytes
        self.estimated_decoded_bytes += decoded
        self.largest_estimated_decode = max(self.largest_estimated_decode, decoded)
        self.reads += 1
        return values


def bounded_crop(
    group: zarr.Group,
    origin: tuple[int, int, int],
    budget: ReadBudget,
    seismic_key: str,
    label_z_offset: int,
) -> dict[str, np.ndarray]:
    label_origin = (origin[0], origin[1], origin[2] + label_z_offset)
    if label_origin[2] < 0:
        raise ValueError("Label offset places the crop outside array bounds.")
    result = {seismic_key: budget.read(require_array(group, seismic_key), origin, PATCH_SHAPE)}
    for key in LABEL_KEYS:
        result[key] = budget.read(require_array(group, key), label_origin, PATCH_SHAPE)
    for key in CONTEXT_KEYS:
        if key in group:
            result[key] = budget.read(require_array(group, key), label_origin, PATCH_SHAPE)
    return result


def crop_statistics(crop: dict[str, np.ndarray], seismic_key: str) -> dict:
    age = crop["geologic_age_faulted"]
    lithology = crop["faulted_lithology"]
    raw_sand_fraction, raw_shale_fraction, raw_water_fraction = compute_lithology_fractions(lithology)
    facies = lithology_masks(lithology, crop[LABEL_KEYS[-1]])
    masks = np.stack([crop[key] > 0 for key in FLUID_KEYS])
    closure_union = masks.any(axis=0)
    rock_count = int(np.count_nonzero(facies["rock"]))
    sand_count = int(np.count_nonzero(facies["sand"]))
    fluid_values = {key: np.unique(crop[key]).tolist() for key in FLUID_KEYS}
    return {
        "seismic_std": float(crop[seismic_key].std()),
        "age_min": float(age.min()),
        "age_max": float(age.max()),
        "age_noninteger_fraction": float(np.mean(np.abs(age - np.rint(age)) > 1e-4)),
        "age_decreasing_depth_fraction": float(np.mean(np.diff(age, axis=2) < -1e-4)),
        "sand_fraction_over_rock_at_diagnostic_threshold": sand_count / rock_count if rock_count else 0.0,
        "shale_fraction_over_rock_at_diagnostic_threshold": (
            int(np.count_nonzero(facies["shale"])) / rock_count if rock_count else 0.0
        ),
        "water_fraction": float(np.mean(facies["water"])),
        "raw_lithology_sand_fraction_over_rock": raw_sand_fraction,
        "raw_lithology_shale_fraction_over_rock": raw_shale_fraction,
        "raw_lithology_water_fraction": raw_water_fraction,
        "sand_mask_policy": "faulted_lithology >= 0.5 OR faults/faulted_channel_labels > 0.5",
        "channel_added_sand_voxels": int(np.count_nonzero(facies["channel"] & ~facies["raw_sand"])),
        "channel_over_raw_water_voxels": int(np.count_nonzero(facies["channel"] & (lithology < 0))),
        "fluid_interpretation_status": "Closure indicators only; final seismic fluid requires generator-version verification.",
        "fluid_mask_values": fluid_values,
        "nonbinary_fluid_mask_voxels": {
            key: int(np.count_nonzero(~np.isin(crop[key], (0, 1)))) for key in FLUID_KEYS
        },
        "fluid_positive_voxels": {
            key: int(np.count_nonzero(crop[key])) for key in FLUID_KEYS
        },
        "fluid_conflict_voxels": int(np.count_nonzero(masks.sum(axis=0) > 1)),
        "closure_positive_outside_corrected_sand_voxels": int(
            np.count_nonzero(closure_union & ~facies["sand"])
        ),
        "sand_without_any_closure_mask_voxels": int(
            np.count_nonzero(facies["sand"] & ~closure_union)
        ),
        "channel_hydrocarbon_closure_overlap_voxels": int(
            np.count_nonzero(facies["channel"] & (masks[0] | masks[1]))
        ),
        "rock_voxels": rock_count,
        "channel_positive_voxels": int(np.count_nonzero(facies["channel"])),
        "context_positive_voxels": {
            key: int(np.count_nonzero(crop[key])) if key in crop else None
            for key in CONTEXT_KEYS
        },
        "semantic_target": None,
        "semantic_target_status": PENDING,
    }


def trace_profiles(crop: dict[str, np.ndarray], seismic_key: str) -> list[dict]:
    profiles = []
    for x in (4, 16, 27):
        for y in (4, 16, 27):
            profiles.append({
                "local_xy": [x, y],
                "values": {
                    key: values[x, y, :].tolist()
                    for key, values in crop.items()
                },
            })
    return profiles


def sample_source(
    source: Source, budget: ReadBudget, seismic_key: str, offset: int, seed: int,
    *, generator_sha: str | None = None,
) -> dict:
    start = time.perf_counter()
    group = zarr.open_group(str(source.path), mode="r")
    shape = require_array(group, seismic_key).shape
    label_shapes = [require_array(group, key).shape for key in LABEL_KEYS]
    max_z = min([shape[2] - 128, *(value[2] - offset - 128 for value in label_shapes)])
    if min(shape[:2]) < 32 or max_z < max(0, -offset):
        raise ValueError(f"Source too small for audit pair/context: {source.path}")
    rng = np.random.default_rng(seed)
    probe_records = []
    selected = None
    rock_control = None
    water_control = None
    observed_max_integer_age = 0

    def centered_origin(x: int, y: int, point: np.ndarray) -> tuple[int, int, int]:
        return (
            min(max(0, x + int(point[0]) - 16), shape[0] - 32),
            min(max(0, y + int(point[1]) - 16), shape[1] - 32),
            min(max(max(0, -offset), int(point[2]) - offset - 32), max_z),
        )

    # Thin full-depth columns find examples without materializing a source cube.
    for fraction in (0.25, 0.5, 0.75):
        x = min(shape[0] - 4, int(shape[0] * fraction))
        y = min(shape[1] - 4, int(shape[1] * fraction))
        probe_masks = [
            budget.read(require_array(group, key), (x, y, 0),
                        (4, 4, require_array(group, key).shape[2]))
            for key in FLUID_KEYS
        ]
        points = np.argwhere(np.stack(probe_masks).any(axis=0))
        lithology = budget.read(require_array(group, "faulted_lithology"), (x, y, 0),
                                (4, 4, require_array(group, "faulted_lithology").shape[2]))
        channel_labels = budget.read(require_array(group, LABEL_KEYS[-1]), (x, y, 0),
                                     (4, 4, require_array(group, LABEL_KEYS[-1]).shape[2]))
        facies = lithology_masks(lithology, channel_labels)
        age_probe = budget.read(require_array(group, "geologic_age_faulted"), (x, y, 0),
                                (4, 4, require_array(group, "geologic_age_faulted").shape[2]))
        observed_max_integer_age = max(
            observed_max_integer_age, int((age_probe + 0.00001).max())
        )
        rock_points = np.argwhere(facies["rock"])
        water_points = np.argwhere(facies["water"])
        probe_records.append({
            "xy": [x, y],
            "positive_voxels": {key: int(np.count_nonzero(mask))
                                for key, mask in zip(FLUID_KEYS, probe_masks)},
            "rock_voxels": len(rock_points),
            "water_voxels": len(water_points),
        })
        if selected is None and len(points):
            selected = centered_origin(x, y, points[len(points) // 2])
        if rock_control is None and len(rock_points):
            rock_control = centered_origin(x, y, rock_points[len(rock_points) // 2])
        if water_control is None and len(water_points):
            water_control = centered_origin(x, y, water_points[len(water_points) // 2])
    if selected is None and rock_control is not None:
        selected = rock_control
        selection = "Bounded rock control; no fluid found in probes (not proof of absence)."
    elif selected is not None:
        selection = "Centered near fluid seen in bounded probes, clipped to valid audit support."
    if selected is None:
        selected = (
            int(rng.integers(0, shape[0] - 31)),
            int(rng.integers(0, shape[1] - 31)),
            int(rng.integers(max(0, -offset), max_z + 1)),
        )
        selection = "Seeded crop; no fluid found in the bounded probes (not proof of absence)."
    origins = {
        "A": selected,
        "B_shift_z16": (selected[0], selected[1], selected[2] + 16),
        "B_disjoint_z64": (selected[0], selected[1], selected[2] + 64),
    }
    if water_control is not None:
        origins["C_water_control"] = water_control
    crops = {}
    for name, origin in origins.items():
        crop = bounded_crop(group, origin, budget, seismic_key, offset)
        fluid = inspect_fluid_fill(crop, generator_sha, observed_max_integer_age or None)
        crops[name] = {
            "origin_xyz": list(origin),
            "statistics": crop_statistics(crop, seismic_key),
            "trace_profiles": trace_profiles(crop, seismic_key),
            "fluid_diagnostics": fluid.summary(),
        }
    return {
        "source_id": source.identifier,
        "path": str(source.path),
        "split": source.split,
        "selection_policy": selection,
        "fluid_probes": probe_records,
        "label_z_offset_hypothesis": offset,
        "alignment_verified": False,
        "generator_sha": generator_sha,
        "historical_fluid_rule_recognized": generator_sha in VERIFIED_SEISMIC_FILES,
        "observed_max_integer_age_lower_bound": observed_max_integer_age,
        "views": crops,
        "physical_overlap_A_to_shift": canonical_overlap(
            selected, origins["B_shift_z16"], PATCH_SHAPE, True
        ),
        "physical_overlap_A_to_disjoint": canonical_overlap(
            selected, origins["B_disjoint_z64"], PATCH_SHAPE, True
        ),
        "elapsed_seconds": time.perf_counter() - start,
    }


def write_json(path: Path, payload: object) -> None:
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_visual_report(path: Path, summary: dict, seismic_key: str) -> None:
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Bounded source audit</title>",
        "<style>body{font:15px sans-serif;max-width:1200px;margin:2em auto}"
        "svg{border:1px solid #ccc;margin:3px}summary{cursor:pointer}"
        ".profiles{display:grid;grid-template-columns:repeat(3,1fr)}</style>",
        f"<h1>{PENDING}</h1>",
        "<p>No training or semantic target generation. XYZ and label offset are hypotheses. "
        "These fluid-seeking examples are not a representative prevalence sample.</p>",
        "<p>Each panel is one trace (depth increases downward). Blue: geologic age "
        "(shared scale within source); gray: lithology (clipped for display only); "
        "orange: channel labels > 0.5 (sand override); "
        "red/green/cyan: gas/oil/brine closure masks, NOT complete lithology or final fluid labels. "
        "Seismic is black, "
        "independently scaled for display only. All nine XY traces are shown; "
        "statistics use the full 32x32x64 crop. JSON retains exact trace values.</p>",
    ]
    for example in summary["examples"]:
        parts.append(f"<h2>{html.escape(example['source_id'])}</h2>")
        parts.append(f"<p>{html.escape(example['selection_policy'])}; "
                     f"label offset hypothesis: {example['label_z_offset_hypothesis']}</p>")
        age_min = min(view["statistics"]["age_min"] for view in example["views"].values())
        age_max = max(view["statistics"]["age_max"] for view in example["views"].values())
        age_span = max(age_max - age_min, 1e-6)
        for name, view in example["views"].items():
            parts.append(
                f"<details open><summary>{html.escape(name)} origin "
                f"{view['origin_xyz']}; corrected sand outside closure masks: "
                f"{view['statistics']['sand_without_any_closure_mask_voxels']} voxels; "
                f"channel-added sand: {view['statistics']['channel_added_sand_voxels']}; "
                f"provisionally fluid-consistent: "
                f"{view['fluid_diagnostics']['provisionally_consistent_sand_voxels']}"
                "</summary><div class='profiles'>"
            )
            for trace in view["trace_profiles"]:
                values = trace["values"]
                seismic_scale = max(max(abs(value) for value in values[seismic_key]), 1e-6)
                tracks = [
                    ("black", [30 + 24 * value / seismic_scale for value in values[seismic_key]]),
                    ("blue", [65 + 70 * (value - age_min) / age_span
                              for value in values["geologic_age_faulted"]]),
                    ("gray", [150 + 20 * min(1, max(-1, value))
                              for value in values["faulted_lithology"]]),
                    ("orange", [178 + 15 * (value > 0.5) for value in values[LABEL_KEYS[-1]]]),
                ]
                for color, key, left in zip(("red", "green", "cyan"), FLUID_KEYS, (200, 245, 290)):
                    tracks.append((color, [left + 20 * (value > 0) for value in values[key]]))
                svg = [f"<svg viewBox='0 0 330 170'><text x='5' y='12'>XY {trace['local_xy']}</text>"]
                for color, coordinates in tracks:
                    points = " ".join(
                        f"{x:.2f},{20 + z * 2.2:.2f}" for z, x in enumerate(coordinates)
                    )
                    svg.append(f"<polyline points='{points}' fill='none' stroke='{color}'/>")
                parts.append("".join(svg) + "</svg>")
            parts.append("</div></details>")
    with path.open("x") as stream:
        stream.write("\n".join(parts) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_audit(config: dict, output: Path) -> dict:
    started = time.perf_counter()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing audit output: {output}")
    allowed = {
        "source_root", "seismic_key", "sample_sources_per_split",
        "label_z_offset_hypothesis", "seed",
    }
    if set(config) != allowed:
        raise ValueError(f"Audit config must contain exactly {sorted(allowed)}")
    count = config["sample_sources_per_split"]
    seed = config["seed"]
    offset = config["label_z_offset_hypothesis"]
    for name, value in (("sample_sources_per_split", count), ("seed", seed),
                        ("label_z_offset_hypothesis", offset)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer.")
    if not 1 <= count <= 5 or seed < 0 or not -8 <= offset <= 8:
        raise ValueError("Audit requires 1-5 sources per split, nonnegative seed, offset in [-8,8].")
    if config["seismic_key"] != DEFAULT_SEISMIC_KEY:
        raise ValueError("This audit phase inspects the approved fullstack key only.")
    source_root = Path(config["source_root"]).resolve(strict=True)
    if output.resolve().is_relative_to(source_root):
        raise ValueError("Audit output must not be written into the read-only source tree.")
    sources = discover_sources(source_root)
    records = []
    for index, source in enumerate(sources):
        record = inspect_source(source, config["seismic_key"])
        records.append(record)
        print(
            f"INVENTORY {index + 1}/{len(sources)} {source.identifier}: "
            f"complete_semantic_arrays={record['semantic_arrays_complete']}, "
            f"issues={record['issues']}", flush=True,
        )
    inventory_seconds = time.perf_counter() - started
    budget = ReadBudget()
    examples = []
    for split in ("train", "validation"):
        candidates = [
            (source, record) for source, record in zip(sources, records)
            if source.split == split and record["semantic_arrays_complete"]
        ]
        if not candidates:
            raise ValueError(f"No complete semantic source available for bounded {split} inspection.")
        indices = np.linspace(0, len(candidates) - 1, min(count, len(candidates)), dtype=int)
        for index in indices:
            source, record = candidates[index]
            print(f"BOUNDED CROP INSPECTION {source.identifier}", flush=True)
            example_seed = np.random.SeedSequence([seed, int(index), split == "validation"])
            examples.append(sample_source(
                source, budget, config["seismic_key"], offset,
                int(example_seed.generate_state(1)[0]),
                generator_sha=record["parameters"]["generator_sha"] if record["parameters"] else None,
            ))
    manifest = {
        "schema_version": 4,
        "source_root": str(Path(config["source_root"]).resolve()),
        "training_exclusions": sorted(HOLDOUT_NAMES),
        "final_test_crops_read": 0,
        "sources": records,
        "status": PENDING,
    }
    summary = {
        "schema_version": 5,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": PENDING,
        "production_training_allowed": False,
        "target_contract_approved": False,
        "label_policy": {
            "sand": "faulted_lithology >= 0.5 OR faults/faulted_channel_labels > 0.5",
            "channel_overrides_raw_lithology": True,
            "channel_identity_preserved_under_fluid_fill": True,
            "closures_define_lithology": False,
            "absence_of_closure_invalidates_lithology": False,
            "closure_values_are_final_seismic_fluid": False,
            "fluid_policy": (
                "Use oil/gas closure fractions as a coarse operational proxy over "
                "selected support; other is the mathematical remainder, not a physical "
                "brine label. Channel identity is retained."
            ),
            "generator_consistency": (
                "The current neighboring generator inserts channel N/G into "
                "faulted_net_to_gross before closure/property construction, and its "
                "synthetic channel tests pass. Per-source dirty-worktree and command "
                "provenance are unavailable, so this does not establish how any saved "
                "source was generated or make closures physical-fluid ground truth."
            ),
            "provisional_fluid_policy": (
                "Do not infer physical brine from absence of an oil/gas closure; "
                "other = 1 - oil - gas is only the target formula's remainder."
            ),
            "historical_seismic_file_sha256": VERIFIED_SEISMIC_FILES,
            "owner_approved_closure_fraction_proxy": True,
        },
        "owner_decisions": {
            "physical_overlap": "Canonical 3D window intersection divided by one window's volume.",
            "directional_event_coverage": {
                "reference": "A",
                "detector": (
                    "Apply scipy.ndimage.uniform_filter1d(size=5, axis=2, "
                    "mode='nearest'), then blend 0.2 * original + 0.8 * filtered. "
                    "Detect strict Z-axis local maxima/minima by comparing each sample "
                    "with its immediately shallower and deeper sample; no prominence threshold."
                ),
                "event_indices": (
                    "Use center=volume[:,:,1:-1], compare with volume[:,:,:-2] and "
                    "volume[:,:,2:], then add one to returned k indices. Combine peak "
                    "and trough (i,j,k) arrays."
                ),
                "events": (
                    "Detect on full original A traces streamed from Zarr in XY chunks; "
                    "exclude source endpoints and count events in shared XYZ support."
                ),
                "numerator": "Events in A's shared XYZ support.",
                "denominator": "Events across full A depth on traces in the shared XY footprint.",
                "fluid_adjustment": (
                    "1 - 0.5 * (abs(oil_A-oil_B) + abs(gas_A-gas_B) "
                    "+ abs(other_A-other_B))"
                ),
                "fluid_fractions": (
                    "Fractions use all selected age-range voxels in each full window; "
                    "oil/gas-positive voxels are counted by their masks, with an "
                    "equal split if both masks are positive. other = 1 - oil - gas."
                ),
                "age_range": (
                    "Average geologic_age_faulted at shallowest and deepest Z indices "
                    "in the shared region; select that range in each full window."
                ),
                "age_label_support": (
                    "Use same-index native property/label crop support; ignore padded "
                    "geologic_age_faulted indices 1500:1510 and do not apply a global "
                    "correlation-derived Z shift."
                ),
                "depth_shape_policy": (
                    "Do not require equal source Z dimensions. Use fully supported "
                    "same-index windows; mark missing full support unavailable rather "
                    "than padding, clipping, or shifting arrays."
                ),
                "zero_target_pairs": [
                    "Same-source windows with no 3D intersection.",
                    "Windows from different source volumes.",
                ],
                "unavailable_cases": [
                    "Overlapping pair with zero events in A's denominator.",
                    "Unsupported native index dimensions, missing/non-finite required labels, or empty age support.",
                ],
                "computed_before_augmentation": True,
            },
            "noise_and_filtering": {
                "source_filter": (
                    "Use the actual bandpass_bandlimits from each source's "
                    "model_parameters text file, constrained by bandwidth_low/high "
                    "ranges and bandwidth_ord in its matched model_config JSON."
                ),
                "source_noise_snr": (
                    "Record sn_db as noise already present in the stored source and "
                    "compare it with the JSON signal_to_noise_ratio_db range; do not "
                    "use source sn_db as the independent augmentation SNR."
                ),
                "augmentation": (
                    "Keep independent 12-60 dB noise at probability 0.5; use the "
                    "source-specific Butterworth band/order and pole-decay context."
                ),
            },
            "source_cohort_policy": (
                "Use /Volumes/CrucialX9/fake_data train and validation as the primary "
                "cohort. The owner reports these datasets were recreated; SHA/timestamps "
                "are not unambiguous generation-history evidence. Keep "
                "/Volumes/CrucialX9/fake_data_validation_v2 separate until regeneration "
                "is complete and it receives its own audit."
            ),
            "cross_volume_age_equivalence": False,
            "primary_retrieval_selection": "Directional pair-head ranking; report mu-cosine separately.",
            "training_progress_monitor": (
                "At every 100 completed optimizer steps and each epoch boundary if "
                "not already logged, run no-gradient inference on fixed deterministic "
                "train/validation monitor batches; emit component losses to stdout "
                "and TensorBoard."
            ),
            "rotation_and_scaling": "Applied to every view; parameters remain independently sampled.",
        },
        "manifest_sha256": content_hash(manifest),
        "config_sha256": content_hash(config),
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "zarr": zarr.__version__,
            "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "timing": {
            "inventory_seconds": inventory_seconds,
            "total_before_report_write_seconds": time.perf_counter() - started,
            "includes_training": False,
        },
        "inventory": {
            split: {
                "sources": sum(source.split == split for source in sources),
                "complete_semantic_sources": sum(
                    record["split"] == split and record["semantic_arrays_complete"]
                    for record in records
                ),
                "parameter_metadata_issues": sum(
                    record["split"] == split and record["parameters"] is None
                    for record in records
                ),
                "source_noise_snr_range_warnings": sum(
                    record["split"] == split
                    and any("source sn_db falls outside" in warning for warning in record["warnings"])
                    for record in records
                ),
            } for split in ("train", "validation")
        },
        "read_budget": dict(vars(budget)),
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * (1 if sys.platform == "darwin" else 1024),
        "examples": examples,
        "limitations": [
            "Bounded, intentionally fluid-seeking examples are not prevalence estimates.",
            "Generator-derived native index support is recorded; filtered seismic events are not age-horizon identities.",
            "Current neighboring generator code and passing synthetic tests do not prove the generation worktree or command for each stored source.",
            "The approved detector is implemented only as a helper; the pair target, validity yield and evaluation manifest remain to be implemented and checked.",
            "Source sn_db metadata falls outside its JSON SNR range in some stores; the actual text band limits remain inside configured bandwidth ranges.",
            "No model forward/backward pass, production training or final-test prediction was run.",
            "Decoded bytes are conservative chunk estimates, not measured physical I/O.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "source_manifest.json", manifest)
    write_json(output / "audit_report.json", summary)
    write_visual_report(output / "audit_examples.html", summary, config["seismic_key"])
    write_json(output / "target_contract.json", {
        "schema_version": 6,
        "status": PENDING,
        "approved": False,
        "manifest_sha256": summary["manifest_sha256"],
        "label_policy": summary["label_policy"],
        "owner_decisions": summary["owner_decisions"],
        "unresolved": [
            "Source validity and target yield from the planned pair sampler; measure after implementation and before authorizing training.",
            "Freeze and validate the deterministic evaluation manifest/cohorts against the approved eligibility and success criteria.",
        ],
    })
    write_json(output / "resolved_audit_config.json", config)
    print(f"AUDIT COMPLETE; {PENDING}; production training remains blocked.", flush=True)
    print(f"Read-only evidence: {output / 'audit_report.json'}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing existing audit output: {args.output}")
    config = json.loads(args.config.read_text())
    if not isinstance(config, dict):
        raise ValueError("Audit config must be a JSON object.")
    # Bound simultaneous chunk decompression, including thin-column inspection.
    with zarr.config.set({"async.concurrency": 2, "threading.max_workers": 2}):
        run_audit(config, args.output)


if __name__ == "__main__":
    main()
