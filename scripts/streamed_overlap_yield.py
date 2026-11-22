"""Read-only diagnostic yield sampling; not a frozen evaluation manifest."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time

import numpy as np
import zarr

from scripts.streamed_overlap_audit import (
    LABEL_KEYS,
    PATCH_SHAPE,
    discover_sources,
    inspect_source,
)
from scripts.streamed_overlap_targets import (
    LABEL_DEPTH_LIMIT,
    TARGET_CONTRACT_VERSION,
    Window,
    pair_target,
)


SEISMIC_KEY = "seismicCubes_cumsum_fullstack"


def _origin(group, rng, *, depth_margin=0):
    keys = (SEISMIC_KEY, *LABEL_KEYS)
    if any(key not in group for key in keys):
        raise ValueError("Source is missing required seismic or semantic arrays.")
    shapes = [group[key].shape for key in keys]
    if any(len(shape) != 3 for shape in shapes):
        raise ValueError("Source arrays must use 3D XYZ shapes.")
    limits = [min(shape[axis] for shape in shapes) for axis in range(3)]
    limits[2] = min(limits[2], LABEL_DEPTH_LIMIT)
    maxima = [limit - size for limit, size in zip(limits, PATCH_SHAPE)]
    maxima[2] -= depth_margin
    if any(value < 0 for value in maxima):
        raise ValueError("Source has insufficient fully supported same-index windows.")
    return tuple(int(rng.integers(0, maximum + 1)) for maximum in maxima)


def run_preflight(root: Path, seed: int, samples_per_source: int) -> dict:
    if samples_per_source < 1:
        raise ValueError("samples_per_source must be positive.")
    started = time.monotonic()
    rng = np.random.default_rng(seed)
    sources = discover_sources(root)
    if len({source.identifier for source in sources}) != len(sources):
        raise ValueError("Source identifiers must uniquely identify stores across both splits.")
    records = []
    inventory = []
    for split in ("train", "validation"):
        split_sources = [source for source in sources if source.split == split]
        if len(split_sources) < 2:
            raise ValueError(f"{split} requires at least two sources for cross-source negatives.")
        for index, source in enumerate(split_sources):
            metadata = inspect_source(source, SEISMIC_KEY)
            inventory.append(metadata)
            if metadata["issues"]:
                raise ValueError(
                    f"Source inventory issues for {source.identifier}: {metadata['issues']}"
                )
            other = split_sources[(index + 1) % len(split_sources)]
            group = zarr.open_group(str(source.path), mode="r")
            other_group = zarr.open_group(str(other.path), mode="r")
            for sample in range(samples_per_source):
                origin = _origin(group, rng, depth_margin=64)
                a = Window(source.identifier, origin, PATCH_SHAPE)
                for category, target_group, b in (
                    ("same_zero_shift", group, a),
                    ("same_z16", group, Window(
                        source.identifier, (origin[0], origin[1], origin[2] + 16), PATCH_SHAPE
                    )),
                    ("same_disjoint", group, Window(
                        source.identifier, (origin[0], origin[1], origin[2] + 64), PATCH_SHAPE
                    )),
                    ("cross_source", other_group, Window(
                        other.identifier, _origin(other_group, rng), PATCH_SHAPE
                    )),
                ):
                    result = pair_target(group, target_group, a, b)
                    records.append({
                        "split": split, "source": source.identifier, "sample": sample,
                        "category": category, "a": asdict(a), "b": asdict(b),
                        "target": asdict(result),
                    })
    summary = {}
    for split in ("train", "validation"):
        rows = [row for row in records if row["split"] == split]
        summary[split] = {
            "sources": sum(source.split == split for source in sources),
            "pairs_by_category": dict(Counter(row["category"] for row in rows)),
            "unavailable_reasons": dict(Counter(
                row["target"]["unavailable_reason"] for row in rows
                if row["target"]["unavailable_reason"] is not None
            )),
            "positive_coverage_pairs": sum(
                row["target"]["coverage"] is not None and row["target"]["coverage"] > 0
                for row in rows
            ),
        }
    return {
        "status": "LOCAL_TARGET_VALIDATION_PENDING",
        "contract_version": TARGET_CONTRACT_VERSION,
        "limitation": "Diagnostic samples are not a retrieval eligibility manifest or training authorization; closure fractions are a coarse proxy with unknown generation-time worktrees.",
        "seed": seed,
        "samples_per_source": samples_per_source,
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "summary": summary,
        "source_inventory": inventory,
        "pairs": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--samples-per-source", required=True, type=int)
    args = parser.parse_args()
    root = args.source_root.resolve(strict=True)
    if args.output.resolve().is_relative_to(root):
        raise ValueError("Preflight output must be outside the source root.")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    # Reserve the destination before any source reads; never overwrite a prior report.
    with args.output.open("x") as handle:
        report = run_preflight(root, args.seed, args.samples_per_source)
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        key: report[key] for key in (
            "status", "summary", "elapsed_seconds", "max_rss_platform_units"
        )
    }, indent=2))


if __name__ == "__main__":
    main()
