"""Volume-balanced, single-worker streaming pair sampler for the opt-in experiment."""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import zarr

from scripts.streamed_overlap_audit import (
    HOLDOUT_NAMES,
    LABEL_KEYS,
    PATCH_SHAPE,
    Source,
    content_hash,
    discover_sources,
    inspect_source,
)
from scripts.streamed_overlap_augmentation import augment_view
from scripts.streamed_overlap_targets import LABEL_DEPTH_LIMIT, Window, pair_target
from scripts.streamed_overlap_yield import SEISMIC_KEY


GEOLOGY_SEEDS = ("horizon_seed", "fault_seed", "closure_seed", "fan_seed", "channel_sim_seed")
MAX_SHIFTS = (8, 8, 16)


def _context_margin() -> tuple[int, int, int]:
    half_x, half_y, half_z = ((size - 1) / 2 for size in PATCH_SHAPE)
    maximum_xy = math.hypot(half_x, half_y) / 0.9
    return (
        math.ceil(maximum_xy - half_x),
        math.ceil(maximum_xy - half_y),
        math.ceil(half_z / 0.9 - half_z),
    )


@dataclass(frozen=True)
class SourceInfo:
    source: Source
    metadata: dict
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]
    geology_seeds: tuple[int, ...]


@dataclass(frozen=True)
class PairSpec:
    category: str
    a: Window
    b: Window
    augmentation_seed_a: int
    augmentation_seed_b: int


@dataclass(frozen=True)
class PairBatch:
    input_a: np.ndarray
    input_b: np.ndarray
    dense_a: np.ndarray
    overlap: np.ndarray
    coverage: np.ndarray
    coverage_valid: np.ndarray
    unavailable_reasons: tuple[str | None, ...]
    provenance: tuple[PairSpec, ...]
    augmentation_stats: tuple[tuple[dict, dict], ...]


def source_info(source: Source, metadata: dict) -> SourceInfo:
    if metadata["issues"]:
        raise ValueError(f"Source inventory issues for {source.identifier}: {metadata['issues']}")
    arrays = metadata["arrays"]
    keys = (SEISMIC_KEY, *LABEL_KEYS)
    if any(key not in arrays or len(arrays[key]["shape"]) != 3 for key in keys):
        raise ValueError(f"Missing or non-XYZ source arrays: {source.identifier}")
    seismic_shape = arrays[SEISMIC_KEY]["shape"]
    margins = _context_margin()
    lower = margins
    upper = tuple(
        min(
            seismic_shape[axis] - PATCH_SHAPE[axis] - margins[axis],
            *(arrays[key]["shape"][axis] - PATCH_SHAPE[axis] for key in LABEL_KEYS),
            *([LABEL_DEPTH_LIMIT - PATCH_SHAPE[axis]] if axis == 2 else []),
        )
        for axis in range(3)
    )
    if any(hi - lo < shift for lo, hi, shift in zip(lower, upper, MAX_SHIFTS)):
        raise ValueError(f"Insufficient supported geometric context or shifts: {source.identifier}")
    if not any(hi - lo >= size for lo, hi, size in zip(lower, upper, PATCH_SHAPE)):
        raise ValueError(f"Source cannot support disjoint windows: {source.identifier}")
    parameters = metadata.get("parameters")
    seeds = parameters.get("generator_seeds") if isinstance(parameters, dict) else None
    if not isinstance(seeds, dict) or any(
        isinstance(seeds.get(key), bool) or not isinstance(seeds.get(key), int)
        for key in GEOLOGY_SEEDS
    ):
        raise ValueError(f"Missing recorded geology seeds: {source.identifier}")
    return SourceInfo(source, metadata, lower, (upper[0], upper[1], upper[2]),
                      tuple(seeds[key] for key in GEOLOGY_SEEDS))


def _related(a: SourceInfo, b: SourceInfo) -> bool:
    return any(x == y for x, y in zip(a.geology_seeds, b.geology_seeds))


class StreamedPairSampler:
    def __init__(self, root: Path, split: str, seed: int, max_open_handles: int = 2):
        if split not in {"train", "validation"}:
            raise ValueError("Only train and validation splits are supported.")
        if seed < 0 or max_open_handles < 2:
            raise ValueError("Seed must be nonnegative and at least two handles are required.")
        self.root = root.resolve(strict=True)
        discovered = discover_sources(self.root)
        inventory = [inspect_source(source, SEISMIC_KEY) for source in discovered]
        infos = [source_info(source, metadata) for source, metadata in zip(discovered, inventory)]
        if len({info.source.identifier for info in infos}) != len(infos):
            raise ValueError("Source identifiers must be unique.")
        train = [info for info in infos if info.source.split == "train"]
        validation = [info for info in infos if info.source.split == "validation"]
        if any(_related(a, b) for a in train for b in validation):
            raise ValueError("Recorded geology seeds overlap between train and validation.")
        self.sources = [info for info in infos if info.source.split == split]
        self.by_id = {info.source.identifier: info for info in self.sources}
        for info in self.sources:
            path = info.source.path.resolve(strict=True)
            if not path.is_relative_to(self.root):
                raise ValueError(f"Source escapes root: {path}")
            if split == "train" and any(part in HOLDOUT_NAMES for part in path.relative_to(self.root).parts):
                raise ValueError(f"Training source is a holdout: {path}")
            if not any(other is not info and not _related(info, other) for other in self.sources):
                raise ValueError(f"No provenance-distinct cross-source partner: {info.source.identifier}")
        self.source_manifest_hash = content_hash(inventory)
        self.split = split
        self.max_open_handles = max_open_handles
        self.handles: OrderedDict[str, zarr.Group] = OrderedDict()
        streams = np.random.SeedSequence(seed).spawn(2)
        self.pair_rng = np.random.default_rng(streams[0])
        self.augmentation_rng = np.random.default_rng(streams[1])
        self.batches_drawn = 0

    def _open(self, source_id: str):
        if source_id not in self.by_id:
            raise ValueError(f"Source is outside sampler split: {source_id}")
        if source_id in self.handles:
            self.handles.move_to_end(source_id)
        else:
            if len(self.handles) >= self.max_open_handles:
                self.handles.popitem(last=False)
            self.handles[source_id] = zarr.open_group(str(self.by_id[source_id].source.path), mode="r")
        return self.handles[source_id]

    def _source(self) -> SourceInfo:
        return self.sources[int(self.pair_rng.integers(len(self.sources)))]

    def _origin(self, info: SourceInfo) -> tuple[int, int, int]:
        values = [int(self.pair_rng.integers(lo, hi + 1)) for lo, hi in zip(info.lower, info.upper)]
        return values[0], values[1], values[2]

    def _overlapping(self, info: SourceInfo, zero: bool) -> tuple[Window, Window]:
        if zero:
            offset = (0, 0, 0)
        else:
            while True:
                values = [int(self.pair_rng.integers(-shift, shift + 1)) for shift in MAX_SHIFTS]
                offset = (values[0], values[1], values[2])
                if offset != (0, 0, 0):
                    break
        low = [max(lo, lo - delta) for lo, delta in zip(info.lower, offset)]
        high = [min(hi, hi - delta) for hi, delta in zip(info.upper, offset)]
        values = [int(self.pair_rng.integers(lo, hi + 1)) for lo, hi in zip(low, high)]
        a = (values[0], values[1], values[2])
        b = (a[0] + offset[0], a[1] + offset[1], a[2] + offset[2])
        return Window(info.source.identifier, a, PATCH_SHAPE), Window(info.source.identifier, b, PATCH_SHAPE)

    def _disjoint(self, info: SourceInfo) -> tuple[Window, Window]:
        axes = [axis for axis, (lo, hi, size) in enumerate(zip(info.lower, info.upper, PATCH_SHAPE))
                if hi - lo >= size]
        axis = axes[int(self.pair_rng.integers(len(axes)))]
        first = list(self._origin(info))
        second = list(self._origin(info))
        first[axis] = int(self.pair_rng.integers(info.lower[axis], info.upper[axis] - PATCH_SHAPE[axis] + 1))
        second[axis] = int(self.pair_rng.integers(first[axis] + PATCH_SHAPE[axis], info.upper[axis] + 1))
        if self.pair_rng.random() < 0.5:
            first, second = second, first
        return (
            Window(info.source.identifier, (first[0], first[1], first[2]), PATCH_SHAPE),
            Window(info.source.identifier, (second[0], second[1], second[2]), PATCH_SHAPE),
        )

    def next_specs(self) -> tuple[PairSpec, ...]:
        specs = []
        for index in range(12):
            info = self._source()
            if index < 6:
                category = "same_overlap"
                a, b = self._overlapping(info, zero=index < 2)
            elif index < 9:
                category = "same_disjoint"
                a, b = self._disjoint(info)
            else:
                category = "cross_source"
                eligible = [other for other in self.sources if other is not info and not _related(info, other)]
                other = eligible[int(self.pair_rng.integers(len(eligible)))]
                a = Window(info.source.identifier, self._origin(info), PATCH_SHAPE)
                b = Window(other.source.identifier, self._origin(other), PATCH_SHAPE)
            seeds = self.augmentation_rng.integers(0, 2**63, size=2, dtype=np.int64)
            specs.append(PairSpec(category, a, b, int(seeds[0]), int(seeds[1])))
        self.batches_drawn += 1
        return tuple(specs)

    def state_dict(self) -> dict:
        return {
            "schema": "streamed-pair-sampler-v1",
            "split": self.split,
            "source_manifest_hash": self.source_manifest_hash,
            "batches_drawn": self.batches_drawn,
            "pair_rng": deepcopy(self.pair_rng.bit_generator.state),
            "augmentation_rng": deepcopy(self.augmentation_rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict) -> None:
        if (state.get("schema") != "streamed-pair-sampler-v1"
                or state.get("split") != self.split
                or state.get("source_manifest_hash") != self.source_manifest_hash):
            raise ValueError("Sampler state is incompatible with the current source manifest.")
        batches = state.get("batches_drawn")
        if isinstance(batches, bool) or not isinstance(batches, int) or batches < 0:
            raise ValueError("Invalid sampler batch position.")
        pair_rng = np.random.default_rng()
        augmentation_rng = np.random.default_rng()
        pair_rng.bit_generator.state = deepcopy(state["pair_rng"])
        augmentation_rng.bit_generator.state = deepcopy(state["augmentation_rng"])
        self.pair_rng, self.augmentation_rng = pair_rng, augmentation_rng
        self.batches_drawn = batches

    def load_batch(self, specs: tuple[PairSpec, ...], *, sparse_probability: float) -> PairBatch:
        if len(specs) != 12 or not math.isfinite(sparse_probability) or not 0 <= sparse_probability <= 1:
            raise ValueError("A batch requires 12 pairs and a valid sparse-input probability.")
        targets = []
        for spec in specs:
            target = pair_target(self._open(spec.a.source_id), self._open(spec.b.source_id), spec.a, spec.b)
            if target.overlap is None:
                raise ValueError(f"Unavailable physical overlap: {target.unavailable_reason}")
            targets.append(target)
        inputs_a, inputs_b, dense_a, stats = [], [], [], []
        for spec in specs:
            views = []
            for window, seed in ((spec.a, spec.augmentation_seed_a), (spec.b, spec.augmentation_seed_b)):
                rng = np.random.default_rng(seed)
                mode = "sparse" if rng.random() < sparse_probability else "extrema"
                group = self._open(window.source_id)
                metadata = self.by_id[window.source_id].metadata["parameters"]
                views.append(augment_view(group[SEISMIC_KEY], window, rng, metadata, input_mode=mode))
            inputs_a.append(views[0][0])
            inputs_b.append(views[1][0])
            dense_a.append(views[0][1])
            stats.append((views[0][2], views[1][2]))
        return PairBatch(
            np.stack(inputs_a)[:, None], np.stack(inputs_b)[:, None], np.stack(dense_a)[:, None],
            np.asarray([target.overlap for target in targets], dtype=np.float32),
            np.asarray([target.coverage if target.coverage is not None else np.nan for target in targets],
                       dtype=np.float32),
            np.asarray([target.coverage is not None for target in targets], dtype=bool),
            tuple(target.unavailable_reason for target in targets), specs, tuple(stats),
        )
