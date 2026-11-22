import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.streamed_overlap_audit import LABEL_KEYS, Source, canonical_overlap
from scripts.streamed_overlap_augmentation import ViewTransform, spatial_view
from scripts.streamed_overlap_sampler import GEOLOGY_SEEDS, StreamedPairSampler
from scripts.streamed_overlap_targets import Window
from scripts.streamed_overlap_yield import SEISMIC_KEY


class StreamedSamplerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sources = []
        self.metadata = {}
        self.groups = {}
        shape = (64, 64, 256)
        signal = np.broadcast_to(np.sin(np.arange(shape[2], dtype=np.float32)), shape)
        age = np.broadcast_to(np.arange(shape[2], dtype=np.float32), shape)
        zeros = np.broadcast_to(np.float32(0), shape)
        for split in ("train", "validation"):
            for index in range(2):
                path = self.root / split / f"s{index}" / "model_data.zarr"
                path.mkdir(parents=True)
                source = Source(path, split)
                self.sources.append(source)
                seed_base = len(self.sources) * 100
                self.metadata[source.identifier] = {
                    "issues": [],
                    "arrays": {key: {"shape": shape} for key in (SEISMIC_KEY, *LABEL_KEYS)},
                    "parameters": {
                        "generator_seeds": {key: seed_base + n for n, key in enumerate(GEOLOGY_SEEDS)},
                        "bandlimits_hz": [5, 45], "sample_spacing_ms_from_digi": 4,
                        "filter_order": 4,
                    },
                }
                self.groups[str(path)] = {
                    SEISMIC_KEY: signal, "geologic_age_faulted": age,
                    "oil_closures": zeros, "gas_closures": zeros,
                }
        self.discover = patch("scripts.streamed_overlap_sampler.discover_sources", return_value=self.sources)
        self.inspect = patch("scripts.streamed_overlap_sampler.inspect_source",
                             side_effect=lambda source, _key: self.metadata[source.identifier])
        self.open = patch("scripts.streamed_overlap_sampler.zarr.open_group",
                          side_effect=lambda path, mode: self.groups[path])
        self.discover.start()
        self.inspect.start()
        self.open.start()
        self.addCleanup(self.discover.stop)
        self.addCleanup(self.inspect.stop)
        self.addCleanup(self.open.stop)

    def sampler(self, split="train"):
        return StreamedPairSampler(self.root, split, seed=41)

    def test_batch_categories_context_and_exact_resume(self):
        sampler = self.sampler()
        specs = sampler.next_specs()
        self.assertEqual([spec.category for spec in specs],
                         ["same_overlap"] * 6 + ["same_disjoint"] * 3 + ["cross_source"] * 3)
        self.assertEqual(sum(spec.a.origin == spec.b.origin for spec in specs[:6]), 2)
        for spec in specs[:6]:
            self.assertTrue(all(abs(a - b) <= limit for a, b, limit
                                in zip(spec.a.origin, spec.b.origin, (8, 8, 16))))
        for spec in specs[6:9]:
            self.assertEqual(canonical_overlap(spec.a.origin, spec.b.origin, spec.a.shape, True), 0)
        self.assertTrue(all(spec.a.source_id != spec.b.source_id for spec in specs[9:]))
        for spec in specs:
            for window in (spec.a, spec.b):
                info = sampler.by_id[window.source_id]
                self.assertTrue(all(lo <= coordinate <= hi
                                    for coordinate, lo, hi in zip(window.origin, info.lower, info.upper)))
        state = sampler.state_dict()
        expected = sampler.next_specs()
        self.assertNotEqual(specs, expected)
        restored = self.sampler()
        restored.load_state_dict(state)
        self.assertEqual(expected, restored.next_specs())
        state["source_manifest_hash"] = "different"
        with self.assertRaisesRegex(ValueError, "incompatible"):
            restored.load_state_dict(state)

    def test_worst_case_rotation_and_scale_have_real_context(self):
        sampler = self.sampler()
        info = sampler.sources[0]
        source = self.groups[str(info.source.path)][SEISMIC_KEY]
        for origin in (info.lower, info.upper):
            window = Window(info.source.identifier, origin, (32, 32, 64))
            for angle in (0, 45, 90, 135, 225, 315):
                view = spatial_view(source, window, ViewTransform(angle, (0.9, 0.9, 0.9), False, False, False))
                self.assertTrue(np.isfinite(view).all())

    def test_targets_are_computed_before_all_input_augmentation(self):
        sampler = self.sampler()
        specs = sampler.next_specs()
        from scripts.streamed_overlap_sampler import pair_target

        with patch("scripts.streamed_overlap_sampler.pair_target", wraps=pair_target) as target:
            def fake_augment(_array, window, _rng, _metadata, *, input_mode):
                self.assertEqual(target.call_count, 12)
                dense = np.ones(window.shape, dtype=np.float32)
                return dense, dense, {"input_mode": input_mode}

            with patch("scripts.streamed_overlap_sampler.augment_view", side_effect=fake_augment):
                batch = sampler.load_batch(specs, sparse_probability=0.5)
        self.assertEqual(batch.input_a.shape, (12, 1, 32, 32, 64))
        self.assertTrue(batch.coverage_valid.all())
        self.assertEqual(len(batch.provenance), 12)
        self.assertLessEqual(len(sampler.handles), 2)

    def test_resume_reproduces_loaded_inputs_and_targets(self):
        sampler = self.sampler()
        state = sampler.state_dict()
        original = sampler.load_batch(sampler.next_specs(), sparse_probability=0.5)
        resumed = self.sampler()
        resumed.load_state_dict(state)
        replay = resumed.load_batch(resumed.next_specs(), sparse_probability=0.5)
        self.assertEqual(original.provenance, replay.provenance)
        self.assertEqual(original.augmentation_stats, replay.augmentation_stats)
        self.assertEqual(original.unavailable_reasons, replay.unavailable_reasons)
        for field in ("input_a", "input_b", "dense_a", "overlap", "coverage", "coverage_valid"):
            np.testing.assert_array_equal(getattr(original, field), getattr(replay, field))

    def test_shared_recorded_geology_seed_across_splits_fails_closed(self):
        train = self.sources[0]
        validation = self.sources[2]
        self.metadata[validation.identifier]["parameters"]["generator_seeds"]["fault_seed"] = \
            self.metadata[train.identifier]["parameters"]["generator_seeds"]["fault_seed"]
        with self.assertRaisesRegex(ValueError, "overlap between train and validation"):
            self.sampler()


if __name__ == "__main__":
    unittest.main()
