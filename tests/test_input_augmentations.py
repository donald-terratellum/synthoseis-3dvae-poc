import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace

import numpy as np
import zarr

from scripts import train as train_script
from scripts.train import ZarrPatchDataset
from src import augmentations
import scripts.train as train_mod


class InputAugmentationTests(unittest.TestCase):
    def test_sparse_keep_count_and_edge_fraction_behavior(self):
        x = np.arange(1, 1 + 8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)

        out_fixed = augmentations.apply_input_random_sparse_keep(
            x,
            fraction_min=0.25,
            fraction_max=0.25,
            method='poisson',
        )
        expected_fixed = int(np.clip(np.rint(0.25 * x.size), 1, x.size))
        self.assertEqual(int(np.count_nonzero(out_fixed)), expected_fixed)

        out_all = augmentations.apply_input_random_sparse_keep(
            x,
            fraction_min=1.0,
            fraction_max=1.0,
            method='uniform',
        )
        self.assertEqual(int(np.count_nonzero(out_all)), int(x.size))

        out_low = augmentations.apply_input_random_sparse_keep(
            x,
            fraction_min=0.01,
            fraction_max=0.01,
            method='poisson',
        )
        expected_low = int(np.clip(np.rint(0.01 * x.size), 1, x.size))
        self.assertEqual(int(np.count_nonzero(out_low)), expected_low)

    def test_sparse_random_method_is_seed_deterministic(self):
        x = np.arange(1, 1 + 6 * 6 * 6, dtype=np.float32).reshape(6, 6, 6)
        np.random.seed(123)
        out_a = augmentations.apply_input_random_sparse_keep(
            x,
            fraction_min=0.30,
            fraction_max=0.30,
            method='random',
        )
        np.random.seed(123)
        out_b = augmentations.apply_input_random_sparse_keep(
            x,
            fraction_min=0.30,
            fraction_max=0.30,
            method='random',
        )
        self.assertTrue(np.array_equal(out_a, out_b))

    def test_sparse_uniform_method_is_seed_deterministic(self):
        x = np.arange(1, 1 + 6 * 6 * 6, dtype=np.float32).reshape(6, 6, 6)
        np.random.seed(321)
        out_a = augmentations.apply_input_random_sparse_keep(
            x,
            fraction_min=0.20,
            fraction_max=0.20,
            method='uniform',
        )
        np.random.seed(321)
        out_b = augmentations.apply_input_random_sparse_keep(
            x,
            fraction_min=0.20,
            fraction_max=0.20,
            method='uniform',
        )
        self.assertTrue(np.array_equal(out_a, out_b))

    def test_decimate_trilinear_preserves_anchors_and_finite_outputs(self):
        x = np.arange(1, 1 + 8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
        for parity in (0, 1):
            out = augmentations.apply_input_decimate_trilinear(x, parity=parity)
            self.assertEqual(out.shape, x.shape)
            self.assertEqual(out.dtype, np.float32)
            self.assertTrue(np.isfinite(out).all())
            idx = np.arange(parity, x.shape[0], 2)
            self.assertTrue(np.allclose(out[np.ix_(idx, idx, idx)], x[np.ix_(idx, idx, idx)]))

    def test_decimate_random_parity_is_seed_deterministic(self):
        x = np.arange(1, 1 + 7 * 7 * 7, dtype=np.float32).reshape(7, 7, 7)
        np.random.seed(999)
        out_a = augmentations.apply_input_decimate_trilinear(x)
        np.random.seed(999)
        out_b = augmentations.apply_input_decimate_trilinear(x)
        self.assertTrue(np.array_equal(out_a, out_b))

    def test_dataset_applies_exactly_one_input_transform_per_sample(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zarr_path = Path(tmp_dir) / 'data.zarr'
            root = zarr.open(str(zarr_path), mode='w')
            data = np.arange(1, 1 + 3 * 4 * 4 * 4, dtype=np.float32).reshape(3, 4, 4, 4)
            root.create_array('patches', data=data)

            ds = ZarrPatchDataset(
                zarr_path,
                augment=False,
                extrema_only=None,
                input_extrema_prob=0.4,
                input_sparse_keep_prob=0.3,
                input_decimate_trilinear_prob=0.3,
                sparse_keep_fraction_min=0.25,
                sparse_keep_fraction_max=0.25,
                mixup_augment_prob=0.0,
            )

            counters = {'extrema': 0, 'sparse': 0, 'decimate': 0}
            orig_extrema = train_mod.keep_trace_extrema_only
            orig_sparse = train_mod.apply_input_random_sparse_keep
            orig_decimate = train_mod.apply_input_decimate_trilinear

            def fake_extrema(x):
                counters['extrema'] += 1
                return x

            def fake_sparse(x, **kwargs):
                counters['sparse'] += 1
                return x

            def fake_decimate(x):
                counters['decimate'] += 1
                return x

            train_mod.keep_trace_extrema_only = fake_extrema
            train_mod.apply_input_random_sparse_keep = fake_sparse
            train_mod.apply_input_decimate_trilinear = fake_decimate
            try:
                for i in range(20):
                    counters['extrema'] = 0
                    counters['sparse'] = 0
                    counters['decimate'] = 0
                    _ = ds[i % len(ds)]
                    self.assertEqual(counters['extrema'] + counters['sparse'] + counters['decimate'], 1)
            finally:
                train_mod.keep_trace_extrema_only = orig_extrema
                train_mod.apply_input_random_sparse_keep = orig_sparse
                train_mod.apply_input_decimate_trilinear = orig_decimate

    def test_backward_compat_extrema_only_matches_probability_mode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zarr_path = Path(tmp_dir) / 'data.zarr'
            root = zarr.open(str(zarr_path), mode='w')
            data = np.arange(1, 1 + 2 * 5 * 5 * 5, dtype=np.float32).reshape(2, 5, 5, 5)
            root.create_array('patches', data=data)

            ds_legacy = ZarrPatchDataset(
                zarr_path,
                augment=False,
                extrema_only=True,
                mixup_augment_prob=0.0,
            )
            ds_prob = ZarrPatchDataset(
                zarr_path,
                augment=False,
                extrema_only=None,
                input_extrema_prob=1.0,
                input_sparse_keep_prob=0.0,
                input_decimate_trilinear_prob=0.0,
                mixup_augment_prob=0.0,
            )

            x_legacy, y_legacy = ds_legacy[0]
            x_prob, y_prob = ds_prob[0]
            self.assertTrue(np.array_equal(x_legacy.numpy(), x_prob.numpy()))
            self.assertTrue(np.array_equal(y_legacy.numpy(), y_prob.numpy()))

    def test_conflict_guard_for_legacy_extrema_and_probability_controls(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zarr_path = Path(tmp_dir) / 'data.zarr'
            root = zarr.open(str(zarr_path), mode='w')
            data = np.arange(1, 1 + 1 * 4 * 4 * 4, dtype=np.float32).reshape(1, 4, 4, 4)
            root.create_array('patches', data=data)

            with self.assertRaises(ValueError):
                ZarrPatchDataset(
                    zarr_path,
                    augment=False,
                    extrema_only=True,
                    input_extrema_prob=0.7,
                    input_sparse_keep_prob=0.3,
                    input_decimate_trilinear_prob=0.0,
                    mixup_augment_prob=0.0,
                )

    def test_validation_uses_same_input_transform_weights_as_training(self):
        captured = {}
        original_dataset = train_script.ZarrPatchDataset
        original_dataloader = train_script.DataLoader

        class StopValidation(Exception):
            pass

        class FakeDataset:
            patch_shape = (32, 32, 32)

            def __init__(self, *args, **kwargs):
                captured.update(kwargs)

            def __len__(self):
                return 1

        class FakeLoader:
            def __init__(self, dataset, *args, **kwargs):
                raise StopValidation()

        train_script.ZarrPatchDataset = FakeDataset
        train_script.DataLoader = FakeLoader
        args = SimpleNamespace(
            validation_data='data/validation.zarr',
            input_scaling='none',
            input_mean=0.0,
            input_std=1.0,
            batch_size=1,
            validation_extrema_only=True,
            input_extrema_prob=0.2,
            input_sparse_keep_prob=0.3,
            input_decimate_trilinear_prob=0.5,
            sparse_keep_fraction_min=0.10,
            sparse_keep_fraction_max=0.30,
            sparse_poisson_radius_scale=0.85,
            current_kl_weight=0.0,
            latent_pred_target_weight=0.0,
            latent_pred_input_weight=0.0,
            latent_alignment_detach_targets=True,
            patch_size_xyz=(32, 32, 32),
        )

        try:
            with self.assertRaises(StopValidation):
                train_script.validate(model=None, args=args, device='cpu', train_steps_per_epoch=1)
        finally:
            train_script.ZarrPatchDataset = original_dataset
            train_script.DataLoader = original_dataloader

        self.assertIsNone(captured['extrema_only'])
        self.assertEqual(captured['input_extrema_prob'], 0.2)
        self.assertEqual(captured['input_sparse_keep_prob'], 0.3)
        self.assertEqual(captured['input_decimate_trilinear_prob'], 0.5)


if __name__ == '__main__':
    unittest.main()
