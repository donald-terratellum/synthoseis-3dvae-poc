import tempfile
from pathlib import Path
import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import zarr

from scripts import train as train_script
from scripts.train import ZarrPatchDataset
from src import augmentations
import scripts.train as train_mod
from scripts import sample_patches as sample_patches_script


class InputAugmentationTests(unittest.TestCase):
    def test_patch_sampling_seed_controls_origins(self):
        geoscore = np.ones((16, 16, 16), dtype=np.float32)

        random.seed(123)
        np.random.seed(123)
        first = sample_patches_script.pick_weighted_positions(
            geoscore, geoscore.shape, (8, 8, 8), n_picks=8, n_candidates=40
        )
        random.seed(123)
        np.random.seed(123)
        replay = sample_patches_script.pick_weighted_positions(
            geoscore, geoscore.shape, (8, 8, 8), n_picks=8, n_candidates=40
        )
        random.seed(456)
        np.random.seed(456)
        different = sample_patches_script.pick_weighted_positions(
            geoscore, geoscore.shape, (8, 8, 8), n_picks=8, n_candidates=40
        )

        self.assertEqual(first, replay)
        self.assertNotEqual(first, different)

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

    def test_dataset_can_optional_return_patch_metadata_for_geology_labels(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zarr_path = Path(tmp_dir) / 'data.zarr'
            root = zarr.open(str(zarr_path), mode='w')
            data = np.arange(1, 1 + 2 * 4 * 4 * 4, dtype=np.float32).reshape(2, 4, 4, 4)
            root.create_array('patches', data=data)
            root.create_array('fault_summary', data=np.array([0.1, 0.9], dtype=np.float32))
            root.create_array('geologic_score', data=np.array([0.2, 0.8], dtype=np.float32))

            ds = ZarrPatchDataset(
                zarr_path,
                augment=False,
                include_metadata=True,
                geology_metadata_keys=('fault_summary', 'geologic_score'),
                mixup_augment_prob=0.0,
            )

            x, y, meta = ds[0]
            self.assertEqual(x.shape, (1, 4, 4, 4))
            self.assertEqual(y.shape, (1, 4, 4, 4))
            self.assertAlmostEqual(float(meta['fault_summary']), 0.1)
            self.assertAlmostEqual(float(meta['geologic_score']), 0.2)

    def test_geology_similarity_loss_matches_latent_and_metadata_geometry(self):
        metadata_batch = [
            {'fault_summary': 1.0, 'geologic_score': 0.0},
            {'fault_summary': 0.0, 'geologic_score': 1.0},
            {'fault_summary': 1.0, 'geologic_score': 1.0},
        ]
        latent = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
        )

        aligned_loss = train_script.compute_geology_similarity_loss(
            latent,
            metadata_batch,
            ('fault_summary', 'geologic_score'),
        )
        self.assertLess(aligned_loss.item(), 1e-6)

    def test_geology_similarity_loss_accepts_collated_dict_metadata_batch(self):
        latent = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        metadata_batch = {
            'fault_summary': torch.tensor([1.0, 0.0], dtype=torch.float32),
            'geologic_score': torch.tensor([0.0, 1.0], dtype=torch.float32),
        }
        aligned_loss = train_script.compute_geology_similarity_loss(
            latent,
            metadata_batch,
            ('fault_summary', 'geologic_score'),
        )
        self.assertLess(aligned_loss.item(), 1e-6)

    def test_latent_geology_diagnostics_detect_aligned_cosine_geometry(self):
        metadata = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=torch.float32,
        )

        metrics = train_script.compute_latent_geology_diagnostics(
            latent_vectors=metadata.clone(),
            metadata_vectors=metadata,
            neighbor_k=1,
        )

        self.assertGreater(metrics['pair_cosine_correlation'], 0.999)
        self.assertGreater(metrics['cosine_separation'], 0.9)
        self.assertEqual(metrics['neighbor_overlap'], 1.0)

    def test_geology_similarity_loss_excludes_diagonal_and_background_pairs(self):
        latent = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        metadata_batch = [
            {'fault_summary': 1.0, 'geologic_score': 0.0},
            {'fault_summary': 0.0, 'geologic_score': 1.0},
            {'fault_summary': 0.0, 'geologic_score': 0.0},
        ]

        # Third sample is treated as background via key index 0 and threshold.
        loss = train_script.compute_geology_similarity_loss(
            latent,
            metadata_batch,
            ('fault_summary', 'geologic_score'),
            background_threshold=1e-6,
            background_key_indices=(0, 1),
            offdiag_only=True,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_geology_similarity_loss_supports_huber_mode(self):
        latent = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        metadata_batch = [
            {'fault_summary': 1.0, 'geologic_score': 0.0},
            {'fault_summary': 0.0, 'geologic_score': 1.0},
            {'fault_summary': 0.0, 'geologic_score': 1.0},
        ]
        mse_loss = train_script.compute_geology_similarity_loss(
            latent,
            metadata_batch,
            ('fault_summary', 'geologic_score'),
            loss_type='mse',
            offdiag_only=True,
        )
        huber_loss = train_script.compute_geology_similarity_loss(
            latent,
            metadata_batch,
            ('fault_summary', 'geologic_score'),
            loss_type='huber',
            huber_delta=0.1,
            offdiag_only=True,
        )
        self.assertTrue(torch.isfinite(mse_loss))
        self.assertTrue(torch.isfinite(huber_loss))
        self.assertGreaterEqual(float(mse_loss.item()), 0.0)
        self.assertGreaterEqual(float(huber_loss.item()), 0.0)

    def test_fit_geology_metadata_calibration_and_prepare_vectors(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zarr_path = Path(tmp_dir) / 'data.zarr'
            root = zarr.open(str(zarr_path), mode='w')
            patches = np.zeros((4, 4, 4, 4), dtype=np.float32)
            root.create_array('patches', data=patches)
            root.create_array('fault_summary', data=np.array([0.0, 0.2, 0.4, 0.8], dtype=np.float32))
            root.create_array('geologic_score', data=np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32))

            ds = ZarrPatchDataset(
                zarr_path,
                augment=False,
                include_metadata=True,
                geology_metadata_keys=('fault_summary', 'geologic_score'),
                mixup_augment_prob=0.0,
            )
            calibration = train_script.fit_geology_metadata_calibration(
                ds,
                ('fault_summary', 'geologic_score'),
                strategy='robust_log1p',
                eps=1e-6,
                clip=6.0,
                background_keys=('fault_summary',),
            )

            metadata_batch = {
                'fault_summary': torch.tensor([0.0, 0.4], dtype=torch.float32),
                'geologic_score': torch.tensor([0.0, 0.2], dtype=torch.float32),
            }
            vectors, has_selected = train_script.prepare_geology_metadata_vectors(
                metadata_batch,
                ('fault_summary', 'geologic_score'),
                device='cpu',
                dtype=torch.float32,
                geology_metadata_calibration=calibration,
                background_threshold=1e-6,
                background_key_indices=(0,),
            )

            self.assertEqual(tuple(vectors.shape), (2, 2))
            self.assertFalse(bool(has_selected[0].item()))
            self.assertTrue(bool(has_selected[1].item()))

    def test_latent_geology_diagnostics_reports_multiple_topk_metrics(self):
        samples = []
        for idx in range(32):
            angle = (2.0 * np.pi * float(idx)) / 32.0
            samples.append([np.cos(angle), np.sin(angle), 0.1 * np.cos(2.0 * angle)])
        metadata = torch.tensor(samples, dtype=torch.float32)
        mask = torch.ones((metadata.shape[0],), dtype=torch.bool)

        metrics = train_script.compute_latent_geology_diagnostics(
            latent_vectors=metadata.clone(),
            metadata_vectors=metadata,
            neighbor_k=5,
            neighbor_ks=[5, 10, 20],
            has_selected_geology=mask,
        )

        self.assertIn('neighbor_overlap', metrics)
        self.assertIn('neighbor_overlap_at_5', metrics)
        self.assertIn('neighbor_overlap_at_10', metrics)
        self.assertIn('neighbor_overlap_at_20', metrics)
        self.assertEqual(metrics['neighbor_overlap_at_5'], 1.0)
        self.assertEqual(metrics['neighbor_overlap_at_10'], 1.0)
        self.assertEqual(metrics['neighbor_overlap_at_20'], 1.0)

    def test_sample_patches_can_return_derived_metadata_vectors(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zarr_path = Path(tmp_dir) / 'model_data.zarr'
            root = zarr.open(str(zarr_path), mode='w')
            shape = (16, 16, 16)
            seismic = np.random.RandomState(0).normal(size=shape).astype(np.float32)
            geoscore = np.abs(np.random.RandomState(1).normal(size=shape)).astype(np.float32)

            # A ramp in depth plus gentle x/y trend yields non-trivial dip and azimuth.
            x = np.linspace(0.0, 1.0, shape[0], dtype=np.float32)[:, None, None]
            y = np.linspace(0.0, 1.0, shape[1], dtype=np.float32)[None, :, None]
            z = np.linspace(0.0, 1.0, shape[2], dtype=np.float32)[None, None, :]
            geologic_age_faulted = (0.2 * x + 0.1 * y + 0.7 * z).astype(np.float32)
            fault_segments = np.ones(shape, dtype=np.uint32)
            fault_intersection = np.zeros(shape, dtype=np.float32)
            fault_intersection[4:12, 4:12, 4:12] = 1.0
            faulted_lithology = np.linspace(-1.0, 1.0, shape[0], dtype=np.float32)[:, None, None] * np.ones(shape, dtype=np.float32)
            # Cover the whole volume so any 8^3 patch will contain these features.
            flat_spot = np.ones(shape, dtype=np.uint8)
            onlap_segments = np.ones(shape, dtype=np.float32)
            channel_labels = np.full(shape, 2, dtype=np.uint8)

            root.create_array('seismicCubes_cumsum_fullstack', data=seismic)
            root.create_array('geologic_score', data=geoscore)
            root.create_array('geologic_age_faulted', data=geologic_age_faulted)
            root.create_array('fault_segments_id', data=fault_segments)
            root.create_array('fault_intersection_segments', data=fault_intersection)
            root.create_array('faulted_lithology', data=faulted_lithology)
            root.create_array('flat_spot', data=flat_spot)
            root.create_array('onlap_segments', data=onlap_segments)
            faults_group = root.create_group('faults')
            faults_group.create_array('faulted_channel_labels', data=channel_labels)

            items = sample_patches_script.sample_patches_from_model(
                root,
                'seismicCubes_cumsum_fullstack',
                'geologic_score',
                (8, 8, 8),
                n_patches_per_vol=2,
                allow_overlap=False,
                return_metadata=True,
            )
            self.assertTrue(items)
            patch, metadata = items[0]
            self.assertEqual(patch.shape, (8, 8, 8))
            for key in sample_patches_script.DERIVED_METADATA_KEYS:
                self.assertIn(key, metadata)
                self.assertTrue(np.isfinite(float(metadata[key])))
            self.assertEqual(float(metadata['meta_fault_fraction']), 1.0)
            self.assertGreater(float(metadata['meta_fault_intersection_fraction']), 0.0)
            self.assertGreaterEqual(float(metadata['meta_sand_fraction']), 0.0)
            self.assertGreaterEqual(float(metadata['meta_shale_fraction']), 0.0)
            self.assertGreater(float(metadata['meta_flat_spot_fraction']), 0.0)
            self.assertGreater(float(metadata['meta_onlap_fraction']), 0.0)
            self.assertGreater(float(metadata['meta_channel_fraction']), 0.0)

    def test_vae_supports_residual_encoder_variant(self):
        model = train_script.VAE3D(
            patch_shape=(32, 32, 32),
            latent_dim=16,
            base_ch=8,
            residual_encoder=True,
        )
        x = torch.randn(2, 1, 32, 32, 32)
        recon, mu, logvar = model(x)
        self.assertEqual(recon.shape, x.shape)
        self.assertEqual(mu.shape, (2, 16))
        self.assertEqual(logvar.shape, (2, 16))

    def test_discover_model_data_volumes_prefers_synthetic_run_folders(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / 'fake_data'
            good_run = root / 'seismic__2026.123__synthoseis_run_42' / 'model_data.zarr'
            good_run.mkdir(parents=True)
            bad_run = root / 'seismic__2026.456__synthoseis_run_43' / 'model_data.zarr'
            bad_run.mkdir(parents=True)
            (bad_run.parent.parent / 'temp_folder__2026.456__synthoseis_run_43').mkdir()

            discovered = train_script.discover_model_data_volumes(root)
            self.assertIn(good_run, discovered)
            self.assertNotIn(bad_run, discovered)

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
            geology_loss_weight=0.0,
            geology_metadata_keys=(),
            current_kl_weight=0.0,
            deep_supervision=False,
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
