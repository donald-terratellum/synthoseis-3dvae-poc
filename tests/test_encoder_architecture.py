import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts import train as train_script
from scripts import diagnose_p5_p6_reconstruction as diagnostics
from scripts import p7_control
from src.model import VAE3D
from src.tokenizer.core.model_adapter import VaeLatentAdapter
from src.tokenizer.core.preprocess import preprocess_for_token


class EncoderArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def test_default_config_round_trip_preserves_legacy_state_dict(self):
        original = VAE3D(base_ch=8, latent_dim=32, patch_shape=(8, 8, 8))
        payload = train_script.build_checkpoint_payload(original)
        config = payload['model_config']
        rebuilt = VAE3D(
            base_ch=8,
            latent_dim=32,
            patch_shape=(8, 8, 8),
            encoder_arch=config['encoder_arch'],
            encoder_hidden_dims=config['encoder_hidden_dims'],
            encoder_depth_profile=config['encoder_depth_profile'],
            encoder_stage_blocks=config['encoder_stage_blocks'],
            encoder_norm=config['encoder_norm'],
            encoder_stem=config['encoder_stem'],
            encoder_input_axes=config['encoder_input_axes'],
            decoder_hidden_dims=config['decoder_hidden_dims'],
            decoder_block=config['decoder_block'],
        )

        self.assertEqual(set(original.state_dict()), set(rebuilt.state_dict()))
        rebuilt.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(config['encoder_arch'], 'conv')
        self.assertIsNone(config['decoder_hidden_dims'])

    def _e2(self, **overrides):
        config = dict(
            patch_shape=(32, 32, 64),
            encoder_arch='resnetv2',
            encoder_hidden_dims=(8, 16, 32),
            encoder_depth_profile='deeper',
            encoder_stage_blocks=(1, 1, 1),
            encoder_norm='instance',
            encoder_input_axes='zxy',
            decoder_block='res',
        )
        config.update(overrides)
        return VAE3D(**config)

    def test_e2_forward_keeps_patch_and_latent_contract(self):
        model = self._e2().eval()
        inputs = torch.randn(2, 1, 32, 32, 64)

        with torch.no_grad():
            reconstruction, mu, _ = model(inputs)

        self.assertEqual(tuple(reconstruction.shape), tuple(inputs.shape))
        self.assertEqual(tuple(mu.shape), (2, 128))
        self.assertEqual(model.encoder.fc_mu.in_features, 2048)
        self.assertTrue(torch.isfinite(reconstruction).all())

    def test_e1_e3_e4_candidates_forward_with_128d_latents(self):
        candidates = (
            VAE3D(patch_shape=(32, 32, 64), encoder_arch='residual'),
            VAE3D(
                patch_shape=(32, 32, 64), encoder_arch='resnetv2',
                encoder_hidden_dims=(32, 64, 128), encoder_stage_blocks=(3, 5, 8),
                encoder_depth_profile='deeper', encoder_norm='instance', encoder_stem='light',
                decoder_hidden_dims=(256, 128, 16),
            ),
            VAE3D(
                patch_shape=(32, 32, 64), encoder_arch='resnetv2',
                encoder_hidden_dims=(40, 80, 160), encoder_stage_blocks=(3, 5, 8),
                encoder_depth_profile='deeper', encoder_norm='instance', encoder_stem='pretrain_v2',
            ),
        )
        expected_flat_dims = (8192, 65536, 10240)
        inputs = torch.randn(1, 1, 32, 32, 64)

        for model, flat_dim in zip(candidates, expected_flat_dims):
            model.eval()
            with torch.no_grad():
                reconstruction, mu, _ = model(inputs)
            self.assertEqual(model.encoder.fc_mu.in_features, flat_dim)
            self.assertEqual(tuple(reconstruction.shape), tuple(inputs.shape))
            self.assertEqual(tuple(mu.shape), (1, 128))
            self.assertTrue(torch.isfinite(reconstruction).all())
            self.assertTrue(torch.isfinite(mu).all())

    def test_zxy_adapter_matches_manual_axis_permutation(self):
        model = self._e2().eval()
        inputs = torch.randn(1, 1, 32, 32, 64)

        with torch.no_grad():
            actual = model.encoder(inputs)
            network_input = inputs.permute(0, 1, 4, 2, 3).contiguous()
            features = model.encoder.trunk(network_input).flatten(1)
            expected = (model.encoder.fc_mu(features), model.encoder.fc_logvar(features))

        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    def test_instance_norm_is_finite_for_low_variance_patch(self):
        model = self._e2().eval()
        inputs = torch.randn(2, 1, 32, 32, 64) * 1e-4

        with torch.no_grad():
            mu, logvar = model.encoder(inputs)

        self.assertTrue(torch.isfinite(mu).all())
        self.assertTrue(torch.isfinite(logvar).all())
        self.assertGreater(float(mu.std()), 0.0)

    def test_pretrained_encoder_transfer_uses_ema_and_rejects_mismatch(self):
        source = self._e2()
        target = self._e2()
        source_weights = {
            f'encoder.{key}': value.detach().clone()
            for key, value in source.encoder.trunk.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'pretrain.pt'
            torch.save({
                'epoch': 84,
                'train_paths': ['a', 'b'],
                'model': {},
                'ema_state': {'shadow': source_weights},
            }, checkpoint)
            result = train_script.load_pretrained_encoder(target, checkpoint)
            self.assertEqual(result['source_state'], 'ema_state.shadow')
            self.assertEqual(result['epoch'], 84)
            self.assertEqual(result['train_paths_count'], 2)
            self.assertEqual(result['tensors_loaded'], len(source.encoder.trunk.state_dict()))
            for key, value in source.encoder.trunk.state_dict().items():
                torch.testing.assert_close(target.encoder.trunk.state_dict()[key], value)

            source_weights.pop(next(iter(source_weights)))
            torch.save({'model': source_weights}, checkpoint)
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                train_script.load_pretrained_encoder(target, checkpoint)

        incompatible = self._e2(encoder_norm='group')
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'pretrain.pt'
            torch.save({'model': source_weights}, checkpoint)
            with self.assertRaisesRegex(ValueError, 'encoder_norm instance'):
                train_script.load_pretrained_encoder(incompatible, checkpoint)

    def test_scheduled_freeze_keeps_optimizer_params_and_unfreezes(self):
        model = self._e2()
        args = SimpleNamespace(
            freeze_encoder=False,
            freeze_decoder=False,
            freeze_encoder_epochs=2,
            encoder_lr_mult=1.0,
            decoder_lr_mult=1.0,
            learning_rate=1e-4,
            weight_decay=1e-4,
        )
        train_script.apply_parameter_freezing(model, args)
        optimizer = train_script.build_optimizer(model, args)
        encoder_params = list(model.encoder.parameters())
        optimizer_params = [parameter for group in optimizer.param_groups for parameter in group['params']]

        self.assertTrue(all(not parameter.requires_grad for parameter in encoder_params))
        self.assertTrue(all(any(parameter is item for item in optimizer_params) for parameter in encoder_params))
        self.assertFalse(train_script.unfreeze_encoder_after_warmup(model, optimizer, 1, args))
        self.assertTrue(train_script.unfreeze_encoder_after_warmup(model, optimizer, 2, args))
        self.assertTrue(all(parameter.requires_grad for parameter in encoder_params))

    def test_resume_rejects_different_encoder_config(self):
        model = self._e2()
        matching = dict(model.model_config)
        train_script.validate_resume_model_config(matching, model)
        changed = dict(matching, encoder_input_axes='xyz')

        with self.assertRaisesRegex(ValueError, 'architecture config'):
            train_script.validate_resume_model_config(changed, model)

        with self.assertRaisesRegex(ValueError, 'legacy conv encoder'):
            train_script.validate_resume_model_config(None, model)

    def test_tokenizer_adapter_loads_architecture_config(self):
        model = self._e2().eval()
        payload = train_script.build_checkpoint_payload(model)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'e2.pt'
            torch.save(payload, checkpoint)
            adapter = VaeLatentAdapter(checkpoint, device='cpu')
            cube = np.random.default_rng(10).normal(size=(32, 32, 64)).astype(np.float32)
            actual = adapter.encode_batch(cube[None])[0]
            actual_geo = adapter.encode_geo_batch(cube[None])[0]
            with torch.no_grad():
                mu = model.encoder(torch.from_numpy(cube[None, None]))[0]
                expected = mu[0].numpy()
                expected_geo = model.encode_geo(mu)[0].numpy()

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(actual_geo, expected_geo, rtol=1e-5, atol=1e-6)

    def test_tokenizer_zgeo_round_trip_for_e1_e3_e4(self):
        candidates = (
            VAE3D(patch_shape=(32, 32, 64), encoder_arch='residual', geology_projection=True).eval(),
            VAE3D(
                patch_shape=(32, 32, 64), encoder_arch='resnetv2',
                encoder_hidden_dims=(8, 16, 32), encoder_stage_blocks=(1, 1, 1),
                encoder_norm='instance', encoder_stem='light', encoder_depth_profile='deeper',
                encoder_input_axes='xyz', decoder_hidden_dims=(16, 16, 16), geology_projection=True,
            ).eval(),
            VAE3D(
                patch_shape=(32, 32, 64), encoder_arch='resnetv2',
                encoder_hidden_dims=(8, 16, 32), encoder_stage_blocks=(1, 1, 1),
                encoder_norm='instance', encoder_stem='pretrain_v2', encoder_depth_profile='deeper',
                encoder_input_axes='zxy', geology_projection=True,
            ).eval(),
        )
        inputs = np.random.default_rng(20).normal(size=(1, 32, 32, 64)).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            for index, model in enumerate(candidates):
                checkpoint = Path(tmp) / f'e{index}.pt'
                torch.save(train_script.build_checkpoint_payload(model), checkpoint)
                adapter = VaeLatentAdapter(checkpoint, device='cpu')
                actual = adapter.encode_geo_batch(inputs)
                with torch.no_grad():
                    mu = model.encoder(torch.from_numpy(inputs[:, None]))[0]
                    expected = model.encode_geo(mu).numpy()
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_diagnostic_selection_is_seeded_and_unique(self):
        indices = diagnostics.select_indices(100, 12, 20261003)
        self.assertEqual(indices, diagnostics.select_indices(100, 12, 20261003))
        self.assertEqual(len(set(indices)), 12)
        with self.assertRaises(ValueError):
            diagnostics.select_indices(10, 11, 1)

    def test_explicit_extrema_input_preserves_clean_targets(self):
        targets = torch.tensor([1.0, 3.0, 2.0, -1.0, 0.0, 4.0, 2.0, 1.0]).reshape(1, 1, 1, 1, 8)
        before = targets.clone()
        masked = diagnostics.evaluation_inputs(targets, 'extrema')
        expected = train_script.keep_trace_extrema_only(targets[0, 0].numpy())
        np.testing.assert_array_equal(masked[0, 0].numpy(), expected)
        torch.testing.assert_close(targets, before)
        self.assertIs(diagnostics.evaluation_inputs(targets, 'clean'), targets)

    def test_tokenizer_inputs_match_production_and_preserve_target_rng(self):
        generator = torch.Generator().manual_seed(22)
        typical = torch.randn(1, 32, 32, 64, generator=generator)
        targets = torch.stack((typical, typical * 1e-9, torch.zeros_like(typical)))
        before = targets.clone()
        rng = torch.get_rng_state().clone()
        actual = diagnostics.evaluation_inputs(targets, 'tokenizer')
        expected = np.stack([preprocess_for_token(cube[0].numpy()) for cube in targets])[:, None]
        np.testing.assert_array_equal(actual.numpy(), expected)
        self.assertTrue(actual.is_contiguous())
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(targets, before)
        torch.testing.assert_close(torch.get_rng_state(), rng)

    def test_tokenizer_adapter_parity_uses_prepared_inputs_once(self):
        model = VAE3D(base_ch=2, latent_dim=4, patch_shape=(8, 8, 8), geology_projection=True).eval()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'checkpoint.pt'
            torch.save(train_script.build_checkpoint_payload(model), destination)
            adapter = VaeLatentAdapter(destination, device='cpu')
            result = diagnostics.verify_tokenizer_adapter_parity(adapter, torch.randn(2, 1, 8, 8, 8))
        self.assertTrue(result['passed'])
        self.assertEqual(result['preprocessing_count'], 1)

    def test_contract_evaluation_reports_volume_balanced_mae(self):
        model = VAE3D(base_ch=2, latent_dim=4, patch_shape=(8, 8, 8)).eval()
        targets = torch.randn(3, 1, 8, 8, 8)
        result = diagnostics.evaluate(model, targets, torch.device('cpu'), 2, 10, None,
                                      'tokenizer', [0, 0, 1], ['first', 'second'])
        grouped = result['per_volume']['mean']
        self.assertEqual(grouped['volumes']['0']['count'], 2)
        self.assertEqual(grouped['volumes']['1']['count'], 1)
        self.assertAlmostEqual(grouped['volume_balanced_mae'],
                               (grouped['volumes']['0']['mae'] + grouped['volumes']['1']['mae']) / 2)

    def test_sampler_exposure_is_not_full_epochs_or_observed_coverage(self):
        result = diagnostics.exposure_summary(108000)
        self.assertEqual(result['examples_per_epoch'], 5400)
        self.assertEqual(result['total_exposures'], 810000)
        self.assertEqual(result['corpus_equivalent_passes'], 7.5)
        self.assertIn('unknown', result['actual_historical_index_coverage'])

    def test_p7_sample_schedule_is_independent_and_tracks_consumed_only(self):
        control = p7_control.PairedBatchSampler(100, 4, 3, 10)
        candidate = p7_control.PairedBatchSampler(100, 4, 3, 10)
        control.set_epoch(2)
        candidate.set_epoch(2)
        schedule = list(control)
        torch.randn(1000)
        self.assertEqual(schedule, list(candidate))
        control.record_consumed()
        self.assertEqual(control.consumed, schedule[0])
        self.assertEqual(len(control.consumed), 4)

    def test_p7_state_round_trip_restores_rng_optimizer_and_scheduler(self):
        parameter = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.AdamW([parameter], lr=0.01)
        parameter.sum().backward()
        optimizer.step()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        scheduler.step(0.5)
        generator = torch.Generator().manual_seed(12)
        early = SimpleNamespace(best_val_loss=0.5, epochs_without_improvement=2)
        state = p7_control.capture_training_state(optimizer, scheduler, generator, early)
        expected = torch.rand(3)
        expected_numpy = np.random.random(3)
        expected_python = p7_control.random.random()
        expected_loader = torch.rand(3, generator=generator)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'state.pt'
            torch.save(state, destination)
            loaded = torch.load(destination, weights_only=True)
        optimizer.param_groups[0]['lr'] = 0.5
        early.best_val_loss = 99
        p7_control.restore_training_state(loaded, optimizer, scheduler, generator, early)
        torch.testing.assert_close(torch.rand(3), expected)
        np.testing.assert_array_equal(np.random.random(3), expected_numpy)
        self.assertEqual(p7_control.random.random(), expected_python)
        torch.testing.assert_close(torch.rand(3, generator=generator), expected_loader)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.01)
        self.assertEqual(early.best_val_loss, 0.5)

    def test_p7_shared_initialization_retains_original_decoder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'initial'
            p7_control.prepare_initializations(root, 10)
            control = torch.load(root / 'e0_initial.pt', weights_only=True)
            candidate = torch.load(root / 'e1_initial.pt', weights_only=True)
            for key, value in control['model_state_dict'].items():
                if key.startswith(('encoder.fc_', 'decoder.', 'geology_head.', 'geology_classifier.')):
                    torch.testing.assert_close(value, candidate['model_state_dict'][key], rtol=0, atol=0)
            self.assertIsNone(candidate['model_config']['decoder_hidden_dims'])

    def test_p7_gate_blocks_regression_and_subset_reports(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = {'reconstruction_gate': {'full_dataset_coverage': True}, 'arms': {}}
            for arm, error in [('p0b', 1.0), ('e0', 1.0), ('e1', 1.03)]:
                report['arms'][arm] = {'evaluation': {
                    'synthetic': {'sampled': {'recipe_total': error}},
                    'real': {'sampled': {'mae': 1.0}},
                }}
            for mode in ('clean', 'extrema'):
                (root / f'reconstruction_{mode}.json').write_text(json.dumps(report))
            self.assertFalse(p7_control.reconstruction_gate(root)['geology_eligible'])
            report['reconstruction_gate']['full_dataset_coverage'] = False
            (root / 'reconstruction_clean.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'full dataset'):
                p7_control.reconstruction_gate(root)

    def test_same_seed_tokenizer_gate_requires_full_coverage_and_respects_two_percent(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evaluation.json'
            report = {'settings': {'input_mode': 'tokenizer', 'contract_audit': True},
                      'reconstruction_gate': {'full_dataset_coverage': True}, 'arms': {}}
            for arm, factor in [('p0b', 1.0), ('e0', 1.03), ('e1', 1.019)]:
                report['arms'][arm] = {'evaluation': {
                    'synthetic': {'sampled': {'recipe_total': 2.0 * factor}},
                    'real': {'sampled': {'mae': 1.0 * factor}},
                }}
            path.write_text(json.dumps(report))
            result = p7_control.tokenizer_reconstruction_gate(path)
            self.assertTrue(result['passed'])
            self.assertTrue((Path(directory) / 'tokenizer_reconstruction_gate.json').exists())
            report['arms']['e1']['evaluation']['real']['sampled']['mae'] = 1.021
            path.write_text(json.dumps(report))
            self.assertFalse(p7_control.tokenizer_reconstruction_gate(path)['passed'])
            report['reconstruction_gate']['full_dataset_coverage'] = False
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'full dataset'):
                p7_control.tokenizer_reconstruction_gate(path)

    def test_seed2_tokenizer_gate_uses_same_seed_control_not_seed1(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'seed2.json'
            arms={}
            for arm,value in [('p0b',1.0),('e0',1.03),('e1',1.01)]:
                arms[arm]={'evaluation':{'synthetic':{'sampled':{'recipe_total':value}},
                                         'real':{'sampled':{'mae':value}}}}
            path.write_text(json.dumps({'settings':{'input_mode':'tokenizer','contract_audit':True},
                                        'reconstruction_gate':{'full_dataset_coverage':True},'arms':arms}))
            result=p7_control.tokenizer_reconstruction_gate(path,candidate='e1',control='p0b')
            self.assertTrue(result['passed'])
            self.assertEqual(result['control'],'p0b')
            arms['e1']['evaluation']['real']['sampled']['mae']=1.021
            path.write_text(json.dumps({'settings':{'input_mode':'tokenizer','contract_audit':True},
                                        'reconstruction_gate':{'full_dataset_coverage':True},'arms':arms}))
            self.assertFalse(p7_control.tokenizer_reconstruction_gate(path,candidate='e1',control='p0b')['passed'])

    def test_tokenizer_gate_accepts_single_candidate_against_control(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'candidate.json'
            arms={}
            for arm,value in [('p0b',1.0),('r006',1.01)]:
                arms[arm]={'evaluation':{'synthetic':{'sampled':{'recipe_total':value}},
                                         'real':{'sampled':{'mae':value}}}}
            path.write_text(json.dumps({'settings':{'input_mode':'tokenizer','contract_audit':True},
                                        'reconstruction_gate':{'full_dataset_coverage':True},'arms':arms}))
            result=p7_control.tokenizer_reconstruction_gate(path,candidate='r006',control='p0b')
            self.assertTrue(result['passed'])

    def test_training_diagnostic_is_disabled_or_skips_nonmilestones(self):
        model = self._e2()
        self.assertIsNone(diagnostics.run_training_diagnostic(model, SimpleNamespace(), torch.device('cpu'), 90, None))
        self.assertIsNone(diagnostics.run_training_diagnostic(model, SimpleNamespace(reconstruction_diagnostic_interval=30), torch.device('cpu'), 61, None))

    def test_training_milestone_preserves_weights_mode_and_rng(self):
        model = VAE3D(base_ch=2, latent_dim=4, patch_shape=(8, 8, 8)).train()
        targets = torch.randn(2, 1, 8, 8, 8)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        rng_before = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(reconstruction_diagnostic_interval=30, reconstruction_diagnostic_samples=2,
                                   validation_data='fixture', real_test_data=None, out_dir=directory,
                                   reconstruction_diagnostic_modes=['tokenizer'])
            with patch.object(diagnostics, 'load_patches', return_value=(targets, {'indices': [0, 1]})):
                result = diagnostics.run_training_diagnostic(model, args, torch.device('cpu'), 90, None)
            self.assertTrue((Path(directory) / 'reconstruction_diagnostic_epoch90.json').exists())
            self.assertEqual(result['epoch'], 90)
            self.assertEqual(result['input_mode'], 'tokenizer')
            self.assertIn('tokenizer', result['evaluation']['synthetic']['metrics_by_input_mode'])
        self.assertTrue(model.training)
        torch.testing.assert_close(torch.get_rng_state(), rng_before)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_diagnostic_recipe_matches_training_kl_normalization(self):
        targets = torch.ones(2, 1, 8, 8, 8)
        prediction = torch.zeros_like(targets)
        mu = torch.ones(2, 4)
        logvar = torch.zeros_like(mu)
        metrics = diagnostics.component_metrics(prediction, targets, mu, logvar, lambda pred, target: pred.new_tensor(2.0))
        self.assertEqual(metrics['mae'], 1.0)
        self.assertEqual(metrics['kl_per_example'], 2.0)
        self.assertAlmostEqual(metrics['kl_per_voxel'], 4 / targets.numel())
        self.assertAlmostEqual(metrics['recipe_total'], 1.2 + 0.001 * 4 / targets.numel(), places=6)

    def test_diagnostic_asymmetric_axis_fixture(self):
        result = diagnostics.axis_audit(self._e2().eval(), torch.device('cpu'))
        self.assertTrue(result['passed'])

    def test_diagnostic_fit_does_not_mutate_checkpoint_modules(self):
        model = VAE3D(base_ch=2, latent_dim=4, patch_shape=(8, 8, 8)).eval()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        result = diagnostics.fit_decoder(model, torch.randn(2, 1, 8, 8, 8), torch.device('cpu'), 2, 10)
        self.assertTrue(result['encoder_unchanged'])
        self.assertTrue(np.isfinite(result['history'][-1]['mae']))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
