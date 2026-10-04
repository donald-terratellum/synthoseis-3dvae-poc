import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts import train as train_script
from src.model import VAE3D
from src.tokenizer.core.model_adapter import VaeLatentAdapter


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


if __name__ == '__main__':
    unittest.main()
