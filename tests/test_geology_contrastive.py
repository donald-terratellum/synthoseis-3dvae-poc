import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from scripts import train as train_script
from src.model import VAE3D, GeologyProjectionHead
from src.tokenizer.core.model_adapter import VaeLatentAdapter


class GeologyProjectionHeadTests(unittest.TestCase):
    def test_projection_head_outputs_unit_norm_embeddings(self):
        head = GeologyProjectionHead(latent_dim=128, proj_hidden=64, proj_dim=32)
        mu = torch.randn(7, 128)
        z_geo = head(mu)
        self.assertEqual(tuple(z_geo.shape), (7, 32))
        norms = z_geo.norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_vae_encode_geo_uses_head_when_enabled(self):
        model = VAE3D(
            patch_shape=(8, 8, 8),
            geology_projection=True,
            geology_proj_hidden=32,
            geology_proj_dim=16,
        )
        self.assertIsNotNone(model.geology_head)
        mu = torch.randn(4, model.latent_dim)
        z_geo = model.encode_geo(mu)
        self.assertEqual(tuple(z_geo.shape), (4, 16))
        norms = z_geo.norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_vae_encode_geo_falls_back_to_normalized_mu(self):
        model = VAE3D(patch_shape=(8, 8, 8), geology_projection=False)
        self.assertIsNone(model.geology_head)
        mu = torch.randn(3, model.latent_dim)
        z_geo = model.encode_geo(mu)
        self.assertEqual(tuple(z_geo.shape), (3, model.latent_dim))
        norms = z_geo.norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))


class SupervisedContrastiveLossTests(unittest.TestCase):
    def _unit(self, vectors):
        t = torch.tensor(vectors, dtype=torch.float32)
        return torch.nn.functional.normalize(t, dim=1)

    def test_perfectly_clustered_embeddings_have_lower_loss_than_scrambled(self):
        # Two tight clusters aligned to their labels vs. the same points mislabeled.
        clustered = self._unit(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
                [0.01, 0.99],
            ]
        )
        labels_aligned = np.asarray([1, 1, 2, 2], dtype=np.int64)
        labels_scrambled = np.asarray([1, 2, 1, 2], dtype=np.int64)

        loss_aligned = train_script.compute_supervised_contrastive_loss(
            clustered, labels_aligned, temperature=0.1
        )
        loss_scrambled = train_script.compute_supervised_contrastive_loss(
            clustered, labels_scrambled, temperature=0.1
        )
        self.assertLess(float(loss_aligned), float(loss_scrambled))

    def test_background_label_excluded_returns_zero_when_no_positive(self):
        embeddings = self._unit([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        # All background (label 0) => no valid positives => zero loss.
        labels = np.asarray([0, 0, 0], dtype=np.int64)
        loss = train_script.compute_supervised_contrastive_loss(embeddings, labels)
        self.assertEqual(float(loss), 0.0)

    def test_single_positive_pair_is_finite_and_nonnegative(self):
        embeddings = self._unit([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        labels = np.asarray([1, 1, 0], dtype=np.int64)
        loss = train_script.compute_supervised_contrastive_loss(embeddings, labels)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertGreaterEqual(float(loss), 0.0)

    def test_loss_is_differentiable_wrt_embeddings(self):
        raw = torch.randn(6, 8, requires_grad=True)
        embeddings = torch.nn.functional.normalize(raw, dim=1)
        labels = np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int64)
        loss = train_script.compute_supervised_contrastive_loss(embeddings, labels)
        loss.backward()
        self.assertIsNotNone(raw.grad)
        self.assertTrue(torch.isfinite(raw.grad).all())


class GeologyAdapterProjectionTests(unittest.TestCase):
    def test_adapter_loads_projection_checkpoint_and_encodes_unit_geo(self):
        patch_shape = (8, 8, 8)
        model = VAE3D(
            base_ch=8,
            latent_dim=32,
            patch_shape=patch_shape,
            geology_projection=True,
            geology_proj_hidden=32,
            geology_proj_dim=16,
        )
        payload = train_script.build_checkpoint_payload(model)
        with TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / 'vae_proj.pt'
            torch.save(payload, ckpt_path)
            adapter = VaeLatentAdapter(ckpt_path, device='cpu')
            self.assertTrue(adapter.geology_projection)
            self.assertEqual(adapter.geology_proj_dim, 16)

            cube = np.random.randn(*patch_shape).astype(np.float32)
            z_geo = adapter.encode_geo_cube(cube)
            self.assertEqual(z_geo.shape, (16,))
            self.assertAlmostEqual(float(np.linalg.norm(z_geo)), 1.0, places=4)

    def test_adapter_without_projection_falls_back_to_normalized_mu(self):
        patch_shape = (8, 8, 8)
        model = VAE3D(base_ch=8, latent_dim=32, patch_shape=patch_shape)
        payload = train_script.build_checkpoint_payload(model)
        with TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / 'vae_plain.pt'
            torch.save(payload, ckpt_path)
            adapter = VaeLatentAdapter(ckpt_path, device='cpu')
            self.assertFalse(adapter.geology_projection)
            cube = np.random.randn(*patch_shape).astype(np.float32)
            z_geo = adapter.encode_geo_cube(cube)
            self.assertEqual(z_geo.shape, (32,))
            self.assertAlmostEqual(float(np.linalg.norm(z_geo)), 1.0, places=4)


class ParameterFreezingTests(unittest.TestCase):
    def _args(self, **overrides):
        base = dict(
            freeze_encoder=False,
            freeze_decoder=False,
            encoder_lr_mult=1.0,
            decoder_lr_mult=1.0,
            learning_rate=1e-4,
            weight_decay=1e-4,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_freeze_encoder_excludes_encoder_params_from_optimizer(self):
        model = VAE3D(
            base_ch=8,
            latent_dim=32,
            patch_shape=(8, 8, 8),
            geology_projection=True,
            geology_proj_hidden=16,
            geology_proj_dim=8,
        )
        args = self._args(freeze_encoder=True)
        frozen = train_script.apply_parameter_freezing(model, args)
        self.assertEqual(frozen, ['encoder'])
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))

        opt = train_script.build_optimizer(model, args)
        opt_param_ids = {id(p) for group in opt.param_groups for p in group['params']}
        encoder_ids = {id(p) for p in model.encoder.parameters()}
        head_ids = {id(p) for p in model.geology_head.parameters()}
        self.assertTrue(opt_param_ids.isdisjoint(encoder_ids))
        self.assertTrue(head_ids.issubset(opt_param_ids))

    def test_freezing_everything_raises(self):
        model = VAE3D(base_ch=8, latent_dim=32, patch_shape=(8, 8, 8))
        args = self._args(freeze_encoder=True, freeze_decoder=True)
        train_script.apply_parameter_freezing(model, args)
        with self.assertRaises(ValueError):
            train_script.build_optimizer(model, args)


if __name__ == '__main__':
    unittest.main()
