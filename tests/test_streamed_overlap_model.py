import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.streamed_overlap_model import (
    OrderedCoverageHead,
    StreamedPairVAE,
    build_optimizer,
    fresh_e1_model,
    joint_loss,
)
from src.model import Decoder, VAE3D
from src.tokenizer.core.model_adapter import VaeLatentAdapter
from scripts.train import build_checkpoint_payload


class StreamedPairModelTests(unittest.TestCase):
    def test_matched_fresh_vae_initialization_does_not_consume_cpu_rng(self):
        torch.manual_seed(31)
        before = torch.get_rng_state().clone()
        control = fresh_e1_model("S1", model_seed=7, head_seed=11)
        candidate = fresh_e1_model("S3", model_seed=7, head_seed=11)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(candidate.vae.latent_dim, 128)
        self.assertEqual(candidate.vae.encoder_arch, "residual")
        self.assertIsInstance(candidate.vae.decoder, Decoder)
        for key, value in control.vae.state_dict().items():
            self.assertTrue(torch.equal(value, candidate.vae.state_dict()[key]), key)
        self.assertIsNone(control.overlap_head)
        self.assertIsNone(control.coverage_head)

    def test_ordered_head_can_produce_directional_scores(self):
        head = OrderedCoverageHead(latent_dim=2, hidden_dim=1)
        with torch.no_grad():
            head.network[0].weight.zero_()
            head.network[0].bias.zero_()
            head.network[0].weight[0, 0] = 1
            head.network[2].weight.fill_(1)
            head.network[2].bias.zero_()
        a = torch.tensor([[2.0, 0.0]])
        b = torch.tensor([[0.0, 0.0]])
        self.assertGreater(head(a, b).item(), head(b, a).item())

    def test_vae_checkpoint_encoder_matches_tokenizer_adapter(self):
        torch.manual_seed(19)
        vae = VAE3D(base_ch=2, latent_dim=4, patch_shape=(8, 8, 8), encoder_arch="residual")
        vae.eval()
        cubes = np.random.default_rng(4).normal(size=(2, 8, 8, 8)).astype(np.float32)
        with torch.no_grad():
            expected, _ = vae.encoder(torch.from_numpy(cubes[:, None]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vae.pt"
            torch.save(build_checkpoint_payload(vae), path)
            adapter = VaeLatentAdapter(path, device="cpu")
            actual = adapter.encode_batch(cubes)
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-6, atol=1e-6)

    def test_each_active_objective_reaches_encoder_and_its_head(self):
        torch.manual_seed(13)
        vae = VAE3D(base_ch=2, latent_dim=4, patch_shape=(8, 8, 8), encoder_arch="residual")
        model = StreamedPairVAE(vae, "S3")
        a = torch.randn(2, 1, 8, 8, 8)
        b = torch.randn_like(a)
        dense_a = torch.randn_like(a)
        output = model(a, b)
        losses = joint_loss(
            output, dense_a, lambda x, y: F.mse_loss(x, y),
            overlap_targets=torch.tensor([0.0, 0.5]),
            coverage_targets=torch.tensor([0.9, float("nan")]),
            coverage_valid=torch.tensor([True, False]),
        )
        self.assertEqual(losses.valid_coverage, 1)
        encoder = tuple(model.vae.encoder.parameters())
        decoder = tuple(model.vae.decoder.parameters())

        def nonzero(loss, parameters):
            grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            return any(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
                       for grad in grads)

        for loss in (losses.reconstruction, losses.lpips):
            self.assertTrue(nonzero(loss, encoder))
            self.assertTrue(nonzero(loss, decoder))
        self.assertTrue(nonzero(losses.kl, encoder))
        self.assertIsNotNone(losses.overlap)
        self.assertIsNotNone(losses.coverage)
        self.assertTrue(nonzero(losses.overlap, encoder))
        self.assertTrue(nonzero(losses.overlap, tuple(model.overlap_head.parameters())))
        self.assertTrue(nonzero(losses.coverage, encoder))
        self.assertTrue(nonzero(losses.coverage, tuple(model.coverage_head.parameters())))
        losses.total.backward()
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for parameter in model.vae.encoder.parameters()))
        optimizer = build_optimizer(model)
        rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
        self.assertEqual(rates, {
            "encoder": 5e-5, "decoder": 5e-4, "overlap": 5e-4, "coverage": 5e-4,
        })
        self.assertTrue(all(parameter.requires_grad for parameter in model.vae.encoder.parameters()))
        before = model.vae.encoder.fc_mu.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.vae.encoder.fc_mu.weight))

    def test_missing_targets_and_nonfinite_valid_values_fail(self):
        torch.manual_seed(7)
        model = StreamedPairVAE(
            VAE3D(base_ch=2, latent_dim=4, patch_shape=(8, 8, 8), encoder_arch="residual"),
            "S3",
        )
        a = torch.randn(2, 1, 8, 8, 8)
        output = model(a, torch.randn_like(a))
        with self.assertRaisesRegex(ValueError, "overlap head"):
            joint_loss(output, a, lambda x, y: F.mse_loss(x, y))
        with self.assertRaisesRegex(ValueError, "Valid coverage"):
            joint_loss(output, a, lambda x, y: F.mse_loss(x, y),
                       torch.zeros(2), torch.tensor([float("nan"), 0]),
                       torch.tensor([True, False]))


if __name__ == "__main__":
    unittest.main()
