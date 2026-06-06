import unittest

import torch
import torch.nn.functional as F

from src.deep_supervision import DeepSupervisionLoss
from src.model import VAE3D


class DeepSupervisionTests(unittest.TestCase):
    def test_forward_outputs_and_loss_weights(self):
        torch.manual_seed(0)
        model = VAE3D(patch_shape=(32, 32, 32), deep_supervision=True)
        model.train()

        x = torch.randn(2, 1, 32, 32, 32)
        y = torch.randn(2, 1, 32, 32, 32)

        recon, mu, logvar, ds_outputs = model(x, return_deep_supervision=True)

        self.assertEqual(recon.shape, (2, 1, 32, 32, 32))
        self.assertEqual(mu.shape, (2, 128))
        self.assertEqual(logvar.shape, (2, 128))
        self.assertEqual(len(ds_outputs), 3)
        for pred in ds_outputs:
            self.assertEqual(pred.shape, (2, 1, 32, 32, 32))

        ds_loss = DeepSupervisionLoss(torch.nn.MSELoss(), weights=(1.0, 0.5, 0.25))
        loss = ds_loss(ds_outputs, y)

        expected = (
            1.0 * F.mse_loss(ds_outputs[0], y)
            + 0.5 * F.mse_loss(ds_outputs[1], y)
            + 0.25 * F.mse_loss(ds_outputs[2], y)
        )
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))

    def test_inference_contract_and_grad_flow(self):
        torch.manual_seed(1)
        model = VAE3D(patch_shape=(32, 32, 32), deep_supervision=True)

        x = torch.randn(2, 1, 32, 32, 32)
        y = torch.randn(2, 1, 32, 32, 32)

        model.eval()
        output = model(x)
        self.assertEqual(len(output), 3)
        recon_eval, mu_eval, logvar_eval = output
        self.assertEqual(recon_eval.shape, (2, 1, 32, 32, 32))
        self.assertEqual(mu_eval.shape, (2, 128))
        self.assertEqual(logvar_eval.shape, (2, 128))

        model.train()
        ds_loss = DeepSupervisionLoss(torch.nn.MSELoss(), weights=(1.0, 0.5, 0.25))
        recon, mu, logvar, ds_outputs = model(x, return_deep_supervision=True)
        rec_loss = ds_loss(ds_outputs, y)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / y.numel()
        total_loss = rec_loss + (1e-3 * kld)

        model.zero_grad(set_to_none=True)
        total_loss.backward()

        self.assertIsNotNone(model.decoder.aux_head_mid.weight.grad)
        self.assertIsNotNone(model.decoder.aux_head_coarse.weight.grad)
        self.assertIsNotNone(model.encoder.fc_mu.weight.grad)


if __name__ == "__main__":
    unittest.main()
