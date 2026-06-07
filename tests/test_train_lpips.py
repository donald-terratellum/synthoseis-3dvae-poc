import copy
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest import mock

import torch
import torch.nn as nn

from scripts import train as train_script


class _RecordingLPIPS(nn.Module):
    def __init__(self, net='alex'):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(self, pred, target):
        self.calls.append(
            {
                'pred_shape': tuple(pred.shape),
                'target_shape': tuple(target.shape),
                'pred_max_abs': float(pred.detach().abs().max().item()),
                'target_max_abs': float(target.detach().abs().max().item()),
                'target_requires_grad': bool(target.requires_grad),
            }
        )
        return ((pred - target) ** 2).mean(dim=(1, 2, 3), keepdim=True)


class _TinyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.75))

    def forward(self, x, return_deep_supervision=False):
        recon = x * self.scale
        mu = x.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1) * self.scale
        logvar = torch.zeros_like(mu)
        if return_deep_supervision:
            return recon, mu, logvar, (recon, recon, recon)
        return recon, mu, logvar


class _TinyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        pooled = x.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1)
        return self.linear(pooled)


class TrainLPIPSTests(unittest.TestCase):
    def test_compute_vae_losses_skips_lpips_when_weight_zero(self):
        recon = torch.randn(2, 1, 8, 8, 8)
        target = torch.randn_like(recon)
        mu = torch.zeros(2, 4)
        logvar = torch.zeros(2, 4)
        lpips_mock = mock.Mock(side_effect=AssertionError('lpips should not be called'))

        total_loss, rec_loss, kld, lpips_loss = train_script.compute_vae_losses(
            recon,
            target,
            mu,
            logvar,
            kl_weight=1e-3,
            rec_loss_fn=nn.MSELoss(),
            lpips_loss_fn=lpips_mock,
            lpips_weight=0.0,
        )

        self.assertEqual(lpips_mock.call_count, 0)
        self.assertEqual(float(lpips_loss.item()), 0.0)
        self.assertTrue(torch.allclose(total_loss, rec_loss + (1e-3 * kld)))

    def test_train_one_epoch_gan_metrics_match_when_lpips_weight_zero(self):
        torch.manual_seed(7)
        inputs = torch.randn(2, 1, 8, 8, 8)
        targets = torch.randn_like(inputs)
        dataloader = [(inputs, targets)]

        base_generator = _TinyGenerator()
        base_discriminator = _TinyDiscriminator()

        generator_a = copy.deepcopy(base_generator)
        generator_b = copy.deepcopy(base_generator)
        discriminator_a = copy.deepcopy(base_discriminator)
        discriminator_b = copy.deepcopy(base_discriminator)

        optimizer_a = torch.optim.SGD(generator_a.parameters(), lr=1e-2)
        optimizer_b = torch.optim.SGD(generator_b.parameters(), lr=1e-2)
        disc_optimizer_a = torch.optim.SGD(discriminator_a.parameters(), lr=1e-2)
        disc_optimizer_b = torch.optim.SGD(discriminator_b.parameters(), lr=1e-2)
        lpips_mock = mock.Mock(side_effect=AssertionError('lpips should not be called'))

        torch.manual_seed(11)
        baseline_metrics = train_script.train_one_epoch(
            generator_a,
            discriminator_a,
            dataloader,
            device='cpu',
            optimizer=optimizer_a,
            disc_optimizer=disc_optimizer_a,
            steps_per_epoch=1,
            grad_clip=2.0,
            kl_weight=1e-3,
            gan_weight=1e-3,
            rec_loss_fn=nn.MSELoss(),
        )

        torch.manual_seed(11)
        lpips_zero_metrics = train_script.train_one_epoch(
            generator_b,
            discriminator_b,
            dataloader,
            device='cpu',
            optimizer=optimizer_b,
            disc_optimizer=disc_optimizer_b,
            steps_per_epoch=1,
            grad_clip=2.0,
            kl_weight=1e-3,
            gan_weight=1e-3,
            rec_loss_fn=nn.MSELoss(),
            lpips_loss_fn=lpips_mock,
            lpips_weight=0.0,
        )

        self.assertEqual(lpips_mock.call_count, 0)
        for baseline_value, lpips_value in zip(baseline_metrics[:-1], lpips_zero_metrics[:-1]):
            self.assertAlmostEqual(float(baseline_value), float(lpips_value), places=6)

    def test_slice_lpips_normalizes_upsamples_and_preserves_prediction_grad(self):
        fake_lpips_module = types.SimpleNamespace(LPIPS=_RecordingLPIPS)
        with mock.patch.object(train_script, 'lpips_lib', fake_lpips_module):
            loss_module = train_script.SliceLPIPSLoss(min_spatial_size=64)

        self.assertFalse(loss_module.network.scale.requires_grad)

        recon = torch.linspace(-3.0, 3.0, steps=32 * 32 * 32, dtype=torch.float32).reshape(1, 1, 32, 32, 32)
        recon.requires_grad_(True)
        target = (-0.5 * recon.detach()).clone().requires_grad_(True)

        loss = loss_module(recon, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        self.assertIsNotNone(recon.grad)
        self.assertIsNone(target.grad)
        self.assertEqual(len(loss_module.network.calls), 2)
        for call in loss_module.network.calls:
            self.assertEqual(call['pred_shape'], (1, 3, 64, 64))
            self.assertEqual(call['target_shape'], (1, 3, 64, 64))
            self.assertLessEqual(call['pred_max_abs'], 1.0)
            self.assertLessEqual(call['target_max_abs'], 1.0)
            self.assertFalse(call['target_requires_grad'])

    def test_metrics_csv_migration_backfills_new_lpips_columns(self):
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'training_metrics.csv'
            with csv_path.open('w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    'epoch',
                    'examples_this_epoch',
                    'cumulative_examples',
                    'train_loss',
                    'val_loss',
                    'kl_weight',
                    'learning_rate',
                    'discriminator_learning_rate',
                    'gan_weight',
                    'g_gan_loss',
                    'd_gan_loss',
                    'd_gan_acc_pct',
                    'best_model',
                ])
                writer.writerow(['1', '100', '100', '0.1', '0.2', '0.001', '0.0001', '0.0001', '0.001', '0.0', '0.0', '50.0', 'best'])

            migrated = train_script.migrate_metrics_csv_if_needed(csv_path, train_script.METRICS_CSV_COLUMNS)

            self.assertTrue(migrated)
            with csv_path.open('r', newline='') as csv_file:
                rows = list(csv.reader(csv_file))

            self.assertEqual(rows[0], train_script.METRICS_CSV_COLUMNS)
            self.assertEqual(rows[1][0], '1')
            self.assertEqual(rows[1][4], '')
            self.assertEqual(rows[1][6], '')


if __name__ == '__main__':
    unittest.main()