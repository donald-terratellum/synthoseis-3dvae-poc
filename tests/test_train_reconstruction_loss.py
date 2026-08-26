import unittest

import torch

from scripts import train as train_script
from src.deep_supervision import DeepSupervisionLoss


class TrainReconstructionLossTests(unittest.TestCase):
    def test_mae_loss_and_per_example_scores_match_voxelwise_l1(self):
        prediction = torch.tensor([[[[[1.0, -1.0]]]], [[[[2.0, 4.0]]]]])
        target = torch.tensor([[[[[0.0, 1.0]]]], [[[[1.0, 1.0]]]]])

        loss_fn = train_script.build_reconstruction_loss('mae')
        expected_per_example = torch.tensor([1.5, 2.0])

        self.assertTrue(torch.allclose(loss_fn(prediction, target), expected_per_example.mean()))
        self.assertTrue(
            torch.allclose(
                train_script.compute_per_example_recon_loss(
                    prediction,
                    target,
                    loss_type='mae',
                    mse_weight=0.6,
                ),
                expected_per_example,
            )
        )

    def test_mae_is_used_by_deep_supervision(self):
        target = torch.zeros(1, 1, 1, 1, 2)
        outputs = (torch.ones_like(target), 2.0 * torch.ones_like(target))
        weights = (1.0, 0.5)
        loss_fn = train_script.build_reconstruction_loss('mae')

        scalar_loss = DeepSupervisionLoss(loss_fn, weights=weights)(outputs, target)
        per_example_loss = train_script.compute_per_example_deep_supervision_recon_loss(
            outputs,
            target,
            weights=weights,
            loss_type='mae',
            mse_weight=0.6,
        )

        self.assertAlmostEqual(float(scalar_loss.item()), 2.0)
        self.assertTrue(torch.allclose(per_example_loss, torch.tensor([2.0])))

    def test_mse_pmse_remains_default_loss_family(self):
        loss_fn = train_script.build_reconstruction_loss('mse_pmse', mse_weight=0.6)

        self.assertIsInstance(loss_fn, train_script.CombinedReconLoss)
        self.assertEqual(loss_fn.mse_weight, 0.6)


if __name__ == '__main__':
    unittest.main()