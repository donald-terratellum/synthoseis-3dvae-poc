import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts import p9_mu_scaling, train as train_script
from src.model import GEOLOGY_PRESENCE_CLASSES, VAE3D
from src.tokenizer.core.model_adapter import VaeLatentAdapter


class LatentWidthExperimentTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def _model(self, latent_dim):
        return VAE3D(
            base_ch=2,
            latent_dim=latent_dim,
            patch_shape=(8, 8, 8),
            geology_projection=True,
            geology_proj_hidden=8,
            geology_proj_dim=64,
            geology_classifier=True,
            geology_classifier_hidden=16,
        )

    def test_latent_dim_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, 'latent_dim must be positive'):
            self._model(0)

    def test_default_latent_dimension_remains_128(self):
        model = VAE3D(base_ch=2, patch_shape=(8, 8, 8))
        self.assertEqual(model.latent_dim, 128)

    def test_reconstruction_and_geology_losses_backpropagate_for_wide_latents(self):
        for latent_dim in (512, 2048):
            with self.subTest(latent_dim=latent_dim):
                torch.manual_seed(19)
                model = self._model(latent_dim)
                inputs = torch.randn(4, 1, 8, 8, 8)
                reconstruction, mu, logvar = model(inputs)
                z_geo = model.encode_geo(mu)
                classifier_logits = model.classify(mu)
                labels = {
                    f'label_presence_{name}': torch.tensor([0, 1, 0, 1])
                    for name in GEOLOGY_PRESENCE_CLASSES
                }
                labels.update({
                    'meta_dip_mean_class': torch.tensor([0, 1, 2, 3]),
                    'meta_dip_range_class': torch.tensor([1, 2, 3, 4]),
                })
                classifier_loss, _ = train_script.compute_geology_classifier_loss(
                    classifier_logits, labels, loss_type='focal'
                )
                contrastive_loss = train_script.compute_supervised_contrastive_loss(
                    z_geo, [1, 1, 2, 2], temperature=0.2
                )
                metadata_loss = train_script.compute_geology_similarity_loss(
                    mu,
                    {'meta_fault_fraction': torch.tensor([0.0, 0.2, 0.8, 1.0])},
                    ('meta_fault_fraction',),
                )
                reconstruction_loss = F.l1_loss(reconstruction, inputs)
                kl_loss = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum() / inputs.numel()
                total = (
                    reconstruction_loss + 0.001 * kl_loss + 0.1 * classifier_loss
                    + 0.5 * contrastive_loss + 0.03 * metadata_loss
                )
                self.assertEqual(tuple(reconstruction.shape), tuple(inputs.shape))
                self.assertEqual(tuple(mu.shape), (4, latent_dim))
                self.assertEqual(tuple(logvar.shape), (4, latent_dim))
                self.assertEqual(tuple(z_geo.shape), (4, 64))
                self.assertEqual(tuple(classifier_logits['presence'].shape), (4, 7))
                self.assertTrue(torch.isfinite(total))
                self.assertTrue(torch.isfinite(reconstruction).all())

                total.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                geology_head = model.geology_head
                geology_classifier = model.geology_classifier
                self.assertIsNotNone(geology_head)
                self.assertIsNotNone(geology_classifier)
                self.assertIsNotNone(model.encoder.fc_mu.weight.grad)
                self.assertIsNotNone(model.decoder.fc.weight.grad)
                assert geology_head is not None
                assert geology_classifier is not None
                self.assertIsNotNone(geology_head.net[0].weight.grad)
                self.assertIsNotNone(geology_classifier.presence.weight.grad)

    def test_wide_checkpoint_round_trip_and_tokenizer_adapter(self):
        cubes = torch.randn(2, 8, 8, 8).numpy()
        for latent_dim in (512, 2048):
            with self.subTest(latent_dim=latent_dim), tempfile.TemporaryDirectory() as directory:
                model = self._model(latent_dim).eval()
                checkpoint = Path(directory) / 'wide.pt'
                torch.save(train_script.build_checkpoint_payload(model), checkpoint)
                adapter = VaeLatentAdapter(checkpoint, device='cpu')
                self.assertEqual(adapter.latent_dim, latent_dim)
                self.assertEqual(adapter.encode_batch(cubes).shape, (2, latent_dim))
                self.assertEqual(adapter.encode_geo_batch(cubes).shape, (2, 64))

    def test_comparison_gate_and_csv_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'e1_mu512/stage1_reconstruction').mkdir(parents=True)
            torch.save({'model_state_dict': {'weight': torch.ones(3, 4)}},
                       root / 'e1_mu512/stage1_reconstruction/vae_best.pt')

            def reconstruction_arm(recipe, mae):
                return {
                    'evaluation': {
                        'synthetic': {
                            'sampled': {'recipe_total': recipe},
                            'mean': {'kl_per_example': 0.2, 'kl_per_voxel': 0.001},
                        },
                        'real': {'sampled': {'mae': mae}},
                    },
                }

            stage1 = {
                'settings': {'input_mode': 'tokenizer'},
                'reconstruction_gate': {'full_dataset_coverage': True},
                'arms': {
                    'p0b': reconstruction_arm(1.0, 0.7),
                    'mu128': reconstruction_arm(1.0, 0.7),
                    'mu512': reconstruction_arm(1.01, 0.71),
                },
            }
            (root / 'stage1_tokenizer_reconstruction.json').write_text(json.dumps(stage1))
            gate_args = argparse.Namespace(
                report=str(root / 'stage1_tokenizer_reconstruction.json'),
                candidate='mu512', controls=['p0b', 'mu128'], maximum_regression=0.02,
            )
            p9_mu_scaling.reconstruction_gate(gate_args)

            epochs = (10, 20, 30, 40, 50)
            for epoch in epochs:
                stage2 = {
                    'settings': {'input_mode': 'tokenizer'},
                    'reconstruction_gate': {'full_dataset_coverage': True},
                    'arms': {
                        'p0b': reconstruction_arm(1.0, 0.7),
                        'mu128_ep30': reconstruction_arm(1.0, 0.7),
                        f'mu512_ep{epoch}': reconstruction_arm(1.01, 0.71),
                    },
                }
                (root / f'mu512_ep{epoch}_tokenizer.json').write_text(json.dumps(stage2))
                benchmark = {
                    'diagnostics': {'neighbor_overlap_at_5': 0.11, 'neighbor_overlap_at_10': 0.2},
                    'global_metrics': {'query_count': 768},
                    'classifier_metrics': {'macro_auroc': 0.8},
                }
                (root / f'mu512_ep{epoch}_zgeo.json').write_text(json.dumps(benchmark))
            (root / 'latent_reconstruction_gate.json').write_text(json.dumps({'passed': True}))
            p0b_benchmark = root / 'p0b.json'
            p0b_benchmark.write_text(json.dumps({
                'diagnostics': {'neighbor_overlap_at_5': 0.1, 'neighbor_overlap_at_10': 0.19},
            }))

            p9_mu_scaling.finalize(argparse.Namespace(
                root=str(root), seed=20260926, p0b_benchmark=str(p0b_benchmark),
                epochs=epochs,
            ))
            with (root / 'latent_size_comparison.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[0]['parameter_count'], '12')
            self.assertEqual(rows[-1]['retrieval_gate'], 'True')
            decision = json.loads((root / 'p9_mu512_decision.json').read_text())
            self.assertTrue(decision['screening_success'])
            self.assertFalse(decision['adoption'])


if __name__ == '__main__':
    unittest.main()
