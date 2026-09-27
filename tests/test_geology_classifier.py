import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import zarr
from torch import nn

from scripts import train as train_script
from src.geology_classifier import (
    PRESENCE_TARGET_KEYS,
    classifier_target_keys,
    compute_geology_classifier_loss,
    compute_pos_weight,
    focal_bce_with_logits,
)
from src.model import GEOLOGY_PRESENCE_CLASSES, GeologyClassifierDecoder, VAE3D
from src.tokenizer.core.model_adapter import VaeLatentAdapter

PATCH = (8, 8, 8)


def _small_model(classifier=False, seed=0):
    torch.manual_seed(seed)
    return VAE3D(
        base_ch=8, latent_dim=32, patch_shape=PATCH,
        geology_projection=True, geology_proj_hidden=16, geology_proj_dim=8,
        geology_classifier=classifier, geology_classifier_hidden=32,
    )


def _batch_targets(n, rng):
    batch = {PRESENCE_TARGET_KEYS[c]: torch.as_tensor(rng.integers(0, 2, n), dtype=torch.uint8) for c in GEOLOGY_PRESENCE_CLASSES}
    batch["meta_dip_mean_class"] = torch.as_tensor(rng.integers(0, 6, n), dtype=torch.float32)
    batch["meta_dip_range_class"] = torch.as_tensor(rng.integers(0, 6, n), dtype=torch.float32)
    return batch


def _one_epoch(model, batches, **kw):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.manual_seed(2)
    stats = {}
    out = train_script.train_one_epoch(
        model, None, batches, 'cpu', opt, None, len(batches), 1.0, 1e-3, 0.0,
        rec_loss_fn=nn.MSELoss(), epoch_stats=stats, **kw,
    )
    return out, stats


def _plain_batches():
    g = torch.Generator().manual_seed(1)
    batches = []
    for _ in range(2):
        x = torch.randn(2, 1, *PATCH, generator=g)
        batches.append((x, x.clone()))
    return batches


# Pre-WP3 (commit 0831677) train_one_epoch on _small_model() / _plain_batches().
GOLDEN_OFF_LOSS = 1.2509312629699707
GOLDEN_OFF_PARAM_SUM = 136.0178985595703
GOLDEN_OFF_PARAM_ABS_SUM = 1971.98095703125


class TestClassifierHead(unittest.TestCase):
    def test_output_shapes_and_finite(self):
        head = GeologyClassifierDecoder(latent_dim=32, hidden=64)
        out = head(torch.randn(5, 32))
        self.assertEqual(tuple(out["presence"].shape), (5, 7))
        self.assertEqual(tuple(out["dip_mean"].shape), (5, 6))
        self.assertEqual(tuple(out["dip_range"].shape), (5, 6))
        self.assertTrue(all(torch.isfinite(v).all() for v in out.values()))

    def test_vae_classify_and_forward_signature(self):
        model = _small_model(classifier=True)
        x = torch.randn(3, 1, *PATCH)
        out = model(x)
        self.assertEqual(len(out), 3)
        logits = model.classify(out[1])
        self.assertEqual(tuple(logits["presence"].shape), (3, 7))
        with self.assertRaises(RuntimeError):
            _small_model(classifier=False).classify(out[1])

    def test_voxel_mode_not_implemented(self):
        with self.assertRaises(ValueError):
            VAE3D(patch_shape=PATCH, geology_classifier=True, geology_classifier_mode='voxel')

    def test_no_classifier_keys_when_disabled(self):
        keys = _small_model().state_dict().keys()
        self.assertFalse(any(k.startswith('geology_classifier.') for k in keys))


class TestTargetsAndLoss(unittest.TestCase):
    def test_target_keys(self):
        keys = classifier_target_keys()
        self.assertEqual(len(keys), 9)
        self.assertIn("label_presence_fault_x", keys)
        self.assertIn("meta_dip_range_class", keys)
        with self.assertRaises(ValueError):
            classifier_target_keys(["bogus"])

    def test_pos_weight_known_counts(self):
        y = np.zeros((100, 4))
        y[:10, 0] = 1        # 90/10 = 9
        y[:1, 1] = 1         # 99 -> clipped to 50
        y[:, 2] = 1          # 0/100 -> clipped up to 1
        pw = compute_pos_weight(y)
        np.testing.assert_allclose(pw.numpy(), [9.0, 50.0, 1.0, 1.0])

    def test_focal_reduces_to_bce_at_gamma_zero(self):
        logits = torch.randn(6, 3)
        y = torch.randint(0, 2, (6, 3)).float()
        bce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none')
        self.assertTrue(torch.allclose(focal_bce_with_logits(logits, y, gamma=0.0), bce))
        self.assertTrue(torch.all(focal_bce_with_logits(logits, y, gamma=2.0) <= bce + 1e-7))

    def test_loss_subset_of_targets(self):
        rng = np.random.default_rng(0)
        logits = GeologyClassifierDecoder(latent_dim=8, hidden=16)(torch.randn(4, 8))
        batch = _batch_targets(4, rng)
        loss_all, parts_all = compute_geology_classifier_loss(logits, batch)
        self.assertEqual(set(parts_all), {"presence", "dip_mean", "dip_range"})
        loss_p, parts_p = compute_geology_classifier_loss(logits, batch, targets=("fault", "sand"))
        self.assertEqual(set(parts_p), {"presence"})
        self.assertTrue(torch.isfinite(loss_all) and torch.isfinite(loss_p))
        with self.assertRaises(ValueError):
            compute_geology_classifier_loss(logits, batch, loss_type="bogus")

    def test_loss_decreases_on_separable_batch(self):
        torch.manual_seed(0)
        n, d = 32, 16
        mu = torch.randn(n, d)
        w = torch.randn(d, 7)
        batch = {PRESENCE_TARGET_KEYS[c]: (mu @ w[:, i] > 0).to(torch.uint8) for i, c in enumerate(GEOLOGY_PRESENCE_CLASSES)}
        batch["meta_dip_mean_class"] = (mu[:, 0] > 0).float() * 3
        batch["meta_dip_range_class"] = (mu[:, 1] > 0).float() * 5
        head = GeologyClassifierDecoder(latent_dim=d, hidden=64, dropout=0.0)
        opt = torch.optim.Adam(head.parameters(), lr=1e-2)
        losses = []
        for _ in range(50):
            loss, _ = compute_geology_classifier_loss(head(mu), batch, loss_type="focal")
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        self.assertLess(losses[-1], 0.5 * losses[0])


class TestTrainingIntegration(unittest.TestCase):
    def test_classifier_off_matches_pre_wp3_golden(self):
        model = _small_model()
        out, stats = _one_epoch(model, _plain_batches())
        params = torch.cat([p.detach().flatten() for p in model.parameters()])
        self.assertAlmostEqual(out[0], GOLDEN_OFF_LOSS, places=5)
        self.assertAlmostEqual(float(params.sum()), GOLDEN_OFF_PARAM_SUM, places=3)
        self.assertAlmostEqual(float(params.abs().sum()), GOLDEN_OFF_PARAM_ABS_SUM, places=2)
        self.assertEqual(stats["geology_classifier_loss"], 0.0)

    def test_classifier_weight_zero_is_noop(self):
        base = _small_model()
        with_head = _small_model(classifier=True, seed=5)
        with_head.load_state_dict(base.state_dict(), strict=False)
        out_a, _ = _one_epoch(base, _plain_batches())
        out_b, stats_b = _one_epoch(with_head, [(x, y, _batch_targets(2, np.random.default_rng(0))) for x, y in _plain_batches()])
        self.assertAlmostEqual(out_a[0], out_b[0], places=6)
        sd_b = with_head.state_dict()
        for k, v in base.state_dict().items():
            self.assertTrue(torch.allclose(v, sd_b[k]), k)
        self.assertEqual(stats_b["geology_classifier_loss"], 0.0)

    def test_classifier_loss_trains_head_and_encoder(self):
        model = _small_model(classifier=True)
        before_enc = [p.detach().clone() for p in model.encoder.parameters()]
        before_head = [p.detach().clone() for p in model.geology_classifier.parameters()]
        batches = [(x, y, _batch_targets(2, np.random.default_rng(i))) for i, (x, y) in enumerate(_plain_batches())]
        _, stats = _one_epoch(model, batches, geology_classifier_weight=0.1, geology_classifier_loss='focal')
        self.assertGreater(stats["geology_classifier_loss"], 0.0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_head, model.geology_classifier.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_enc, model.encoder.parameters())))


class TestCheckpoints(unittest.TestCase):
    def test_warm_start_without_classifier_only_misses_classifier_keys(self):
        old = _small_model()
        new = _small_model(classifier=True, seed=3)
        result = new.load_state_dict(old.state_dict(), strict=False)
        self.assertTrue(result.missing_keys)
        self.assertTrue(all(k.startswith('geology_classifier.') for k in result.missing_keys))
        self.assertEqual(train_script.resume_state_dict_incompatibilities(result.missing_keys, result.unexpected_keys), ([], []))
        # Loading a classifier checkpoint into a model without the head is tolerated too.
        result = old.load_state_dict(new.state_dict(), strict=False)
        self.assertEqual(train_script.resume_state_dict_incompatibilities(result.missing_keys, result.unexpected_keys), ([], []))
        self.assertEqual(
            train_script.resume_state_dict_incompatibilities(['encoder.fc_mu.weight'], []),
            (['encoder.fc_mu.weight'], []),
        )

    def test_resume_with_classifier_restores_exactly(self):
        src = _small_model(classifier=True, seed=1)
        src.geology_classifier.pos_weight.copy_(torch.arange(1, 8, dtype=torch.float32))
        payload = train_script.build_checkpoint_payload(src)
        self.assertTrue(payload['geology_classifier'])
        self.assertEqual(payload['geology_classifier_mode'], 'patch')
        self.assertEqual(payload['geology_classifier_hidden'], 32)
        dst = _small_model(classifier=True, seed=9)
        dst.load_state_dict(payload['model_state_dict'], strict=True)
        for k, v in src.state_dict().items():
            self.assertTrue(torch.equal(v, dst.state_dict()[k]), k)

    def test_tokenizer_adapter_ignores_classifier(self):
        model = _small_model(classifier=True, seed=4)
        full = train_script.build_checkpoint_payload(model)
        stripped = dict(full)
        stripped['model_state_dict'] = {k: v for k, v in full['model_state_dict'].items() if not k.startswith('geology_classifier.')}
        stripped['geology_classifier'] = False
        cubes = np.random.default_rng(0).normal(size=(3,) + PATCH).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name, payload in (("full", full), ("stripped", stripped)):
                path = Path(tmp) / f"{name}.pt"
                torch.save(payload, path)
                paths.append(path)
            z_full = VaeLatentAdapter(paths[0], device='cpu').encode_geo_batch(cubes)
            z_stripped = VaeLatentAdapter(paths[1], device='cpu').encode_geo_batch(cubes)
        np.testing.assert_array_equal(z_full, z_stripped)


class TestDatasetAndGuards(unittest.TestCase):
    def test_contrastive_loss_loads_metadata_arrays(self):
        args = SimpleNamespace(
            geology_metadata_keys=['meta_fault_fraction'],
            geology_loss_weight=0.0,
            geology_contrastive_weight=0.5,
            geology_uniformity_weight=0.0,
            geology_batch_sampler=False,
            geology_classifier_weight=0.0,
            geology_classifier_classes=[],
            input_scaling='none', input_mean=0.0, input_std=1.0,
            swap_xy_prob=0.0, flip_x_prob=0.0, flip_y_prob=0.0,
            vertical_warp_prob=0.0, zero_cluster_min=0, zero_cluster_max=0,
            input_extrema_prob=1.0, input_sparse_keep_prob=0.0,
            input_decimate_trilinear_prob=0.0, sparse_keep_fraction_min=0.1,
            sparse_keep_fraction_max=0.3, sparse_poisson_radius_scale=0.85,
            mixup_augment_prob=0.0,
        )
        with mock.patch.object(train_script, 'ZarrPatchDataset', return_value='dataset') as dataset_cls:
            self.assertEqual(train_script.build_dataset(args, 'unused.zarr'), 'dataset')
        self.assertTrue(dataset_cls.call_args.kwargs['include_metadata'])
        self.assertEqual(dataset_cls.call_args.kwargs['geology_metadata_keys'], ('meta_fault_fraction',))

    def _write_dataset(self, root, with_presence=True, n=6):
        g = zarr.open_group(str(root), mode='w')
        g.create_array('patches', data=np.random.default_rng(0).normal(size=(n,) + PATCH).astype(np.float32))
        g.create_array('meta_dip_mean_class', data=np.arange(n, dtype=np.float32) % 6)
        g.create_array('meta_dip_range_class', data=np.zeros(n, dtype=np.float32))
        if with_presence:
            for c in GEOLOGY_PRESENCE_CLASSES:
                g.create_array(f'label_presence_{c}', data=(np.arange(n) % 2).astype(np.uint8))

    def test_dataset_returns_classifier_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'd.zarr'
            self._write_dataset(path)
            ds = train_script.ZarrPatchDataset(path, label_target_keys=classifier_target_keys())
            self.assertTrue(ds.include_metadata)
            _, _, meta = ds[3]
            self.assertEqual(int(meta['label_presence_fault']), 1)
            self.assertEqual(float(meta['meta_dip_mean_class']), 3.0)
            plain = train_script.ZarrPatchDataset(path)
            self.assertEqual(len(plain[0]), 2)
            # Metadata keys requested without a metadata loss must not be read when only the classifier is on.
            cls_only = train_script.ZarrPatchDataset(
                path, geology_metadata_keys=('meta_missing_key',), label_target_keys=classifier_target_keys(),
            )
            self.assertNotIn('meta_missing_key', cls_only[0][2])

    def test_missing_presence_arrays_fail_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'd.zarr'
            self._write_dataset(path, with_presence=False)
            with self.assertRaisesRegex(KeyError, 'label_presence_fault'):
                train_script.ZarrPatchDataset(path, label_target_keys=classifier_target_keys())

    def test_weight_requires_flag(self):
        args = SimpleNamespace(geology_classifier_weight=0.1, geology_classifier=False)
        with self.assertRaisesRegex(ValueError, '--geology_classifier'):
            train_script.train(args)


if __name__ == '__main__':
    unittest.main()
