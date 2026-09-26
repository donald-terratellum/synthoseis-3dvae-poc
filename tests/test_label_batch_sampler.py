import hashlib
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from scripts import train as train_script
from src.geology_classifier import PRESENCE_TARGET_KEYS
from src.geology_sampler import GeologyAwareBatchSampler, build_presence_strata, presence_rarity_weights
from src.model import GEOLOGY_PRESENCE_CLASSES, VAE3D

# Batches from the pre-WP4 sampler (commit b36b457) for the config in _baseline_sampler.
GOLDEN_SAMPLER_HASH = "c403c333a7e1836f"


def _baseline_sampler(**kw):
    labels = np.random.default_rng(0).integers(0, 5, 200)
    s = GeologyAwareBatchSampler(labels, batch_size=12, num_batches=20, seed=7, background_fraction=0.1, hard_fraction=0.2, **kw)
    s.set_epoch(3)
    return s


def _hash(batches):
    return hashlib.sha256(np.asarray(batches).astype(np.int64).tobytes()).hexdigest()[:16]


class TestPresenceStrata(unittest.TestCase):
    classes = ("fault", "fault_x", "flat_spot")

    def test_rarity_weights(self):
        presence = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0]])
        np.testing.assert_allclose(presence_rarity_weights(presence), [4 / 3, 4.0, 0.0])

    def test_signatures_keep_rarest_classes(self):
        presence = np.array([
            [0, 0, 0],  # background
            [1, 0, 0],  # fault
            [1, 1, 0],  # fault+fault_x
            [1, 1, 1],  # three present, keep two rarest: fault_x+flat_spot
            [0, 1, 1],  # fault_x+flat_spot
            [1, 0, 0],  # fault
        ])
        rarity = np.array([1.0, 3.0, 6.0])
        strata = build_presence_strata(presence, self.classes, rarity, max_active_keys_per_stratum=2)
        names = [strata.label_to_name[int(v)] for v in strata.labels]
        self.assertEqual(names, [
            "background", "fault", "fault+fault_x", "fault_x+flat_spot", "fault_x+flat_spot", "fault",
        ])
        single = build_presence_strata(presence, self.classes, rarity, max_active_keys_per_stratum=1)
        self.assertEqual(single.label_to_name[int(single.labels[3])], "flat_spot")

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            build_presence_strata(np.zeros((3, 2)), self.classes, np.ones(3))


class TestClassQuota(unittest.TestCase):
    def test_no_quota_matches_pre_wp4_sampler(self):
        self.assertEqual(_hash(list(iter(_baseline_sampler()))), GOLDEN_SAMPLER_HASH)
        membership = {"fault_x": np.random.default_rng(1).random(200) < 0.1}
        self.assertEqual(_hash(list(iter(_baseline_sampler(class_membership=membership)))), GOLDEN_SAMPLER_HASH)

    def test_quota_met_when_feasible(self):
        n = 300
        rng = np.random.default_rng(0)
        fx = np.zeros(n, bool)
        fx[rng.choice(n, 15, replace=False)] = True
        fs = np.zeros(n, bool)
        fs[rng.choice(n, 20, replace=False)] = True
        sampler = GeologyAwareBatchSampler(
            rng.integers(0, 4, n), batch_size=12, num_batches=40, seed=1,
            class_membership={"fault_x": fx, "flat_spot": fs}, class_quota={"fault_x": 2, "flat_spot": 1},
        )
        for batch in sampler:
            self.assertEqual(len(batch), 12)
            self.assertEqual(len(set(batch)), 12)
            self.assertGreaterEqual(int(fx[batch].sum()), 2)
            self.assertGreaterEqual(int(fs[batch].sum()), 1)
        stats = sampler.get_last_epoch_stats()
        self.assertEqual(stats["class_quota_met_batch_rate"], 1.0)
        self.assertEqual(stats["class_quota_fallback_fault_x"], 0.0)
        self.assertGreaterEqual(stats["class_share_fault_x"], 2 / 12)

    def test_quota_fallback_reported_when_infeasible(self):
        n = 100
        fx = np.zeros(n, bool)
        fx[5] = True
        sampler = GeologyAwareBatchSampler(
            np.zeros(n, dtype=np.int64), batch_size=8, num_batches=10, seed=0,
            class_membership={"fault_x": fx}, class_quota={"fault_x": 3},
        )
        batches = list(iter(sampler))
        self.assertTrue(all(5 in b for b in batches))
        stats = sampler.get_last_epoch_stats()
        self.assertEqual(stats["class_quota_fallback_fault_x"], 20.0)
        self.assertEqual(stats["class_quota_met_batch_rate"], 0.0)

    def test_quota_validation(self):
        labels = np.zeros(10, dtype=np.int64)
        with self.assertRaises(ValueError):
            GeologyAwareBatchSampler(labels, 4, 1, 0, class_membership={"a": np.ones(10, bool)}, class_quota={"a": 5})
        with self.assertRaises(ValueError):
            GeologyAwareBatchSampler(labels, 4, 1, 0, class_quota={"a": 1})
        with self.assertRaises(ValueError):
            GeologyAwareBatchSampler(labels, 4, 1, 0, class_membership={"a": np.ones(3, bool)})

    def test_parse_class_quota(self):
        self.assertEqual(train_script.parse_class_quota(["fault_x=2", "flat_spot=1"]), {"fault_x": 2, "flat_spot": 1})
        self.assertEqual(train_script.parse_class_quota(None), {})
        with self.assertRaises(ValueError):
            train_script.parse_class_quota(["bogus=1"])


class TestWeightedCalibration(unittest.TestCase):
    keys = ("meta_a", "meta_b")

    def _natural(self, n=4000):
        rng = np.random.default_rng(0)
        cols = {}
        for key, rate in zip(self.keys, (0.10, 0.30)):
            vals = np.zeros(n)
            pos = rng.random(n) < rate
            vals[pos] = rng.lognormal(-3.0, 0.7, int(pos.sum()))
            cols[key] = vals
        return cols

    def _fit(self, cols, weights=None):
        ds = SimpleNamespace(_metadata_arrays=cols)
        return train_script.fit_geology_metadata_calibration(ds, self.keys, sample_weights=weights)

    def test_weighted_percentile_unit_weights_match_numpy(self):
        v = np.random.default_rng(1).normal(size=101)
        for q in (0, 10, 25, 50, 75, 99, 100):
            self.assertAlmostEqual(train_script.weighted_percentile(v, q, np.ones_like(v)), float(np.percentile(v, q)), places=12)

    def test_unit_weights_identical_to_unweighted(self):
        cols = self._natural()
        self.assertEqual(self._fit(cols), self._fit(cols, np.ones(4000)))

    def test_inclusion_weight_recovers_natural_prevalence(self):
        natural = self._natural()
        # Anchored-style dataset: every patch with a positive meta_a is included 4 times at weight 1/4.
        pos = natural["meta_a"] > 0
        rows = np.concatenate([np.flatnonzero(~pos)] + [np.flatnonzero(pos)] * 4)
        anchored = {k: v[rows] for k, v in natural.items()}
        weights = np.where(pos[rows], 0.25, 1.0)

        ref = self._fit(natural)
        weighted = self._fit(anchored, weights)
        biased = self._fit(anchored)
        for field in ("nonzero_scale", "center", "scale"):
            np.testing.assert_allclose(weighted[field], ref[field], rtol=0.02, atol=1e-3, err_msg=field)
        # Without weights the rebalanced set distorts the spread of meta_a.
        self.assertGreater(abs(biased["scale"][0] - ref["scale"][0]), 0.1)

    def test_invalid_weights(self):
        with self.assertRaises(ValueError):
            self._fit(self._natural(10), -np.ones(10))


class TestTrainingWiring(unittest.TestCase):
    def test_resolve_label_target_keys(self):
        args = SimpleNamespace(
            geology_classifier_weight=0.0, geology_batch_sampler=True,
            geology_strata_source="presence_labels", geology_strata_classes=["fault", "fault_x"],
            geology_batch_class_quota=["fault_x=1", "flat_spot=1"],
        )
        self.assertEqual(
            train_script.resolve_label_target_keys(args),
            ("label_presence_fault", "label_presence_fault_x", "label_presence_flat_spot"),
        )
        args.geology_batch_sampler = False
        self.assertEqual(train_script.resolve_label_target_keys(args), ())

    def test_presence_matrix_from_tensors(self):
        batch = {PRESENCE_TARGET_KEYS["fault"]: torch.tensor([1, 0, 1], dtype=torch.uint8),
                 PRESENCE_TARGET_KEYS["fault_x"]: np.array([0, 0, 1])}
        m = train_script.presence_matrix_from_labels(batch, ("fault", "fault_x"))
        np.testing.assert_array_equal(m, [[1, 0], [0, 0], [1, 1]])

    def test_supcon_uses_presence_strata(self):
        torch.manual_seed(0)
        model = VAE3D(base_ch=8, latent_dim=32, patch_shape=(8, 8, 8), geology_projection=True, geology_proj_hidden=16, geology_proj_dim=8)
        x = torch.randn(4, 1, 8, 8, 8)
        labels = {PRESENCE_TARGET_KEYS[c]: torch.zeros(4, dtype=torch.uint8) for c in GEOLOGY_PRESENCE_CLASSES}
        labels[PRESENCE_TARGET_KEYS["fault_x"]] = torch.tensor([1, 1, 0, 0], dtype=torch.uint8)
        labels[PRESENCE_TARGET_KEYS["fault"]] = torch.tensor([0, 0, 1, 1], dtype=torch.uint8)
        classes = ("fault", "fault_x")
        out = train_script.train_one_epoch(
            model, None, [(x, x.clone(), labels)], 'cpu', torch.optim.Adam(model.parameters(), lr=1e-3), None, 1,
            1.0, 1e-3, 0.0, rec_loss_fn=nn.MSELoss(), geology_contrastive_weight=0.5,
            geology_presence_strata=(classes, np.array([2.0, 2.0])),
        )
        self.assertGreater(out[5], 0.0)

    def test_presence_strata_require_batch_sampler(self):
        args = SimpleNamespace(
            geology_classifier_weight=0.0, geology_classifier=False, geology_loss_weight=0.0, geology_metadata_keys=[],
            geology_diagnostic_max_samples=512, geology_diagnostic_neighbor_k=5, geology_huber_delta=0.1,
            geology_background_threshold=1e-6, geology_diagnostic_topk=[5], geology_batch_background_fraction=0.2,
            geology_batch_hard_fraction=0.2, geology_batch_hard_top_quantile=0.2, geology_batch_min_negative_strata=2,
            geology_strata_max_active_keys=2, geology_strata_source="presence_labels", geology_batch_class_quota=None,
            geology_batch_sampler=False, batch_size=12,
        )
        with self.assertRaisesRegex(ValueError, "--geology_batch_sampler"):
            train_script.train(args)


if __name__ == "__main__":
    unittest.main()
