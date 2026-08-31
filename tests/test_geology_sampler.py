import unittest

import numpy as np

from src.geology_sampler import GeologyAwareBatchSampler, build_multilabel_strata


class GeologySamplerTests(unittest.TestCase):
    def _toy_metadata(self):
        # columns: fault, channel, flat
        return np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def test_build_multilabel_strata_assigns_background_and_mixed_labels(self):
        matrix = self._toy_metadata()
        strata = build_multilabel_strata(
            metadata_matrix=matrix,
            metadata_keys=("fault", "channel", "flat"),
            threshold=1e-6,
            active_key_indices=(0, 1, 2),
            max_active_keys_per_stratum=2,
        )

        self.assertEqual(int(strata.background_label), 0)
        self.assertEqual(int(strata.labels[6]), 0)
        self.assertEqual(int(strata.labels[7]), 0)
        self.assertNotEqual(int(strata.labels[0]), int(strata.labels[2]))
        self.assertEqual(int(strata.labels[4]), int(strata.labels[5]))

    def test_geology_batch_sampler_is_seed_deterministic(self):
        matrix = self._toy_metadata()
        strata = build_multilabel_strata(
            metadata_matrix=matrix,
            metadata_keys=("fault", "channel", "flat"),
            threshold=1e-6,
            active_key_indices=(0, 1, 2),
        )
        weights = np.linspace(1.0, 2.0, num=matrix.shape[0], dtype=np.float64)

        sampler_a = GeologyAwareBatchSampler(
            strata_labels=strata.labels,
            batch_size=6,
            num_batches=4,
            seed=123,
            sample_weights=weights,
            background_fraction=0.2,
            hard_fraction=0.2,
            min_negative_strata=2,
            require_positive_pair=True,
            allow_duplicates=False,
        )
        sampler_a.set_epoch(5)
        seq_a = [batch for batch in sampler_a]

        sampler_b = GeologyAwareBatchSampler(
            strata_labels=strata.labels,
            batch_size=6,
            num_batches=4,
            seed=123,
            sample_weights=weights,
            background_fraction=0.2,
            hard_fraction=0.2,
            min_negative_strata=2,
            require_positive_pair=True,
            allow_duplicates=False,
        )
        sampler_b.set_epoch(5)
        seq_b = [batch for batch in sampler_b]

        self.assertEqual(seq_a, seq_b)

    def test_geology_batch_sampler_reports_epoch_stats(self):
        matrix = self._toy_metadata()
        strata = build_multilabel_strata(
            metadata_matrix=matrix,
            metadata_keys=("fault", "channel", "flat"),
            threshold=1e-6,
            active_key_indices=(0, 1, 2),
        )
        sampler = GeologyAwareBatchSampler(
            strata_labels=strata.labels,
            batch_size=6,
            num_batches=4,
            seed=17,
            background_fraction=0.2,
            hard_fraction=0.2,
            min_negative_strata=2,
            require_positive_pair=True,
            allow_duplicates=False,
        )
        sampler.set_epoch(0)
        batches = [batch for batch in sampler]

        self.assertEqual(len(batches), 4)
        stats = sampler.get_last_epoch_stats()
        self.assertIn("positive_pair_batch_rate", stats)
        self.assertIn("negative_strata_batch_rate", stats)
        self.assertIn("avg_unique_strata_per_batch", stats)
        self.assertGreaterEqual(float(stats["positive_pair_batch_rate"]), 0.0)
        self.assertLessEqual(float(stats["positive_pair_batch_rate"]), 1.0)

    def test_geology_batch_sampler_respects_quota_preferences_when_feasible(self):
        sample_count = 120
        labels = np.zeros((sample_count,), dtype=np.int64)
        labels[40:80] = 1
        labels[80:120] = 2

        # Make hard examples concentrated in non-background strata.
        sample_weights = np.linspace(0.1, 0.9, num=sample_count, dtype=np.float64)
        sample_weights[80:120] += 0.5

        sampler = GeologyAwareBatchSampler(
            strata_labels=labels,
            batch_size=20,
            num_batches=200,
            seed=42,
            sample_weights=sample_weights,
            background_fraction=0.10,
            hard_fraction=0.20,
            hard_top_quantile=0.20,
            min_negative_strata=1,
            require_positive_pair=False,
            allow_duplicates=False,
        )
        sampler.set_epoch(0)
        _ = [batch for batch in sampler]
        stats = sampler.get_last_epoch_stats()

        self.assertLessEqual(float(stats["background_fraction_achieved"]), 0.18)
        self.assertLessEqual(float(stats["hard_fraction_achieved"]), 0.30)

    def test_geology_batch_sampler_hard_pool_matches_top_quantile_count(self):
        labels = np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
        weights = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64)

        sampler = GeologyAwareBatchSampler(
            strata_labels=labels,
            batch_size=5,
            num_batches=20,
            seed=7,
            sample_weights=weights,
            hard_fraction=0.2,
            hard_top_quantile=0.2,
            require_positive_pair=False,
            allow_duplicates=False,
        )
        sampler.set_epoch(0)
        _ = [batch for batch in sampler]
        stats = sampler.get_last_epoch_stats()

        self.assertGreaterEqual(float(stats["hard_fraction_achieved"]), 0.15)
        self.assertLessEqual(float(stats["hard_fraction_achieved"]), 0.40)


if __name__ == "__main__":
    unittest.main()
