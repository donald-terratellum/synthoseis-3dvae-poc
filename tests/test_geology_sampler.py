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


if __name__ == "__main__":
    unittest.main()
