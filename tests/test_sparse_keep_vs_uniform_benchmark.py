import time
import unittest
import os
from pathlib import Path
from typing import cast

import numpy as np

from src import augmentations


class SparseKeepVsUniformBenchmarkTests(unittest.TestCase):
    def test_sparse_keep_vs_uniform_indices_and_cross_sections(self):
        shape = (32, 32, 64)
        target_fraction = float(os.environ.get('SPARSE_KEEP_TARGET_FRACTION', '0.30'))
        n_trials = int(os.environ.get('SPARSE_KEEP_BENCH_TRIALS', '3'))
        nvox = int(np.prod(shape))
        self.assertGreater(target_fraction, 0.0)
        self.assertLessEqual(target_fraction, 1.0)
        self.assertGreaterEqual(n_trials, 2)
        target_count = int(np.clip(np.rint(target_fraction * nvox), 1, nvox))

        poisson_times = []
        uniform_times = []

        representative_poisson_indices = None
        representative_uniform_mask = None

        for trial_idx in range(n_trials):
            np.random.seed(20260603 + trial_idx)

            t_poisson = time.perf_counter()
            poisson_indices = augmentations._build_sparse_indices_poisson_like(
                shape=shape,
                target_count=target_count,
                radius_scale=0.85,
            )
            poisson_times.append(time.perf_counter() - t_poisson)

            t_uniform = time.perf_counter()
            uniform_indices = augmentations._build_sparse_indices_uniform_threshold(shape, target_fraction)
            uniform_mask = np.zeros(shape, dtype=bool)
            uniform_mask.reshape(-1)[uniform_indices] = True
            uniform_times.append(time.perf_counter() - t_uniform)

            if representative_poisson_indices is None:
                representative_poisson_indices = poisson_indices
            if representative_uniform_mask is None:
                representative_uniform_mask = uniform_mask

            self.assertEqual(int(poisson_indices.size), target_count)

        poisson_times = np.asarray(poisson_times, dtype=np.float64)
        uniform_times = np.asarray(uniform_times, dtype=np.float64)

        self.assertIsNotNone(representative_poisson_indices)
        self.assertIsNotNone(representative_uniform_mask)
        poisson_indices_ref = cast(np.ndarray, representative_poisson_indices)
        uniform_mask_ref = cast(np.ndarray, representative_uniform_mask)

        poisson_mask = np.zeros(shape, dtype=bool)
        poisson_mask.reshape(-1)[poisson_indices_ref] = True

        x_mid = shape[0] // 2
        y_mid = shape[1] // 2

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            self.fail(f"matplotlib is required for PNG output in this benchmark test: {exc}")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

        axes[0, 0].imshow(poisson_mask[x_mid, :, :].T.astype(np.uint8), cmap='gray', origin='lower', aspect='auto')
        axes[0, 0].set_title(f"Poisson-like x={x_mid}")
        axes[0, 0].set_xlabel('y')
        axes[0, 0].set_ylabel('z')

        axes[0, 1].imshow(poisson_mask[:, y_mid, :].T.astype(np.uint8), cmap='gray', origin='lower', aspect='auto')
        axes[0, 1].set_title(f"Poisson-like y={y_mid}")
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('z')

        axes[1, 0].imshow(uniform_mask_ref[x_mid, :, :].T.astype(np.uint8), cmap='gray', origin='lower', aspect='auto')
        axes[1, 0].set_title(f"Uniform<={target_fraction:.2f} x={x_mid}")
        axes[1, 0].set_xlabel('y')
        axes[1, 0].set_ylabel('z')

        axes[1, 1].imshow(uniform_mask_ref[:, y_mid, :].T.astype(np.uint8), cmap='gray', origin='lower', aspect='auto')
        axes[1, 1].set_title(f"Uniform<={target_fraction:.2f} y={y_mid}")
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('z')

        fraction_tag = str(int(round(target_fraction * 100.0))).zfill(2)
        out_path = Path(f'docs/plans/sparse_keep_vs_uniform_cross_sections_32x32x64_{fraction_tag}pct.png')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

        print(
            '\nPoisson-like sparse benchmark: '
            f"trials={n_trials}, target_fraction={target_fraction:.2f}, target_count={target_count}, "
            f"mean_s={poisson_times.mean():.6f}, std_s={poisson_times.std(ddof=1):.6f}"
        )
        print(
            'Uniform threshold benchmark: '
            f"trials={n_trials}, threshold={target_fraction:.2f}, mean_s={uniform_times.mean():.6f}, "
            f"std_s={uniform_times.std(ddof=1):.6f}, sample_kept={int(uniform_mask_ref.sum())}"
        )
        print(f"PNG written to: {out_path}")

        self.assertTrue(out_path.exists())


if __name__ == '__main__':
    unittest.main()
