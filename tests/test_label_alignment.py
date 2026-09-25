import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from scripts import verify_label_alignment as vla


def _ricker(f=0.08, half=20):
    t = np.arange(-half, half + 1, dtype=np.float64)
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def _synthetic_traces(n_traces=24, seis_depth=300, pad=10, seed=0):
    """Synthoseis-like traces: labels padded by `pad`, rfc is Z-1, cumsum of bandlimited rfc."""
    rng = np.random.default_rng(seed)
    label_depth = seis_depth + 1 + pad
    labels = np.empty((n_traces, label_depth), dtype=np.float32)
    for i in range(n_traces):
        seabed = int(rng.integers(10, 30))
        labels[i, :seabed] = -1.0
        z = seabed
        while z < label_depth:
            thick = int(rng.integers(4, 25))
            labels[i, z : z + thick] = float(rng.integers(0, 2))
            z += thick
    return labels, rng


def _seismic_from_labels(labels, seis_depth, rng):
    imp_map = {-1.0: 1.5, 0.0: 5.0, 1.0: 6.5}
    wav = _ricker()
    out = np.empty((labels.shape[0], seis_depth), dtype=np.float32)
    for i, lab in enumerate(labels):
        imp = np.vectorize(imp_map.get)(lab[: seis_depth + 1]) + rng.normal(0, 0.05, seis_depth + 1)
        rfc = (imp[1:] - imp[:-1]) / (imp[1:] + imp[:-1])
        out[i] = np.cumsum(np.convolve(rfc, wav, mode="same"))
    return out


class TestEnvelope(unittest.TestCase):
    def test_envelope_of_cosine_is_amplitude(self):
        t = np.arange(512)
        x = 3.0 * np.cos(2 * np.pi * 16 * t / 512)
        env = vla.analytic_envelope(x)
        np.testing.assert_allclose(env, 3.0, atol=1e-6)


class TestRefinePeak(unittest.TestCase):
    def test_parabolic_refinement(self):
        offsets = [-2, -1, 0, 1, 2]
        curve = -((np.array(offsets) - 0.3) ** 2)
        best, refined = vla.refine_peak(curve, offsets)
        self.assertEqual(best, 0)
        self.assertAlmostEqual(refined, 0.3, places=6)

    def test_edge_peak_not_refined(self):
        best, refined = vla.refine_peak([3, 2, 1], [-1, 0, 1])
        self.assertEqual(best, -1)
        self.assertEqual(refined, -1.0)


class TestOffsetRecovery(unittest.TestCase):
    offsets = list(range(-15, 16))

    def _check(self, labels, seis, expected):
        curves = vla.offset_correlation_curves(seis, labels, self.offsets)
        for name, curve in curves.items():
            best, _ = vla.refine_peak(curve, self.offsets)
            self.assertEqual(best, expected, msg=f"{name}: {best} != {expected}")

    def test_synthoseis_layout_gives_plus_one(self):
        labels, rng = _synthetic_traces(seis_depth=300)
        seis = _seismic_from_labels(labels, 300, rng)
        self.assertEqual(labels.shape[-1] - seis.shape[-1], 11)
        self._check(labels, seis, expected=1)

    def test_positive_shift(self):
        labels, rng = _synthetic_traces(seis_depth=300, pad=20)
        seis = _seismic_from_labels(labels[:, 6:], 300, rng)
        self._check(labels, seis, expected=7)

    def test_negative_shift(self):
        labels, rng = _synthetic_traces(seis_depth=300)
        seis = _seismic_from_labels(labels, 300, rng)
        self._check(labels[:, 4:], seis, expected=-3)

    def test_rejects_offsets_without_overlap(self):
        with self.assertRaises(ValueError):
            vla.offset_correlation_curves(np.zeros((2, 20)), np.zeros((2, 20)), range(-15, 16))


class TestVolumeAndSummary(unittest.TestCase):
    def _write_volume(self, root, seed):
        nx = ny = 10
        seis_depth = 200
        labels, rng = _synthetic_traces(n_traces=nx * ny, seis_depth=seis_depth, seed=seed)
        seis = _seismic_from_labels(labels, seis_depth, rng)
        g = zarr.open_group(str(root), mode="w")
        g.create_array("seismicCubes_cumsum_fullstack", data=seis.reshape(nx, ny, -1), chunks=(5, 5, seis_depth))
        g.create_array("faulted_lithology", data=labels.reshape(nx, ny, -1), chunks=(5, 5, labels.shape[-1]))

    def test_measure_and_summarize(self):
        offsets = list(range(-15, 16))
        with tempfile.TemporaryDirectory() as tmp:
            per_volume = []
            for seed in range(5):
                root = Path(tmp) / f"seismic__v{seed}" / "model_data.zarr"
                self._write_volume(root, seed)
                zvol = zarr.open_group(str(root), mode="r")
                res = vla.measure_volume(
                    zvol, "seismicCubes_cumsum_fullstack", "faulted_lithology", offsets,
                    n_blocks=4, trace_stride=1, rng=random.Random(seed),
                )
                self.assertEqual(res["n_traces"], 100)
                self.assertEqual(res["label_depth"] - res["seismic_depth"], 11)
                per_volume.append(res)
            self.assertEqual(len(vla.list_volumes(tmp)), 5)

        summary = vla.summarize(per_volume, offsets)
        self.assertEqual(summary["recommended_label_z_offset"], 1)
        self.assertTrue(summary["verified"])
        self.assertEqual(summary["envelope"]["n_agree_within_tolerance"], 5)

    def test_summary_not_verified_with_few_volumes(self):
        offsets = [0, 1, 2]
        vol = {name: {"best_offset": 1, "curve": [0.1, 0.9, 0.1]} for name in ("envelope", "signed_diff")}
        summary = vla.summarize([vol] * 4, offsets)
        self.assertEqual(summary["recommended_label_z_offset"], 1)
        self.assertFalse(summary["verified"])


if __name__ == "__main__":
    unittest.main()
