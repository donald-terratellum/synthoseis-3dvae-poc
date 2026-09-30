import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
import zarr

from scripts import sample_patches as sp
from scripts import add_dip_samples as ads
from scripts import train as train_script
from src.augmentations import (
    apply_pair_augmentations,
    apply_vertical_warp_to_cube,
    sample_vertical_warp_target_indices,
)
from src.geology_classifier import compute_geology_classifier_loss
from src.geology_label_augment import (
    DIP_IGNORE_CLASS,
    adjust_azimuth_deg,
    adjusted_dip_values_for_stretch,
    dip_field_deg,
    sample_dip_field,
    vertical_warp_stretch_at_source,
)

PATCH = (16, 16, 32)


def curved_age_field(shape=PATCH):
    """Geologic age increasing with depth, with dips varying across the patch."""
    x = np.arange(shape[0], dtype=np.float32)[:, None, None]
    y = np.arange(shape[1], dtype=np.float32)[None, :, None]
    z = np.arange(shape[2], dtype=np.float32)[None, None, :]
    return (z * (1.0 + 0.03 * x) + 0.4 * x + 1.5 * np.sin(y / 3.0) + 0.2 * z * np.cos(y / 5.0)).astype(np.float32)


def dip_stats_exact(field):
    mean, std, dip_range, azimuth, variance = sp._compute_dip_azimuth_features(field)
    return {"meta_dip_mean_deg": mean, "meta_dip_std_deg": std, "meta_dip_range_deg": dip_range}, azimuth, variance


class TestAzimuth(unittest.TestCase):
    def test_matches_recomputed_azimuth_for_all_flip_swap_combinations(self):
        field = curved_age_field()
        _, az0, _ = dip_stats_exact(field)
        for swap in (False, True):
            for fx in (False, True):
                for fy in (False, True):
                    f = field
                    if swap:
                        f = np.swapaxes(f, 0, 1)
                    if fx:
                        f = f[::-1, :, :]
                    if fy:
                        f = f[:, ::-1, :]
                    _, expected, _ = dip_stats_exact(np.ascontiguousarray(f))
                    got = adjust_azimuth_deg(az0, swap, fx, fy)
                    diff = abs((got - expected + 180.0) % 360.0 - 180.0)
                    self.assertLess(diff, 1e-3, msg=f"swap={swap} flip_x={fx} flip_y={fy}")


class TestVerticalWarpDip(unittest.TestCase):
    def test_identity_warp_returns_stored_values(self):
        field = curved_age_field()
        stored, _, _ = dip_stats_exact(field)
        dips, z_idx = sample_dip_field(dip_field_deg(field)[0], 512, np.random.default_rng(0))
        stretch = vertical_warp_stretch_at_source(np.arange(PATCH[2], dtype=np.float32), z_idx)
        new = adjusted_dip_values_for_stretch(stored, dips, stretch)
        for key, value in stored.items():
            self.assertAlmostEqual(new[key], value, places=6)

    def test_constant_stretch_matches_tan_rule(self):
        stored = {"meta_dip_mean_deg": 45.0, "meta_dip_std_deg": 0.0, "meta_dip_range_deg": 0.0}
        new = adjusted_dip_values_for_stretch(stored, np.full(64, 45.0), np.full(64, 2.0))
        self.assertAlmostEqual(new["meta_dip_mean_deg"], np.degrees(np.arctan(2.0)), places=3)
        self.assertEqual(new["meta_dip_mean_class"], 5.0)

    def test_adjusted_stats_match_dips_recomputed_on_warped_field(self):
        field = curved_age_field()
        stored, _, _ = dip_stats_exact(field)
        dips, z_idx = sample_dip_field(dip_field_deg(field)[0], 512, np.random.default_rng(1))
        np.random.seed(3)
        worst = {k: 0.0 for k in stored}
        for _ in range(10):
            t = sample_vertical_warp_target_indices(PATCH[2], min_step=0.5, max_step=2.0,
                                                    z_low_ratio=0.5, z_mode_ratio=1.0, z_high_ratio=1.5)
            truth, _, _ = dip_stats_exact(apply_vertical_warp_to_cube(field, t))
            new = adjusted_dip_values_for_stretch(stored, dips, vertical_warp_stretch_at_source(t, z_idx))
            for k in stored:
                worst[k] = max(worst[k], abs(new[k] - truth[k]))
        self.assertLess(worst["meta_dip_mean_deg"], 1.0, worst)
        self.assertLess(worst["meta_dip_std_deg"], 1.0, worst)
        self.assertLess(worst["meta_dip_range_deg"], 2.0, worst)

    def test_pair_augmentations_params_and_rng_unchanged(self):
        x = np.random.default_rng(0).normal(size=PATCH).astype(np.float32)
        np.random.seed(11)
        a = apply_pair_augmentations(x.copy(), x.copy(), 0.5, 0.5, 0.5, 1.0)
        np.random.seed(11)
        b = apply_pair_augmentations(x.copy(), x.copy(), 0.5, 0.5, 0.5, 1.0, return_params=True)
        np.testing.assert_array_equal(a[1], b[1])
        self.assertEqual(
            set(b[2]),
            {"swap_xy", "flip_x", "flip_y", "vertical_warp_indices", "phase_deg"},
        )
        self.assertIsNone(b[2]["phase_deg"])
        self.assertIsNotNone(b[2]["vertical_warp_indices"])


def make_dataset_store(root, with_dip_samples=True, n=6):
    field = curved_age_field()
    stored, az, var = dip_stats_exact(field)
    g = zarr.open_group(str(root), mode="w")
    g.create_array("patches", data=np.random.default_rng(0).normal(size=(n,) + PATCH).astype(np.float32))
    for key, value in stored.items():
        g.create_array(key, data=np.full(n, value, np.float32))
    g.create_array("meta_dip_mean_class", data=np.full(n, sp.quantize(stored["meta_dip_mean_deg"], sp.DIP_MEAN_CLASS_EDGES_DEG), np.float32))
    g.create_array("meta_dip_range_class", data=np.full(n, sp.quantize(stored["meta_dip_range_deg"], sp.DIP_RANGE_CLASS_EDGES_DEG), np.float32))
    g.create_array("meta_azimuth_mean_deg", data=np.full(n, az, np.float32))
    g.create_array("meta_azimuth_circular_variance", data=np.full(n, var, np.float32))
    if with_dip_samples:
        dips, z_idx = sample_dip_field(dip_field_deg(field)[0], 512, np.random.default_rng(2))
        sp.write_dip_samples(g, np.tile(dips, (n, 1)), np.tile(z_idx, (n, 1)))
    return field, stored, az


class TestDatasetLabelAdjustment(unittest.TestCase):
    DIP_KEYS = ("meta_dip_mean_class", "meta_dip_range_class")

    def _dataset(self, path, policy, warp=1.0, swap=0.0, flip=0.0, metadata_keys=("meta_azimuth_mean_deg",)):
        return train_script.ZarrPatchDataset(
            path, augment=True, swap_xy_prob=swap, flip_x_prob=flip, flip_y_prob=flip,
            vertical_warp_prob=warp, mixup_augment_prob=0.0, include_metadata=True,
            geology_metadata_keys=metadata_keys, label_target_keys=self.DIP_KEYS, dip_label_policy=policy,
        )

    def test_adjust_matches_recomputed_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.zarr"
            field, _, _ = make_dataset_store(path)
            ds = self._dataset(path, "adjust")
            captured = {}
            original = sample_vertical_warp_target_indices

            def spy(nz, *a, **k):
                captured["t"] = original(nz, *a, **k)
                return captured["t"]

            np.random.seed(5)
            with mock.patch("src.augmentations.sample_vertical_warp_target_indices", side_effect=spy):
                _, _, meta = ds[0]
            truth, _, _ = dip_stats_exact(apply_vertical_warp_to_cube(field, captured["t"]))
            self.assertAlmostEqual(
                float(meta["meta_dip_mean_class"]), sp.quantize(truth["meta_dip_mean_deg"], sp.DIP_MEAN_CLASS_EDGES_DEG)
            )

    def test_mask_and_ignore_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.zarr"
            make_dataset_store(path, with_dip_samples=False)
            _, _, meta = self._dataset(path, "mask")[0]
            for key in self.DIP_KEYS:
                self.assertEqual(float(meta[key]), DIP_IGNORE_CLASS)
            stored = float(zarr.open_group(str(path), mode="r")["meta_dip_mean_class"][0])
            _, _, meta = self._dataset(path, "ignore")[0]
            self.assertEqual(float(meta["meta_dip_mean_class"]), stored)

    def test_adjust_without_dip_samples_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.zarr"
            make_dataset_store(path, with_dip_samples=False)
            with self.assertRaisesRegex(KeyError, "add_dip_samples"):
                self._dataset(path, "adjust")
            self._dataset(path, "adjust", warp=0.0)

    def test_azimuth_adjusted_for_swap_and_flips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.zarr"
            _, _, az = make_dataset_store(path)
            _, _, meta = self._dataset(path, "adjust", warp=0.0, swap=1.0, flip=1.0)[0]
            self.assertAlmostEqual(float(meta["meta_azimuth_mean_deg"]), adjust_azimuth_deg(az, True, True, True), places=3)
            _, _, meta = self._dataset(path, "adjust", warp=0.0)[0]
            self.assertAlmostEqual(float(meta["meta_azimuth_mean_deg"]), az, places=4)


class TestClassifierIgnoresMaskedDip(unittest.TestCase):
    def test_masked_rows_are_skipped(self):
        torch.manual_seed(0)
        logits = {"presence": torch.randn(4, 7), "dip_mean": torch.randn(4, 6), "dip_range": torch.randn(4, 6)}
        batch = {"meta_dip_mean_class": torch.tensor([1.0, -1.0, 3.0, -1.0])}
        _, parts = compute_geology_classifier_loss(logits, batch, targets=("dip_mean",), label_smoothing=0.0)
        expected = F.cross_entropy(logits["dip_mean"][[0, 2]], torch.tensor([1, 3]))
        self.assertAlmostEqual(parts["dip_mean"], float(expected), places=5)
        batch = {"meta_dip_mean_class": torch.tensor([-1.0] * 4)}
        total, parts = compute_geology_classifier_loss(logits, batch, targets=("dip_mean",))
        self.assertNotIn("dip_mean", parts)
        self.assertEqual(float(total), 0.0)


def make_source_volume(root, shape=(24, 24, 48)):
    g = zarr.open_group(str(root), mode="w")
    rng = np.random.default_rng(0)
    g.create_array(sp.DEFAULT_SEISMIC_KEY, data=rng.normal(size=shape).astype(np.float32))
    lshape = (shape[0], shape[1], shape[2] + 2)
    g.create_array("geologic_score", data=np.abs(rng.normal(size=lshape)).astype(np.float32))
    g.create_array("geologic_age_faulted", data=curved_age_field(lshape))


class TestSamplePatchesDipSamples(unittest.TestCase):
    def test_main_writes_dip_samples_and_backfill_reproduces_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_source_volume(Path(tmp) / "src" / "v0" / "model_data.zarr")
            out = Path(tmp) / "out.zarr"
            argv = ["sample_patches.py", "--source", str(Path(tmp) / "src"), "--out", str(out),
                    "--patch_size", "8", "8", "16", "--n_patches", "5", "--n_per_volume", "5",
                    "--seed", "3", "--label_z_offset", "1", "--dip_sample_count", "64"]
            with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
                sp.main()
            g = zarr.open_group(str(out), mode="r")
            dips = np.asarray(g["dip_samples_deg"])
            self.assertEqual(dips.shape, (5, 64))
            self.assertEqual(np.asarray(g["dip_samples_z"]).max() < 16, True)
            self.assertEqual(int(g.attrs["dip_sample_count"]), 64)

            with mock.patch("builtins.print"):
                worst = ads.add_dip_samples(out, n_samples=64, verify=5)
            self.assertLess(worst, 1e-4)
            np.testing.assert_array_equal(np.asarray(zarr.open_group(str(out), mode="r")["dip_samples_deg"]), dips)


if __name__ == "__main__":
    unittest.main()
