import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import zarr

from scripts import sample_patches as sp

SEIS_SHAPE = (24, 24, 40)
LABEL_PAD = 11


def make_labeled_volume(root, seed=0, shape=SEIS_SHAPE, with_fault_x=True, chunks=(8, 8, None)):
    """Synthoseis-like volume: labels are LABEL_PAD samples deeper than seismic."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = shape
    lshape = (nx, ny, nz + LABEL_PAD)
    lchunks = (chunks[0], chunks[1], lshape[2])
    g = zarr.open_group(str(root), mode="w")

    def put(key, data):
        g.create_array(key, data=data, chunks=lchunks if data.shape == lshape else (chunks[0], chunks[1], data.shape[2]))

    put(sp.DEFAULT_SEISMIC_KEY, rng.normal(size=shape).astype(np.float32))
    put("geologic_score", np.abs(rng.normal(size=lshape)).astype(np.float32))
    x = np.arange(nx, dtype=np.float32)[:, None, None]
    z = np.arange(lshape[2], dtype=np.float32)[None, None, :]
    put("geologic_age_faulted", np.broadcast_to(z + 0.3 * x, lshape).astype(np.float32))

    lith = np.zeros(lshape, dtype=np.float32)
    lith[..., :4] = -1.0
    for z0 in range(4, lshape[2], 6):
        lith[..., z0:z0 + 6] = float(rng.integers(0, 2))
    put("faulted_lithology", lith)

    fault = np.zeros(lshape, dtype=np.uint32)
    fault[5, :, :] = 3
    fault[:, 15, :] = 7
    put("fault_segments_id", fault)
    fx = np.zeros(lshape, dtype=np.float32)
    if with_fault_x:
        fx[5, 15, 20:30] = 1.0
    put("fault_intersection_segments", fx)

    closure = np.zeros(lshape, dtype=np.uint16)
    closure[10:14, 2:6, 25:35] = 5
    put("closure_segments_id", closure)
    onlap = np.zeros(lshape, dtype=np.float32)
    onlap[18:22, 18:22, 10:20] = 0.8
    onlap[0:2, 0:2, 10:12] = 0.2
    put("onlap_segments", onlap)
    flat = np.zeros(lshape, dtype=np.uint8)
    flat[12:20, 8:12, 40:42] = 1
    put("flat_spot", flat)

    chan = np.zeros(lshape, dtype=np.uint8)
    chan[2:10, 18:22, 30:34] = 1
    chan[4:8, 19:21, 31:33] = 2
    g.create_group("faults").create_array("faulted_channel_labels", data=chan, chunks=lchunks)
    return g


def _run_main(argv):
    with mock.patch.object(sys, "argv", ["sample_patches.py"] + argv), mock.patch("builtins.print"):
        sp.main()


def geoscore_output_hashes(out):
    """Hashes of every array the pre-WP2 sampler wrote."""
    dst = zarr.open_group(str(out), mode="r")
    keys = ["patches", "source_volume_index", "origin_x", "origin_y", "origin_z"] + [
        k for k in sp.DERIVED_METADATA_KEYS
    ]
    return {k: hashlib.sha256(np.ascontiguousarray(np.asarray(dst[k])).tobytes()).hexdigest()[:16] for k in keys}


REGRESSION_ARGV = ["--patch_size", "8", "8", "16", "--n_patches", "20", "--n_per_volume", "10", "--seed", "7"]
# Generated with the pre-WP2 (commit a79799a) sampler on make_labeled_volume seeds 0 and 1.
GOLDEN_GEOSCORE_HASHES = {
    "patches": "7b690124f90d7bc6",
    "source_volume_index": "e1c3f510b9538862",
    "origin_x": "d29a75532dfeaa69",
    "origin_y": "8ce10adc3ff69afd",
    "origin_z": "9dc6b18c3f2efbc0",
    "meta_dip_mean_deg": "2d127a5930ebccc1",
    "meta_dip_std_deg": "1024607505b6da9f",
    "meta_azimuth_mean_deg": "5b6fb58e61fa4759",
    "meta_azimuth_circular_variance": "5b6fb58e61fa4759",
    "meta_fault_fraction": "11a0c8c0e14167f7",
    "meta_fault_intersection_fraction": "c1f78f74444b1708",
    "meta_geologic_score_mean": "a4b5718a368ef9b5",
    "meta_sand_fraction": "f8ff9998d6f89ab3",
    "meta_shale_fraction": "d2183f0346579940",
    "meta_flat_spot_fraction": "5b6fb58e61fa4759",
    "meta_onlap_fraction": "5e5f003a16e07600",
    "meta_onlap_variability": "ab44d00318b393f9",
    "meta_channel_fraction": "8766ec8796e6b71d",
    "meta_channel_core_fraction": "67ae92c256a1cf3f",
    "meta_structural_complexity": "365b8344677220e7",
    "meta_dip_range_deg": "a5ecb879aa362dc0",
    "meta_dip_mean_class": "822d06342bcaebfe",
    "meta_dip_range_class": "5b6fb58e61fa4759",
    "meta_water_fraction": "d11aabc21d8dff66",
    "meta_closure_fraction": "3815b79c324f30bc",
}


class TestSharesAndSlots(unittest.TestCase):
    def test_parse_and_normalize(self):
        q = sp.parse_class_quotas(["fault=0.2", "fault_x=0.2"])
        shares = sp.normalize_shares(q, 0.6)
        self.assertAlmostEqual(sum(shares.values()), 1.0)
        self.assertAlmostEqual(shares["fault"], 0.2)
        self.assertEqual(shares["sand"], 0.0)
        with self.assertRaises(ValueError):
            sp.parse_class_quotas(["bogus=0.1"])
        with self.assertRaises(ValueError):
            sp.parse_class_quotas(["fault"])

    def test_default_shares(self):
        shares = sp.normalize_shares(sp.DEFAULT_CLASS_QUOTAS, sp.DEFAULT_BACKGROUND_FRACTION)
        self.assertAlmostEqual(shares[sp.BACKGROUND], 0.25)
        self.assertAlmostEqual(shares["fault_x"], 0.125)

    def test_allocate_slots_exact_counts(self):
        shares = {"a": 0.5, "b": 0.3, "c": 0.2}
        slots = sp.allocate_slots(10, shares, np.random.default_rng(0))
        self.assertEqual(len(slots), 10)
        self.assertEqual({k: slots.count(k) for k in shares}, {"a": 5, "b": 3, "c": 2})
        self.assertEqual(len(sp.allocate_slots(7, shares, np.random.default_rng(0))), 7)


class CountingArray:
    """Array wrapper that records the size of every read."""

    def __init__(self, arr):
        self._arr = arr
        self.shape = arr.shape
        self.chunks = arr.chunks
        self.read_sizes = []

    def __getitem__(self, item):
        out = np.asarray(self._arr[item])
        self.read_sizes.append(out.size)
        return out


class TestAnchorIndex(unittest.TestCase):
    def test_coords_satisfy_rule_and_are_in_seismic_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = make_labeled_volume(Path(tmp) / "v.zarr")
            index = sp.build_anchor_index(g, SEIS_SHAPE, label_z_offset=1, max_coords=10**6)
            for name, entry in index.items():
                key = sp.LABEL_CLASS_SOURCES[name]
                arr = np.asarray(g[key])
                coords = entry["coords"]
                self.assertGreater(len(coords), 0, name)
                self.assertTrue(np.all(coords >= 0))
                self.assertTrue(np.all(coords < np.array(SEIS_SHAPE)))
                vals = arr[coords[:, 0], coords[:, 1], coords[:, 2] + 1]
                self.assertTrue(np.all(sp.class_mask(name, vals)), name)
                self.assertEqual(entry["count"], len(coords))
            self.assertEqual(set(np.unique(index["fault"]["ids"])), {3, 7})
            # Onlap below threshold (0.2) is not an anchor.
            self.assertEqual(index["onlap"]["count"], 4 * 4 * 10)

    def test_reservoir_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = make_labeled_volume(Path(tmp) / "v.zarr")
            index = sp.build_anchor_index(g, SEIS_SHAPE, max_coords=50, classes=("fault",))
            self.assertEqual(len(index["fault"]["coords"]), 50)
            self.assertGreater(index["fault"]["count"], 50)

    def test_built_chunk_by_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = make_labeled_volume(Path(tmp) / "v.zarr")
            wrapped = {k: CountingArray(g[k]) for k in ("fault_segments_id",)}
            sp.build_anchor_index(wrapped, SEIS_SHAPE, classes=("fault",))
            arr = wrapped["fault_segments_id"]
            chunk_size = int(np.prod(arr.chunks))
            self.assertEqual(len(arr.read_sizes), 9)
            self.assertLessEqual(max(arr.read_sizes), chunk_size)
            self.assertLess(max(arr.read_sizes), int(np.prod(arr.shape)))

    def test_missing_class_array(self):
        index = sp.build_anchor_index({}, SEIS_SHAPE)
        self.assertTrue(all(len(e["coords"]) == 0 for e in index.values()))


class TestAnchoredSampling(unittest.TestCase):
    patch = (8, 8, 16)

    def _items(self, g, n=400, shares=None, **kw):
        rng = np.random.default_rng(3)
        return sp.sample_labeled_patches(
            g, sp.DEFAULT_SEISMIC_KEY, self.patch, n, rng, shares=shares, label_z_offset=1, **kw
        )

    def test_anchor_class_is_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = make_labeled_volume(Path(tmp) / "v.zarr")
            items, stats = self._items(g, max_patches_per_object=0)
            anchored = [it for it in items if it["anchor_class"] >= 0]
            self.assertGreater(len(anchored), 0)
            for it in anchored:
                name = sp.LABEL_CLASS_ORDER[it["anchor_class"]]
                self.assertGreaterEqual(it["counts"][name], 1, name)

    def test_center_jitter_contains_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = make_labeled_volume(Path(tmp) / "v.zarr")
            items, _ = self._items(g, n=100, anchor_jitter="center", max_patches_per_object=0)
            for it in items:
                if it["anchor_class"] >= 0:
                    self.assertGreaterEqual(it["counts"][sp.LABEL_CLASS_ORDER[it["anchor_class"]]], 1)

    def test_achieved_shares_match_quotas(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = make_labeled_volume(Path(tmp) / "v.zarr")
            shares = sp.normalize_shares(sp.DEFAULT_CLASS_QUOTAS, sp.DEFAULT_BACKGROUND_FRACTION)
            n = 400
            items, stats = self._items(g, n=n, shares=shares, max_patches_per_object=0)
            anchors = [it["anchor_class"] for it in items]
            for ci, name in enumerate(sp.LABEL_CLASS_ORDER):
                self.assertLessEqual(abs(anchors.count(ci) / n - shares[name]), 0.03, name)
            self.assertLessEqual(abs(anchors.count(-1) / n - shares[sp.BACKGROUND]), 0.03)
            self.assertEqual(sum(stats["fallback"].values()), 0)

    def test_fallback_when_class_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = make_labeled_volume(Path(tmp) / "v.zarr", with_fault_x=False)
            shares = sp.normalize_shares(sp.DEFAULT_CLASS_QUOTAS, sp.DEFAULT_BACKGROUND_FRACTION)
            items, stats = self._items(g, n=80, shares=shares, max_patches_per_object=0)
            self.assertEqual(stats["fallback"]["fault_x"], 10)
            self.assertEqual(stats["anchored"]["fault_x"], 0)
            self.assertEqual([it["anchor_class"] for it in items].count(-1), 20 + 10)

    def test_max_patches_per_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = make_labeled_volume(Path(tmp) / "v.zarr")
            shares = sp.normalize_shares({"closure": 1.0}, 0.0)
            items, stats = self._items(g, n=20, shares=shares, max_patches_per_object=3)
            self.assertEqual(stats["anchored"]["closure"], 3)
            self.assertEqual(stats["object_cap_fallback"]["closure"], 17)

    def test_inclusion_weight_and_label_patches(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = make_labeled_volume(Path(tmp) / "v.zarr")
            items, _ = self._items(g, n=60, store_label_patches=True)
            for it in items:
                w = it["inclusion_weight"]
                self.assertTrue(np.isfinite(w) and w > 0)
                lp = it["label_patch"]
                self.assertEqual(lp.shape, (len(sp.LABEL_CLASS_ORDER),) + self.patch)
                self.assertEqual(lp.dtype, np.uint8)
                for ci, name in enumerate(sp.LABEL_CLASS_ORDER):
                    self.assertEqual(int(lp[ci].sum()), it["counts"][name])
            uni, _ = self._items(g, n=10, sampling_mode="uniform")
            self.assertTrue(all(it["inclusion_weight"] == 1.0 and it["anchor_class"] == -1 for it in uni))

    def test_inclusion_weight_formula(self):
        realized = {sp.BACKGROUND: 0.5, "fault": 0.5}
        densities = {"fault": 0.01}
        # A patch with the average fault count gets q/p = 0.5 + 0.5 * 1 -> weight 1.
        self.assertAlmostEqual(sp.compute_inclusion_weight({"fault": 10}, realized, densities, 1000), 1.0)
        self.assertAlmostEqual(sp.compute_inclusion_weight({"fault": 0}, realized, densities, 1000), 2.0)

    def test_presence_threshold_and_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = make_labeled_volume(Path(tmp) / "v.zarr")
            # Flat spot occupies label z 40..41 -> seismic z 39..40 with offset 1.
            origin = (12, 8, 24)
            counts, presence, _ = sp.compute_label_presence(g, origin, self.patch, label_z_offset=1, presence_min_voxels=32)
            self.assertEqual(counts["flat_spot"], 8 * 4 * 1)
            self.assertEqual(presence["flat_spot"], 1)
            counts0, _, _ = sp.compute_label_presence(g, origin, self.patch, label_z_offset=0)
            self.assertEqual(counts0["flat_spot"], 0)
            _, presence_hi, _ = sp.compute_label_presence(g, origin, self.patch, label_z_offset=1, presence_min_voxels=33)
            self.assertEqual(presence_hi["flat_spot"], 0)


class TestMain(unittest.TestCase):
    def _make_source(self, tmp, **kw):
        root = Path(tmp) / "src"
        make_labeled_volume(root / "seismic__a" / "model_data.zarr", seed=0, **kw)
        make_labeled_volume(root / "seismic__b" / "model_data.zarr", seed=1, **kw)
        return root

    def test_geoscore_mode_matches_pre_wp2_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_source(tmp)
            out = Path(tmp) / "out.zarr"
            _run_main(["--source", str(root), "--out", str(out)] + REGRESSION_ARGV)
            self.assertEqual(geoscore_output_hashes(out), GOLDEN_GEOSCORE_HASHES)

    def test_class_anchored_main_writes_arrays_and_attrs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_source(tmp, with_fault_x=False)
            out = Path(tmp) / "out.zarr"
            _run_main([
                "--source", str(root), "--out", str(out), "--patch_size", "8", "8", "16",
                "--n_patches", "32", "--n_per_volume", "16", "--seed", "5",
                "--sampling_mode", "class_anchored", "--label_z_offset", "1",
                "--class_quotas", "fault=0.25", "fault_x=0.25", "closure=0.25",
                "--background_fraction", "0.25", "--store_label_patches",
            ])
            dst = zarr.open_group(str(out), mode="r")
            attrs = dict(dst.attrs)
            self.assertEqual(attrs["sampling_mode"], "class_anchored")
            self.assertEqual(attrs["label_z_offset"], 1)
            self.assertEqual(attrs["label_class_order"], list(sp.LABEL_CLASS_ORDER))
            self.assertEqual(attrs["fallback_counts"]["fault_x"], 8)
            self.assertEqual(attrs["anchored_counts"]["fault"], 8)
            self.assertAlmostEqual(sum(attrs["class_quotas"].values()) + attrs["background_fraction"], 1.0)
            for name in sp.LABEL_CLASS_ORDER:
                arr = dst[f"label_presence_{name}"]
                self.assertEqual(arr.shape, (32,))
                self.assertEqual(arr.dtype, np.uint8)
            self.assertEqual(dst["anchor_class"].dtype, np.int8)
            w = np.asarray(dst["inclusion_weight"])
            self.assertTrue(np.all(np.isfinite(w)) and np.all(w > 0))
            lp = dst["label_patches"]
            self.assertEqual(lp.shape, (32, 7, 8, 8, 16))
            self.assertEqual(lp.dtype, np.uint8)
            self.assertEqual(tuple(lp.chunks), (1, 7, 8, 8, 16))

    def test_skips_volumes_without_labels_in_labeled_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "src"
            make_labeled_volume(root / "seismic__a" / "model_data.zarr", seed=0)
            bare = zarr.open_group(str(root / "seismic__b" / "model_data.zarr"), mode="w")
            bare.create_array(sp.DEFAULT_SEISMIC_KEY, data=np.ones(SEIS_SHAPE, dtype=np.float32))
            out = Path(tmp) / "out.zarr"
            _run_main(["--source", str(root), "--out", str(out), "--patch_size", "8", "8", "16",
                       "--n_patches", "8", "--n_per_volume", "4", "--seed", "1", "--sampling_mode", "uniform"])
            dst = zarr.open_group(str(out), mode="r")
            self.assertEqual(dst.attrs["skipped_volumes_missing_labels"], [str(root / "seismic__b" / "model_data.zarr")])
            self.assertEqual(dst.attrs["n_written"], 4)
            self.assertTrue(np.all(np.asarray(dst["source_volume_index"])[:4] == 0))

    def test_disjoint_from(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_source(tmp)
            first = Path(tmp) / "first.zarr"
            base = ["--patch_size", "8", "8", "16", "--n_patches", "4", "--n_per_volume", "2", "--seed", "1",
                    "--sampling_mode", "uniform"]
            _run_main(["--source", str(root), "--out", str(first)] + base)
            with self.assertRaises(SystemExit):
                _run_main(["--source", str(root), "--out", str(Path(tmp) / "second.zarr"),
                           "--disjoint_from", str(first)] + base)


if __name__ == "__main__":
    unittest.main()
