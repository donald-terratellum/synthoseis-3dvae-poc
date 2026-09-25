import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import zarr

from scripts import sample_patches as sp


def _write_volume(root, shape=(8, 8, 16), seismic_key=sp.DEFAULT_SEISMIC_KEY):
    g = zarr.open_group(str(root), mode="w")
    rng = np.random.default_rng(0)
    g.create_array(seismic_key, data=rng.normal(size=shape).astype(np.float32))
    g.create_array("geologic_score", data=np.ones(shape, dtype=np.float32))
    # Depth layout: 4 water, 4 shale, 8 sand samples.
    lith = np.empty(shape, dtype=np.float32)
    lith[..., :4] = -1.0
    lith[..., 4:8] = 0.0
    lith[..., 8:] = 1.0
    g.create_array("faulted_lithology", data=lith)
    closure = np.zeros(shape, dtype=np.uint16)
    closure[:4, :, 8:] = 7
    g.create_array("closure_segments_id", data=closure)
    return g


class TestLithologyFractions(unittest.TestCase):
    def test_known_layout(self):
        lith = np.array([-1, -1, 0, 0, 0, 1, 0.3, 0.7], dtype=np.float32)
        sand, shale, water = sp.compute_lithology_fractions(lith)
        self.assertAlmostEqual(water, 2 / 8)
        self.assertAlmostEqual(sand, 2 / 6)
        self.assertAlmostEqual(shale, 4 / 6)

    def test_shale_is_not_half_sand(self):
        sand, shale, water = sp.compute_lithology_fractions(np.zeros((4, 4, 4), dtype=np.float32))
        self.assertEqual((sand, shale, water), (0.0, 1.0, 0.0))

    def test_all_water(self):
        self.assertEqual(sp.compute_lithology_fractions(-np.ones(10, dtype=np.float32)), (0.0, 0.0, 1.0))

    def test_threshold(self):
        lith = np.array([0.4, 0.6], dtype=np.float32)
        self.assertEqual(sp.compute_lithology_fractions(lith, sand_threshold=0.5)[0], 0.5)
        self.assertEqual(sp.compute_lithology_fractions(lith, sand_threshold=0.3)[0], 1.0)


class TestPatchMetadata(unittest.TestCase):
    def test_exact_fractions_from_synthetic_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = _write_volume(Path(tmp) / "model_data.zarr")
            meta = sp.compute_patch_derived_metadata(g, (0, 0, 0), (8, 8, 16), "geologic_score")
            self.assertAlmostEqual(meta["meta_water_fraction"], 0.25)
            self.assertAlmostEqual(meta["meta_sand_fraction"], 8 / 12)
            self.assertAlmostEqual(meta["meta_shale_fraction"], 4 / 12)
            self.assertAlmostEqual(meta["meta_closure_fraction"], 0.25)

            meta = sp.compute_patch_derived_metadata(g, (0, 0, 0), (8, 8, 4), "geologic_score")
            self.assertEqual(meta["meta_water_fraction"], 1.0)
            self.assertEqual(meta["meta_sand_fraction"], 0.0)
            self.assertEqual(meta["meta_shale_fraction"], 0.0)

    def test_new_keys_registered(self):
        for key in ("meta_water_fraction", "meta_closure_fraction"):
            self.assertIn(key, sp.DERIVED_METADATA_KEYS)


class TestVolumeListing(unittest.TestCase):
    def _make_tree(self, root):
        paths = {
            "train": root / "seismic__a" / "model_data.zarr",
            "val": root / "validation" / "seismic__b" / "model_data.zarr",
            "val_nested": root / "validation" / "sub" / "seismic__c" / "model_data.zarr",
            "temp": root / "seismic__d" / "model_data.zarr",
        }
        for p in paths.values():
            p.mkdir(parents=True)
        (root / "temp_folder__d").mkdir()
        return paths

    def test_exclude_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._make_tree(root)
            all_vols = sp.list_source_volumes(root)
            self.assertEqual(all_vols, sorted([paths["train"], paths["val"], paths["val_nested"]]))
            self.assertEqual(sp.list_source_volumes(root, ["validation"]), [paths["train"]])

    def test_exclude_only_matches_below_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._make_tree(root)
            vols = sp.list_source_volumes(root / "validation", ["validation"])
            self.assertEqual(vols, sorted([paths["val"], paths["val_nested"]]))


class TestMainDefaults(unittest.TestCase):
    def test_default_seismic_key_and_exclude_dir_end_to_end(self):
        self.assertEqual(sp.DEFAULT_SEISMIC_KEY, "seismicCubes_cumsum_fullstack")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "src"
            _write_volume(root / "seismic__a" / "model_data.zarr")
            _write_volume(root / "validation" / "seismic__b" / "model_data.zarr")
            out = Path(tmp) / "out.zarr"
            argv = [
                "sample_patches.py", "--source", str(root), "--out", str(out),
                "--patch_size", "4", "4", "8", "--n_patches", "4", "--n_per_volume", "4",
                "--seed", "1", "--exclude_dir", "validation",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
                sp.main()
            dst = zarr.open_group(str(out), mode="r")
            self.assertEqual(dst.attrs["source_volumes"], [str(root / "seismic__a" / "model_data.zarr")])
            self.assertEqual(dst.attrs["exclude_dirs"], ["validation"])
            self.assertEqual(dst.attrs["sand_threshold"], sp.DEFAULT_SAND_THRESHOLD)
            self.assertTrue(np.all(np.asarray(dst["source_volume_index"]) == 0))
            self.assertTrue(np.any(np.asarray(dst["patches"]) != 0))
            for key in ("meta_water_fraction", "meta_closure_fraction"):
                self.assertIn(key, dst)


if __name__ == "__main__":
    unittest.main()
