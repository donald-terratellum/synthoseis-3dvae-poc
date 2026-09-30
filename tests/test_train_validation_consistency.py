import tempfile
import unittest
from pathlib import Path

import zarr

from scripts import train as train_script


def make_zarr(path, source_volumes=(), scaling_mode="divide_by_std", scaling_mean=0.0, scaling_std=100.0, **extra_attrs):
    g = zarr.open_group(str(path), mode="w")
    g.create_array("patches", shape=(1, 4, 4, 4), dtype="f4")
    g.attrs["source_volumes"] = list(source_volumes)
    g.attrs["scaling_mode"] = scaling_mode
    g.attrs["scaling_mean"] = float(scaling_mean)
    g.attrs["scaling_std"] = float(scaling_std)
    for k, v in extra_attrs.items():
        g.attrs[k] = v
    return g


class TestTrainValidationConsistency(unittest.TestCase):
    def test_disjoint_volumes_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_p, val_p = Path(tmp) / "train.zarr", Path(tmp) / "val.zarr"
            make_zarr(train_p, source_volumes=["/a/vol1", "/a/vol2"])
            make_zarr(val_p, source_volumes=["/a/validation/vol3"])
            train_script.assert_train_validation_consistency(train_p, val_p)

    def test_overlapping_volumes_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_p, val_p = Path(tmp) / "train.zarr", Path(tmp) / "val.zarr"
            make_zarr(train_p, source_volumes=["/a/vol1", "/a/vol2"])
            make_zarr(val_p, source_volumes=["/a/vol2", "/a/validation/vol3"])
            with self.assertRaisesRegex(ValueError, "share 1 source"):
                train_script.assert_train_validation_consistency(train_p, val_p)

    def test_mismatched_scaling_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_p, val_p = Path(tmp) / "train.zarr", Path(tmp) / "val.zarr"
            make_zarr(train_p, source_volumes=["/a/vol1"], scaling_mean=0.0, scaling_std=100.0)
            make_zarr(val_p, source_volumes=["/a/validation/vol2"], scaling_mean=0.0, scaling_std=120.0)
            with self.assertRaisesRegex(ValueError, "scaling_std"):
                train_script.assert_train_validation_consistency(train_p, val_p)

    def test_mismatched_mean_ignored_for_divide_by_std(self):
        # divide_by_std never subtracts the mean, so noisy near-zero means must not fail the check.
        with tempfile.TemporaryDirectory() as tmp:
            train_p, val_p = Path(tmp) / "train.zarr", Path(tmp) / "val.zarr"
            make_zarr(train_p, source_volumes=["/a/vol1"], scaling_mode="divide_by_std", scaling_mean=0.0009, scaling_std=100.0)
            make_zarr(val_p, source_volumes=["/a/validation/vol2"], scaling_mode="divide_by_std", scaling_mean=0.0043, scaling_std=100.0)
            train_script.assert_train_validation_consistency(train_p, val_p)

    def test_mismatched_mean_raises_for_zscore(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_p, val_p = Path(tmp) / "train.zarr", Path(tmp) / "val.zarr"
            make_zarr(train_p, source_volumes=["/a/vol1"], scaling_mode="zscore", scaling_mean=5.0, scaling_std=100.0)
            make_zarr(val_p, source_volumes=["/a/validation/vol2"], scaling_mode="zscore", scaling_mean=8.0, scaling_std=100.0)
            with self.assertRaisesRegex(ValueError, "scaling_mean"):
                train_script.assert_train_validation_consistency(train_p, val_p)

    def test_scaling_within_tolerance_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_p, val_p = Path(tmp) / "train.zarr", Path(tmp) / "val.zarr"
            make_zarr(train_p, source_volumes=["/a/vol1"], scaling_std=100.0)
            make_zarr(val_p, source_volumes=["/a/validation/vol2"], scaling_std=100.00001)
            train_script.assert_train_validation_consistency(train_p, val_p)

    def test_mismatched_label_z_offset_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_p, val_p = Path(tmp) / "train.zarr", Path(tmp) / "val.zarr"
            make_zarr(train_p, source_volumes=["/a/vol1"], label_z_offset=1)
            make_zarr(val_p, source_volumes=["/a/validation/vol2"], label_z_offset=0)
            with self.assertRaisesRegex(ValueError, "label_z_offset"):
                train_script.assert_train_validation_consistency(train_p, val_p)

    def test_mismatched_dip_class_edges_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_p, val_p = Path(tmp) / "train.zarr", Path(tmp) / "val.zarr"
            make_zarr(train_p, source_volumes=["/a/vol1"], dip_mean_class_edges_deg=[10.0, 20.0])
            make_zarr(val_p, source_volumes=["/a/validation/vol2"], dip_mean_class_edges_deg=[10.0, 25.0])
            with self.assertRaisesRegex(ValueError, "dip_mean_class_edges_deg"):
                train_script.assert_train_validation_consistency(train_p, val_p)

    def test_skip_bypasses_all_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_p, val_p = Path(tmp) / "train.zarr", Path(tmp) / "val.zarr"
            make_zarr(train_p, source_volumes=["/a/vol1"], scaling_std=100.0)
            make_zarr(val_p, source_volumes=["/a/vol1"], scaling_std=999.0)
            train_script.assert_train_validation_consistency(train_p, val_p, skip=True)

    def test_missing_source_volumes_attr_warns_but_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_p, val_p = Path(tmp) / "train.zarr", Path(tmp) / "val.zarr"
            g1 = zarr.open_group(str(train_p), mode="w")
            g1.create_array("patches", shape=(1, 4, 4, 4), dtype="f4")
            g2 = zarr.open_group(str(val_p), mode="w")
            g2.create_array("patches", shape=(1, 4, 4, 4), dtype="f4")
            train_script.assert_train_validation_consistency(train_p, val_p)


if __name__ == "__main__":
    unittest.main()
