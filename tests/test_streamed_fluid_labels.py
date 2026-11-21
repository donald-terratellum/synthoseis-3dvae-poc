import unittest

import numpy as np

from scripts.streamed_fluid_labels import (
    CHANNEL_KEY,
    FLUID_KEYS,
    VERIFIED_SEISMIC_FILES,
    FluidCode,
    FluidReason,
    inspect_fluid_fill,
)


class FluidDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.sha = next(iter(VERIFIED_SEISMIC_FILES))
        self.shape = (1, 1, 6)
        self.crop = {
            "faulted_lithology": np.ones(self.shape, dtype="f4"),
            "geologic_age_faulted": np.full(self.shape, 2, dtype="f4"),
            CHANNEL_KEY: np.zeros(self.shape, dtype="f4"),
            **{key: np.zeros(self.shape, dtype="f4") for key in FLUID_KEYS},
        }

    def test_background_brine_requires_recognized_version_and_modeled_domain(self):
        result = inspect_fluid_fill(self.crop, self.sha, 10)
        self.assertTrue(np.all(result.intended_code == FluidCode.BRINE))
        self.assertTrue(result.provisional_valid.all())
        unknown = inspect_fluid_fill(self.crop, "unverified-version", 10)
        self.assertTrue(np.all(unknown.intended_code == FluidCode.UNRESOLVED))
        self.assertFalse(unknown.provisional_valid.any())
        self.assertTrue(np.all(unknown.reason_bits & FluidReason.UNVERIFIED_GENERATOR))
        no_domain = inspect_fluid_fill(self.crop, self.sha, None)
        self.assertFalse(no_domain.provisional_valid.any())
        self.assertTrue(np.all(no_domain.reason_bits & FluidReason.UNPROVEN_MODELED_DOMAIN))

    def test_oil_gas_and_brine_with_channel_identity_and_inconsistency(self):
        self.crop[FLUID_KEYS[1]][0, 0, :2] = 1
        self.crop[FLUID_KEYS[0]][0, 0, 2:4] = 1
        self.crop[FLUID_KEYS[2]][0, 0, 4] = 1
        self.crop[CHANNEL_KEY][0, 0, [1, 3, 5]] = [1, 2, 1]
        result = inspect_fluid_fill(self.crop, self.sha, 10)
        np.testing.assert_array_equal(result.intended_code.ravel(), [
            FluidCode.OIL, FluidCode.OIL, FluidCode.GAS,
            FluidCode.GAS, FluidCode.BRINE, FluidCode.BRINE,
        ])
        np.testing.assert_array_equal(result.provisional_valid.ravel(), [
            True, False, True, False, True, True,
        ])
        self.assertEqual(result.summary()["reason_voxel_counts"]["channel_hydrocarbon_synthesis"], 2)
        np.testing.assert_array_equal(self.crop[CHANNEL_KEY].ravel(), [0, 1, 0, 2, 0, 1])

    def test_shale_water_and_channels_use_facies_not_closure_union(self):
        self.crop["faulted_lithology"][0, 0, :3] = [0, -1, 0]
        self.crop[CHANNEL_KEY][0, 0, 2] = 1
        self.crop[FLUID_KEYS[1]][0, 0, :3] = 1
        result = inspect_fluid_fill(self.crop, self.sha, 10)
        np.testing.assert_array_equal(result.intended_code.ravel()[:3], [
            FluidCode.NOT_SAND, FluidCode.NOT_SAND, FluidCode.OIL,
        ])
        self.assertEqual(result.summary()["reason_voxel_counts"]["closure_outside_sand"], 2)
        self.assertFalse(result.provisional_valid[0, 0, :3].any())

    def test_conflicts_malformed_and_nonfinite_closures_are_explicitly_invalid(self):
        self.crop[FLUID_KEYS[0]][0, 0, 0] = 1
        self.crop[FLUID_KEYS[1]][0, 0, 0] = 1
        self.crop[FLUID_KEYS[2]][0, 0, 1] = .25
        self.crop[FLUID_KEYS[0]][0, 0, 2] = np.nan
        result = inspect_fluid_fill(self.crop, self.sha, 10)
        self.assertTrue(np.all(result.intended_code[0, 0, :3] == FluidCode.UNRESOLVED))
        self.assertFalse(result.provisional_valid[0, 0, :3].any())
        counts = result.summary()["reason_voxel_counts"]
        self.assertEqual(counts["conflicting_closures"], 1)
        self.assertEqual(counts["malformed_closure"], 2)

    def test_domain_uses_probed_global_lower_bound_not_crop_local_maximum(self):
        self.crop["geologic_age_faulted"][0, 0, :] = [0, 1, 2, 3, 4, 5]
        result = inspect_fluid_fill(self.crop, self.sha, 5)
        np.testing.assert_array_equal(result.provisional_valid.ravel(), [
            False, True, True, True, True, False,
        ])
        for bad in (0, -1, True, 2.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                inspect_fluid_fill(self.crop, self.sha, bad)

    def test_misaligned_labels_and_invalid_age_fail_explicitly(self):
        self.crop[FLUID_KEYS[0]] = np.zeros((1, 1, 1))
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            inspect_fluid_fill(self.crop, self.sha, 10)
        self.crop[FLUID_KEYS[0]] = np.zeros(self.shape)
        self.crop["geologic_age_faulted"][0, 0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            inspect_fluid_fill(self.crop, self.sha, 10)


if __name__ == "__main__":
    unittest.main()
