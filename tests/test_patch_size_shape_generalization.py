import unittest

from src.model import VAE3D

from scripts.sample_patches import normalize_patch_size as normalize_sample_patch_size
from scripts.train import normalize_patch_size as normalize_train_patch_size


class PatchSizeGeneralizationTests(unittest.TestCase):
    def test_single_patch_size_broadcasts_to_three_axes(self):
        self.assertEqual(normalize_sample_patch_size([32]), (32, 32, 32))
        self.assertEqual(normalize_train_patch_size([24]), (24, 24, 24))

    def test_three_patch_sizes_are_preserved(self):
        self.assertEqual(normalize_sample_patch_size([16, 24, 32]), (16, 24, 32))
        self.assertEqual(normalize_train_patch_size([8, 16, 24]), (8, 16, 24))

    def test_vae3d_accepts_generalized_patch_shape(self):
        model = VAE3D(patch_shape=(32, 40, 48))
        self.assertEqual(model.patch_shape, (32, 40, 48))


if __name__ == "__main__":
    unittest.main()