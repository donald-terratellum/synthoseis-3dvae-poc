import unittest
from pathlib import Path

from src.tokenizer.config.defaults import RuntimeConfig, SearchConfig


class TokenizerConfigTests(unittest.TestCase):
    def test_runtime_config_defaults_validate(self):
        cfg = RuntimeConfig()
        cfg.validate()

    def test_runtime_config_stride_must_divide_patch(self):
        cfg = RuntimeConfig(patch_size=32, stride=7)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_search_config_requires_all_paths(self):
        cfg = SearchConfig(
            source_volume=Path("source.zarr"),
            search_volume=Path("search.zarr"),
            output_volume=Path("out.zarr"),
        )
        cfg.validate()


if __name__ == "__main__":
    unittest.main()
