from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_PATH = Path("checkpoints_gan_vwarp2/vae_final.pt")
DEFAULT_TEMP_ZARR_PATH = Path("/tmp/temp_seismic/input_seismic.zarr")


@dataclass(frozen=True)
class RuntimeConfig:
    model_path: Path = DEFAULT_MODEL_PATH
    temp_input_zarr: Path = DEFAULT_TEMP_ZARR_PATH
    latent_dim: int = 128
    patch_size: int = 32
    stride: int = 16

    def validate(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if self.stride <= 0:
            raise ValueError("stride must be > 0")
        if self.patch_size % self.stride != 0:
            raise ValueError("patch_size must be divisible by stride")


@dataclass(frozen=True)
class SearchConfig:
    source_volume: Path
    search_volume: Path
    output_volume: Path

    def validate(self) -> None:
        if not self.source_volume:
            raise ValueError("source_volume must be provided")
        if not self.search_volume:
            raise ValueError("search_volume must be provided")
        if not self.output_volume:
            raise ValueError("output_volume must be provided")
