import numpy as np
import torch

from src.model import VAE3D


def cube_to_latent_128(cube: np.ndarray) -> np.ndarray:
    """Map a 32^3 preprocessed cube to a deterministic 128-D latent vector.

    This adapter is a stable fallback used for Phase 4 integration while full
    model-backed encoder inference is integrated.
    """
    arr = np.asarray(cube, dtype=np.float32)
    if arr.shape != (32, 32, 32):
        raise ValueError(f"expected cube shape (32, 32, 32), got {arr.shape}")

    # Mean-pool into 8x8x2 blocks => 128 features.
    pooled = arr.reshape(8, 4, 8, 4, 2, 16).mean(axis=(1, 3, 5))
    latent = pooled.reshape(128)
    return np.ascontiguousarray(latent, dtype=np.float32)


def choose_torch_device(requested: str = "auto") -> torch.device:
    req = requested.lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if req == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        return torch.device("mps")
    if req == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of: auto, cuda, mps, cpu")


class VaeLatentAdapter:
    def __init__(self, checkpoint_path, device: str = "auto"):
        self.device = choose_torch_device(device)

        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Checkpoint {checkpoint_path} is invalid. Expected dict with keys "
                "['model_state_dict', 'patch_shape', 'latent_dim', 'base_ch']."
            )
        required_keys = {"model_state_dict", "patch_shape", "latent_dim", "base_ch"}
        missing = required_keys.difference(checkpoint.keys())
        if missing:
            raise ValueError(f"Checkpoint {checkpoint_path} missing required keys: {sorted(missing)}")

        self.patch_shape = tuple(int(v) for v in checkpoint["patch_shape"])
        self.latent_dim = int(checkpoint["latent_dim"])
        self.base_ch = int(checkpoint["base_ch"])
        self.model = VAE3D(
            in_ch=1,
            out_ch=1,
            base_ch=self.base_ch,
            latent_dim=self.latent_dim,
            patch_shape=self.patch_shape,
        )
        state_dict = checkpoint["model_state_dict"]
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_batch(self, cubes: np.ndarray) -> np.ndarray:
        arr = np.asarray(cubes, dtype=np.float32)
        if arr.ndim != 4 or arr.shape[1:] != self.patch_shape:
            raise ValueError(f"expected cubes shape (B,{self.patch_shape[0]},{self.patch_shape[1]},{self.patch_shape[2]}), got {arr.shape}")

        batch = torch.from_numpy(arr[:, None, :, :, :]).to(self.device)
        mu, _ = self.model.encoder(batch)
        return mu.detach().cpu().numpy().astype(np.float32, copy=False)

    @torch.inference_mode()
    def encode_cube(self, cube: np.ndarray) -> np.ndarray:
        arr = np.asarray(cube, dtype=np.float32)
        if arr.shape != self.patch_shape:
            raise ValueError(
                f"expected cube shape ({self.patch_shape[0]},{self.patch_shape[1]},{self.patch_shape[2]}), got {arr.shape}"
            )
        out = self.encode_batch(arr[None, ...])
        return np.ascontiguousarray(out[0], dtype=np.float32)
