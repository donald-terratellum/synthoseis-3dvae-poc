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
    def __init__(self, checkpoint_path, device: str = "auto", load_classifier: bool = False):
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
        self.geology_projection = bool(checkpoint.get("geology_projection", False))
        self.geology_proj_hidden = int(checkpoint.get("geology_proj_hidden", 128))
        self.geology_proj_dim = int(checkpoint.get("geology_proj_dim", 64))
        model_config = checkpoint.get("model_config") or {}
        self.geology_classifier = bool(load_classifier)
        if self.geology_classifier and not bool(checkpoint.get("geology_classifier", False)):
            raise ValueError(f"Checkpoint {checkpoint_path} has no geology classifier head (train with --geology_classifier).")
        self.model = VAE3D(
            in_ch=1,
            out_ch=1,
            base_ch=self.base_ch,
            latent_dim=self.latent_dim,
            patch_shape=self.patch_shape,
            geology_projection=self.geology_projection,
            geology_proj_hidden=self.geology_proj_hidden,
            geology_proj_dim=self.geology_proj_dim,
            geology_classifier=self.geology_classifier,
            geology_classifier_mode=str(checkpoint.get("geology_classifier_mode", "patch")),
            geology_classifier_hidden=int(checkpoint.get("geology_classifier_hidden", 256)),
            encoder_arch=str(model_config.get('encoder_arch', 'conv')),
            encoder_hidden_dims=model_config.get('encoder_hidden_dims'),
            encoder_depth_profile=str(model_config.get('encoder_depth_profile', 'baseline')),
            encoder_stage_blocks=model_config.get('encoder_stage_blocks'),
            encoder_norm=model_config.get('encoder_norm'),
            encoder_stem=str(model_config.get('encoder_stem', 'pretrain_v2')),
            encoder_input_axes=str(model_config.get('encoder_input_axes', 'xyz')),
            decoder_hidden_dims=model_config.get('decoder_hidden_dims'),
            decoder_block=str(model_config.get('decoder_block', 'conv')),
        )
        state_dict = checkpoint["model_state_dict"]
        load_result = self.model.load_state_dict(state_dict, strict=False)
        # The geology classifier head is training-only; retrieval never builds it.
        ignored_prefixes = ("decoder.aux_head_", "geology_head.", "geology_classifier.")
        invalid_missing = [
            k for k in load_result.missing_keys
            if not k.startswith(ignored_prefixes) or (self.geology_classifier and k.startswith("geology_classifier."))
        ]
        invalid_unexpected = [k for k in load_result.unexpected_keys if not k.startswith(ignored_prefixes)]
        if invalid_missing or invalid_unexpected:
            raise ValueError(
                "Checkpoint state_dict is incompatible with tokenizer adapter model. "
                f"invalid missing keys={invalid_missing}, invalid unexpected keys={invalid_unexpected}"
            )
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

    @torch.inference_mode()
    def encode_geo_batch(self, cubes: np.ndarray) -> np.ndarray:
        """Return unit-norm geology embeddings (z_geo) for a batch of cubes.

        Uses the trained projection head when present; otherwise falls back to
        L2-normalized mu so callers always receive comparable unit-norm vectors.
        """
        arr = np.asarray(cubes, dtype=np.float32)
        if arr.ndim != 4 or arr.shape[1:] != self.patch_shape:
            raise ValueError(f"expected cubes shape (B,{self.patch_shape[0]},{self.patch_shape[1]},{self.patch_shape[2]}), got {arr.shape}")

        batch = torch.from_numpy(arr[:, None, :, :, :]).to(self.device)
        mu, _ = self.model.encoder(batch)
        z_geo = self.model.encode_geo(mu)
        return z_geo.detach().cpu().numpy().astype(np.float32, copy=False)

    @torch.inference_mode()
    def encode_geo_cube(self, cube: np.ndarray) -> np.ndarray:
        arr = np.asarray(cube, dtype=np.float32)
        if arr.shape != self.patch_shape:
            raise ValueError(
                f"expected cube shape ({self.patch_shape[0]},{self.patch_shape[1]},{self.patch_shape[2]}), got {arr.shape}"
            )
        out = self.encode_geo_batch(arr[None, ...])
        return np.ascontiguousarray(out[0], dtype=np.float32)

    @torch.inference_mode()
    def classify_batch(self, cubes: np.ndarray) -> dict:
        """Presence probabilities (B, 7) and dip-class probabilities (B, 6) from the classifier head on mu."""
        if not self.geology_classifier:
            raise RuntimeError("classify_batch requires VaeLatentAdapter(..., load_classifier=True).")
        arr = np.asarray(cubes, dtype=np.float32)
        if arr.ndim != 4 or arr.shape[1:] != self.patch_shape:
            raise ValueError(f"expected cubes shape (B,{self.patch_shape[0]},{self.patch_shape[1]},{self.patch_shape[2]}), got {arr.shape}")
        batch = torch.from_numpy(arr[:, None, :, :, :]).to(self.device)
        mu, _ = self.model.encoder(batch)
        logits = self.model.classify(mu)
        return {
            "presence": torch.sigmoid(logits["presence"]).cpu().numpy(),
            "dip_mean": torch.softmax(logits["dip_mean"], dim=-1).cpu().numpy(),
            "dip_range": torch.softmax(logits["dip_range"], dim=-1).cpu().numpy(),
        }

    @torch.inference_mode()
    def reconstruct_batch(self, cubes: np.ndarray) -> np.ndarray:
        """Reconstruct a batch of preprocessed cubes using encoder(mu)+decoder."""
        arr = np.asarray(cubes, dtype=np.float32)
        if arr.ndim != 4 or arr.shape[1:] != self.patch_shape:
            raise ValueError(
                f"expected cubes shape (B,{self.patch_shape[0]},{self.patch_shape[1]},{self.patch_shape[2]}), got {arr.shape}"
            )

        batch = torch.from_numpy(arr[:, None, :, :, :]).to(self.device)
        mu, _ = self.model.encoder(batch)
        recon, _ = self.model.decoder(mu)
        out = recon[:, 0, :, :, :].detach().cpu().numpy().astype(np.float32, copy=False)
        return out

    @torch.inference_mode()
    def reconstruct_cube(self, cube: np.ndarray) -> np.ndarray:
        arr = np.asarray(cube, dtype=np.float32)
        if arr.shape != self.patch_shape:
            raise ValueError(
                f"expected cube shape ({self.patch_shape[0]},{self.patch_shape[1]},{self.patch_shape[2]}), got {arr.shape}"
            )
        out = self.reconstruct_batch(arr[None, ...])
        return np.ascontiguousarray(out[0], dtype=np.float32)
