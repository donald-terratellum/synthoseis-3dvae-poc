"""Opt-in joint VAE and ordered pair objectives; not a training entry point."""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from scripts.train import build_reconstruction_loss, compute_vae_losses
from src.model import VAE3D


class CalibratedCosine(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_scale = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, mu_a: torch.Tensor, mu_b: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(F.softplus(self.raw_scale) * F.cosine_similarity(mu_a, mu_b) + self.bias)


class OrderedCoverageHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        if latent_dim < 1 or hidden_dim < 1:
            raise ValueError("Coverage-head dimensions must be positive.")
        self.network = nn.Sequential(
            nn.Linear(4 * latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, mu_a: torch.Tensor, mu_b: torch.Tensor) -> torch.Tensor:
        features = torch.cat((mu_a, mu_b, torch.abs(mu_a - mu_b), mu_a * mu_b), dim=-1)
        return torch.sigmoid(self.network(features).squeeze(-1))


class StreamedPairVAE(nn.Module):
    def __init__(self, vae: VAE3D, arm: str):
        super().__init__()
        if arm not in {"S1", "S2", "S3"}:
            raise ValueError("arm must be S1, S2 or S3.")
        self.vae = vae
        self.arm = arm
        self.overlap_head = CalibratedCosine() if arm in {"S2", "S3"} else None
        self.coverage_head = OrderedCoverageHead(vae.latent_dim) if arm == "S3" else None

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor | None]:
        if a.shape != b.shape or a.ndim != 5 or a.shape[0] == 0:
            raise ValueError("A and B must have the same nonempty five-dimensional batch shape.")
        if a.shape[1:] != (1, *self.vae.patch_shape):
            raise ValueError("Input views must match the VAE channel and XYZ patch shape.")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Input views must be finite.")
        count = a.shape[0]
        mu, logvar = self.vae.encoder(torch.cat((a, b), dim=0))
        mu_a, mu_b = mu[:count], mu[count:]
        logvar_a = logvar[:count]
        reconstruction, _ = self.vae.decoder(self.vae.reparameterize(mu_a, logvar_a))
        return {
            "reconstruction": reconstruction,
            "mu_a": mu_a,
            "mu_b": mu_b,
            "logvar_a": logvar_a,
            "overlap": self.overlap_head(mu_a, mu_b) if self.overlap_head is not None else None,
            "coverage": self.coverage_head(mu_a, mu_b) if self.coverage_head is not None else None,
        }


def fresh_e1_model(arm: str, model_seed: int, head_seed: int) -> StreamedPairVAE:
    """Create the same fresh E1 VAE for each arm without consuming caller RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(model_seed)
        vae = VAE3D(
            patch_shape=(32, 32, 64),
            latent_dim=128,
            encoder_arch="residual",
            encoder_norm="batch",
            encoder_input_axes="xyz",
        )
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(head_seed)
        model = StreamedPairVAE(vae, arm)
    return model


def build_optimizer(model: StreamedPairVAE) -> torch.optim.AdamW:
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("The VAE and every enabled pair head must remain trainable.")
    groups = [
        {"params": list(model.vae.encoder.parameters()), "lr": 5e-5, "name": "encoder"},
        {"params": list(model.vae.decoder.parameters()), "lr": 5e-4, "name": "decoder"},
    ]
    if model.overlap_head is not None:
        groups.append({"params": list(model.overlap_head.parameters()), "lr": 5e-4, "name": "overlap"})
    if model.coverage_head is not None:
        groups.append({"params": list(model.coverage_head.parameters()), "lr": 5e-4, "name": "coverage"})
    assigned = {id(parameter) for group in groups for parameter in group["params"]}
    if (assigned != {id(parameter) for parameter in model.parameters()}
            or len(assigned) != sum(len(group["params"]) for group in groups)):
        raise ValueError("All model parameters must belong to exactly one optimizer group.")
    return torch.optim.AdamW(groups, lr=5e-4, weight_decay=1e-4)


@dataclass(frozen=True)
class JointLoss:
    total: torch.Tensor
    reconstruction: torch.Tensor
    lpips: torch.Tensor
    kl: torch.Tensor
    overlap: torch.Tensor | None
    coverage: torch.Tensor | None
    valid_coverage: int


def joint_loss(
    outputs: dict[str, torch.Tensor | None],
    dense_a: torch.Tensor,
    lpips_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    overlap_targets: torch.Tensor | None = None,
    coverage_targets: torch.Tensor | None = None,
    coverage_valid: torch.Tensor | None = None,
) -> JointLoss:
    reconstruction, mu_a, logvar_a = (
        outputs["reconstruction"], outputs["mu_a"], outputs["logvar_a"]
    )
    if reconstruction is None or mu_a is None or logvar_a is None:
        raise ValueError("VAE reconstruction, mu and log variance are required.")
    if not callable(lpips_fn):
        raise ValueError("A callable LPIPS loss is required.")
    if reconstruction.shape != dense_a.shape or not torch.isfinite(dense_a).all():
        raise ValueError("Dense reconstruction targets must be finite and match output shape.")
    total, mae, kl, lpips, _ = compute_vae_losses(
        reconstruction, dense_a, mu_a, logvar_a, kl_weight=0.001,
        rec_loss_fn=build_reconstruction_loss("mae"), lpips_loss_fn=lpips_fn, lpips_weight=0.1,
    )
    overlap_loss = None
    prediction = outputs["overlap"]
    if prediction is not None:
        if overlap_targets is None or overlap_targets.shape != prediction.shape:
            raise ValueError("Enabled overlap head requires one target per pair.")
        if not torch.isfinite(overlap_targets).all() or ((overlap_targets < 0) | (overlap_targets > 1)).any():
            raise ValueError("Overlap targets must be finite values in [0, 1].")
        overlap_loss = F.mse_loss(prediction, overlap_targets)
        total = total + 0.1 * overlap_loss
    coverage_loss = None
    valid_count = 0
    prediction = outputs["coverage"]
    if prediction is not None:
        if (coverage_targets is None or coverage_valid is None
                or coverage_targets.shape != prediction.shape
                or coverage_valid.shape != prediction.shape
                or coverage_valid.dtype != torch.bool):
            raise ValueError("Enabled coverage head requires targets and a boolean validity mask.")
        valid_count = int(coverage_valid.sum().item())
        if valid_count:
            targets = coverage_targets[coverage_valid]
            if not torch.isfinite(targets).all() or ((targets < 0) | (targets > 1)).any():
                raise ValueError("Valid coverage targets must be finite values in [0, 1].")
            coverage_loss = F.mse_loss(prediction[coverage_valid], targets)
            total = total + 0.1 * coverage_loss
    if not torch.isfinite(total):
        raise ValueError("Joint loss is non-finite.")
    return JointLoss(total, mae, lpips, kl, overlap_loss, coverage_loss, valid_count)
