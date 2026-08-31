import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _normalize_patch_shape(patch_shape):
    if len(patch_shape) != 3:
        raise ValueError("patch_shape must contain exactly 3 values")
    dims = tuple(int(v) for v in patch_shape)
    if any(v <= 0 for v in dims):
        raise ValueError("patch_shape values must be positive")
    if any(v % 8 != 0 for v in dims):
        raise ValueError("patch_shape values must be divisible by 8 for VAE3D")
    return dims


class Conv3dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualConv3dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.act = nn.GELU()
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.residual = nn.Identity()
        if in_ch != out_ch or s != 1:
            self.residual = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=s, padding=0),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        residual = self.residual(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class Encoder(nn.Module):
    def __init__(self, in_ch=1, base_ch=16, latent_dim=128, patch_shape=(32, 32, 32), residual_encoder=False):
        super().__init__()
        px, py, pz = _normalize_patch_shape(patch_shape)
        self._encoded_shape = (base_ch * 4, px // 8, py // 8, pz // 8)
        flat_dim = int(self._encoded_shape[0] * self._encoded_shape[1] * self._encoded_shape[2] * self._encoded_shape[3])
        block_cls = ResidualConv3dBlock if residual_encoder else Conv3dBlock
        self.enc = nn.Sequential(
            block_cls(in_ch, base_ch),
            block_cls(base_ch, base_ch*2, s=2),
            block_cls(base_ch*2, base_ch*2),
            block_cls(base_ch*2, base_ch*4, s=2),
            block_cls(base_ch*4, base_ch*4, s=2),
        )
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        x = self.enc(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, out_ch=1, base_ch=16, latent_dim=128, patch_shape=(32, 32, 32), deep_supervision=False):
        super().__init__()
        px, py, pz = _normalize_patch_shape(patch_shape)
        self._encoded_shape = (base_ch * 4, px // 8, py // 8, pz // 8)
        flat_dim = int(self._encoded_shape[0] * self._encoded_shape[1] * self._encoded_shape[2] * self._encoded_shape[3])
        self.deep_supervision = bool(deep_supervision)
        self.fc = nn.Linear(latent_dim, flat_dim)
        self.unflatten = nn.Unflatten(1, self._encoded_shape)
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.block1 = Conv3dBlock(base_ch*4, base_ch*2)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.block2 = Conv3dBlock(base_ch*2, base_ch)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.block3 = Conv3dBlock(base_ch, base_ch)
        self.out_conv = nn.Conv3d(base_ch, out_ch, kernel_size=3, padding=1)
        if self.deep_supervision:
            # MONAI-style auxiliary prediction heads from intermediate decoder levels.
            self.aux_head_coarse = nn.Conv3d(base_ch * 2, out_ch, kernel_size=1)
            self.aux_head_mid = nn.Conv3d(base_ch, out_ch, kernel_size=1)
        else:
            self.aux_head_coarse = None
            self.aux_head_mid = None

    def forward(self, z, return_deep_supervision=False):
        x = self.fc(z)
        x = self.unflatten(x)
        x = self.up1(x)
        coarse_features = self.block1(x)
        x = self.up2(coarse_features)
        mid_features = self.block2(x)
        x = self.up3(mid_features)
        fine_features = self.block3(x)
        out = self.out_conv(fine_features)

        if (
            return_deep_supervision
            and self.deep_supervision
            and self.aux_head_coarse is not None
            and self.aux_head_mid is not None
        ):
            target_size = out.shape[2:]
            pred_mid = self.aux_head_mid(mid_features)
            pred_coarse = self.aux_head_coarse(coarse_features)
            pred_mid_up = F.interpolate(pred_mid, size=target_size, mode='trilinear', align_corners=False)
            pred_coarse_up = F.interpolate(pred_coarse, size=target_size, mode='trilinear', align_corners=False)
            return out, [out, pred_mid_up, pred_coarse_up]

        return out, None


class GeologyProjectionHead(nn.Module):
    """MLP projection head mapping encoder ``mu`` to a unit-norm geology embedding.

    The head is kept separate from ``mu`` so a contrastive geology objective can shape
    the retrieval embedding (``z_geo``) without competing with the reconstruction
    objective that owns ``mu``.
    """

    def __init__(self, latent_dim=128, proj_hidden=128, proj_dim=64):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.proj_hidden = int(proj_hidden)
        self.proj_dim = int(proj_dim)
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, self.proj_hidden),
            nn.GELU(),
            nn.Linear(self.proj_hidden, self.proj_dim),
        )

    def forward(self, mu):
        z = self.net(mu)
        return F.normalize(z, p=2, dim=-1, eps=1e-8)


class VAE3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=16, latent_dim=128, patch_shape=(32, 32, 32), deep_supervision=False, residual_encoder=False, geology_projection=False, geology_proj_hidden=128, geology_proj_dim=64):
        super().__init__()
        self.base_ch = int(base_ch)
        self.latent_dim = int(latent_dim)
        self.patch_shape = _normalize_patch_shape(patch_shape)
        self.deep_supervision = bool(deep_supervision)
        self.residual_encoder = bool(residual_encoder)
        self.geology_projection = bool(geology_projection)
        self.geology_proj_hidden = int(geology_proj_hidden)
        self.geology_proj_dim = int(geology_proj_dim)
        self.encoder = Encoder(
            in_ch,
            self.base_ch,
            self.latent_dim,
            patch_shape=self.patch_shape,
            residual_encoder=self.residual_encoder,
        )
        self.decoder = Decoder(
            out_ch,
            self.base_ch,
            self.latent_dim,
            patch_shape=self.patch_shape,
            deep_supervision=self.deep_supervision,
        )
        if self.geology_projection:
            self.geology_head = GeologyProjectionHead(
                latent_dim=self.latent_dim,
                proj_hidden=self.geology_proj_hidden,
                proj_dim=self.geology_proj_dim,
            )
        else:
            self.geology_head = None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_geo(self, mu):
        """Return the unit-norm geology embedding ``z_geo`` for a batch of ``mu``.

        Falls back to L2-normalized ``mu`` when the projection head is disabled so
        downstream consumers always receive a comparable unit-norm embedding.
        """
        if self.geology_head is not None:
            return self.geology_head(mu)
        return F.normalize(mu, p=2, dim=-1, eps=1e-8)

    def forward(self, x, return_deep_supervision=False):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        out, ds_outputs = self.decoder(z, return_deep_supervision=return_deep_supervision)
        if return_deep_supervision and self.deep_supervision:
            return out, mu, logvar, ds_outputs
        return out, mu, logvar
