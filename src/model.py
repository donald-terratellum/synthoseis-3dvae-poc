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


class Encoder(nn.Module):
    def __init__(self, in_ch=1, base_ch=16, latent_dim=128, patch_shape=(32, 32, 32)):
        super().__init__()
        px, py, pz = _normalize_patch_shape(patch_shape)
        self._encoded_shape = (base_ch * 4, px // 8, py // 8, pz // 8)
        flat_dim = int(self._encoded_shape[0] * self._encoded_shape[1] * self._encoded_shape[2] * self._encoded_shape[3])
        self.enc = nn.Sequential(
            Conv3dBlock(in_ch, base_ch),
            Conv3dBlock(base_ch, base_ch*2, s=2),
            Conv3dBlock(base_ch*2, base_ch*2),
            Conv3dBlock(base_ch*2, base_ch*4, s=2),
            Conv3dBlock(base_ch*4, base_ch*4, s=2),
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


class VAE3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=16, latent_dim=128, patch_shape=(32, 32, 32), deep_supervision=False):
        super().__init__()
        self.base_ch = int(base_ch)
        self.latent_dim = int(latent_dim)
        self.patch_shape = _normalize_patch_shape(patch_shape)
        self.deep_supervision = bool(deep_supervision)
        self.encoder = Encoder(in_ch, self.base_ch, self.latent_dim, patch_shape=self.patch_shape)
        self.decoder = Decoder(
            out_ch,
            self.base_ch,
            self.latent_dim,
            patch_shape=self.patch_shape,
            deep_supervision=self.deep_supervision,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, return_deep_supervision=False):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        out, ds_outputs = self.decoder(z, return_deep_supervision=return_deep_supervision)
        if return_deep_supervision and self.deep_supervision:
            return out, mu, logvar, ds_outputs
        return out, mu, logvar
