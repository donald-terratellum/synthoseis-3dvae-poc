import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, out_ch=1, base_ch=16, latent_dim=128, patch_shape=(32, 32, 32)):
        super().__init__()
        px, py, pz = _normalize_patch_shape(patch_shape)
        self._encoded_shape = (base_ch * 4, px // 8, py // 8, pz // 8)
        flat_dim = int(self._encoded_shape[0] * self._encoded_shape[1] * self._encoded_shape[2] * self._encoded_shape[3])
        self.fc = nn.Linear(latent_dim, flat_dim)
        self.dec = nn.Sequential(
            nn.Unflatten(1, self._encoded_shape),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            Conv3dBlock(base_ch*4, base_ch*2),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            Conv3dBlock(base_ch*2, base_ch),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            Conv3dBlock(base_ch, base_ch),
            nn.Conv3d(base_ch, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, z):
        x = self.fc(z)
        x = self.dec(x)
        return x


class VAE3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=16, latent_dim=128, patch_shape=(32, 32, 32)):
        super().__init__()
        self.base_ch = int(base_ch)
        self.latent_dim = int(latent_dim)
        self.patch_shape = _normalize_patch_shape(patch_shape)
        self.encoder = Encoder(in_ch, self.base_ch, self.latent_dim, patch_shape=self.patch_shape)
        self.decoder = Decoder(out_ch, self.base_ch, self.latent_dim, patch_shape=self.patch_shape)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar
