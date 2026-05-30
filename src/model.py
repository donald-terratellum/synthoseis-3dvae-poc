import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Encoder(nn.Module):
    def __init__(self, in_ch=1, base_ch=16, latent_dim=128):
        super().__init__()
        self.enc = nn.Sequential(
            Conv3dBlock(in_ch, base_ch),
            Conv3dBlock(base_ch, base_ch*2, s=2),
            Conv3dBlock(base_ch*2, base_ch*2),
            Conv3dBlock(base_ch*2, base_ch*4, s=2),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc_mu = nn.Linear(base_ch*4, latent_dim)
        self.fc_logvar = nn.Linear(base_ch*4, latent_dim)

    def forward(self, x):
        x = self.enc(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, out_ch=1, base_ch=16, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, base_ch*4)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (base_ch*4, 1, 1, 1)),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            Conv3dBlock(base_ch*4, base_ch*2),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            Conv3dBlock(base_ch*2, base_ch),
            nn.Conv3d(base_ch, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, z):
        x = self.fc(z)
        x = self.dec(x)
        return x


class VAE3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=16, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(in_ch, base_ch, latent_dim)
        self.decoder = Decoder(out_ch, base_ch, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar
