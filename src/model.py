import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from src.encoders_resnetv2 import ResNetV2Trunk3d


def _normalize_patch_shape(patch_shape):
    if len(patch_shape) != 3:
        raise ValueError("patch_shape must contain exactly 3 values")
    dims = tuple(int(v) for v in patch_shape)
    if any(v <= 0 for v in dims):
        raise ValueError("patch_shape values must be positive")
    if any(v % 8 != 0 for v in dims):
        raise ValueError("patch_shape values must be divisible by 8 for VAE3D")
    return dims


def _normalization(norm_type, channels):
    if norm_type == 'batch':
        return nn.BatchNorm3d(channels)
    if norm_type == 'instance':
        return nn.InstanceNorm3d(channels, affine=True)
    if norm_type == 'group':
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError("normalization must be one of: batch, instance, group")


class Conv3dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, norm_type='batch'):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        self.bn = _normalization(norm_type, out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualConv3dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, norm_type='batch'):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        self.bn1 = _normalization(norm_type, out_ch)
        self.act = nn.GELU()
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = _normalization(norm_type, out_ch)
        self.residual = nn.Identity()
        if in_ch != out_ch or s != 1:
            self.residual = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=s, padding=0),
                _normalization(norm_type, out_ch),
            )

    def forward(self, x):
        residual = self.residual(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class ResBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.GELU()
        self.proj = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, inputs):
        residual = self.proj(inputs)
        output = self.act(self.norm1(self.conv1(inputs)))
        output = self.norm2(self.conv2(output))
        return self.act(output + residual)


class Encoder(nn.Module):
    def __init__(
        self, in_ch=1, base_ch=16, latent_dim=128, patch_shape=(32, 32, 32), residual_encoder=False,
        encoder_arch='conv', hidden_dims=None, depth_profile='baseline', stage_blocks=None,
        norm_type=None, stem='pretrain_v2', input_axes='xyz',
    ):
        super().__init__()
        px, py, pz = _normalize_patch_shape(patch_shape)
        if encoder_arch == 'conv' and residual_encoder:
            encoder_arch = 'residual'
        if encoder_arch not in {'conv', 'residual', 'resnetv2'}:
            raise ValueError("encoder_arch must be one of: conv, residual, resnetv2")
        self.arch = str(encoder_arch)
        self.input_axes = str(input_axes).lower()
        if self.input_axes not in {'xyz', 'zxy'}:
            raise ValueError("encoder_input_axes must be xyz or zxy")
        default_norm = 'instance' if self.arch == 'resnetv2' else 'batch'
        self.norm_type = str(norm_type or default_norm)
        if self.arch == 'resnetv2':
            widths = tuple(int(v) for v in (hidden_dims or (32, 64, 128)))
            if stage_blocks is None:
                profiles = {'baseline': (3, 4, 6), 'deeper': (3, 5, 8)}
                if depth_profile not in profiles:
                    raise ValueError("three-stage resnetv2 depth_profile must be baseline or deeper")
                stage_blocks = profiles[depth_profile]
            blocks = tuple(int(v) for v in stage_blocks)
            self.trunk = ResNetV2Trunk3d(in_ch, widths, blocks, stem=stem, norm=self.norm_type)
            input_shape = (px, py, pz)
            if self.input_axes == 'zxy':
                input_shape = (pz, px, py)
            was_training = self.trunk.training
            self.trunk.eval()
            with torch.no_grad():
                feature_shape = tuple(self.trunk(torch.zeros((1, in_ch) + input_shape)).shape[1:])
            self.trunk.train(was_training)
            self._encoded_shape = feature_shape
            self.hidden_dims = widths
            self.stage_blocks = blocks
        else:
            widths = tuple(int(v) for v in (hidden_dims or (base_ch, base_ch * 2, base_ch * 2, base_ch * 4, base_ch * 4)))
            if len(widths) != 5 or any(v <= 0 for v in widths):
                raise ValueError("conv and residual encoders require five positive encoder_hidden_dims")
            default_widths = (base_ch, base_ch * 2, base_ch * 2, base_ch * 4, base_ch * 4)
            use_legacy = widths == default_widths and self.norm_type == 'batch'

            def make_block(in_width, out_width, stride=1):
                if use_legacy:
                    block_class = ResidualConv3dBlock if self.arch == 'residual' else Conv3dBlock
                    return block_class(in_width, out_width, s=stride) if stride != 1 else block_class(in_width, out_width)
                if self.arch == 'residual':
                    return ResidualConv3dBlock(in_width, out_width, s=stride, norm_type=self.norm_type)
                return Conv3dBlock(in_width, out_width, s=stride, norm_type=self.norm_type)

            self.enc = nn.Sequential(
                make_block(in_ch, widths[0]),
                make_block(widths[0], widths[1], stride=2),
                make_block(widths[1], widths[2]),
                make_block(widths[2], widths[3], stride=2),
                make_block(widths[3], widths[4], stride=2),
            )
            self._encoded_shape = (widths[-1], px // 8, py // 8, pz // 8)
            self.hidden_dims = widths
            self.stage_blocks = ()
        flat_dim = int(self._encoded_shape[0] * self._encoded_shape[1] * self._encoded_shape[2] * self._encoded_shape[3])
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        if self.arch == 'resnetv2':
            if self.input_axes == 'zxy':
                x = x.permute(0, 1, 4, 2, 3).contiguous()
            x = self.trunk(x)
        else:
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


class FlexibleDecoder(nn.Module):
    def __init__(
        self, in_channels, encoded_shape, patch_shape, latent_dim, hidden_dims,
        block_type='conv', output_axes='xyz', out_channels=1,
    ):
        super().__init__()
        spatial_steps = []
        for encoded, target in zip(encoded_shape[1:], patch_shape):
            ratio = target // encoded
            if target % encoded or ratio < 1 or ratio & (ratio - 1):
                raise ValueError("encoder feature map must divide patch_shape by a power of two")
            spatial_steps.append(int(math.log2(ratio)))
        if len(set(spatial_steps)) != 1 or spatial_steps[0] <= 0:
            raise ValueError("encoder feature map must have a uniform positive power-of-two scale")
        steps = spatial_steps[0]
        if hidden_dims is None:
            widths = tuple(max(16, in_channels // (2 ** (index + 1))) for index in range(steps - 1)) + (16,)
        else:
            widths = tuple(int(value) for value in hidden_dims)
        if len(widths) != steps or any(value <= 0 for value in widths):
            raise ValueError(f"decoder_hidden_dims must contain {steps} positive widths")
        if block_type not in {'conv', 'res'}:
            raise ValueError("decoder_block must be conv or res")
        self.encoded_shape = tuple(int(value) for value in encoded_shape)
        self.hidden_dims = widths
        self.output_axes = str(output_axes)
        self.deep_supervision = False
        self.fc = nn.Linear(latent_dim, int(math.prod(self.encoded_shape)))
        self.unflatten = nn.Unflatten(1, self.encoded_shape)
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        in_width = int(in_channels)
        block_cls = Conv3dBlock if block_type == 'conv' else ResBlock3d
        for width in widths:
            self.ups.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False))
            self.blocks.append(block_cls(in_width, width))
            in_width = width
        self.out_conv = nn.Conv3d(in_width, int(out_channels), kernel_size=3, padding=1)
        self.aux_head_coarse = None
        self.aux_head_mid = None

    def forward(self, z, return_deep_supervision=False):
        features = self.unflatten(self.fc(z))
        for upsample, block in zip(self.ups, self.blocks):
            features = block(upsample(features))
        output = self.out_conv(features)
        if self.output_axes == 'zxy':
            output = output.permute(0, 1, 3, 4, 2).contiguous()
        return output, None


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


GEOLOGY_PRESENCE_CLASSES = ("fault", "fault_x", "channel", "closure", "onlap", "sand", "flat_spot")
GEOLOGY_DIP_CLASSES = 6


class GeologyClassifierDecoder(nn.Module):
    """Patch-level multi-task classifier on ``mu``: class presence logits plus dip-mean/dip-range classes."""

    pos_weight: torch.Tensor

    def __init__(self, latent_dim=128, hidden=256, dropout=0.1, n_presence=len(GEOLOGY_PRESENCE_CLASSES), n_dip_classes=GEOLOGY_DIP_CLASSES):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        self.trunk = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.presence = nn.Linear(self.hidden, int(n_presence))
        self.dip_mean = nn.Linear(self.hidden, int(n_dip_classes))
        self.dip_range = nn.Linear(self.hidden, int(n_dip_classes))
        # clip(n_neg / n_pos, 1, 50) from the training split; saved with the state dict.
        self.register_buffer("pos_weight", torch.ones(int(n_presence)))

    def forward(self, mu):
        h = self.trunk(mu)
        return {
            "presence": self.presence(h),
            "dip_mean": self.dip_mean(h),
            "dip_range": self.dip_range(h),
        }


class VAE3D(nn.Module):
    def __init__(
        self, in_ch=1, out_ch=1, base_ch=16, latent_dim=128, patch_shape=(32, 32, 32),
        deep_supervision=False, residual_encoder=False, geology_projection=False, geology_proj_hidden=128,
        geology_proj_dim=64, geology_classifier=False, geology_classifier_mode='patch',
        geology_classifier_hidden=256, encoder_arch='conv', encoder_hidden_dims=None,
        encoder_depth_profile='baseline', encoder_stage_blocks=None, encoder_norm=None,
        encoder_stem='pretrain_v2', encoder_input_axes='xyz', decoder_hidden_dims=None,
        decoder_block='conv',
    ):
        super().__init__()
        self.base_ch = int(base_ch)
        self.latent_dim = int(latent_dim)
        self.patch_shape = _normalize_patch_shape(patch_shape)
        self.deep_supervision = bool(deep_supervision)
        resolved_encoder_arch = 'residual' if residual_encoder and encoder_arch == 'conv' else str(encoder_arch)
        self.encoder_arch = resolved_encoder_arch
        self.encoder_depth_profile = str(encoder_depth_profile)
        self.encoder_norm = str(encoder_norm or ('instance' if resolved_encoder_arch == 'resnetv2' else 'batch'))
        self.encoder_stem = str(encoder_stem)
        self.encoder_input_axes = str(encoder_input_axes)
        self.decoder_block = str(decoder_block)
        self.residual_encoder = resolved_encoder_arch == 'residual'
        self.geology_projection = bool(geology_projection)
        self.geology_proj_hidden = int(geology_proj_hidden)
        self.geology_proj_dim = int(geology_proj_dim)
        self.encoder = Encoder(
            in_ch,
            self.base_ch,
            self.latent_dim,
            patch_shape=self.patch_shape,
            residual_encoder=False,
            encoder_arch=resolved_encoder_arch,
            hidden_dims=encoder_hidden_dims,
            depth_profile=self.encoder_depth_profile,
            stage_blocks=encoder_stage_blocks,
            norm_type=self.encoder_norm,
            stem=self.encoder_stem,
            input_axes=self.encoder_input_axes,
        )
        self.encoder_hidden_dims = tuple(int(v) for v in self.encoder.hidden_dims)
        self.encoder_stage_blocks = tuple(int(v) for v in self.encoder.stage_blocks)
        self.decoder_hidden_dims = None if decoder_hidden_dims is None else tuple(int(v) for v in decoder_hidden_dims)
        use_flexible_decoder = (
            resolved_encoder_arch == 'resnetv2'
            or self.encoder.hidden_dims != (self.base_ch, self.base_ch * 2, self.base_ch * 2, self.base_ch * 4, self.base_ch * 4)
            or self.decoder_hidden_dims is not None
            or self.decoder_block != 'conv'
        )
        if use_flexible_decoder:
            if self.deep_supervision:
                raise ValueError('deep_supervision is not supported with configurable encoder decoders.')
            decoder_patch_shape = self.patch_shape
            if self.encoder_input_axes == 'zxy':
                decoder_patch_shape = (self.patch_shape[2], self.patch_shape[0], self.patch_shape[1])
            self.decoder = FlexibleDecoder(
                self.encoder._encoded_shape[0],
                self.encoder._encoded_shape,
                decoder_patch_shape,
                self.latent_dim,
                self.decoder_hidden_dims,
                block_type=self.decoder_block,
                output_axes=self.encoder_input_axes,
                out_channels=out_ch,
            )
            if self.decoder_hidden_dims is None:
                self.decoder_hidden_dims = tuple(int(value) for value in self.decoder.hidden_dims)
        else:
            self.decoder = Decoder(
                out_ch,
                self.base_ch,
                self.latent_dim,
                patch_shape=self.patch_shape,
                deep_supervision=self.deep_supervision,
            )
        self.model_config = {
            'encoder_arch': self.encoder_arch,
            'encoder_hidden_dims': list(self.encoder_hidden_dims),
            'encoder_depth_profile': self.encoder_depth_profile,
            'encoder_stage_blocks': list(self.encoder_stage_blocks),
            'encoder_norm': self.encoder_norm,
            'encoder_stem': self.encoder_stem,
            'encoder_input_axes': self.encoder_input_axes,
            'decoder_hidden_dims': list(self.decoder_hidden_dims) if self.decoder_hidden_dims is not None else None,
            'decoder_block': self.decoder_block,
        }
        if self.geology_projection:
            self.geology_head = GeologyProjectionHead(
                latent_dim=self.latent_dim,
                proj_hidden=self.geology_proj_hidden,
                proj_dim=self.geology_proj_dim,
            )
        else:
            self.geology_head = None
        self.geology_classifier_enabled = bool(geology_classifier)
        self.geology_classifier_mode = str(geology_classifier_mode)
        self.geology_classifier_hidden = int(geology_classifier_hidden)
        if self.geology_classifier_enabled:
            if self.geology_classifier_mode != 'patch':
                raise ValueError(f"geology_classifier_mode '{self.geology_classifier_mode}' is not implemented; use 'patch'.")
            self.geology_classifier = GeologyClassifierDecoder(
                latent_dim=self.latent_dim,
                hidden=self.geology_classifier_hidden,
            )
        else:
            self.geology_classifier = None

    def classify(self, mu):
        """Return a dict of geology classifier logits for a batch of ``mu``."""
        if self.geology_classifier is None:
            raise RuntimeError('classify() requires geology_classifier=True.')
        return self.geology_classifier(mu)

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
