"""ResNetV2 3D encoder trunk compatible with the pretrain-v2 r006 layout."""

from __future__ import annotations

import torch
from torch import nn


def _norm3d(kind: str, channels: int) -> nn.Module:
    if kind == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    if kind == "batch":
        return nn.BatchNorm3d(channels)
    if kind == "group":
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError("encoder_norm must be one of: batch, instance, group")


class BottleneckV2_3d(nn.Module):
    """Pre-activation bottleneck; parameter names match the r006 encoder."""

    expansion = 4

    def __init__(self, in_channels: int, planes: int, stride: int = 1, norm: str = "instance"):
        super().__init__()
        out_channels = int(planes) * self.expansion
        self.norm1 = _norm3d(norm, in_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, planes, kernel_size=1, bias=False)
        self.norm2 = _norm3d(norm, planes)
        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm3 = _norm3d(norm, planes)
        self.act3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(planes, out_channels, kernel_size=1, bias=False)
        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        activated = self.act1(self.norm1(inputs))
        residual = self.shortcut(activated if not isinstance(self.shortcut, nn.Identity) else inputs)
        output = self.conv1(activated)
        output = self.conv2(self.act2(self.norm2(output)))
        output = self.conv3(self.act3(self.norm3(output)))
        return output + residual


class ResNetV2Stage3d(nn.Module):
    def __init__(self, in_channels: int, planes: int, num_blocks: int, stride: int, norm: str = "instance"):
        super().__init__()
        blocks = [BottleneckV2_3d(in_channels, planes, stride=stride, norm=norm)]
        out_channels = int(planes) * BottleneckV2_3d.expansion
        blocks.extend(
            BottleneckV2_3d(out_channels, planes, norm=norm)
            for _ in range(1, int(num_blocks))
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.blocks(inputs)


class ResNetV2Trunk3d(nn.Module):
    """Stem and stages only; FC layers are deliberately owned by the VAE encoder."""

    def __init__(
        self,
        in_channels: int,
        hidden_dims: tuple[int, ...],
        stage_blocks: tuple[int, ...],
        stem: str = "pretrain_v2",
        norm: str = "instance",
    ):
        super().__init__()
        if len(hidden_dims) < 3 or len(hidden_dims) != len(stage_blocks):
            raise ValueError("resnetv2 requires matching hidden_dims and stage_blocks with at least 3 stages")
        if stem not in {"pretrain_v2", "light"}:
            raise ValueError("encoder_stem must be one of: pretrain_v2, light")
        first_width = int(hidden_dims[0])
        stem_kernel = 7 if stem == "pretrain_v2" else 3
        stem_padding = stem_kernel // 2
        self.stem_conv = nn.Conv3d(
            in_channels, first_width, kernel_size=stem_kernel, stride=2, padding=stem_padding, bias=False
        )
        self.stem_norm = _norm3d(norm, first_width)
        self.stem_act = nn.ReLU(inplace=True)
        self.stem_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1) if stem == "pretrain_v2" else nn.Identity()
        stages = []
        for index, (width, depth) in enumerate(zip(hidden_dims, stage_blocks)):
            if index == 0:
                stage_in, stride = first_width, 1
            else:
                stage_in = int(hidden_dims[index - 1]) * BottleneckV2_3d.expansion
                stride = 2
            stages.append(ResNetV2Stage3d(stage_in, int(width), int(depth), stride, norm=norm))
        self.stages = nn.ModuleList(stages)
        self.output_channels = int(hidden_dims[-1]) * BottleneckV2_3d.expansion
        self.stage_block_schedule = tuple(int(v) for v in stage_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.stem_act(self.stem_norm(self.stem_conv(inputs)))
        output = self.stem_pool(output)
        for stage in self.stages:
            output = stage(output)
        return output
