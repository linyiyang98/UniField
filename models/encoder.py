"""3-D residual encoder for UniField.

Extracts multi-scale feature maps from a low-quality MRI volume.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class ResBlock3D(nn.Module):
    """3-D pre-activation residual block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(min(32, channels), channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, channels), channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class EncoderStage(nn.Module):
    """One encoder stage: strided downsampling + N residual blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        if downsample:
            layers.append(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            )
        else:
            layers.append(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            )
        for _ in range(num_res_blocks):
            layers.append(ResBlock3D(out_channels))
        self.stage = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage(x)


class Encoder(nn.Module):
    """Multi-scale 3-D encoder.

    Produces a list of feature maps at successive spatial resolutions,
    from finest (stem output) to coarsest (deepest stage).

    Parameters
    ----------
    in_channels:
        Number of input channels (1 for single-contrast MRI).
    base_channels:
        Number of channels in the first stage; doubles at each stage.
    num_stages:
        Number of downsampling stages.
    num_res_blocks:
        Residual blocks per stage.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_stages: int = 4,
        num_res_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=False)

        self.stages = nn.ModuleList()
        ch = base_channels
        for i in range(num_stages):
            out_ch = ch * 2
            self.stages.append(
                EncoderStage(ch, out_ch, num_res_blocks=num_res_blocks, downsample=True)
            )
            ch = out_ch
        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns
        -------
        bottleneck:
            Coarsest feature map.
        skips:
            List of feature maps from stem to second-deepest stage
            (ordered finest → coarsest), used as skip connections.
        """
        skips: List[torch.Tensor] = []
        feat = self.stem(x)
        skips.append(feat)
        for stage in self.stages[:-1]:
            feat = stage(feat)
            skips.append(feat)
        bottleneck = self.stages[-1](feat)
        return bottleneck, skips
