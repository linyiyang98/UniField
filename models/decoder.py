"""FiLM-conditioned 3-D decoder for UniField.

Each decoder stage:
  1. Trilinear up-sampling
  2. Concatenation with skip connection
  3. Feature-wise Linear Modulation (FiLM) conditioned on B₀ embedding
  4. Residual blocks
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ResBlock3D


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM).

    Computes per-channel scale (γ) and shift (β) from a conditioning vector
    and applies them to spatial feature maps:

        out = γ · features + β

    Parameters
    ----------
    num_channels:
        Number of spatial feature-map channels to modulate.
    cond_dim:
        Dimensionality of the conditioning vector (B₀ embedding).
    """

    def __init__(self, num_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, num_channels)
        self.beta_proj = nn.Linear(cond_dim, num_channels)
        nn.init.ones_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(
        self, features: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features:
            Spatial feature map of shape (B, C, H, W, D).
        cond:
            Conditioning vector of shape (B, cond_dim).

        Returns
        -------
        torch.Tensor
            Modulated feature map of shape (B, C, H, W, D).
        """
        gamma = self.gamma_proj(cond)            # (B, C)
        beta = self.beta_proj(cond)              # (B, C)
        # Reshape for broadcasting over spatial dims.
        gamma = gamma.view(-1, gamma.shape[1], 1, 1, 1)
        beta = beta.view(-1, beta.shape[1], 1, 1, 1)
        return gamma * features + beta


class DecoderStage(nn.Module):
    """One FiLM-conditioned decoder upsampling stage."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        cond_dim: int,
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        merged_ch = in_channels + skip_channels
        self.proj = nn.Conv3d(merged_ch, out_channels, kernel_size=1, bias=False)
        self.film = FiLMLayer(out_channels, cond_dim)
        self.res_blocks = nn.Sequential(
            *[ResBlock3D(out_channels) for _ in range(num_res_blocks)]
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        x = self.film(x, cond)
        x = self.res_blocks(x)
        return x


class Decoder(nn.Module):
    """Multi-stage FiLM-conditioned 3-D decoder.

    Parameters
    ----------
    bottleneck_channels:
        Channel count of the encoder bottleneck.
    skip_channels_list:
        List of skip-connection channel counts, from coarsest to finest.
    out_channels:
        Number of output channels (1 for single-contrast reconstruction).
    cond_dim:
        Dimensionality of the B₀ conditioning embedding.
    num_res_blocks:
        Residual blocks per decoder stage.
    """

    def __init__(
        self,
        bottleneck_channels: int,
        skip_channels_list: List[int],
        out_channels: int = 1,
        cond_dim: int = 32,
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.stages = nn.ModuleList()
        in_ch = bottleneck_channels
        for skip_ch in skip_channels_list:
            stage_out_ch = max(skip_ch, out_channels)
            self.stages.append(
                DecoderStage(in_ch, skip_ch, stage_out_ch, cond_dim, num_res_blocks)
            )
            in_ch = stage_out_ch

        self.head = nn.Sequential(
            nn.GroupNorm(min(32, in_ch), in_ch),
            nn.GELU(),
            nn.Conv3d(in_ch, out_channels, kernel_size=1),
        )

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: List[torch.Tensor],
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        bottleneck:
            Coarsest feature map from the encoder.
        skips:
            Skip connections ordered from coarsest to finest.
        cond:
            B₀ embedding of shape (B, cond_dim).
        """
        x = bottleneck
        for stage, skip in zip(self.stages, skips):
            x = stage(x, skip, cond)
        return self.head(x)
