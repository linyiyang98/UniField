"""Loss functions for UniField.

Combines:
  - L1 reconstruction loss
  - Structural Similarity (SSIM) loss
  - A lightweight perceptual loss using a small 3-D VGG-style feature network
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    """Return a 1-D Gaussian kernel as a 1×1×size tensor."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _ssim_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
    sigma: float = 1.5,
    data_range: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Differentiable 3-D SSIM."""
    k1, k2 = 0.01, 0.03
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # Build a separable 3-D Gaussian window.
    k1d = _gaussian_kernel_1d(window_size, sigma).to(pred.device)
    win = k1d[:, None, None] * k1d[None, :, None] * k1d[None, None, :]  # (W, W, W)
    win = win.unsqueeze(0).unsqueeze(0)  # (1, 1, W, W, W)

    channels = pred.shape[1]
    win = win.expand(channels, 1, window_size, window_size, window_size).contiguous()

    pad = window_size // 2

    mu1 = F.conv3d(pred, win, padding=pad, groups=channels)
    mu2 = F.conv3d(target, win, padding=pad, groups=channels)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(pred * pred, win, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv3d(target * target, win, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv3d(pred * target, win, padding=pad, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    if reduction == "mean":
        return ssim_map.mean()
    return ssim_map


class SSIMLoss(nn.Module):
    """1 − SSIM loss for 3-D volumes."""

    def __init__(self, window_size: int = 7, sigma: float = 1.5) -> None:
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - _ssim_3d(pred, target, self.window_size, self.sigma)


class PerceptualFeatureNet(nn.Module):
    """Lightweight 3-D feature network for perceptual loss.

    Uses the first two convolutional blocks as fixed feature extractors
    (no pre-trained weights — trained jointly in early iterations).
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AvgPool3d(2)
        self.block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.block1(x)
        f2 = self.block2(self.pool(f1))
        return [f1, f2]


class PerceptualLoss(nn.Module):
    """Multi-scale feature matching loss."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.net = PerceptualFeatureNet(in_channels)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        feats_pred = self.net(pred)
        feats_tgt = self.net(target)
        loss = sum(
            F.l1_loss(fp, ft) for fp, ft in zip(feats_pred, feats_tgt)
        )
        return loss / len(feats_pred)


class UniFieldLoss(nn.Module):
    """Combined reconstruction loss for UniField.

    L_total = w_l1·L1 + w_ssim·(1−SSIM) + w_perceptual·L_perceptual

    Parameters
    ----------
    l1_weight:
        Weight for the pixel-level L1 term.
    ssim_weight:
        Weight for the SSIM term.
    perceptual_weight:
        Weight for the perceptual feature-matching term.
    in_channels:
        Input channels (used to build the perceptual network).
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.1,
        perceptual_weight: float = 0.1,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight

        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.perceptual = PerceptualLoss(in_channels)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys 'total', 'l1', 'ssim', 'perceptual'.
        """
        l1_loss = self.l1(pred, target)
        ssim_loss = self.ssim(pred, target)
        perc_loss = self.perceptual(pred, target)

        total = (
            self.l1_weight * l1_loss
            + self.ssim_weight * ssim_loss
            + self.perceptual_weight * perc_loss
        )
        return {
            "total": total,
            "l1": l1_loss,
            "ssim": ssim_loss,
            "perceptual": perc_loss,
        }
