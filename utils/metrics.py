"""Image quality metrics for MRI enhancement evaluation.

All functions accept PyTorch tensors with values in [0, 1] and shape
(B, C, H, W, D) or (B, C, H, W).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Compute batch-averaged PSNR (in dB).

    Parameters
    ----------
    pred, target:
        Tensors of the same shape with values in [0, data_range].
    data_range:
        Maximum possible signal value (default 1.0).
    """
    mse = F.mse_loss(pred, target, reduction="none")
    # Average over all spatial/channel dims but keep batch dim.
    dims = list(range(1, mse.ndim))
    mse = mse.mean(dim=dims)
    psnr = 10 * torch.log10(data_range ** 2 / (mse + 1e-8))
    return psnr.mean()


def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Compute batch-averaged SSIM for 3-D (or 2-D) volumes."""
    k1, k2 = 0.01, 0.03
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    k1d = _gaussian_kernel_1d(window_size, sigma, pred.device)

    if pred.ndim == 5:
        # 3-D: (B, C, H, W, D)
        win = k1d[:, None, None] * k1d[None, :, None] * k1d[None, None, :]
        win = win.unsqueeze(0).unsqueeze(0)
        channels = pred.shape[1]
        win = win.expand(channels, 1, window_size, window_size, window_size).contiguous()
        pad = window_size // 2
        conv = lambda x: F.conv3d(x, win, padding=pad, groups=channels)
    else:
        # 2-D fallback: (B, C, H, W)
        win = k1d[:, None] * k1d[None, :]
        win = win.unsqueeze(0).unsqueeze(0)
        channels = pred.shape[1]
        win = win.expand(channels, 1, window_size, window_size).contiguous()
        pad = window_size // 2
        conv = lambda x: F.conv2d(x, win, padding=pad, groups=channels)

    mu1 = conv(pred)
    mu2 = conv(target)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = conv(pred * pred) - mu1_sq
    sigma2_sq = conv(target * target) - mu2_sq
    sigma12 = conv(pred * target) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> Dict[str, float]:
    """Return a dict of {'psnr': float, 'ssim': float}."""
    with torch.no_grad():
        psnr = compute_psnr(pred, target, data_range).item()
        ssim = compute_ssim(pred, target, data_range=data_range).item()
    return {"psnr": psnr, "ssim": ssim}
