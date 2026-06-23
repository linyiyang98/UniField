"""Visualization helpers for UniField.

Utility functions for logging images to TensorBoard and saving output
volumes as NIfTI files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[misc,assignment]

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None  # type: ignore[assignment]


def log_images_to_tb(
    writer: "SummaryWriter",
    tag: str,
    lq: torch.Tensor,
    pred: torch.Tensor,
    hq: torch.Tensor,
    step: int,
    slice_idx: Optional[int] = None,
    n_samples: int = 4,
) -> None:
    """Log LQ / prediction / HQ slice triplets to TensorBoard.

    Parameters
    ----------
    writer:
        An open ``SummaryWriter`` instance.
    tag:
        Image tag prefix.
    lq, pred, hq:
        Batched 3-D volumes, shape (B, 1, H, W, D), values in [0, 1].
    step:
        Global training step.
    slice_idx:
        Axial slice to visualise (default: middle slice).
    n_samples:
        Number of samples from the batch to display.
    """
    if writer is None:
        return

    b = min(n_samples, lq.shape[0])
    d = lq.shape[-1]
    si = slice_idx if slice_idx is not None else d // 2

    # Extract axial slices: (B, 1, H, W)
    lq_sl = lq[:b, :, :, :, si].float().cpu().clamp(0, 1)
    pred_sl = pred[:b, :, :, :, si].float().cpu().clamp(0, 1)
    hq_sl = hq[:b, :, :, :, si].float().cpu().clamp(0, 1)

    # Concatenate horizontally: (B, 1, H, 3W)
    grid = torch.cat([lq_sl, pred_sl, hq_sl], dim=-1)

    writer.add_images(tag, grid, global_step=step)


def save_nifti(
    volume: torch.Tensor,
    save_path: str,
    affine: Optional[np.ndarray] = None,
) -> None:
    """Save a 3-D tensor as a NIfTI file.

    Parameters
    ----------
    volume:
        Tensor of shape (1, H, W, D) or (H, W, D).
    save_path:
        Destination file path (should end in .nii or .nii.gz).
    affine:
        Optional 4×4 affine matrix. Uses identity if ``None``.
    """
    if nib is None:
        raise ImportError("nibabel is required to save NIfTI files.")

    if volume.ndim == 4:
        volume = volume.squeeze(0)

    arr = volume.detach().cpu().float().numpy()
    affine_mat = affine if affine is not None else np.eye(4)
    img = nib.Nifti1Image(arr, affine_mat)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, save_path)
