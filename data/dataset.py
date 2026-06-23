"""Dataset utilities for UniField.

Expected directory layout::

    root/
        0.35T/
            subject_001/
                image.nii.gz
                ...
        1.5T/
            subject_001/
                image.nii.gz
                ...
        3T/  ...
        7T/  ...

Each sub-directory name is used to infer the B₀ field strength label.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

# Mapping from folder-name strings to integer labels.
FIELD_STRENGTH_LABELS: Dict[str, int] = {
    "0.35T": 0,
    "0.35t": 0,
    "1.5T": 1,
    "1.5t": 1,
    "3T": 2,
    "3t": 2,
    "7T": 3,
    "7t": 3,
}


def _load_nifti(path: str) -> np.ndarray:
    """Load a NIfTI volume and return a float32 numpy array."""
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    return data


def _normalise(volume: np.ndarray) -> np.ndarray:
    """Min-max normalise a volume to [0, 1]."""
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(volume)
    return (volume - vmin) / (vmax - vmin)


def _random_patch(
    volume: np.ndarray,
    patch_size: Tuple[int, int, int],
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Extract a random 3-D patch from *volume* (H×W×D)."""
    h, w, d = volume.shape
    ph, pw, pd = patch_size
    if h < ph or w < pw or d < pd:
        # Pad if the volume is smaller than the requested patch.
        pad_h = max(0, ph - h)
        pad_w = max(0, pw - w)
        pad_d = max(0, pd - d)
        volume = np.pad(
            volume,
            ((0, pad_h), (0, pad_w), (0, pad_d)),
            mode="reflect",
        )
        h, w, d = volume.shape

    z0 = random.randint(0, h - ph)
    z1 = random.randint(0, w - pw)
    z2 = random.randint(0, d - pd)
    patch = volume[z0 : z0 + ph, z1 : z1 + pw, z2 : z2 + pd]
    return patch, (z0, z1, z2)


class MRIEnhancementDataset(Dataset):
    """PyTorch Dataset for field-aware MRI super-resolution / denoising.

    Parameters
    ----------
    root:
        Root directory containing per-field-strength sub-directories.
    patch_size:
        Spatial size (H, W, D) of training patches.
    scale_factor:
        Down-sampling factor for super-resolution tasks. Ignored when
        ``task == 'denoise'``.
    task:
        One of ``'sr'``, ``'denoise'``, or ``'sr+denoise'``.
    noise_sigma:
        Standard deviation of additive Gaussian noise (denoising tasks).
    transform:
        Optional callable applied to the *high-quality* patch (after
        normalisation but before degradation).
    is_train:
        If ``True`` extract random patches; otherwise use the full volume.
    """

    def __init__(
        self,
        root: str,
        patch_size: Tuple[int, int, int] = (64, 64, 32),
        scale_factor: int = 4,
        task: str = "sr",
        noise_sigma: float = 0.05,
        transform: Optional[Callable] = None,
        is_train: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.patch_size = tuple(patch_size)
        self.scale_factor = scale_factor
        self.task = task
        self.noise_sigma = noise_sigma
        self.transform = transform
        self.is_train = is_train

        self.samples: List[Tuple[str, int]] = []
        self._collect_samples()

    def _collect_samples(self) -> None:
        """Walk *root* and collect (file_path, field_label) pairs."""
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        for field_dir in sorted(self.root.iterdir()):
            if not field_dir.is_dir():
                continue
            label = FIELD_STRENGTH_LABELS.get(field_dir.name)
            if label is None:
                continue
            for fpath in sorted(field_dir.rglob("*.nii.gz")):
                self.samples.append((str(fpath), label))
            for fpath in sorted(field_dir.rglob("*.nii")):
                self.samples.append((str(fpath), label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No NIfTI files found under {self.root}. "
                "Expected sub-directories named one of: "
                + ", ".join(FIELD_STRENGTH_LABELS.keys())
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _degrade(
        self, hq: np.ndarray
    ) -> np.ndarray:
        """Apply degradation to produce the low-quality input."""
        lq = hq.copy()

        if "sr" in self.task:
            sf = self.scale_factor
            # Simulate low-resolution by downsampling then upsampling.
            lr_shape = tuple(max(1, s // sf) for s in lq.shape)
            from scipy.ndimage import zoom as ndimage_zoom
            lq = ndimage_zoom(lq, [ls / s for ls, s in zip(lr_shape, lq.shape)], order=1)
            lq = ndimage_zoom(lq, [s / ls for s, ls in zip(hq.shape, lr_shape)], order=1)

        if "denoise" in self.task:
            noise = np.random.normal(0.0, self.noise_sigma, size=lq.shape).astype(
                np.float32
            )
            lq = np.clip(lq + noise, 0.0, 1.0)

        return lq.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fpath, field_label = self.samples[idx]
        volume = _load_nifti(fpath)
        volume = _normalise(volume)

        if self.is_train:
            hq_patch, _ = _random_patch(volume, self.patch_size)
        else:
            hq_patch = volume

        if self.transform is not None:
            hq_patch = self.transform(hq_patch)

        lq_patch = self._degrade(hq_patch)

        # Add channel dimension: (1, H, W, D)
        hq_tensor = torch.from_numpy(hq_patch).unsqueeze(0)
        lq_tensor = torch.from_numpy(lq_patch).unsqueeze(0)
        field_tensor = torch.tensor(field_label, dtype=torch.long)

        return {
            "lq": lq_tensor,
            "hq": hq_tensor,
            "field": field_tensor,
            "path": fpath,
        }
