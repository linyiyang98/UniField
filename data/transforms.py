"""Data transforms / augmentations for UniField.

All transforms operate on raw float32 NumPy arrays of shape (H, W, D)
representing normalised MRI patches.
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional

import numpy as np


class RandomFlip:
    """Randomly flip a 3-D volume along one or more axes."""

    def __init__(self, axes: Optional[List[int]] = None, p: float = 0.5) -> None:
        self.axes = axes if axes is not None else [0, 1, 2]
        self.p = p

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        for axis in self.axes:
            if random.random() < self.p:
                volume = np.flip(volume, axis=axis).copy()
        return volume


class RandomRotate90:
    """Randomly rotate 90° in the axial plane (axes 0–1)."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            k = random.randint(1, 3)
            volume = np.rot90(volume, k=k, axes=(0, 1)).copy()
        return volume


class RandomIntensityShift:
    """Add a small random global intensity offset."""

    def __init__(self, shift_range: float = 0.05) -> None:
        self.shift_range = shift_range

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        shift = random.uniform(-self.shift_range, self.shift_range)
        return np.clip(volume + shift, 0.0, 1.0).astype(np.float32)


class RandomIntensityScale:
    """Multiply intensities by a random scale factor near 1."""

    def __init__(self, scale_range: float = 0.1) -> None:
        self.scale_range = scale_range

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        scale = random.uniform(1.0 - self.scale_range, 1.0 + self.scale_range)
        return np.clip(volume * scale, 0.0, 1.0).astype(np.float32)


class Compose:
    """Sequentially apply a list of transforms."""

    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            volume = t(volume)
        return volume


def build_train_transforms() -> Compose:
    """Return a standard set of training augmentations."""
    return Compose(
        [
            RandomFlip(axes=[0, 1, 2], p=0.5),
            RandomRotate90(p=0.3),
            RandomIntensityShift(shift_range=0.05),
            RandomIntensityScale(scale_range=0.1),
        ]
    )


def build_val_transforms() -> None:
    """No augmentation at validation time (returns ``None``)."""
    return None
