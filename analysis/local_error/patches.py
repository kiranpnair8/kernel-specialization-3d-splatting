from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class Patch:
    x: int
    y: int
    width: int
    height: int


def iter_patches(height: int, width: int, patch_size: int, stride: int) -> Iterator[Patch]:
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")
    if patch_size > height or patch_size > width:
        raise ValueError(f"patch_size={patch_size} does not fit image shape {height}x{width}")

    y_positions = list(range(0, height - patch_size + 1, stride))
    x_positions = list(range(0, width - patch_size + 1, stride))
    if y_positions[-1] != height - patch_size:
        y_positions.append(height - patch_size)
    if x_positions[-1] != width - patch_size:
        x_positions.append(width - patch_size)

    for y in y_positions:
        for x in x_positions:
            yield Patch(x=x, y=y, width=patch_size, height=patch_size)


def crop(image: np.ndarray, patch: Patch) -> np.ndarray:
    return image[patch.y : patch.y + patch.height, patch.x : patch.x + patch.width, ...]
