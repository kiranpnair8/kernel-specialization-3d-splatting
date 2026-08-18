from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from PIL import Image

from analysis.local_error.patches import Patch


def _normalize(values: np.ndarray, symmetric: bool = False) -> np.ndarray:
    values = values.astype(np.float64)
    if symmetric:
        scale = max(float(np.max(np.abs(values))), 1e-12)
        return np.clip((values / scale + 1.0) * 0.5, 0.0, 1.0)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _heatmap(values: np.ndarray) -> np.ndarray:
    t = _normalize(values)
    return np.stack([t, 0.25 * (1.0 - t), 1.0 - t], axis=-1)


def _signed_map(values: np.ndarray) -> np.ndarray:
    t = _normalize(values, symmetric=True)
    red = np.clip((t - 0.5) * 2.0, 0.0, 1.0)
    blue = np.clip((0.5 - t) * 2.0, 0.0, 1.0)
    green = 1.0 - np.maximum(red, blue)
    return np.stack([red, green, blue], axis=-1)


def save_float_map(path: Path, values: np.ndarray, signed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = _signed_map(values) if signed else _heatmap(values)
    Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(path)


def rasterize_patch_values(shape: Tuple[int, int], patches: Iterable[Patch], values: Iterable[float]) -> np.ndarray:
    height, width = shape
    accum = np.zeros((height, width), dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.float64)
    for patch, value in zip(patches, values):
        accum[patch.y : patch.y + patch.height, patch.x : patch.x + patch.width] += value
        counts[patch.y : patch.y + patch.height, patch.x : patch.x + patch.width] += 1.0
    return accum / np.maximum(counts, 1.0)


def save_winner_map(
    path: Path,
    shape: Tuple[int, int],
    patches: Iterable[Patch],
    winners: Iterable[str],
    method_colors: Dict[str, Tuple[int, int, int]],
    tie_color: Tuple[int, int, int] = (160, 160, 160),
) -> None:
    height, width = shape
    accum = np.zeros((height, width, 3), dtype=np.float64)
    counts = np.zeros((height, width, 1), dtype=np.float64)
    for patch, winner in zip(patches, winners):
        color = method_colors.get(winner, tie_color)
        accum[patch.y : patch.y + patch.height, patch.x : patch.x + patch.width, :] += np.asarray(color)
        counts[patch.y : patch.y + patch.height, patch.x : patch.x + patch.width, :] += 1.0
    rgb = accum / np.maximum(counts, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)
