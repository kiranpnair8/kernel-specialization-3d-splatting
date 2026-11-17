from __future__ import annotations

from typing import Dict

import numpy as np

from analysis.local_error.metrics import grayscale


def _integral_image(values: np.ndarray) -> np.ndarray:
    return np.pad(values.astype(np.float64), ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)


def _rectangle_sums(integral: np.ndarray, x: np.ndarray, y: np.ndarray, width: np.ndarray, height: np.ndarray) -> np.ndarray:
    x1, y1 = x + width, y + height
    return integral[y1, x1] - integral[y, x1] - integral[y1, x] + integral[y, x]


def image_patch_complexity_descriptors(
    rgb: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    width: np.ndarray,
    height: np.ndarray,
    edge_threshold: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Vectorized GT descriptors used by the multi-seed rebuttal analyses."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image, found {rgb.shape}")
    x, y, width, height = (np.asarray(value, dtype=np.int64) for value in (x, y, width, height))
    if np.any(x < 0) or np.any(y < 0) or np.any(width <= 0) or np.any(height <= 0):
        raise ValueError("Invalid patch coordinates")
    if np.any(x + width > rgb.shape[1]) or np.any(y + height > rgb.shape[0]):
        raise ValueError("Patch coordinates exceed image dimensions")
    gray = grayscale(rgb).astype(np.float64)
    gradient_y, gradient_x = np.gradient(gray)
    gradient = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    area = (width * height).astype(np.float64)
    gradient_sum = _rectangle_sums(_integral_image(gradient), x, y, width, height)
    edge_sum = _rectangle_sums(_integral_image(gradient > edge_threshold), x, y, width, height)
    gray_sum = _rectangle_sums(_integral_image(gray), x, y, width, height)
    square_sum = _rectangle_sums(_integral_image(gray * gray), x, y, width, height)
    mean = gray_sum / area
    return {
        "mean_gradient_magnitude": gradient_sum / area,
        "edge_density": edge_sum / area,
        "local_texture_variance": np.maximum(square_sum / area - mean * mean, 0.0),
    }


def _box_blur(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, 1, mode="reflect")
    out = np.zeros_like(gray)
    for dy in range(3):
        for dx in range(3):
            out += padded[dy : dy + gray.shape[0], dx : dx + gray.shape[1]]
    return out / 9.0


def _laplacian(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, 1, mode="reflect")
    center = padded[1:-1, 1:-1] * -4.0
    return center + padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]


def _entropy(gray: np.ndarray, bins: int = 32) -> float:
    hist, _ = np.histogram(gray, bins=bins, range=(0.0, 1.0), density=False)
    probs = hist.astype(np.float64)
    total = probs.sum()
    if total <= 0:
        return 0.0
    probs = probs[probs > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def patch_descriptors(gt_patch: np.ndarray) -> Dict[str, float]:
    gray = grayscale(gt_patch).astype(np.float64)
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    lap = _laplacian(gray)
    low = _box_blur(gray)
    high = gray - low

    return {
        "mean_gradient_magnitude": float(np.mean(grad_mag)),
        "edge_strength": float(np.percentile(grad_mag, 90)),
        "laplacian_energy": float(np.mean(lap * lap)),
        "local_variance": float(np.var(gray)),
        "high_frequency_energy": float(np.mean(high * high)),
        "entropy": _entropy(gray),
    }
