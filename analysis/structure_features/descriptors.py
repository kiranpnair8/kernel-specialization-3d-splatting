from __future__ import annotations

from typing import Dict

import numpy as np

from analysis.local_error.metrics import grayscale


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
