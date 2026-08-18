from __future__ import annotations

import math
from typing import Dict

import numpy as np


EPS = 1e-12


def abs_rgb_error(gt: np.ndarray, render: np.ndarray) -> np.ndarray:
    return np.abs(render - gt)


def mse(gt: np.ndarray, render: np.ndarray) -> float:
    return float(np.mean((render - gt) ** 2))


def psnr_from_mse(value: float) -> float:
    if value <= EPS:
        return float("inf")
    return float(-10.0 * math.log10(value))


def grayscale(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def ssim_simple(gt: np.ndarray, render: np.ndarray) -> float:
    x = grayscale(gt).astype(np.float64)
    y = grayscale(render).astype(np.float64)
    c1 = 0.01**2
    c2 = 0.03**2

    mux = float(np.mean(x))
    muy = float(np.mean(y))
    varx = float(np.var(x))
    vary = float(np.var(y))
    cov = float(np.mean((x - mux) * (y - muy)))

    numerator = (2.0 * mux * muy + c1) * (2.0 * cov + c2)
    denominator = (mux * mux + muy * muy + c1) * (varx + vary + c2)
    return float(numerator / max(denominator, EPS))


def image_metrics(gt: np.ndarray, render: np.ndarray) -> Dict[str, float]:
    value = mse(gt, render)
    abs_error = abs_rgb_error(gt, render)
    return {
        "mse": value,
        "psnr": psnr_from_mse(value),
        "mae": float(np.mean(abs_error)),
        "ssim_simple": ssim_simple(gt, render),
    }
