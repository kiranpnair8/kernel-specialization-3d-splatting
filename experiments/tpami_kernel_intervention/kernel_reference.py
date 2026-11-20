"""CPU reference math for the fixed-beta raster kernels.

This module is deliberately independent of PyTorch/CUDA so the defining
equations and support policy can be regression-tested on any development host.
"""

from __future__ import annotations

import math
from typing import Sequence

SUPPORTED_BETAS = (1, 2, 4)
ALPHA_THRESHOLD = 1.0 / 255.0
BETA1_GRADIENT_Q_FLOOR = 1.0e-8


def validate_beta(beta: int) -> int:
    beta = int(beta)
    if beta not in SUPPORTED_BETAS:
        raise ValueError(f"kernel_beta must be one of {SUPPORTED_BETAS}; got {beta}")
    return beta


def quadratic_form(delta: Sequence[float], conic: Sequence[float]) -> float:
    dx, dy = delta
    a, b, c = conic
    return a * dx * dx + 2.0 * b * dx * dy + c * dy * dy


def kernel_value(q: float, beta: int) -> float:
    validate_beta(beta)
    if q < 0.0:
        raise ValueError(f"q must be nonnegative; got {q}")
    return math.exp(-0.5 * q ** (beta / 2.0))


def kernel_derivative_q(q: float, beta: int) -> float:
    """Return dK/dq using the CUDA center convention for beta=1."""
    validate_beta(beta)
    if q < 0.0:
        raise ValueError(f"q must be nonnegative; got {q}")
    value = kernel_value(q, beta)
    if beta == 1:
        if q == 0.0:
            return 0.0
        return -0.25 * value / math.sqrt(max(q, BETA1_GRADIENT_Q_FLOOR))
    if beta == 2:
        return -0.5 * value
    return -q * value


def evaluate_conic(delta: Sequence[float], conic: Sequence[float], beta: int) -> float:
    """Evaluate the 2D radial kernel, retaining baseline beta=2 arithmetic."""
    validate_beta(beta)
    dx, dy = delta
    a, b, c = conic
    if beta == 2:
        return math.exp(-0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy)
    return kernel_value(quadratic_form(delta, conic), beta)


def screen_and_conic_derivatives(
    delta: Sequence[float], conic: Sequence[float], beta: int
) -> tuple[tuple[float, float], tuple[float, float, float]]:
    """Return dK/d(delta) and CUDA's symmetric-conic gradient representation.

    The off-diagonal conic derivative is one half of the derivative with
    respect to the single scalar ``b`` because the downstream covariance code
    treats the two symmetric matrix entries together, matching baseline 3DGS.
    """
    dx, dy = delta
    a, b, c = conic
    q = quadratic_form(delta, conic)
    derivative_q = kernel_derivative_q(q, beta)
    d_delta = (
        2.0 * derivative_q * (a * dx + b * dy),
        2.0 * derivative_q * (b * dx + c * dy),
    )
    d_conic = (
        derivative_q * dx * dx,
        derivative_q * dx * dy,
        derivative_q * dy * dy,
    )
    return d_delta, d_conic


def support_radius_multiplier(beta: int, effective_opacity: float) -> float | None:
    """Return the multiplier applied to sqrt(max projected eigenvalue).

    Beta 2 intentionally returns the original fixed 3-sigma radius. For beta
    1/4, the returned radius reaches the contour where opacity*K=1/255.
    """
    validate_beta(beta)
    if beta == 2:
        return 3.0
    if effective_opacity < ALPHA_THRESHOLD:
        return None
    log_ratio = 2.0 * math.log(effective_opacity / ALPHA_THRESHOLD)
    if log_ratio < 0.0:
        return None
    return log_ratio if beta == 1 else log_ratio ** 0.25
