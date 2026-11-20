#!/usr/bin/env python3
"""Regression checks for contiguous and zero-stride rasterizer gradients."""

from __future__ import annotations

import argparse
import math

import torch

from gpu_smoke_test import render


def depth_upstream(depth: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "contiguous":
        gradient = torch.full_like(depth, 0.01)
        assert gradient.is_contiguous()
        return gradient
    if layout == "expanded":
        gradient = torch.tensor(0.01, dtype=depth.dtype, device=depth.device).expand_as(depth)
        assert not gradient.is_contiguous()
        assert 0 in gradient.stride()
        return gradient
    raise ValueError(layout)


def rgb_upstream(image: torch.Tensor, layout: str) -> torch.Tensor:
    channel_weights = torch.tensor([0.2, 0.6, 1.0], dtype=image.dtype, device=image.device).view(3, 1, 1)
    if layout == "contiguous":
        gradient = channel_weights.expand_as(image).clone()
        assert gradient.is_contiguous()
        return gradient
    if layout == "expanded":
        gradient = channel_weights.expand_as(image)
        assert not gradient.is_contiguous()
        assert 0 in gradient.stride()
        return gradient
    raise ValueError(layout)


def analytical_depth_gradients(beta: int, layout: str) -> tuple[float, float]:
    _, _, depth, tensors = render(beta)
    torch.autograd.backward(depth, depth_upstream(depth, layout))
    torch.cuda.synchronize()
    return (
        tensors["opacities"].grad[0, 0].item(),
        tensors["cov3D_precomp"].grad[0, 0].item(),
    )


def analytical_rgb_gradients(beta: int, layout: str) -> tuple[float, float]:
    image, _, _, tensors = render(beta)
    torch.autograd.backward(image, rgb_upstream(image, layout))
    torch.cuda.synchronize()
    return (
        tensors["opacities"].grad[0, 0].item(),
        tensors["cov3D_precomp"].grad[0, 0].item(),
    )


@torch.no_grad()
def depth_objective(beta: int, opacity: float = 0.45, covariance_xx: float = 0.035) -> float:
    _, _, depth, _ = render(beta, opacity_value=opacity, covariance_xx=covariance_xx)
    return (depth * 0.01).sum().item()


@torch.no_grad()
def rgb_objective(beta: int, opacity: float = 0.45, covariance_xx: float = 0.035) -> float:
    image, _, _, _ = render(beta, opacity_value=opacity, covariance_xx=covariance_xx)
    return (image * rgb_upstream(image, "expanded")).sum().item()


def finite_difference(function, center: float, step: float) -> float:
    return (function(center + step) - function(center - step)) / (2.0 * step)


def assert_close(label: str, analytical: float, numerical: float, rtol: float, atol: float) -> None:
    if not math.isclose(analytical, numerical, rel_tol=rtol, abs_tol=atol):
        raise AssertionError(f"{label}: analytical={analytical:.9g}, numerical={numerical:.9g}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 10:
        raise ValueError("--repeats must be at least 10")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this regression test")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    for beta in (1, 2, 4):
        repeated: dict[str, list[tuple[float, float]]] = {}
        for layout in ("contiguous", "expanded"):
            values = [analytical_depth_gradients(beta, layout) for _ in range(args.repeats)]
            repeated[layout] = values
            reference = torch.tensor(values[0])
            observed = torch.tensor(values)
            if not torch.allclose(observed, reference.expand_as(observed), rtol=1.0e-5, atol=1.0e-6):
                raise AssertionError(f"beta={beta} {layout} depth gradients are nondeterministic: {values}")

        contiguous = torch.tensor(repeated["contiguous"])
        expanded = torch.tensor(repeated["expanded"])
        if not torch.allclose(contiguous, expanded, rtol=1.0e-5, atol=1.0e-6):
            raise AssertionError(
                f"beta={beta} contiguous/expanded depth gradients differ: "
                f"contiguous={repeated['contiguous']}, expanded={repeated['expanded']}"
            )

        analytical_opacity, analytical_covariance = repeated["expanded"][0]
        numerical_opacity = finite_difference(
            lambda value: depth_objective(beta, opacity=value), 0.45, 1.0e-3
        )
        numerical_covariance = finite_difference(
            lambda value: depth_objective(beta, covariance_xx=value), 0.035, 1.0e-4
        )
        assert_close(
            f"beta={beta} depth opacity gradient",
            analytical_opacity,
            numerical_opacity,
            rtol=5.0e-3,
            atol=5.0e-3,
        )
        assert_close(
            f"beta={beta} depth covariance gradient",
            analytical_covariance,
            numerical_covariance,
            rtol=2.0e-2,
            atol=2.0e-2,
        )

        rgb_contiguous = analytical_rgb_gradients(beta, "contiguous")
        rgb_expanded = analytical_rgb_gradients(beta, "expanded")
        if not torch.allclose(torch.tensor(rgb_contiguous), torch.tensor(rgb_expanded), rtol=1.0e-5, atol=1.0e-6):
            raise AssertionError(
                f"beta={beta} contiguous/expanded RGB gradients differ: "
                f"contiguous={rgb_contiguous}, expanded={rgb_expanded}"
            )
        numerical_rgb_opacity = finite_difference(
            lambda value: rgb_objective(beta, opacity=value), 0.45, 1.0e-3
        )
        numerical_rgb_covariance = finite_difference(
            lambda value: rgb_objective(beta, covariance_xx=value), 0.035, 1.0e-4
        )
        assert_close(
            f"beta={beta} RGB opacity gradient",
            rgb_expanded[0],
            numerical_rgb_opacity,
            rtol=5.0e-3,
            atol=5.0e-3,
        )
        assert_close(
            f"beta={beta} RGB covariance gradient",
            rgb_expanded[1],
            numerical_rgb_covariance,
            rtol=2.0e-2,
            atol=2.0e-2,
        )

        print(
            f"beta={beta}: depth_repeat_10x=PASS, layouts=PASS, "
            f"depth_opacity=({analytical_opacity:.8f}, {numerical_opacity:.8f}), "
            f"depth_cov_xx=({analytical_covariance:.8f}, {numerical_covariance:.8f}), "
            f"rgb_opacity=({rgb_expanded[0]:.8f}, {numerical_rgb_opacity:.8f}), "
            f"rgb_cov_xx=({rgb_expanded[1]:.8f}, {numerical_rgb_covariance:.8f})"
        )

    print("Depth-gradient regression PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
