#!/usr/bin/env python3
"""Small CUDA forward/backward smoke test for beta=1,2,4.

Run only after building/installing the vendored rasterizer extension. This test
uses two synthetic primitives and does not require a scene dataset.
"""

from __future__ import annotations

import argparse
import math

import torch

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def settings(beta: int) -> GaussianRasterizationSettings:
    identity = torch.eye(4, dtype=torch.float32, device="cuda")
    return GaussianRasterizationSettings(
        image_height=32,
        image_width=32,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=torch.zeros(3, dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=identity,
        projmatrix=identity,
        sh_degree=0,
        campos=torch.zeros(3, dtype=torch.float32, device="cuda"),
        prefiltered=False,
        debug=False,
        antialiasing=False,
        kernel_beta=beta,
    )


def inputs(opacity_value: float = 0.45, covariance_xx: float = 0.035) -> dict[str, torch.Tensor]:
    return {
        "means3D": torch.tensor(
            [[-0.12, 0.02, 2.0], [0.17, -0.08, 2.3]],
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        ),
        "means2D": torch.zeros((2, 3), dtype=torch.float32, device="cuda", requires_grad=True),
        "colors_precomp": torch.tensor(
            [[0.9, 0.2, 0.1], [0.1, 0.4, 0.9]], dtype=torch.float32, device="cuda", requires_grad=True
        ),
        "opacities": torch.tensor(
            [[opacity_value], [0.35]], dtype=torch.float32, device="cuda", requires_grad=True
        ),
        "cov3D_precomp": torch.tensor(
            [[covariance_xx, 0.0, 0.0, 0.035, 0.0, 0.02], [0.04, 0.0, 0.0, 0.025, 0.0, 0.03]],
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        ),
    }


def scalar_loss(image: torch.Tensor, invdepth: torch.Tensor) -> torch.Tensor:
    weights = torch.linspace(0.2, 1.0, image.numel(), device="cuda").reshape_as(image)
    return (image * weights).sum() + 0.01 * invdepth.sum()


def render(
    beta: int,
    opacity_value: float = 0.45,
    covariance_xx: float = 0.035,
    backward: bool = False,
):
    tensors = inputs(opacity_value, covariance_xx)
    image, radii, invdepth = GaussianRasterizer(settings(beta))(
        shs=None,
        scales=None,
        rotations=None,
        **tensors,
    )
    if backward:
        scalar_loss(image, invdepth).backward()
    return image, radii, invdepth, tensors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finite-difference-step", type=float, default=1.0e-3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    outputs = {}
    for beta in (1, 2, 4):
        image, radii, invdepth, tensors = render(beta, backward=True)
        assert torch.isfinite(image).all() and torch.isfinite(invdepth).all()
        assert (radii > 0).all(), (beta, radii)
        for name, tensor in tensors.items():
            assert tensor.grad is not None, (beta, name)
            assert torch.isfinite(tensor.grad).all(), (beta, name)
        outputs[beta] = image.detach()
        print(
            f"beta={beta}: image_sum={image.sum().item():.8f}, "
            f"radii={radii.tolist()}, finite_forward_backward=PASS"
        )

    beta2_repeat = render(2)[0]
    if not torch.equal(outputs[2], beta2_repeat):
        raise AssertionError("beta=2 repeated forward pass was not bitwise deterministic")

    step = args.finite_difference_step
    for beta in (1, 2, 4):
        center = render(beta, backward=True)
        analytical_opacity = center[3]["opacities"].grad[0, 0].item()
        plus_image, _, plus_depth, _ = render(beta, opacity_value=0.45 + step)
        minus_image, _, minus_depth, _ = render(beta, opacity_value=0.45 - step)
        numerical_opacity = (
            (scalar_loss(plus_image, plus_depth) - scalar_loss(minus_image, minus_depth))
            / (2.0 * step)
        ).item()
        if not math.isclose(analytical_opacity, numerical_opacity, rel_tol=5.0e-3, abs_tol=5.0e-3):
            raise AssertionError(
                f"beta={beta} opacity gradient mismatch: "
                f"analytical={analytical_opacity}, numerical={numerical_opacity}"
            )

        covariance_step = 1.0e-4
        analytical_covariance = center[3]["cov3D_precomp"].grad[0, 0].item()
        plus_image, _, plus_depth, _ = render(beta, covariance_xx=0.035 + covariance_step)
        minus_image, _, minus_depth, _ = render(beta, covariance_xx=0.035 - covariance_step)
        numerical_covariance = (
            (scalar_loss(plus_image, plus_depth) - scalar_loss(minus_image, minus_depth))
            / (2.0 * covariance_step)
        ).item()
        if not math.isclose(analytical_covariance, numerical_covariance, rel_tol=2.0e-2, abs_tol=2.0e-2):
            raise AssertionError(
                f"beta={beta} covariance gradient mismatch: "
                f"analytical={analytical_covariance}, numerical={numerical_covariance}"
            )
        print(
            f"beta={beta} gradients: opacity=({analytical_opacity:.8f}, {numerical_opacity:.8f}), "
            f"cov_xx=({analytical_covariance:.8f}, {numerical_covariance:.8f}), PASS"
        )
    print("GPU smoke test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
