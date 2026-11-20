from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RASTERIZER = ROOT / "vendor" / "gaussian-splatting" / "submodules" / "diff-gaussian-rasterization"


class SourceContractTests(unittest.TestCase):
    def test_beta_reaches_python_and_cuda_entry_points(self) -> None:
        files = {
            "python": RASTERIZER / "diff_gaussian_rasterization" / "__init__.py",
            "binding": RASTERIZER / "rasterize_points.cu",
            "forward": RASTERIZER / "cuda_rasterizer" / "forward.cu",
            "backward": RASTERIZER / "cuda_rasterizer" / "backward.cu",
        }
        for label, path in files.items():
            self.assertIn("kernel_beta", path.read_text(encoding="utf-8"), label)

    def test_gaussian_expression_and_three_sigma_bound_are_retained(self) -> None:
        helper = (RASTERIZER / "cuda_rasterizer" / "kernel_utils.cuh").read_text(encoding="utf-8")
        self.assertIn("-0.5f * (conic_opacity.x * delta.x * delta.x", helper)
        self.assertIn("- conic_opacity.y * delta.x * delta.y", helper)
        self.assertIn("radius_multiplier = 3.0f", helper)

    def test_beta1_center_strategy_is_explicit(self) -> None:
        helper = (RASTERIZER / "cuda_rasterizer" / "kernel_utils.cuh").read_text(encoding="utf-8")
        self.assertIn("q == 0.0f", helper)
        self.assertIn("kBeta1GradientQFloor", helper)

    def test_upstream_gradients_are_materialized_before_cuda(self) -> None:
        binding = (RASTERIZER / "rasterize_points.cu").read_text(encoding="utf-8")
        self.assertIn("dL_dout_color_contiguous = dL_dout_color.contiguous()", binding)
        self.assertIn("dL_dout_invdepth_contiguous = dL_dout_invdepth.contiguous()", binding)
        self.assertIn("dL_dout_invdepth_contiguous.data_ptr<float>()", binding)


if __name__ == "__main__":
    unittest.main()
