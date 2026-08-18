# Local Error Analysis

This package validates paired held-out views and computes image-level and
patch-level errors for any configured set of rendered method outputs.

Patch LPIPS is intentionally not computed here. LPIPS is calibrated as a
learned perceptual image metric, and naive small-patch LPIPS can be unstable,
context-dependent, and sensitive to resize/padding choices. For this pipeline
validation stage, patch analysis uses RGB absolute error, MSE, PSNR-style scores,
and a lightweight SSIM-style local statistic. Image-level LPIPS from upstream
method evaluation remains useful as a global baseline metric.
