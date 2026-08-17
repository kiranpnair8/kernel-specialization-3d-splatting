# 3DGS Baseline — Mip-NeRF360 Garden

## Environment
- GPU: Tesla V100-PCIE-32GB
- PyTorch: 2.5.1+cu121
- CUDA toolkit: 12.3
- GCC/G++: 12.4
- 3DGS commit: 54c035f7834b564019656c3e3fcc3646292f727d
- GPU architecture: sm_70

## Dataset
- Dataset: Mip-NeRF360
- Scene: garden
- Image resolution: images_4
- Evaluation split: LLFF holdout
- Initial COLMAP points: 138,766

## Training
- Iterations: 30,000
- Training wall time: ~44 minutes
- Final train PSNR: 31.4790 dB
- Final test PSNR during training: 27.4850 dB

## Final Evaluation
- PSNR: 27.4781 dB
- SSIM: 0.8680687
- LPIPS: 0.1060733

## Representation
- Final primitive count: 4,146,866
- Output directory size: 2.0 GB

## Output
outputs/3dgs/garden_baseline/
