#pragma once

#include <cuda_runtime.h>
#include <math.h>

namespace RadialKernel
{
constexpr float kAlphaThreshold = 1.0f / 255.0f;
constexpr float kBeta1GradientQFloor = 1.0e-8f;

// Evaluate K_beta(q) and dK_beta/dq. The beta=2 branch intentionally uses
// the historical 3DGS arithmetic expression so its Gaussian path is unchanged.
__forceinline__ __device__ bool evaluate(
	const float4 conic_opacity,
	const float2 delta,
	const int kernel_beta,
	float& kernel_value,
	float& dkernel_dq)
{
	if (kernel_beta == 2)
	{
		const float power = -0.5f * (conic_opacity.x * delta.x * delta.x + conic_opacity.z * delta.y * delta.y)
			- conic_opacity.y * delta.x * delta.y;
		if (power > 0.0f)
			return false;
		kernel_value = exp(power);
		dkernel_dq = -0.5f * kernel_value;
		return true;
	}

	const float q = conic_opacity.x * delta.x * delta.x
		+ 2.0f * conic_opacity.y * delta.x * delta.y
		+ conic_opacity.z * delta.y * delta.y;
	if (q < 0.0f)
		return false;

	if (kernel_beta == 1)
	{
		const float sqrt_q = sqrt(q);
		kernel_value = exp(-0.5f * sqrt_q);
		// K_1 is not differentiable at q=0 after composition with screen
		// coordinates. We use the symmetric zero subgradient exactly at the
		// center and cap the near-center dK/dq magnitude with a q floor.
		dkernel_dq = q == 0.0f
			? 0.0f
			: -0.25f * kernel_value / sqrt(max(q, kBeta1GradientQFloor));
		return true;
	}

	// kernel_beta == 4 (validated at the public C++ entry point).
	const float q_squared = q * q;
	kernel_value = exp(-0.5f * q_squared);
	dkernel_dq = -q * kernel_value;
	return true;
}

// Radius multiplier for sqrt(lambda_max). Beta=2 retains the original 3-sigma
// bound exactly. For beta=1/4, solve opacity*K_beta(q) >= 1/255 for the support
// contour so heavy and compact tails receive appropriate tile coverage.
__forceinline__ __device__ bool supportRadiusMultiplier(
	const int kernel_beta,
	const float effective_opacity,
	float& radius_multiplier)
{
	if (kernel_beta == 2)
	{
		radius_multiplier = 3.0f;
		return true;
	}

	if (effective_opacity < kAlphaThreshold)
		return false;

	const float log_ratio = 2.0f * log(effective_opacity / kAlphaThreshold);
	if (log_ratio < 0.0f)
		return false;

	radius_multiplier = kernel_beta == 1 ? log_ratio : sqrt(sqrt(log_ratio));
	return true;
}
}
