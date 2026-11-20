from __future__ import annotations

import math
import unittest

from experiments.tpami_kernel_intervention.kernel_reference import (
    ALPHA_THRESHOLD,
    BETA1_GRADIENT_Q_FLOOR,
    evaluate_conic,
    kernel_derivative_q,
    kernel_value,
    quadratic_form,
    screen_and_conic_derivatives,
    support_radius_multiplier,
    validate_beta,
)


def central_difference(function, value: float, step: float = 1.0e-6) -> float:
    return (function(value + step) - function(value - step)) / (2.0 * step)


class KernelReferenceTests(unittest.TestCase):
    def test_supported_beta_validation(self) -> None:
        self.assertEqual([validate_beta(beta) for beta in (1, 2, 4)], [1, 2, 4])
        with self.assertRaises(ValueError):
            validate_beta(3)

    def test_beta2_exact_original_forward_expression(self) -> None:
        delta = (0.731, -1.219)
        conic = (1.7, 0.23, 0.9)
        dx, dy = delta
        a, b, c = conic
        original = math.exp(-0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy)
        self.assertEqual(evaluate_conic(delta, conic, 2), original)

    def test_analytical_q_derivative_matches_numerical(self) -> None:
        for beta in (1, 2, 4):
            for q in (0.07, 0.8, 3.2):
                numerical = central_difference(lambda x: kernel_value(x, beta), q)
                self.assertAlmostEqual(kernel_derivative_q(q, beta), numerical, places=7)

    def test_screen_and_conic_gradients_match_numerical(self) -> None:
        delta = [0.63, -0.41]
        conic = [1.4, 0.17, 0.95]
        step = 1.0e-6
        for beta in (1, 2, 4):
            d_delta, d_conic = screen_and_conic_derivatives(delta, conic, beta)
            for index in range(2):
                numerical = central_difference(
                    lambda x: evaluate_conic(
                        [x, delta[1]] if index == 0 else [delta[0], x], conic, beta
                    ),
                    delta[index],
                    step,
                )
                self.assertAlmostEqual(d_delta[index], numerical, places=6)
            for index in (0, 2):
                numerical = central_difference(
                    lambda x: evaluate_conic(
                        delta,
                        [x, conic[1], conic[2]] if index == 0 else [conic[0], conic[1], x],
                        beta,
                    ),
                    conic[index],
                    step,
                )
                self.assertAlmostEqual(d_conic[index], numerical, places=6)
            numerical_b = central_difference(
                lambda x: evaluate_conic(delta, [conic[0], x, conic[2]], beta),
                conic[1],
                step,
            )
            self.assertAlmostEqual(2.0 * d_conic[1], numerical_b, places=6)

    def test_beta1_center_uses_finite_symmetric_subgradient(self) -> None:
        self.assertEqual(kernel_derivative_q(0.0, 1), 0.0)
        d_delta, d_conic = screen_and_conic_derivatives((0.0, 0.0), (1.0, 0.0, 1.0), 1)
        self.assertEqual(d_delta, (0.0, 0.0))
        self.assertEqual(d_conic, (0.0, 0.0, 0.0))
        self.assertTrue(math.isfinite(kernel_derivative_q(BETA1_GRADIENT_Q_FLOOR / 2.0, 1)))

    def test_support_reaches_alpha_threshold_for_generalized_kernels(self) -> None:
        for beta in (1, 4):
            for opacity in (0.1, 0.5, 0.99):
                radius = support_radius_multiplier(beta, opacity)
                self.assertIsNotNone(radius)
                q = radius * radius
                self.assertAlmostEqual(opacity * kernel_value(q, beta), ALPHA_THRESHOLD, places=12)

    def test_gaussian_support_is_original_three_sigma(self) -> None:
        for opacity in (0.001, 0.2, 0.99):
            self.assertEqual(support_radius_multiplier(2, opacity), 3.0)

    def test_subthreshold_generalized_primitive_has_no_support(self) -> None:
        self.assertIsNone(support_radius_multiplier(1, ALPHA_THRESHOLD / 2.0))
        self.assertIsNone(support_radius_multiplier(4, ALPHA_THRESHOLD / 2.0))

    def test_quadratic_form_is_nonnegative_for_positive_definite_conic(self) -> None:
        self.assertGreaterEqual(quadratic_form((0.7, -1.2), (1.5, 0.2, 0.8)), 0.0)


if __name__ == "__main__":
    unittest.main()
