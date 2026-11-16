from __future__ import annotations

from itertools import product
from pathlib import Path
import sys
import unittest

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_cross_family import (
    bootstrap_mean_oracle_gain,
    combination_error_indices,
    combo_record,
    cross_family_combinations,
    precompute_view_oracle_sums,
)
from common import fractions, oracle_metrics, winner_labels


def synthetic_cross_family_fixture() -> tuple[np.ndarray, list[str], np.ndarray]:
    rng = np.random.default_rng(47)
    patch_counts = [2, 5, 3, 7]
    views = np.concatenate([
        np.repeat(f"view_{index:02d}.png", count)
        for index, count in enumerate(patch_counts)
    ])
    errors = rng.uniform(1e-4, 5e-3, size=(len(views), 9))
    run_ids = [f"{method}_seed_{seed}" for method in ("3dgs", "ges", "drk") for seed in range(3)]
    return errors, run_ids, views


def legacy_bootstrap(
    errors: np.ndarray,
    run_ids: list[str],
    views: np.ndarray,
    replicates: int,
    seed: int,
) -> np.ndarray:
    combinations = cross_family_combinations([0, 1, 2])
    indices = combination_error_indices(run_ids, combinations)
    labels = ["3dgs", "ges", "drk"]
    unique_views = sorted(set(views))
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(replicates):
        sampled_views = rng.choice(unique_views, size=len(unique_views), replace=True)
        sampled_indices = np.concatenate([np.flatnonzero(views == view) for view in sampled_views])
        gains = [
            combo_record(errors[sampled_indices][:, combination_indices], labels, 1e-5)[0]["oracle_gain_vs_3dgs_fraction"]
            for combination_indices in indices
        ]
        results.append(float(np.mean(gains)))
    return np.asarray(results)


class RebuttalAnalysisTests(unittest.TestCase):
    def test_local_compare_tie_semantics_are_strictly_less_than_threshold(self) -> None:
        errors = np.asarray([
            [0.10, 0.20, 0.30],
            [0.10, 0.100009, 0.40],
            [0.10, 0.100020, 0.40],
        ])
        winners, margins = winner_labels(errors, ["a", "b", "c"], 1e-5)
        self.assertEqual(winners.tolist(), ["a", "tie", "a"])
        self.assertTrue(np.allclose(margins, [0.1, 9e-6, 2e-5]))

    def test_winner_fractions_include_ties_and_sum_to_one(self) -> None:
        values = fractions(np.asarray(["a", "b", "tie", "a"], dtype=object), ["a", "b"])
        self.assertAlmostEqual(values["a_winner_fraction"], 0.5)
        self.assertAlmostEqual(values["b_winner_fraction"], 0.25)
        self.assertAlmostEqual(values["tie_fraction"], 0.25)
        self.assertAlmostEqual(sum(values.values()), 1.0)

    def test_oracle_never_exceeds_any_reference(self) -> None:
        errors = np.asarray([[0.3, 0.2, 0.4], [0.1, 0.5, 0.2]])
        result = oracle_metrics(errors, range(3))
        for reference in result["references"]:
            self.assertLessEqual(result["oracle_mse"], reference["reference_mse"])
            self.assertGreaterEqual(reference["headroom_fraction"], 0.0)

    def test_cross_family_combinations_are_exactly_27(self) -> None:
        combinations = list(product([0, 1, 2], repeat=3))
        self.assertEqual(len(combinations), 27)
        self.assertEqual(len(set(combinations)), 27)

    def test_same_reference_increment_is_well_defined(self) -> None:
        reference = np.asarray([0.4, 0.2])
        same_family_oracle = np.asarray([0.3, 0.15])
        cross_family_oracle = np.asarray([0.2, 0.10])
        within_gain = (reference.mean() - same_family_oracle.mean()) / reference.mean()
        cross_gain = (reference.mean() - cross_family_oracle.mean()) / reference.mean()
        incremental = (same_family_oracle.mean() - cross_family_oracle.mean()) / reference.mean()
        self.assertAlmostEqual(cross_gain - within_gain, incremental)

    def test_optimized_bootstrap_matches_original_with_unequal_view_sizes(self) -> None:
        errors, run_ids, views = synthetic_cross_family_fixture()
        combinations = cross_family_combinations([0, 1, 2])
        indices = combination_error_indices(run_ids, combinations)
        unique_views, patch_counts, reference_sums, oracle_sums = precompute_view_oracle_sums(errors, views, indices)
        self.assertEqual(unique_views.tolist(), sorted(set(views)))
        self.assertEqual(patch_counts.tolist(), [2, 5, 3, 7])

        expected = legacy_bootstrap(errors, run_ids, views, replicates=80, seed=2027)
        actual = bootstrap_mean_oracle_gain(
            patch_counts, reference_sums, oracle_sums, replicates=80, seed=2027
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)

    def test_optimized_bootstrap_is_deterministic_for_fixed_seed(self) -> None:
        errors, run_ids, views = synthetic_cross_family_fixture()
        combinations = cross_family_combinations([0, 1, 2])
        indices = combination_error_indices(run_ids, combinations)
        _, patch_counts, reference_sums, oracle_sums = precompute_view_oracle_sums(errors, views, indices)
        first = bootstrap_mean_oracle_gain(
            patch_counts, reference_sums, oracle_sums, replicates=50, seed=19
        )
        second = bootstrap_mean_oracle_gain(
            patch_counts, reference_sums, oracle_sums, replicates=50, seed=19
        )
        np.testing.assert_array_equal(first, second)

    def test_precomputation_retains_all_27_combinations(self) -> None:
        errors, run_ids, views = synthetic_cross_family_fixture()
        combinations = cross_family_combinations([0, 1, 2])
        indices = combination_error_indices(run_ids, combinations)
        _, patch_counts, reference_sums, oracle_sums = precompute_view_oracle_sums(errors, views, indices)
        self.assertEqual(len(combinations), 27)
        self.assertEqual(len(set(combinations)), 27)
        self.assertEqual(indices.shape, (27, 3))
        self.assertEqual(reference_sums.shape, (len(patch_counts), 27))
        self.assertEqual(oracle_sums.shape, (len(patch_counts), 27))


if __name__ == "__main__":
    unittest.main()
