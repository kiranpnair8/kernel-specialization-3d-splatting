from __future__ import annotations

from itertools import product
from pathlib import Path
import sys
import unittest

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import fractions, oracle_metrics, winner_labels


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


if __name__ == "__main__":
    unittest.main()
