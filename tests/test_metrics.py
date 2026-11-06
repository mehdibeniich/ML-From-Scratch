import unittest

import numpy as np

from ml_from_scratch.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class TestRegressionMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([1, 2, 3], dtype=float)
        self.y_pred = np.array([1, 3, 5], dtype=float)

    def test_mean_squared_error(self):
        self.assertAlmostEqual(mean_squared_error(self.y_true, self.y_pred), 5 / 3)

    def test_mean_absolute_error(self):
        self.assertAlmostEqual(mean_absolute_error(self.y_true, self.y_pred), 1.0)

    def test_r2_score_can_be_negative(self):
        self.assertAlmostEqual(r2_score(self.y_true, self.y_pred), -1.5)

    def test_r2_score_for_perfect_constant_target(self):
        self.assertEqual(r2_score([2, 2], [2, 2]), 1.0)

    def test_r2_score_for_imperfect_constant_target(self):
        self.assertEqual(r2_score([2, 2], [1, 3]), 0.0)

    def test_empty_inputs_raise(self):
        with self.assertRaises(ValueError):
            mean_squared_error([], [])

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            mean_absolute_error([1, 2], [1])

    def test_non_finite_values_raise(self):
        with self.assertRaises(ValueError):
            r2_score([1, np.nan], [1, 2])


if __name__ == "__main__":
    unittest.main()
