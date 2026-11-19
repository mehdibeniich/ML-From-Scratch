import unittest

import numpy as np

from ml_from_scratch.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class TestClassificationMetrics(unittest.TestCase):
    def test_accuracy_is_one_for_perfect_predictions(self):
        self.assertEqual(accuracy_score([0, 1, 2], [0, 1, 2]), 1.0)

    def test_accuracy_supports_string_multiclass_labels(self):
        self.assertAlmostEqual(
            accuracy_score(["cat", "dog", "bird"], ["cat", "bird", "bird"]),
            2 / 3,
        )

    def test_accuracy_compares_mixed_label_types_without_coercion(self):
        self.assertEqual(accuracy_score([1, "1"], ["1", "1"]), 0.5)

    def test_accuracy_rejects_empty_mismatched_and_non_1d_inputs(self):
        invalid_cases = (
            ([], []),
            ([0, 1], [0]),
            ([[0], [1]], [0, 1]),
            ([0, 1], [[0], [1]]),
        )
        for y_true, y_pred in invalid_cases:
            with self.subTest(y_true=y_true, y_pred=y_pred):
                with self.assertRaises(ValueError):
                    accuracy_score(y_true, y_pred)

    def test_accuracy_rejects_missing_or_non_finite_labels(self):
        invalid_cases = (
            ([0, None], [0, 1]),
            ([0.0, np.nan], [0.0, 1.0]),
            ([0.0, np.inf], [0.0, 1.0]),
            ([0, 1], [0, None]),
        )
        for y_true, y_pred in invalid_cases:
            with self.subTest(y_true=y_true, y_pred=y_pred):
                with self.assertRaises(ValueError):
                    accuracy_score(y_true, y_pred)

    def test_confusion_matrix_binary_counts_true_rows_and_predicted_columns(self):
        matrix = confusion_matrix(
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            labels=[0, 1],
        )
        np.testing.assert_array_equal(matrix, [[1, 1], [1, 1]])

    def test_confusion_matrix_sorts_comparable_multiclass_labels(self):
        matrix = confusion_matrix(
            ["dog", "cat", "bird", "dog"],
            ["cat", "cat", "dog", "dog"],
        )
        np.testing.assert_array_equal(
            matrix,
            [[0, 0, 1], [0, 1, 0], [0, 1, 1]],
        )

    def test_confusion_matrix_preserves_explicit_label_order(self):
        matrix = confusion_matrix(
            [0, 1, 2],
            [1, 1, 0],
            labels=[2, 0, 1],
        )
        np.testing.assert_array_equal(
            matrix,
            [[0, 1, 0], [0, 0, 1], [0, 0, 1]],
        )

    def test_confusion_matrix_handles_mixed_label_types_without_coercion(self):
        matrix = confusion_matrix([1, "a", 1], ["a", "a", 1])
        np.testing.assert_array_equal(matrix, [[1, 1], [0, 1]])

    def test_confusion_matrix_rejects_invalid_or_incomplete_labels(self):
        invalid_labels = (
            [],
            [0, 0],
            [[0], [1]],
            [0, None],
            [0, np.nan],
        )
        for labels in invalid_labels:
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    confusion_matrix([0, 1], [0, 1], labels=labels)

        with self.assertRaisesRegex(ValueError, "include every class"):
            confusion_matrix([0, 1], [0, 2], labels=[0, 1])


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

    def test_r2_score_requires_at_least_two_samples(self):
        with self.assertRaisesRegex(ValueError, "at least two samples"):
            r2_score([1.0], [1.0])
        with self.assertRaisesRegex(ValueError, "at least two samples"):
            r2_score([1.0], [2.0])

    def test_overflowing_metric_calculations_raise(self):
        with self.assertRaises(FloatingPointError):
            mean_squared_error([1e200], [0.0])
        with self.assertRaises(FloatingPointError):
            mean_absolute_error([1e308], [-1e308])
        with self.assertRaises(FloatingPointError):
            r2_score([1e200, -1e200], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
