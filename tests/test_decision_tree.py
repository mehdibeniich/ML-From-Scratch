import unittest

import numpy as np

from ml_from_scratch.decision_tree import best_split, gini_impurity


def exhaustive_best_split(X, y):
    """Reference implementation that evaluates each child label slice."""
    features = np.asarray(X, dtype=float)
    labels = np.asarray(y)
    parent_impurity = gini_impurity(labels)
    if parent_impurity == 0.0:
        return None

    best = None
    best_impurity = parent_impurity
    n_samples = labels.size
    for feature_index in range(features.shape[1]):
        values = features[:, feature_index]
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        sorted_labels = labels[order]
        split_positions = np.flatnonzero(
            sorted_values[:-1] < sorted_values[1:]
        ) + 1
        for split_position in split_positions:
            left_value = sorted_values[split_position - 1]
            right_value = sorted_values[split_position]
            threshold = left_value / 2.0 + right_value / 2.0
            if not (left_value <= threshold < right_value):
                threshold = left_value
            left_labels = sorted_labels[:split_position]
            right_labels = sorted_labels[split_position:]
            weighted_impurity = (
                left_labels.size * gini_impurity(left_labels)
                + right_labels.size * gini_impurity(right_labels)
            ) / n_samples
            if weighted_impurity < best_impurity:
                best_impurity = weighted_impurity
                best = (feature_index, float(threshold), float(weighted_impurity))
    return best


class TestGiniImpurity(unittest.TestCase):
    def test_pure_node_has_zero_impurity(self):
        self.assertEqual(gini_impurity([1, 1, 1, 1]), 0.0)

    def test_balanced_binary_node(self):
        self.assertAlmostEqual(gini_impurity([0, 0, 1, 1]), 0.5)

    def test_multiclass_node(self):
        self.assertAlmostEqual(gini_impurity(["cat", "cat", "dog", "bird"]), 0.625)

    def test_accepts_string_labels(self):
        self.assertAlmostEqual(gini_impurity(["yes", "no", "yes"]), 4.0 / 9.0)

    def test_rejects_empty_labels(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            gini_impurity([])

    def test_rejects_non_1d_labels(self):
        with self.assertRaisesRegex(ValueError, "1D"):
            gini_impurity([[0, 1], [1, 0]])

    def test_rejects_missing_and_non_finite_labels(self):
        invalid_cases = (
            [0, None],
            [0.0, np.nan],
            [0.0, np.inf],
            [0.0, -np.inf],
            np.array([0, np.nan], dtype=object),
        )
        for labels in invalid_cases:
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    gini_impurity(labels)

    def test_rejects_incomparable_object_labels(self):
        labels = np.array([1, "one"], dtype=object)
        with self.assertRaisesRegex(ValueError, "mutually comparable"):
            gini_impurity(labels)


class TestBestSplit(unittest.TestCase):
    def test_finds_perfect_split_and_prefers_first_feature_on_tie(self):
        X = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        result = best_split(X, [0, 0, 1, 1])

        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 1.5)
        self.assertEqual(result[2], 0.0)

    def test_supports_string_class_labels(self):
        result = best_split(
            [[0.0], [1.0], [2.0], [3.0]], ["low", "low", "high", "high"]
        )

        self.assertEqual(result, (0, 1.5, 0.0))

    def test_uses_weighted_child_impurity_and_keeps_first_threshold_on_tie(self):
        result = best_split([[0.0], [1.0], [2.0], [3.0]], [0, 1, 0, 1])

        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 0.5)
        self.assertAlmostEqual(result[2], 1.0 / 3.0)

    def test_duplicate_feature_values_never_create_empty_children(self):
        result = best_split([[0.0], [0.0], [1.0], [1.0]], [0, 1, 1, 1])

        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 0.5)
        self.assertAlmostEqual(result[2], 0.25)

    def test_returns_none_when_features_are_constant(self):
        self.assertIsNone(best_split([[2.0], [2.0], [2.0]], [0, 1, 1]))

    def test_returns_none_for_a_pure_parent_node(self):
        self.assertIsNone(best_split([[0.0], [1.0], [2.0]], ["same"] * 3))

    def test_threshold_calculation_avoids_overflow(self):
        result = best_split([[-1e308], [1e308]], [0, 1])

        self.assertEqual(result[1], 0.0)
        self.assertEqual(result[2], 0.0)

    def test_rejects_invalid_feature_arrays(self):
        invalid_features = (
            [],
            np.empty((0, 1)),
            np.empty((2, 0)),
            [[1.0], [np.nan]],
            [[1.0], [np.inf]],
            [[1.0], [complex(2.0, 1.0)]],
            [[1.0], ["not numeric"]],
        )
        for X in invalid_features:
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    best_split(X, [0, 1])

    def test_rejects_invalid_labels_and_sample_count_mismatch(self):
        X = [[0.0], [1.0]]
        invalid_labels = (
            [0],
            [[0], [1]],
            [0, np.nan],
            [0, None],
            np.array([1, "one"], dtype=object),
        )
        for y in invalid_labels:
            with self.subTest(y=y):
                with self.assertRaises(ValueError):
                    best_split(X, y)

    def test_matches_exhaustive_reference_for_multiclass_data(self):
        rng = np.random.default_rng(2026)
        for case in range(40):
            n_samples = int(rng.integers(2, 14))
            n_features = int(rng.integers(1, 4))
            n_classes = int(rng.integers(2, 5))
            X = rng.integers(-3, 4, size=(n_samples, n_features)).astype(float)
            y = rng.integers(0, n_classes, size=n_samples)

            with self.subTest(case=case):
                expected = exhaustive_best_split(X, y)
                actual = best_split(X, y)
                if expected is None:
                    self.assertIsNone(actual)
                else:
                    self.assertIsNotNone(actual)
                    self.assertEqual(actual[0], expected[0])
                    self.assertEqual(actual[1], expected[1])
                    self.assertAlmostEqual(actual[2], expected[2], places=14)


if __name__ == "__main__":
    unittest.main()
