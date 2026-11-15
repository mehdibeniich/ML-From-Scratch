import unittest

import numpy as np

from ml_from_scratch.decision_tree import gini_impurity


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


if __name__ == "__main__":
    unittest.main()
