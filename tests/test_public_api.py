"""Regression tests for the package-level public API."""

import unittest

import ml_from_scratch


class TestPublicAPI(unittest.TestCase):
    def test_all_supported_top_level_names_remain_exported(self):
        expected_names = {
            "LinearRegression",
            "LogisticRegression",
            "KNeighborsClassifier",
            "DecisionTreeClassifier",
            "StandardScaler",
            "sigmoid",
            "binary_cross_entropy_with_logits",
            "accuracy_score",
            "confusion_matrix",
            "mean_absolute_error",
            "mean_squared_error",
            "r2_score",
            "gini_impurity",
            "best_split",
        }

        self.assertEqual(set(ml_from_scratch.__all__), expected_names)
        self.assertEqual(len(ml_from_scratch.__all__), len(expected_names))

        for name in expected_names:
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(ml_from_scratch, name, None)))


if __name__ == "__main__":
    unittest.main()
