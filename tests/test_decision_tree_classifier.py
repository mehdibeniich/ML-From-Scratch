import unittest

import numpy as np

from ml_from_scratch import DecisionTreeClassifier


class TestDecisionTreeClassifier(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
        self.y = np.array(["low", "low", "middle", "middle", "high", "high"])

    def test_fit_returns_self_and_predicts_multiclass_training_data(self):
        model = DecisionTreeClassifier().fit(self.X, self.y)
        self.assertIs(model.fit(self.X, self.y), model)
        np.testing.assert_array_equal(model.predict(self.X), self.y)
        self.assertEqual(model.classes_.tolist(), ["high", "low", "middle"])
        self.assertEqual(model.n_features_in_, 1)

    def test_prediction_accepts_a_single_sample(self):
        model = DecisionTreeClassifier().fit(self.X, self.y)
        np.testing.assert_array_equal(model.predict([0.5]), ["low"])

    def test_max_depth_zero_returns_majority_class_leaf(self):
        model = DecisionTreeClassifier(max_depth=0).fit(
            [[0.0], [1.0], [2.0], [3.0]], ["z", "a", "z", "a"]
        )
        np.testing.assert_array_equal(model.predict([[0.0], [3.0]]), ["a", "a"])

    def test_max_depth_limits_tree_growth(self):
        model = DecisionTreeClassifier(max_depth=1).fit(self.X, self.y)
        self.assertIn("feature_index", model.tree_)
        self.assertNotIn("feature_index", model.tree_["left"])
        self.assertNotIn("feature_index", model.tree_["right"])

    def test_min_samples_split_stops_small_nodes(self):
        model = DecisionTreeClassifier(min_samples_split=7).fit(self.X, self.y)
        self.assertNotIn("feature_index", model.tree_)

    def test_prediction_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            DecisionTreeClassifier().predict([[0.0]])

    def test_rejects_invalid_hyperparameters(self):
        for max_depth in (-1, 1.5, True, np.bool_(False)):
            with self.subTest(max_depth=max_depth):
                with self.assertRaises(ValueError):
                    DecisionTreeClassifier(max_depth=max_depth)
        for min_samples_split in (0, 1, 2.5, True, np.bool_(False)):
            with self.subTest(min_samples_split=min_samples_split):
                with self.assertRaises(ValueError):
                    DecisionTreeClassifier(min_samples_split=min_samples_split)

    def test_rejects_invalid_training_data(self):
        invalid_cases = (
            ([], []),
            ([[]], [0]),
            ([[1.0], [np.nan]], [0, 1]),
            ([[1.0], [np.inf]], [0, 1]),
            ([[1.0], [2.0]], [0]),
            ([[1.0], [2.0]], [[0], [1]]),
            ([[1.0], [2.0]], [0, np.nan]),
            ([[1.0], [2.0]], [0, None]),
        )
        for X, y in invalid_cases:
            with self.subTest(X=X, y=y):
                with self.assertRaises(ValueError):
                    DecisionTreeClassifier().fit(X, y)

    def test_rejects_invalid_prediction_features(self):
        model = DecisionTreeClassifier().fit(self.X, self.y)
        invalid_inputs = (
            [[1.0, 2.0]],
            np.empty((0, 1)),
            [[np.nan]],
            [[np.inf]],
        )
        for X in invalid_inputs:
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    model.predict(X)

    def test_failed_refit_preserves_previous_tree(self):
        model = DecisionTreeClassifier().fit(self.X, self.y)
        predictions_before = model.predict(self.X)
        tree_before = model.tree_
        with self.assertRaises(ValueError):
            model.fit([[0.0], [1.0]], [0])
        self.assertIs(model.tree_, tree_before)
        np.testing.assert_array_equal(model.predict(self.X), predictions_before)


if __name__ == "__main__":
    unittest.main()
