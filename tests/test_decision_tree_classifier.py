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

    def test_failed_refit_preserves_previous_tree_and_importances(self):
        model = DecisionTreeClassifier().fit(self.X, self.y)
        predictions_before = model.predict(self.X)
        tree_before = model.tree_
        importances_before = model.feature_importances_.copy()
        with self.assertRaises(ValueError):
            model.fit([[0.0], [1.0]], [0])
        self.assertIs(model.tree_, tree_before)
        np.testing.assert_array_equal(model.feature_importances_, importances_before)
        np.testing.assert_array_equal(model.predict(self.X), predictions_before)


    def test_predict_proba_returns_leaf_class_frequencies(self):
        X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
        y = ["A", "A", "B", "B", "A", "B"]
        model = DecisionTreeClassifier(max_depth=1).fit(X, y)

        probabilities = model.predict_proba([[0.5], [4.5]])

        self.assertEqual(model.classes_.tolist(), ["A", "B"])
        np.testing.assert_allclose(
            probabilities,
            [[1.0, 0.0], [0.25, 0.75]],
        )
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        np.testing.assert_array_equal(model.predict([[0.5], [4.5]]), ["A", "B"])

    def test_predict_proba_uses_global_class_order_and_root_leaf_prior(self):
        model = DecisionTreeClassifier(max_depth=0).fit(
            [[0.0], [1.0], [2.0], [3.0]], ["z", "a", "z", "a"]
        )

        self.assertEqual(model.classes_.tolist(), ["a", "z"])
        np.testing.assert_allclose(
            model.predict_proba([[0.0], [3.0]]),
            [[0.5, 0.5], [0.5, 0.5]],
        )
        np.testing.assert_array_equal(model.predict([[0.0]]), ["a"])

    def test_predict_proba_returns_one_hot_for_pure_leaves(self):
        model = DecisionTreeClassifier().fit(self.X, self.y)
        probabilities = model.predict_proba(self.X)

        self.assertEqual(probabilities.shape, (self.X.shape[0], 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        np.testing.assert_array_equal(
            probabilities,
            np.eye(3)[[1, 1, 2, 2, 0, 0]],
        )

    def test_predict_proba_rejects_unfitted_and_invalid_inputs(self):
        with self.assertRaises(RuntimeError):
            DecisionTreeClassifier().predict_proba([[0.0]])

        model = DecisionTreeClassifier().fit(self.X, self.y)
        for X in ([[1.0, 2.0]], np.empty((0, 1)), [[np.nan]], [[np.inf]]):
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    model.predict_proba(X)


    def test_feature_importances_identify_informative_feature(self):
        X = np.array([
            [0.0, 10.0], [1.0, 10.0], [2.0, 10.0],
            [3.0, 10.0], [4.0, 10.0], [5.0, 10.0],
        ])
        y = np.array([0, 0, 0, 1, 1, 1])
        model = DecisionTreeClassifier().fit(X, y)

        np.testing.assert_allclose(model.feature_importances_, [1.0, 0.0])
        self.assertAlmostEqual(float(model.feature_importances_.sum()), 1.0)

    def test_feature_importances_accumulate_weighted_decreases_across_tree(self):
        X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        y = [0, 1, 1, 1]
        model = DecisionTreeClassifier().fit(X, y)

        np.testing.assert_allclose(
            model.feature_importances_, [1.0 / 3.0, 2.0 / 3.0]
        )

    def test_feature_importances_are_zero_when_tree_has_no_split(self):
        model = DecisionTreeClassifier(max_depth=0).fit(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], [0, 1, 0]
        )

        np.testing.assert_array_equal(model.feature_importances_, [0.0, 0.0])


    def test_random_feature_selection_is_seeded_and_limits_candidates(self):
        X = np.array([
            [0.0, 10.0, 100.0], [1.0, 11.0, 101.0],
            [2.0, 12.0, 102.0], [3.0, 13.0, 103.0],
            [4.0, 14.0, 104.0], [5.0, 15.0, 105.0],
        ])
        y = np.array([0, 0, 0, 1, 1, 1])
        first = DecisionTreeClassifier(max_features=1, random_state=8).fit(X, y)
        second = DecisionTreeClassifier(max_features=1, random_state=8).fit(X, y)
        def flatten(node):
            if "prediction" in node:
                return (
                    "leaf",
                    node["prediction"],
                    tuple(node["probabilities"].tolist()),
                )
            return (
                "split",
                int(node["feature_index"]),
                float(node["threshold"]),
                flatten(node["left"]),
                flatten(node["right"]),
            )

        self.assertEqual(flatten(first.tree_), flatten(second.tree_))
        self.assertEqual(first.feature_importances_.shape, (3,))

    def test_rejects_invalid_random_feature_parameters(self):
        for max_features in (0, -1, 1.5, 1.1, True, "bad"):
            with self.subTest(max_features=max_features):
                with self.assertRaises(ValueError):
                    DecisionTreeClassifier(max_features=max_features)
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(max_features=4).fit([[0.0, 1.0]], [0])

    def test_best_split_can_restrict_candidate_features(self):
        from ml_from_scratch import best_split
        X = [[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0]]
        y = [0, 0, 1, 1]
        split = best_split(X, y, feature_indices=np.array([1]))
        self.assertEqual(split[0], 1)
        with self.assertRaises(ValueError):
            best_split(X, y, feature_indices=np.array([1, 1]))


if __name__ == "__main__":
    unittest.main()
