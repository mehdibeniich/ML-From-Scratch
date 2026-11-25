import unittest

import numpy as np

from ml_from_scratch import RandomForestClassifier


class TestRandomForestClassifier(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [0.0, 10.0], [1.0, 10.0], [2.0, 10.0], [3.0, 10.0],
            [4.0, 10.0], [5.0, 10.0], [6.0, 10.0], [7.0, 10.0],
        ])
        self.y = np.array(["low", "low", "low", "low", "high", "high", "high", "high"])

    def test_fit_predict_and_probability_shapes(self):
        model = RandomForestClassifier(
            n_estimators=12, max_features=None, random_state=13
        )
        self.assertIs(model.fit(self.X, self.y), model)

        probabilities = model.predict_proba(self.X)
        predictions = model.predict(self.X)

        self.assertEqual(probabilities.shape, (len(self.y), 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))
        self.assertEqual(predictions.shape, (len(self.y),))
        self.assertTrue(set(predictions).issubset(set(model.classes_)))

    def test_prediction_accepts_a_single_feature_vector(self):
        model = RandomForestClassifier(
            n_estimators=5, max_features=None, bootstrap=False, random_state=2
        ).fit(self.X, self.y)

        self.assertEqual(model.predict([1.5, 10.0]).shape, (1,))
        self.assertEqual(model.predict_proba([1.5, 10.0]).shape, (1, 2))

    def test_seed_reproduces_bootstrap_samples_and_predictions(self):
        first = RandomForestClassifier(n_estimators=10, random_state=29).fit(
            self.X, self.y
        )
        second = RandomForestClassifier(n_estimators=10, random_state=29).fit(
            self.X, self.y
        )

        for first_sample, second_sample in zip(
            first.estimators_samples_, second.estimators_samples_
        ):
            np.testing.assert_array_equal(first_sample, second_sample)
        np.testing.assert_array_equal(first.predict(self.X), second.predict(self.X))
        np.testing.assert_allclose(
            first.predict_proba(self.X), second.predict_proba(self.X)
        )

    def test_bootstrap_samples_have_replacement(self):
        model = RandomForestClassifier(
            n_estimators=12, bootstrap=True, random_state=4
        ).fit(self.X, self.y)

        self.assertEqual(len(model.estimators_samples_), 12)
        for sample_indices in model.estimators_samples_:
            self.assertEqual(sample_indices.shape, (len(self.y),))
            self.assertTrue(np.all((sample_indices >= 0) & (sample_indices < len(self.y))))
        self.assertTrue(
            any(np.unique(indices).size < len(self.y)
                for indices in model.estimators_samples_)
        )

    def test_disabling_bootstrap_uses_every_training_sample(self):
        model = RandomForestClassifier(
            n_estimators=3, bootstrap=False, max_features=None, random_state=4
        ).fit(self.X, self.y)
        expected = np.arange(len(self.y))

        for sample_indices in model.estimators_samples_:
            np.testing.assert_array_equal(sample_indices, expected)

    def test_probability_columns_align_when_a_tree_omits_a_rare_class(self):
        X = np.arange(21, dtype=float).reshape(-1, 1)
        y = np.array(["common"] * 20 + ["rare"])
        model = RandomForestClassifier(n_estimators=40, random_state=0).fit(X, y)

        self.assertTrue(any(estimator.classes_.size == 1 for estimator in model.estimators_))
        self.assertEqual(model.classes_.tolist(), ["common", "rare"])
        probabilities = model.predict_proba([[0.0], [20.0]])
        self.assertEqual(probabilities.shape, (2, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_feature_importances_are_normalized(self):
        model = RandomForestClassifier(
            n_estimators=8, max_features=None, bootstrap=False, random_state=5
        ).fit(self.X, self.y)

        self.assertEqual(model.feature_importances_.shape, (2,))
        np.testing.assert_allclose(model.feature_importances_, [1.0, 0.0])
        self.assertAlmostEqual(float(model.feature_importances_.sum()), 1.0)

    def test_supports_training_data_with_one_class(self):
        model = RandomForestClassifier(n_estimators=4, random_state=3).fit(
            [[0.0], [1.0], [2.0]], ["only", "only", "only"]
        )

        np.testing.assert_array_equal(model.predict([[0.5], [1.5]]), ["only", "only"])
        np.testing.assert_array_equal(model.predict_proba([[0.5], [1.5]]), [[1.0], [1.0]])
        np.testing.assert_array_equal(model.feature_importances_, [0.0])

    def test_prediction_before_fit_raises(self):
        model = RandomForestClassifier(n_estimators=2)
        with self.assertRaises(RuntimeError):
            model.predict([[0.0]])
        with self.assertRaises(RuntimeError):
            model.predict_proba([[0.0]])

    def test_rejects_invalid_hyperparameters(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(n_estimators=value):
                with self.assertRaises(ValueError):
                    RandomForestClassifier(n_estimators=value)

        for kwargs in (
            {"bootstrap": 1},
            {"max_depth": -1},
            {"min_samples_split": 1},
            {"max_features": "invalid"},
            {"random_state": -1},
            {"random_state": True},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    RandomForestClassifier(**kwargs)

    def test_rejects_max_features_larger_than_input_dimension(self):
        model = RandomForestClassifier(n_estimators=2, max_features=3)
        with self.assertRaises(ValueError):
            model.fit([[0.0, 1.0], [1.0, 2.0]], [0, 1])

    def test_rejects_invalid_training_and_prediction_data(self):
        invalid_training = (
            ([], []),
            ([[]], [0]),
            ([[0.0], [np.nan]], [0, 1]),
            ([[0.0], [1.0]], [0]),
            ([[0.0], [1.0]], [[0], [1]]),
        )
        for X, y in invalid_training:
            with self.subTest(X=X, y=y):
                with self.assertRaises(ValueError):
                    RandomForestClassifier(n_estimators=2).fit(X, y)

        model = RandomForestClassifier(n_estimators=3).fit(self.X, self.y)
        for invalid_X in ([[0.0]], [[np.inf, 1.0]], np.empty((0, 2))):
            with self.subTest(X=invalid_X):
                with self.assertRaises(ValueError):
                    model.predict(invalid_X)

        with self.assertRaises(ValueError):
            RandomForestClassifier(n_estimators=2).fit(
                np.array([[1.0 + 2.0j], [2.0 + 1.0j]]), [0, 1]
            )

    def test_failed_refit_preserves_previous_forest(self):
        model = RandomForestClassifier(n_estimators=5, random_state=8).fit(
            self.X, self.y
        )
        old_estimators = model.estimators_
        old_predictions = model.predict(self.X)
        old_importances = model.feature_importances_.copy()

        with self.assertRaises(ValueError):
            model.fit([[0.0], [1.0]], [0])

        self.assertIs(model.estimators_, old_estimators)
        np.testing.assert_array_equal(model.predict(self.X), old_predictions)
        np.testing.assert_array_equal(model.feature_importances_, old_importances)


if __name__ == "__main__":
    unittest.main()
