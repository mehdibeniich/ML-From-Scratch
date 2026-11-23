import unittest

import numpy as np

from ml_from_scratch import GaussianNB


class TestGaussianNB(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [-2.0, -1.0], [-1.5, -0.8], [-1.0, -1.2],
            [1.0, 1.2], [1.5, 0.8], [2.0, 1.0],
        ])
        self.y = np.array(["left", "left", "left", "right", "right", "right"])

    def test_learns_separated_gaussian_classes(self):
        model = GaussianNB().fit(self.X, self.y)
        np.testing.assert_array_equal(
            model.predict([[-1.8, -1.0], [1.8, 1.0]]), ["left", "right"]
        )

    def test_probabilities_are_normalized_and_follow_classes_order(self):
        model = GaussianNB().fit(self.X, self.y)
        probabilities = model.predict_proba([[-1.8, -1.0], [1.8, 1.0]])
        self.assertEqual(model.classes_.tolist(), ["left", "right"])
        self.assertEqual(probabilities.shape, (2, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))

    def test_estimates_class_priors_and_statistics(self):
        X = [[0.0], [2.0], [4.0], [10.0]]
        y = ["a", "a", "a", "b"]
        model = GaussianNB().fit(X, y)
        np.testing.assert_allclose(model.class_count_, [3.0, 1.0])
        np.testing.assert_allclose(model.class_prior_, [0.75, 0.25])
        np.testing.assert_allclose(model.theta_[:, 0], [2.0, 10.0])
        self.assertEqual(model.n_features_in_, 1)
        self.assertGreater(model.epsilon_, 0.0)

    def test_constant_features_receive_positive_variance_floor(self):
        model = GaussianNB(var_smoothing=0.0).fit(
            [[3.0, 7.0], [3.0, 7.0], [3.0, 7.0], [3.0, 7.0]],
            [0, 0, 1, 1],
        )
        self.assertTrue(np.all(model.var_ > 0))
        self.assertTrue(np.all(np.isfinite(model.predict_proba([[3.0, 7.0]]))))

    def test_supports_numeric_and_mixed_incomparable_labels(self):
        model = GaussianNB().fit(
            [[0.0], [0.1], [2.0], [2.1]], [1, "one", 2, "two"]
        )
        self.assertEqual(len(model.classes_), 4)
        self.assertEqual(model.predict([[0.0]]).shape, (1,))

    def test_predict_accepts_single_feature_vector(self):
        model = GaussianNB().fit(self.X, self.y)
        self.assertEqual(model.predict(np.array([-1.8, -1.0])).shape, (1,))

    def test_prediction_before_fit_raises(self):
        model = GaussianNB()
        with self.assertRaises(RuntimeError):
            model.predict(self.X)
        with self.assertRaises(RuntimeError):
            model.predict_proba(self.X)

    def test_rejects_invalid_smoothing(self):
        for value in (-1.0, np.nan, np.inf, -np.inf, True, "invalid"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    GaussianNB(var_smoothing=value)

    def test_rejects_invalid_training_shapes_and_class_count(self):
        cases = [
            ([[1.0], [2.0]], [[0], [1]]),
            ([[1.0], [2.0]], [0]),
            ([], []),
            ([[], []], [0, 1]),
            ([[1.0], [2.0]], [1, 1]),
        ]
        for X, y in cases:
            with self.subTest(X=X, y=y):
                with self.assertRaises(ValueError):
                    GaussianNB().fit(X, y)

    def test_rejects_non_finite_training_data(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(value=bad):
                X = [[0.0], [1.0], [2.0], [3.0]]
                X[0][0] = bad
                with self.assertRaises(ValueError):
                    GaussianNB().fit(X, [0, 0, 1, 1])

    def test_rejects_invalid_prediction_features(self):
        model = GaussianNB().fit(self.X, self.y)
        for X in ([[1.0, 2.0, 3.0]], [[np.nan, 0.0]], np.empty((0, 2))):
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    model.predict(X)

    def test_failed_refit_preserves_previous_model(self):
        model = GaussianNB().fit(self.X, self.y)
        before = model.predict(self.X)
        old_means = model.theta_.copy()
        with self.assertRaises(ValueError):
            model.fit([[0.0], [1.0]], [None, 1])
        np.testing.assert_array_equal(model.theta_, old_means)
        np.testing.assert_array_equal(model.predict(self.X), before)


if __name__ == "__main__":
    unittest.main()
