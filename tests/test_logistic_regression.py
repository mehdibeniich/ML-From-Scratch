import unittest

import numpy as np

from ml_from_scratch import (
    LogisticRegression,
    binary_cross_entropy_with_logits,
    sigmoid,
)


class TestSigmoid(unittest.TestCase):
    def test_sigmoid_at_zero(self):
        self.assertEqual(sigmoid(0), 0.5)

    def test_sigmoid_for_positive_and_negative_values(self):
        values = np.array([-2.0, 0.0, 2.0])
        result = sigmoid(values)
        expected = 1.0 / (1.0 + np.exp(-values))
        np.testing.assert_allclose(result, expected)

    def test_sigmoid_is_stable_for_extreme_values(self):
        result = sigmoid(np.array([-1000.0, 1000.0]))
        np.testing.assert_array_equal(result, np.array([0.0, 1.0]))

    def test_sigmoid_preserves_array_shape_and_bounds(self):
        values = np.array([[-3.0, 1.0], [0.0, 4.0]])
        result = sigmoid(values)
        self.assertEqual(result.shape, values.shape)
        self.assertTrue(np.all((result >= 0.0) & (result <= 1.0)))

    def test_sigmoid_returns_float_for_scalar_input(self):
        self.assertIsInstance(sigmoid(1.0), float)


class TestBinaryCrossEntropyWithLogits(unittest.TestCase):
    def test_matches_known_values(self):
        targets = np.array([0.0, 1.0, 1.0, 0.0])
        logits = np.array([0.0, 0.0, 2.0, -2.0])
        expected = np.log(2.0) + np.logaddexp(0.0, -2.0)
        self.assertAlmostEqual(
            binary_cross_entropy_with_logits(targets, logits), expected / 2.0
        )

    def test_is_stable_for_extreme_logits(self):
        targets = np.array([0.0, 1.0, 0.0, 1.0])
        logits = np.array([-1000.0, 1000.0, 1000.0, -1000.0])
        result = binary_cross_entropy_with_logits(targets, logits)
        self.assertTrue(np.isfinite(result))
        self.assertAlmostEqual(result, 500.0)

    def test_reduction_modes(self):
        targets = np.array([0.0, 1.0])
        logits = np.array([0.0, 0.0])
        losses = binary_cross_entropy_with_logits(targets, logits, reduction="none")
        np.testing.assert_allclose(losses, [np.log(2.0), np.log(2.0)])
        self.assertAlmostEqual(
            binary_cross_entropy_with_logits(targets, logits, reduction="sum"),
            2.0 * np.log(2.0),
        )

    def test_scalar_inputs_return_float(self):
        result = binary_cross_entropy_with_logits(1, 0.5)
        self.assertIsInstance(result, float)

    def test_supports_soft_targets(self):
        targets = np.array([0.25, 0.75])
        logits = np.array([-1.0, 2.0])
        expected = np.logaddexp(0.0, logits) - targets * logits
        losses = binary_cross_entropy_with_logits(targets, logits, reduction="none")
        np.testing.assert_allclose(losses, expected)
        self.assertAlmostEqual(
            binary_cross_entropy_with_logits(targets, logits), float(np.mean(expected))
        )

    def test_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            binary_cross_entropy_with_logits([0, 1], [0.2])

    def test_rejects_empty_inputs(self):
        with self.assertRaises(ValueError):
            binary_cross_entropy_with_logits([], [])
        with self.assertRaises(ValueError):
            binary_cross_entropy_with_logits([1], [])

    def test_rejects_non_finite_inputs(self):
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    binary_cross_entropy_with_logits([1.0], [invalid])
                with self.assertRaises(ValueError):
                    binary_cross_entropy_with_logits([invalid], [0.0])

    def test_rejects_invalid_targets_and_reduction(self):
        with self.assertRaises(ValueError):
            binary_cross_entropy_with_logits([2], [0.2])
        with self.assertRaises(ValueError):
            binary_cross_entropy_with_logits([1], [0.2], reduction="median")


class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        self.y = np.array([0, 0, 1, 1])

    def test_learns_linearly_separable_data(self):
        model = LogisticRegression(learning_rate=0.1, n_iterations=1_000)
        self.assertIs(model.fit(self.X, self.y), model)
        np.testing.assert_array_equal(model.predict(self.X), self.y)
        self.assertLess(model.cost_history[-1], model.cost_history[0])

    def test_failed_refit_preserves_previous_model(self):
        model = LogisticRegression(learning_rate=0.1, n_iterations=1_000).fit(
            self.X, self.y
        )
        weights_before = model.weights.copy()
        bias_before = model.bias
        cost_history_before = model.cost_history.copy()

        with self.assertRaises(FloatingPointError):
            model.fit([[-1e200], [1e200]], [0, 1])

        np.testing.assert_array_equal(model.weights, weights_before)
        self.assertEqual(model.bias, bias_before)
        self.assertEqual(model.cost_history, cost_history_before)
        np.testing.assert_array_equal(model.predict(self.X), self.y)

    def test_predict_proba_returns_two_normalized_columns(self):
        model = LogisticRegression(learning_rate=0.1, n_iterations=1_000).fit(
            self.X, self.y
        )
        probabilities = model.predict_proba(self.X)
        self.assertEqual(probabilities.shape, (4, 2))
        np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0)
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))

    def test_predict_accepts_a_single_feature_vector(self):
        model = LogisticRegression(learning_rate=0.1, n_iterations=1_000).fit(
            self.X, self.y
        )
        np.testing.assert_array_equal(model.predict(np.array([-3.0])), [0])

    def test_prediction_before_fit_raises(self):
        model = LogisticRegression()
        with self.assertRaises(RuntimeError):
            model.predict(self.X)
        with self.assertRaises(RuntimeError):
            model.predict_proba(self.X)

    def test_rejects_invalid_training_data(self):
        model = LogisticRegression()
        with self.assertRaises(ValueError):
            model.fit([1.0, 2.0], self.y)
        with self.assertRaises(ValueError):
            model.fit(self.X, [0, 1])
        with self.assertRaises(ValueError):
            model.fit(np.empty((0, 1)), [])
        with self.assertRaises(ValueError):
            model.fit(np.empty((4, 0)), self.y)

    def test_rejects_non_finite_training_values(self):
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                X = self.X.copy()
                X[0, 0] = invalid
                with self.assertRaises(ValueError):
                    LogisticRegression().fit(X, self.y)
                y = self.y.astype(float)
                y[0] = invalid
                with self.assertRaises(ValueError):
                    LogisticRegression().fit(self.X, y)

    def test_rejects_non_binary_or_single_class_targets(self):
        with self.assertRaises(ValueError):
            LogisticRegression().fit(self.X, [0, 0.5, 1, 1])
        with self.assertRaises(ValueError):
            LogisticRegression().fit(self.X, [1, 1, 1, 1])

    def test_rejects_invalid_prediction_features(self):
        model = LogisticRegression(learning_rate=0.1, n_iterations=100).fit(
            self.X, self.y
        )
        with self.assertRaises(ValueError):
            model.predict([[1.0, 2.0]])
        with self.assertRaises(ValueError):
            model.predict([[np.nan]])
        with self.assertRaises(ValueError):
            model.predict(np.empty((0, 1)))

    def test_validates_hyperparameters(self):
        for invalid in (0.0, -0.1, np.nan, np.inf, -np.inf):
            with self.subTest(learning_rate=invalid):
                with self.assertRaises(ValueError):
                    LogisticRegression(learning_rate=invalid)
        for invalid in (0, -1, 1.5, True):
            with self.subTest(n_iterations=invalid):
                with self.assertRaises(ValueError):
                    LogisticRegression(n_iterations=invalid)


if __name__ == "__main__":
    unittest.main()
