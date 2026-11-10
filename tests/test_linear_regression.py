import unittest

import numpy as np

from ml_from_scratch import LinearRegression


class TestLinearRegression(unittest.TestCase):
    def test_learns_simple_relationship(self):
        X = np.array([[0], [1], [2], [3], [4]], dtype=float)
        y = 2 * X.reshape(-1) + 1
        model = LinearRegression(learning_rate=0.05, n_iterations=2_000).fit(X, y)
        np.testing.assert_allclose(model.predict([[5]])[0], 11.0, atol=0.05)

    def test_training_loss_decreases(self):
        X = np.array([[0], [1], [2], [3]], dtype=float)
        y = 2 * X.reshape(-1) + 1

        model = LinearRegression(learning_rate=0.05, n_iterations=100)
        model.fit(X, y)

        self.assertLess(model.cost_history[-1], model.cost_history[0])

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            LinearRegression().predict([[1]])

    def test_mismatched_samples_raise(self):
        with self.assertRaises(ValueError):
            LinearRegression().fit([[1], [2]], [1])

    def test_non_finite_training_values_raise(self):
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    LinearRegression().fit([[1.0], [invalid]], [1.0, 2.0])
                with self.assertRaises(ValueError):
                    LinearRegression().fit([[1.0], [2.0]], [1.0, invalid])

    def test_non_finite_prediction_values_raise(self):
        model = LinearRegression().fit([[0.0], [1.0]], [0.0, 1.0])
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    model.predict([[invalid]])


if __name__ == "__main__":
    unittest.main()
