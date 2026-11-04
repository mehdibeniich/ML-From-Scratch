import unittest

import numpy as np

from ml_from_scratch import LinearRegression


class TestLinearRegression(unittest.TestCase):
    def test_learns_simple_relationship(self):
        X = np.array([[0], [1], [2], [3], [4]], dtype=float)
        y = 2 * X.reshape(-1) + 1
        model = LinearRegression(learning_rate=0.05, n_iterations=2_000).fit(X, y)
        np.testing.assert_allclose(model.predict([[5]])[0], 11.0, atol=0.05)

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            LinearRegression().predict([[1]])

    def test_mismatched_samples_raise(self):
        with self.assertRaises(ValueError):
            LinearRegression().fit([[1], [2]], [1])


if __name__ == "__main__":
    unittest.main()
