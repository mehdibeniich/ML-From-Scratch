import unittest

import numpy as np

from ml_from_scratch import sigmoid


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


if __name__ == "__main__":
    unittest.main()
