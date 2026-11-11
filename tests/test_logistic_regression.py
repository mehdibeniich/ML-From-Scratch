import unittest

import numpy as np

from ml_from_scratch import binary_cross_entropy_with_logits, sigmoid


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


if __name__ == "__main__":
    unittest.main()
