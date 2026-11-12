import unittest

import numpy as np

from ml_from_scratch import StandardScaler


class TestStandardScaler(unittest.TestCase):
    def setUp(self):
        self.X = np.array(
            [[1.0, 10.0, 5.0], [2.0, 20.0, 5.0], [3.0, 30.0, 5.0]]
        )

    def test_fit_learns_population_statistics(self):
        scaler = StandardScaler().fit(self.X)
        np.testing.assert_allclose(scaler.mean_, [2.0, 20.0, 5.0])
        np.testing.assert_allclose(
            scaler.scale_, [np.sqrt(2.0 / 3.0), np.sqrt(200.0 / 3.0), 1.0]
        )
        self.assertEqual(scaler.n_features_in_, 3)

    def test_transform_standardizes_features_and_constant_columns(self):
        transformed = StandardScaler().fit_transform(self.X)
        np.testing.assert_allclose(np.mean(transformed, axis=0), [0.0, 0.0, 0.0])
        np.testing.assert_allclose(np.std(transformed[:, :2], axis=0), [1.0, 1.0])
        np.testing.assert_array_equal(transformed[:, 2], [0.0, 0.0, 0.0])

    def test_fit_returns_self_and_fit_transform_matches_transform(self):
        scaler = StandardScaler()
        self.assertIs(scaler.fit(self.X), scaler)
        expected = scaler.transform(self.X)
        actual = StandardScaler().fit_transform(self.X)
        np.testing.assert_allclose(actual, expected)

    def test_transform_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            StandardScaler().transform(self.X)

    def test_fit_rejects_invalid_shapes_and_values(self):
        invalid_inputs = (
            [1.0, 2.0],
            np.empty((0, 2)),
            np.empty((2, 0)),
            [[1.0, np.nan]],
            [[1.0, np.inf]],
            [[1.0, -np.inf]],
        )
        for values in invalid_inputs:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    StandardScaler().fit(values)

    def test_transform_rejects_invalid_shapes_and_values(self):
        scaler = StandardScaler().fit(self.X)
        invalid_inputs = (
            [1.0, 2.0, 3.0],
            np.empty((0, 3)),
            [[1.0, 2.0]],
            [[1.0, np.nan, 3.0]],
        )
        for values in invalid_inputs:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    scaler.transform(values)

    def test_failed_refit_preserves_previous_statistics(self):
        scaler = StandardScaler().fit(self.X)
        mean_before = scaler.mean_.copy()
        scale_before = scaler.scale_.copy()

        with self.assertRaises(ValueError):
            scaler.fit([[1.0, np.nan, 5.0]])

        np.testing.assert_array_equal(scaler.mean_, mean_before)
        np.testing.assert_array_equal(scaler.scale_, scale_before)
        np.testing.assert_allclose(scaler.transform(self.X).mean(axis=0), 0.0)


if __name__ == "__main__":
    unittest.main()
