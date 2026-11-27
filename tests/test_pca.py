import unittest

import numpy as np

from ml_from_scratch import PCA


class TestPCA(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]])

    def test_rank_one_projection_and_inverse_reconstruction(self):
        X = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        model = PCA(1).fit(X)
        self.assertEqual(model.components_.shape, (1, 2))
        self.assertAlmostEqual(model.explained_variance_ratio_[0], 1.0)
        reconstructed = model.inverse_transform(model.transform(X))
        np.testing.assert_allclose(reconstructed, X, atol=1e-10)

    def test_fractional_components_reach_variance_target(self):
        model = PCA(0.9).fit(self.X)
        self.assertGreaterEqual(model.explained_variance_ratio_.sum(), 0.9)
        self.assertEqual(model.n_components_, 2)

    def test_none_keeps_all_available_components(self):
        model = PCA().fit(self.X)
        self.assertEqual(model.n_components_, 2)
        self.assertAlmostEqual(model.explained_variance_ratio_.sum(), 1.0)

    def test_fit_transform_matches_transform_after_fit(self):
        model = PCA(1)
        projected = model.fit_transform(self.X)
        np.testing.assert_allclose(projected, model.transform(self.X))
        self.assertEqual(projected.shape, (4, 1))

    def test_single_sample_and_constant_data_are_supported(self):
        model = PCA(0.95).fit([[3.0, 7.0]])
        self.assertEqual(model.n_components_, 1)
        self.assertTrue(np.all(model.explained_variance_ratio_ == 0.0))
        np.testing.assert_allclose(model.transform([[3.0, 7.0]]), [[0.0]])

    def test_validation_and_unfitted_methods(self):
        with self.assertRaises(RuntimeError):
            PCA(1).transform(self.X)
        with self.assertRaises(RuntimeError):
            PCA(1).inverse_transform([[0.0]])
        for bad in (0, -1, 3, 1.5, True, 0.0, -0.2, 1.1, np.nan):
            with self.subTest(n_components=bad):
                with self.assertRaises(ValueError):
                    PCA(bad).fit(self.X)
        for bad_X in ([1.0, 2.0], [], [[np.nan, 1.0]], [[np.inf, 0.0]]):
            with self.subTest(X=bad_X):
                with self.assertRaises(ValueError):
                    PCA(1).fit(bad_X)

    def test_transform_and_inverse_reject_wrong_dimensions(self):
        model = PCA(1).fit(self.X)
        with self.assertRaises(ValueError):
            model.transform([[1.0]])
        with self.assertRaises(ValueError):
            model.inverse_transform([[1.0, 2.0]])

    def test_failed_refit_preserves_previous_model(self):
        model = PCA(1).fit(self.X)
        mean = model.mean_.copy()
        components = model.components_.copy()
        with self.assertRaises(ValueError):
            model.fit([[1.0, np.nan]])
        np.testing.assert_array_equal(model.mean_, mean)
        np.testing.assert_array_equal(model.components_, components)


if __name__ == "__main__":
    unittest.main()