import unittest

import numpy as np

from ml_from_scratch import KMeans


class TestKMeans(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [-5.0, -4.0], [-4.0, -5.0], [-6.0, -5.0],
            [5.0, 4.0], [4.0, 5.0], [6.0, 5.0],
        ])

    def test_finds_separated_clusters_and_low_inertia(self):
        model = KMeans(n_clusters=2, n_init=5, random_state=7).fit(self.X)
        self.assertEqual(model.cluster_centers_.shape, (2, 2))
        self.assertEqual(model.labels_.shape, (6,))
        self.assertLess(model.inertia_, 5.0)
        self.assertEqual(model.predict([[-5.0, -5.0], [5.0, 5.0]]).shape, (2,))
        self.assertEqual(model.n_features_in_, 2)

    def test_seed_is_reproducible_and_fit_predict_matches(self):
        a = KMeans(2, n_init=4, random_state=42).fit(self.X)
        b = KMeans(2, n_init=4, random_state=42).fit(self.X)
        np.testing.assert_allclose(a.cluster_centers_, b.cluster_centers_)
        np.testing.assert_array_equal(a.labels_, b.labels_)
        np.testing.assert_array_equal(
            KMeans(2, n_init=4, random_state=42).fit_predict(self.X), a.labels_)

    def test_transform_returns_euclidean_distances(self):
        model = KMeans(2, random_state=0).fit(self.X)
        distances = model.transform([[0.0, 0.0]])
        self.assertEqual(distances.shape, (1, 2))
        self.assertTrue(np.all(distances >= 0.0))

    def test_duplicate_points_and_empty_cluster_repair(self):
        X = np.array([[0.0], [0.0], [0.0], [1.0], [1.0]])
        model = KMeans(3, n_init=3, random_state=1).fit(X)
        self.assertTrue(np.isfinite(model.inertia_))
        self.assertEqual(model.cluster_centers_.shape, (3, 1))
        self.assertTrue(np.all(np.isfinite(model.labels_)))

    def test_validation_and_unfitted_methods(self):
        with self.assertRaises(RuntimeError):
            KMeans(2).predict(self.X)
        for bad in (0, -1, 7, 1.5, True):
            with self.subTest(n_clusters=bad):
                with self.assertRaises(ValueError):
                    KMeans(bad).fit(self.X)
        for bad_X in ([1, 2], np.empty((0, 2)), [[np.nan, 0.0]], [[np.inf, 1.0]]):
            with self.subTest(X=bad_X):
                with self.assertRaises(ValueError):
                    KMeans(2).fit(bad_X)

    def test_failed_refit_preserves_previous_model(self):
        model = KMeans(2, random_state=3).fit(self.X)
        centers = model.cluster_centers_.copy()
        with self.assertRaises(ValueError):
            model.fit([[1.0, np.nan]])
        np.testing.assert_array_equal(model.cluster_centers_, centers)


if __name__ == "__main__":
    unittest.main()
