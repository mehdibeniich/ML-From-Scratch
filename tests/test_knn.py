import unittest

import numpy as np

from ml_from_scratch import KNeighborsClassifier


class TestKNeighborsClassifier(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
        self.y = np.array(["low", "low", "low", "high", "high", "high"])

    def test_predicts_nearest_class(self):
        model = KNeighborsClassifier(n_neighbors=3).fit(self.X, self.y)
        np.testing.assert_array_equal(
            model.predict([[0.2], [4.8]]), ["low", "high"]
        )

    def test_predict_proba_uses_sorted_class_columns_and_normalized_votes(self):
        model = KNeighborsClassifier(n_neighbors=3).fit(self.X, self.y)
        probabilities = model.predict_proba([[0.2], [4.8]])
        self.assertEqual(model.classes_.tolist(), ["high", "low"])
        np.testing.assert_allclose(probabilities, [[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_single_neighbor_matches_nearest_training_label(self):
        model = KNeighborsClassifier(n_neighbors=1).fit(self.X, self.y)
        np.testing.assert_array_equal(model.predict([2.2]), ["low"])

    def test_vote_ties_are_broken_by_closest_neighbor(self):
        model = KNeighborsClassifier(n_neighbors=2).fit(
            [[-1.0], [1.0]], ["first", "second"]
        )
        np.testing.assert_array_equal(model.predict([[0.0]]), ["first"])
        np.testing.assert_allclose(model.predict_proba([[0.0]]), [[0.5, 0.5]])

    def test_fit_returns_self_and_failed_refit_preserves_model(self):
        model = KNeighborsClassifier(n_neighbors=1)
        self.assertIs(model.fit(self.X, self.y), model)
        predictions_before = model.predict([[0.2], [4.8]])
        with self.assertRaises(ValueError):
            model.fit([[np.nan]], ["only"])
        np.testing.assert_array_equal(model.predict([[0.2], [4.8]]), predictions_before)

    def test_prediction_before_fit_raises(self):
        model = KNeighborsClassifier(n_neighbors=1)
        with self.assertRaises(RuntimeError):
            model.predict([[0.0]])
        with self.assertRaises(RuntimeError):
            model.predict_proba([[0.0]])

    def test_rejects_invalid_hyperparameters(self):
        for invalid in (0, -1, 1.5, True, np.bool_(False)):
            with self.subTest(n_neighbors=invalid):
                with self.assertRaises(ValueError):
                    KNeighborsClassifier(n_neighbors=invalid)

    def test_accepts_numpy_integer_neighbor_count(self):
        model = KNeighborsClassifier(n_neighbors=np.int64(1)).fit(self.X, self.y)
        self.assertEqual(model.n_neighbors, 1)

    def test_rejects_invalid_training_data(self):
        invalid_cases = (
            ([[1.0], [2.0]], [0]),
            ([], []),
            ([[]], [0]),
            ([[1.0], [np.nan]], [0, 1]),
            ([[1.0], [np.inf]], [0, 1]),
            ([[1.0], [2.0]], [0, np.nan]),
            ([[1.0], [2.0]], [0, np.inf]),
            ([[1.0], [2.0]], [0, None]),
        )
        for X, y in invalid_cases:
            with self.subTest(X=X, y=y):
                with self.assertRaises(ValueError):
                    KNeighborsClassifier(n_neighbors=1).fit(X, y)

    def test_rejects_more_neighbors_than_training_samples(self):
        with self.assertRaises(ValueError):
            KNeighborsClassifier(n_neighbors=3).fit([[0.0], [1.0]], [0, 1])

    def test_rejects_invalid_prediction_features(self):
        model = KNeighborsClassifier(n_neighbors=1).fit(self.X, self.y)
        invalid_inputs = (
            [[1.0, 2.0]],
            np.empty((0, 1)),
            [[np.nan]],
            [[np.inf]],
        )
        for X in invalid_inputs:
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    model.predict(X)

    def test_distance_overflow_raises_clear_error(self):
        model = KNeighborsClassifier(n_neighbors=1).fit([[1e200]], [1])
        with self.assertRaises(FloatingPointError):
            model.predict([[-1e200]])


if __name__ == "__main__":
    unittest.main()
