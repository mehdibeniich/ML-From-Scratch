import unittest

import numpy as np

from ml_from_scratch.classification_metrics import (
    f1_score,
    precision_score,
    recall_score,
)


class TestClassificationScores(unittest.TestCase):
    def test_binary_precision_recall_and_f1(self):
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 1, 0]
        self.assertEqual(precision_score(y_true, y_pred), 0.5)
        self.assertEqual(recall_score(y_true, y_pred), 0.5)
        self.assertEqual(f1_score(y_true, y_pred), 0.5)

    def test_binary_supports_string_labels_and_custom_positive_label(self):
        y_true = ["yes", "yes", "no", "no"]
        y_pred = ["yes", "no", "yes", "no"]
        self.assertEqual(precision_score(y_true, y_pred, pos_label="yes"), 0.5)
        self.assertEqual(recall_score(y_true, y_pred, pos_label="yes"), 0.5)
        self.assertEqual(f1_score(y_true, y_pred, pos_label="yes"), 0.5)

    def test_micro_scores_equal_accuracy_for_single_label_multiclass(self):
        y_true = ["a", "a", "b", "b", "c", "c"]
        y_pred = ["a", "b", "b", "c", "c", "c"]
        for score in (precision_score, recall_score, f1_score):
            self.assertAlmostEqual(score(y_true, y_pred, average="micro"), 4 / 6)

    def test_macro_and_weighted_scores(self):
        y_true = ["a", "a", "b", "b", "c", "c"]
        y_pred = ["a", "b", "b", "c", "c", "c"]
        expected_precision = (1.0 + 0.5 + 2 / 3) / 3
        expected_recall = (0.5 + 0.5 + 1.0) / 3
        expected_f1 = (2 / 3 + 0.5 + 0.8) / 3
        self.assertAlmostEqual(precision_score(y_true, y_pred, average="macro"), expected_precision)
        self.assertAlmostEqual(recall_score(y_true, y_pred, average="macro"), expected_recall)
        self.assertAlmostEqual(f1_score(y_true, y_pred, average="macro"), expected_f1)
        self.assertAlmostEqual(precision_score(y_true, y_pred, average="weighted"), expected_precision)

    def test_per_class_output_respects_explicit_label_order(self):
        scores = precision_score(
            ["a", "b", "c"], ["a", "a", "c"], average=None, labels=["c", "b", "a"]
        )
        np.testing.assert_allclose(scores, [1.0, 0.0, 0.5])

    def test_zero_division_is_configurable(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 0, 0]
        self.assertEqual(precision_score(y_true, y_pred, zero_division=0), 0.0)
        self.assertEqual(precision_score(y_true, y_pred, zero_division=1), 1.0)
        self.assertEqual(recall_score(y_true, y_pred, zero_division=1), 0.0)
        self.assertEqual(f1_score(y_true, y_pred, zero_division=1), 0.0)

    def test_invalid_average_and_zero_division_raise(self):
        with self.assertRaises(ValueError):
            f1_score([0, 1], [0, 1], average="samples")
        with self.assertRaises(ValueError):
            precision_score([0, 1], [0, 1], zero_division="warn")

    def test_binary_requires_two_observed_classes_and_valid_positive_label(self):
        with self.assertRaisesRegex(ValueError, "exactly two observed"):
            precision_score([0, 0], [0, 0])
        with self.assertRaisesRegex(ValueError, "pos_label"):
            recall_score([0, 1], [0, 1], pos_label=2)
        with self.assertRaisesRegex(ValueError, "exactly two observed"):
            f1_score([0, 1, 2], [0, 1, 2])

    def test_validation_rejects_empty_mismatched_and_non_1d_inputs(self):
        invalid = (([], []), ([0, 1], [0]), ([[0], [1]], [0, 1]))
        for y_true, y_pred in invalid:
            with self.subTest(y_true=y_true, y_pred=y_pred):
                with self.assertRaises(ValueError):
                    precision_score(y_true, y_pred, average="macro")


if __name__ == "__main__":
    unittest.main()
