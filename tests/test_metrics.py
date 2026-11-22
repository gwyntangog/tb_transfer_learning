"""AUPRC, Youden, and the two false-positive rates."""

import unittest

import numpy as np
from sklearn.metrics import roc_curve

from tb_transfer.metrics import (
    auprc_without_sick,
    average_precision,
    false_positive_rate,
    val_report,
    youden_threshold,
)


class MetricTests(unittest.TestCase):
    def test_perfect_separation_uses_the_lowest_positive_score(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(youden_threshold(labels, scores), 0.8)

    def test_youden_matches_highest_threshold_at_max_j(self):
        labels = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.1, 0.4, 0.4, 0.7, 0.7, 0.9])
        false_positive, true_positive, thresholds = roc_curve(labels, scores)
        finite = np.isfinite(thresholds)
        youden = (true_positive - false_positive)[finite]
        finite_thresholds = thresholds[finite]
        best = youden.max()
        expected = float(finite_thresholds[np.abs(youden - best) <= 1e-12].max())
        self.assertAlmostEqual(youden_threshold(labels, scores), expected)

    def test_removing_sick_films_changes_auprc(self):
        labels = [1, 0, 0]
        scores = [0.4, 0.1, 0.9]
        groups = ["tb", "healthy", "sick"]
        self.assertGreater(
            auprc_without_sick(labels, scores, groups),
            average_precision(labels, scores),
        )

    def test_false_positive_rates_are_separate(self):
        labels = [1, 0, 0, 0, 0]
        scores = [0.95, 0.2, 0.6, 0.9, 0.7]
        groups = ["tb", "healthy", "healthy", "healthy", "sick"]
        report = val_report(labels, scores, groups, threshold=0.5)
        self.assertAlmostEqual(report["fpr_healthy"], 2 / 3)
        self.assertAlmostEqual(report["fpr_sick"], 1.0)
        self.assertEqual(report["n_val"], 5)
        self.assertEqual(report["n_val_no_sick"], 4)
        self.assertEqual(report["val_tp"], 1)
        self.assertEqual(report["val_fn"], 0)
        self.assertEqual(report["val_fp"], 3)
        self.assertEqual(report["val_tn"], 1)
        self.assertAlmostEqual(report["val_sensitivity"], 1.0)
        self.assertAlmostEqual(report["val_specificity"], 0.25)
        self.assertAlmostEqual(report["val_specificity_healthy"], 1 / 3)
        self.assertAlmostEqual(report["val_specificity_sick"], 0.0)
        self.assertAlmostEqual(false_positive_rate([0.2, 0.6], 0.5), 0.5)


if __name__ == "__main__":
    unittest.main()
