"""Selection AUPRC, the Youden threshold, and the val report."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_curve


def average_precision(labels, scores) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0 or np.unique(labels).size < 2:
        raise ValueError("AUPRC needs both positive and negative films.")
    return float(average_precision_score(labels, scores))


def youden_threshold(labels, scores) -> float:
    """Highest finite ROC threshold that maximizes TPR minus FPR.

    A film is called positive when its score is greater than or equal to the
    threshold. That is the same rule used for the false-positive rates.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if np.unique(labels).size < 2:
        raise ValueError("Youden threshold needs both positive and negative films.")
    false_positive, true_positive, thresholds = roc_curve(labels, scores)
    youden = true_positive - false_positive
    finite = np.isfinite(thresholds)
    if not finite.any():
        raise ValueError("ROC curve produced no finite threshold.")
    youden = youden[finite]
    thresholds = thresholds[finite]
    best = youden.max()
    tied = thresholds[np.abs(youden - best) <= 1e-12]
    return float(tied.max())


def false_positive_rate(scores, threshold: float) -> float:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("Cannot compute a false-positive rate on an empty group.")
    return float(np.mean(scores >= threshold))


def auprc_without_sick(labels, scores, groups) -> float:
    groups = np.asarray(groups)
    keep = groups != "sick"
    return average_precision(np.asarray(labels)[keep], np.asarray(scores)[keep])


def val_report(labels, scores, groups, threshold: float) -> dict:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)
    healthy = groups == "healthy"
    sick = groups == "sick"
    return {
        "val_auprc": average_precision(labels, scores),
        "val_auprc_no_sick": auprc_without_sick(labels, scores, groups),
        "fpr_healthy": false_positive_rate(scores[healthy], threshold),
        "fpr_sick": false_positive_rate(scores[sick], threshold),
        "n_val": int(labels.size),
        "n_val_no_sick": int((~sick).sum()),
    }
