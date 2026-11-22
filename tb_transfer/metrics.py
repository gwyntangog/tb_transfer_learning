"""Selection AUPRC, the Youden threshold, and the val report."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


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


def auroc(labels, scores) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0 or np.unique(labels).size < 2:
        raise ValueError("AUROC needs both positive and negative films.")
    return float(roc_auc_score(labels, scores))


def rates_at_threshold(labels, scores, threshold: float) -> dict:
    """Counts at one cutoff. A film is positive when its score is >= threshold.

    Sensitivity is the true positive rate. Specificity is the true negative rate.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    called = scores >= threshold
    positive = labels == 1
    tp = int(np.count_nonzero(called & positive))
    fn = int(np.count_nonzero(~called & positive))
    fp = int(np.count_nonzero(called & ~positive))
    tn = int(np.count_nonzero(~called & ~positive))
    sensitivity = _div(tp, tp + fn)
    specificity = _div(tn, tn + fp)
    youden_j = None
    if sensitivity is not None and specificity is not None:
        youden_j = sensitivity + specificity - 1.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": _div(tp, tp + fp),
        "npv": _div(tn, tn + fn),
        "youden_j": youden_j,
    }


def false_positive_rate(scores, threshold: float) -> float:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("Cannot compute a false-positive rate on an empty group.")
    return float(np.mean(scores >= threshold))


def _div(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


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
    tb = groups == "tb"
    rates = rates_at_threshold(labels, scores, threshold)
    healthy_rates = rates_at_threshold(labels[healthy], scores[healthy], threshold)
    sick_rates = rates_at_threshold(labels[sick], scores[sick], threshold)
    no_sick = ~sick
    return {
        "val_auprc": average_precision(labels, scores),
        "val_auprc_no_sick": auprc_without_sick(labels, scores, groups),
        "val_auroc": auroc(labels, scores),
        "val_auroc_no_sick": auroc(labels[no_sick], scores[no_sick]),
        "fpr_healthy": false_positive_rate(scores[healthy], threshold),
        "fpr_sick": false_positive_rate(scores[sick], threshold),
        "n_val": int(labels.size),
        "n_val_no_sick": int(no_sick.sum()),
        "n_val_tb": int(tb.sum()),
        "n_val_healthy": int(healthy.sum()),
        "n_val_sick": int(sick.sum()),
        "val_sensitivity": rates["sensitivity"],
        "val_specificity": rates["specificity"],
        "val_ppv": rates["ppv"],
        "val_npv": rates["npv"],
        "val_youden_j": rates["youden_j"],
        "val_tp": rates["tp"],
        "val_fp": rates["fp"],
        "val_tn": rates["tn"],
        "val_fn": rates["fn"],
        "val_specificity_healthy": healthy_rates["specificity"],
        "val_specificity_sick": sick_rates["specificity"],
        "val_fp_healthy": healthy_rates["fp"],
        "val_tn_healthy": healthy_rates["tn"],
        "val_fp_sick": sick_rates["fp"],
        "val_tn_sick": sick_rates["tn"],
    }


def holdout_report(labels, scores, groups, threshold: float) -> dict:
    """Operating point on the films that chose the epoch and the cutoff."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)
    rates = rates_at_threshold(labels, scores, threshold)
    healthy = groups == "healthy"
    sick = groups == "sick"
    return {
        "holdout_auprc": average_precision(labels, scores),
        "holdout_auroc": auroc(labels, scores),
        "holdout_sensitivity": rates["sensitivity"],
        "holdout_specificity": rates["specificity"],
        "holdout_youden_j": rates["youden_j"],
        "holdout_tp": rates["tp"],
        "holdout_fp": rates["fp"],
        "holdout_tn": rates["tn"],
        "holdout_fn": rates["fn"],
        "holdout_fpr_healthy": false_positive_rate(scores[healthy], threshold),
        "holdout_fpr_sick": false_positive_rate(scores[sick], threshold),
        "n_holdout_tb": int((groups == "tb").sum()),
        "n_holdout_healthy": int(healthy.sum()),
        "n_holdout_sick": int(sick.sum()),
    }
