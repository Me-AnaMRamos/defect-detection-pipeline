"""Calibration and evaluation metrics.

Protocol
--------
1. **Calibrate** on validation images that are all normal: the threshold is the
   (1 - target_fpr) quantile of their scores, so roughly ``target_fpr`` of normal
   images will be flagged. No defective images are needed for this step.
2. **Evaluate** on the test split (normal + defective) at that fixed threshold:
   recall, realized FPR and precision, plus threshold-free AUROC / AUPR.
3. **Oracle** reference: the best recall reachable on test with FPR <= target if
   the threshold could be chosen using test labels. Not deployable; the gap to
   the calibrated recall measures the cost of calibrating on normal data only.

Convention: label 1 = defective (positive), 0 = normal; higher score = more anomalous.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def calibrate_threshold(normal_scores: np.ndarray, target_fpr: float = 0.05) -> float:
    """Threshold above which about ``target_fpr`` of normal scores fall.

    Uses the 'higher' quantile so the empirical FPR on the calibration set is
    <= target_fpr (a sample is flagged when score > threshold).
    """
    s = np.asarray(normal_scores, dtype=float)
    if s.size == 0:
        raise ValueError("Need at least one normal score to calibrate")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be in (0, 1)")
    return float(np.quantile(s, 1.0 - target_fpr, method="higher"))


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Recall, FPR and precision when flagging ``score > threshold``."""
    y = np.asarray(y_true).astype(int)
    pred = np.asarray(scores, dtype=float) > threshold

    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))

    return {
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.05) -> dict:
    """Highest recall achievable with FPR <= target_fpr (uses labels: oracle only).

    Requires both classes in ``y_true``.
    """
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("recall_at_fpr needs both normal and defective samples")

    fpr, tpr, thresholds = roc_curve(y, scores)
    ok = np.flatnonzero(fpr <= target_fpr + 1e-12)  # fpr[0] == 0, so never empty
    best = ok[np.argmax(tpr[ok])]
    return {
        "recall": float(tpr[best]),
        "fpr": float(fpr[best]),
        "threshold": float(thresholds[best]),
    }


def evaluate(
    y_test: np.ndarray, test_scores: np.ndarray, threshold: float, target_fpr: float = 0.05
) -> dict:
    """Full test-split report at a calibrated threshold."""
    y = np.asarray(y_test).astype(int)
    s = np.asarray(test_scores, dtype=float)

    at_thr = metrics_at_threshold(y, s, threshold)
    oracle = recall_at_fpr(y, s, target_fpr)

    return {
        "auroc": float(roc_auc_score(y, s)),
        "aupr": float(average_precision_score(y, s)),
        "threshold": float(threshold),
        "recall": at_thr["recall"],
        "fpr": at_thr["fpr"],
        "precision": at_thr["precision"],
        "tp": at_thr["tp"],
        "fp": at_thr["fp"],
        "fn": at_thr["fn"],
        "tn": at_thr["tn"],
        "oracle_recall": oracle["recall"],
        "oracle_fpr": oracle["fpr"],
        "n_test_normal": int(np.sum(y == 0)),
        "n_test_defective": int(np.sum(y == 1)),
    }
