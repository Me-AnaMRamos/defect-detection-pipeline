import numpy as np
from sklearn.metrics import roc_curve


def recall_at_fpr(y_true, scores, target_fpr=0.05):
    """
    Compute Recall at fixed False Positive Rate.

    Parameters
    ----------
    y_true : array-like (0 = normal, 1 = defect)
    scores : anomaly scores (higher = more anomalous)
    target_fpr : float

    Returns
    -------
    recall : float
    threshold : float
    """

    fpr, tpr, thresholds = roc_curve(y_true, scores)

    # Find closest FPR
    idx = np.argmin(np.abs(fpr - target_fpr))

    return {
        "recall": tpr[idx],
        "fpr": fpr[idx],
        "threshold": thresholds[idx],
    }
