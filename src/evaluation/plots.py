from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def plot_roc(y_true, scores, target_fpr: float = 0.05, ax=None, label: str | None = None):
    """ROC curve (recall vs FPR) with the FPR budget marked."""
    ax = ax or plt.gca()
    fpr, tpr, _ = roc_curve(y_true, scores)
    ax.plot(fpr, tpr, label=label)
    ax.axvline(target_fpr, linestyle="--", linewidth=1, color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("Recall")
    ax.grid(True, alpha=0.3)
    return ax


def plot_score_hist(y_true, scores, threshold: float, ax=None, bins: int = 30):
    """Score distributions of normal vs defective images with the threshold line."""
    ax = ax or plt.gca()
    y = np.asarray(y_true)
    s = np.asarray(scores)
    ax.hist(s[y == 0], bins=bins, alpha=0.6, label="normal")
    ax.hist(s[y == 1], bins=bins, alpha=0.6, label="defective")
    ax.axvline(threshold, linestyle="--", color="black", label="threshold")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Images")
    ax.legend()
    return ax
