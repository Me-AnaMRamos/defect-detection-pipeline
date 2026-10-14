import numpy as np
import pytest

from src.evaluation.metrics import (
    calibrate_threshold,
    evaluate,
    metrics_at_threshold,
    recall_at_fpr,
)


def test_calibrated_threshold_flags_at_most_target_fraction_of_normals():
    rng = np.random.default_rng(0)
    normal = rng.normal(size=1000)
    thr = calibrate_threshold(normal, target_fpr=0.05)
    assert np.mean(normal > thr) <= 0.05
    assert np.mean(normal > thr) > 0.04


def test_calibration_needs_no_defective_samples():
    # The whole point: a normal-only set is enough to set the threshold.
    thr = calibrate_threshold(np.arange(100, dtype=float), target_fpr=0.1)
    assert thr == 90.0


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1])
def test_calibration_rejects_invalid_fpr(bad):
    with pytest.raises(ValueError):
        calibrate_threshold(np.ones(10), target_fpr=bad)


def test_metrics_at_threshold_counts():
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.4])
    m = metrics_at_threshold(y, s, threshold=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 3)
    assert m["recall"] == 0.5
    assert m["fpr"] == 0.25


def test_recall_at_fpr_never_exceeds_budget():
    # 20 normals, 10 defects, budget 8% -> at most 1 false positive (5%).
    # The old nearest-FPR rule picked FPR 0.10 here, breaking the budget.
    y = np.array([0] * 20 + [1] * 10)
    s = np.concatenate([np.arange(20), [18.5, 17.5, 16.5] + list(range(100, 107))]).astype(float)
    r = recall_at_fpr(y, s, target_fpr=0.08)
    assert r["fpr"] <= 0.08
    # 7 defects beat every normal; the 8th (18.5) beats all but one.
    assert r["recall"] == pytest.approx(0.8)


def test_recall_at_fpr_requires_both_classes():
    with pytest.raises(ValueError):
        recall_at_fpr(np.zeros(10), np.arange(10.0))


def test_evaluate_perfect_separation():
    y = np.array([0] * 50 + [1] * 50)
    s = np.concatenate([np.linspace(0, 1, 50), np.linspace(2, 3, 50)])
    thr = calibrate_threshold(s[:50], 0.05)
    r = evaluate(y, s, thr, 0.05)
    assert r["auroc"] == 1.0
    assert r["recall"] == 1.0
    assert r["oracle_recall"] == 1.0
    assert r["fpr"] <= 0.05
