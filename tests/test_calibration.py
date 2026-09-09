import numpy as np
import pytest

from otd import evaluate as ev


def _shifted_problem(seed=0):
    """Train-like scores against a test period whose base rate has halved --
    the situation this dataset actually presents."""
    rng = np.random.default_rng(seed)
    p_valid = np.clip(rng.beta(2, 8, 6000), 1e-3, 1 - 1e-3)
    y_valid = (rng.random(6000) < p_valid / 2).astype(int)
    p_test = np.clip(rng.beta(2, 8, 6000), 1e-3, 1 - 1e-3)
    y_test = (rng.random(6000) < p_test / 2).astype(int)
    return p_valid, y_valid, p_test, y_test


def test_isotonic_never_reverses_two_scores():
    """The real guarantee. Isotonic is non-decreasing, so p1 <= p2 implies
    f(p1) <= f(p2) for every pair -- calibration cannot undo the model's
    ordering."""
    p_va, y_va, p_te, _ = _shifted_problem()
    cal = ev.IsotonicCalibrator().fit(y_va, p_va)
    order = np.argsort(p_te)
    out = cal.transform(p_te)[order]
    assert np.all(np.diff(out) >= -1e-12)


def test_isotonic_shifts_roc_auc_only_slightly_via_ties():
    """It is non-decreasing but not strictly increasing: runs of scores collapse
    onto one plateau, and those new ties each score half a concordance instead
    of one. So ROC-AUC moves a little. Asserting exact equality here would be
    asserting something false."""
    p_va, y_va, p_te, y_te = _shifted_problem()
    cal = ev.IsotonicCalibrator().fit(y_va, p_va)
    before = ev.roc_auc(y_te, p_te)
    after = ev.roc_auc(y_te, cal.transform(p_te))
    assert after == pytest.approx(before, abs=0.01)
    assert len(np.unique(cal.transform(p_te))) < len(np.unique(p_te))


def test_isotonic_improves_calibration_under_base_rate_shift():
    p_va, y_va, p_te, y_te = _shifted_problem()
    cal = ev.IsotonicCalibrator().fit(y_va, p_va)
    assert ev.ece(y_te, cal.transform(p_te)) < ev.ece(y_te, p_te)
    assert ev.brier(y_te, cal.transform(p_te)) < ev.brier(y_te, p_te)


def test_calibrated_output_is_a_valid_probability():
    p_va, y_va, p_te, _ = _shifted_problem(1)
    out = ev.IsotonicCalibrator().fit(y_va, p_va).transform(p_te)
    assert out.min() >= 0 and out.max() <= 1


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        ev.IsotonicCalibrator().transform([0.1, 0.2])
