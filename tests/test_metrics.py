import numpy as np
import pytest

from otd import evaluate as ev


def test_brier_hand_computed():
    y = [1, 0, 1, 0]
    p = [0.8, 0.2, 0.6, 0.1]
    expected = np.mean([(0.8 - 1) ** 2, (0.2 - 0) ** 2, (0.6 - 1) ** 2, (0.1 - 0) ** 2])
    assert ev.brier(y, p) == pytest.approx(expected)


def test_roc_auc_perfect_and_inverted():
    y = [0, 0, 1, 1]
    assert ev.roc_auc(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert ev.roc_auc(y, [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)


def test_pr_auc_floor_is_the_base_rate():
    """A constant prediction scores the positive rate, not 0.5. This is the
    property that makes PR-AUC readable under heavy imbalance."""
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.05).astype(int)
    assert ev.pr_auc(y, np.full(len(y), 0.5)) == pytest.approx(y.mean(), abs=0.01)


def test_roc_auc_is_blind_to_a_constant_shift():
    """Discrimination and calibration are different things: halving every
    probability cannot change the ranking, so ROC-AUC must not move."""
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.2).astype(int)
    p = np.clip(rng.random(2000), 0.01, 0.99)
    assert ev.roc_auc(y, p) == pytest.approx(ev.roc_auc(y, p / 2))
    assert ev.brier(y, p) != pytest.approx(ev.brier(y, p / 2))


def test_ece_zero_for_a_perfectly_calibrated_model():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.95, 40000)
    y = (rng.random(40000) < p).astype(int)
    assert ev.ece(y, p, n_bins=10) < 0.02


def test_recall_at_k_and_lift_are_consistent():
    rng = np.random.default_rng(3)
    p = rng.random(5000)
    y = (rng.random(5000) < p / 3).astype(int)
    lift = ev.lift_table(y, p, n_bins=10)
    assert lift.cum_recall.iloc[-1] == pytest.approx(1.0)
    assert lift.cum_recall.is_monotonic_increasing
    assert ev.recall_at_k(y, p, 0.10) == pytest.approx(lift.cum_recall.iloc[0], abs=1e-9)


def test_reliability_intervals_bracket_the_observed_rate():
    rng = np.random.default_rng(4)
    p = rng.uniform(0, 1, 3000)
    y = (rng.random(3000) < p).astype(int)
    tab = ev.reliability_table(y, p)
    assert ((tab.lo <= tab.frac_pos) & (tab.frac_pos <= tab.hi)).all()


def test_metrics_reject_mismatched_shapes():
    with pytest.raises(ValueError):
        ev.brier([1, 0], [0.5])
