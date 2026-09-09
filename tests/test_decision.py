import numpy as np
import pytest

from otd.decision import (CostModel, best_threshold, policy_cost,
                          required_effectiveness, sensitivity, sweep_thresholds)


def _problem(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(2, 20, n)
    y = (rng.random(n) < p).astype(int)
    return y, p


def test_naive_threshold_is_the_break_even_risk():
    cm = CostModel(cost_late=60, cost_action=12, effectiveness=0.30)
    assert cm.naive_threshold == pytest.approx(12 / (0.30 * 60))


def test_naive_threshold_is_not_one_half():
    """The default 0.5 cutoff encodes equal costs in both directions, which is
    almost never the situation an operations team is in."""
    assert CostModel().naive_threshold != pytest.approx(0.5)


def test_zero_effectiveness_means_no_threshold_is_worth_acting_on():
    assert CostModel(effectiveness=0.0).naive_threshold == float("inf")


def test_flagging_nothing_costs_exactly_the_do_nothing_cost():
    y, p = _problem()
    cm = CostModel()
    r = policy_cost(y, p, 1.1, cm)
    assert r["n_flagged"] == 0
    assert r["cost_policy"] == pytest.approx(r["cost_do_nothing"])
    assert r["net_saving"] == pytest.approx(0.0)


def test_flagging_everything_prevents_the_expected_share():
    y, p = _problem()
    cm = CostModel(cost_late=60, cost_action=0.0, effectiveness=0.5)
    r = policy_cost(y, p, -1, cm)
    assert r["flag_rate"] == 1.0
    assert r["expected_late_prevented"] == pytest.approx(0.5 * y.sum())
    assert r["cost_policy"] == pytest.approx(60 * 0.5 * y.sum())


def test_a_free_and_perfect_intervention_removes_all_cost():
    y, p = _problem()
    r = policy_cost(y, p, -1, CostModel(cost_action=0.0, effectiveness=1.0))
    assert r["cost_policy"] == pytest.approx(0.0)
    assert r["saving_pct"] == pytest.approx(100.0)


def test_recall_falls_as_the_threshold_rises():
    y, p = _problem()
    sweep = sweep_thresholds(y, p, CostModel()).sort_values("threshold")
    assert sweep.recall.is_monotonic_decreasing
    assert sweep.flag_rate.is_monotonic_decreasing


def test_capacity_constraint_is_respected():
    y, p = _problem()
    best = best_threshold(y, p, CostModel(cost_action=0.01), max_flag_rate=0.10)
    assert best["flag_rate"] <= 0.10


def test_impossible_capacity_raises():
    y, p = _problem()
    with pytest.raises(ValueError):
        best_threshold(y, p, CostModel(), max_flag_rate=-1.0)


def test_required_effectiveness_is_the_true_break_even_point():
    """Setting effectiveness to the returned value must make the policy score
    exactly zero saving."""
    y, p = _problem()
    cm = CostModel(cost_late=100, cost_action=2, effectiveness=0.3)
    t = float(np.quantile(p, 0.9))
    req = required_effectiveness(y, p, t, cm)
    assert 0 < req < 1
    at_break_even = CostModel(cost_late=cm.cost_late, cost_action=cm.cost_action,
                              effectiveness=req)
    assert policy_cost(y, p, t, at_break_even)["net_saving"] == pytest.approx(0.0, abs=1e-6)


def test_required_effectiveness_above_one_means_the_policy_cannot_pay():
    """The useful negative result. When the break-even exceeds 1 the action
    would have to prevent more lateness than exists at that threshold, so no
    intervention -- however good -- makes the policy worth running."""
    y, p = _problem()
    expensive = CostModel(cost_late=20, cost_action=50, effectiveness=0.3)
    t = float(np.quantile(p, 0.9))
    assert required_effectiveness(y, p, t, expensive) > 1.0


def test_sensitivity_covers_the_whole_grid():
    y, p = _problem()
    s = sensitivity(y, p, cost_late=(30, 60), cost_action=(6, 12), effectiveness=(0.2, 0.4))
    assert len(s) == 8
    assert {"cost_late", "cost_action", "effectiveness", "saving_per_order"} <= set(s.columns)


def test_cost_model_rejects_nonsense():
    with pytest.raises(ValueError):
        CostModel(effectiveness=1.5)
    with pytest.raises(ValueError):
        CostModel(cost_late=-1)
