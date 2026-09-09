"""Turning a probability into an operating decision.

A ranked risk score is not yet a recommendation. To act on it you need a
threshold, and to choose a threshold you need to say what being wrong costs in
each direction. That is a business input, not a modelling output, so every
number it depends on is declared in one place, given a default that is argued
for, and swept in a sensitivity grid -- because the honest answer to "what
would this save?" is a range with the assumptions attached.

WHAT IS MEASURED AND WHAT IS ASSUMED
------------------------------------
Measured in this dataset:
  * which orders were late, and by how many days
  * that late orders are scored 1 or 2 stars 62.8% of the time against 9.5%
    for on-time orders -- an association, in observational data
  * how well the model ranks risk on a held-out, chronologically later period

Assumed, and not knowable from this data:
  * ``cost_late``       the cost to the business of one late delivery
  * ``cost_action``     the cost of one proactive intervention
  * ``effectiveness``   the share of late deliveries an intervention prevents

The third is the one that matters most and the one no observational dataset can
answer. Nobody randomised which orders got intervened on, so ``effectiveness``
is a stated scenario parameter. Any figure produced here is a conditional
statement -- "if an intervention prevents 30% of the lateness it is applied to,
then..." -- and never a claim about realised impact. Establishing the real
number needs an experiment; ``required_effectiveness`` below computes the
break-even value that experiment would have to beat, which is the useful thing
to hand an operations team before running it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """Costs in Brazilian reais per order.

    Defaults are illustrative and chosen to be defensible rather than
    flattering: ``cost_late`` at R$60 is roughly a third of the average order
    value in this data, standing in for refunds, support contacts and the
    revenue at risk behind a 1-star review; ``cost_action`` at R$12 is the
    order of magnitude of an upgraded shipping lane or a proactive contact.
    Both should be replaced with the operator's real figures -- the point of
    the class is that they are visible and swept, not that they are right.
    """

    cost_late: float = 60.0
    cost_action: float = 12.0
    effectiveness: float = 0.30

    def __post_init__(self):
        if not 0 <= self.effectiveness <= 1:
            raise ValueError("effectiveness must be in [0, 1]")
        if self.cost_late < 0 or self.cost_action < 0:
            raise ValueError("costs must be non-negative")

    @property
    def naive_threshold(self) -> float:
        """Break-even risk for acting on a single order.

        Act when ``p * effectiveness * cost_late > cost_action``, i.e. when the
        expected lateness avoided is worth more than the action. Note this is
        not 0.5: with the defaults it is 0.667, which is above almost every
        predicted probability in this data -- so a per-order rule would act on
        nothing, and the sensible policy is a capacity-constrained top-k. That
        gap between the textbook threshold and the workable one is exactly what
        gets lost when a model is shipped at its default 0.5 cutoff.
        """
        denom = self.effectiveness * self.cost_late
        return float("inf") if denom <= 0 else self.cost_action / denom


def policy_cost(y_true, p, threshold: float, cm: CostModel) -> dict:
    """Expected cost of flagging every order at or above ``threshold``."""
    y = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(p, dtype=float).ravel()
    flagged = p >= threshold
    n_flag = int(flagged.sum())
    late_flagged = int(y[flagged].sum())
    late_missed = int(y[~flagged].sum())

    prevented = cm.effectiveness * late_flagged
    remaining_late = late_flagged - prevented + late_missed
    cost = cm.cost_action * n_flag + cm.cost_late * remaining_late
    do_nothing = cm.cost_late * int(y.sum())

    return {
        "threshold": float(threshold),
        "n": len(y),
        "n_flagged": n_flag,
        "flag_rate": n_flag / len(y) if len(y) else np.nan,
        "late_caught": late_flagged,
        "late_missed": late_missed,
        "recall": late_flagged / y.sum() if y.sum() else np.nan,
        "precision": late_flagged / n_flag if n_flag else np.nan,
        "expected_late_prevented": float(prevented),
        "cost_policy": float(cost),
        "cost_do_nothing": float(do_nothing),
        "net_saving": float(do_nothing - cost),
        "saving_pct": float((do_nothing - cost) / do_nothing * 100) if do_nothing else np.nan,
        "saving_per_order": float((do_nothing - cost) / len(y)) if len(y) else np.nan,
    }


def sweep_thresholds(y_true, p, cm: CostModel, n_points: int = 200) -> pd.DataFrame:
    """Cost curve over candidate thresholds, taken at quantiles of the score so
    the grid is dense where the orders actually are."""
    p = np.asarray(p, dtype=float).ravel()
    qs = np.unique(np.quantile(p, np.linspace(0, 1, n_points)))
    rows = [policy_cost(y_true, p, t, cm) for t in qs]
    return pd.DataFrame(rows)


def best_threshold(y_true, p, cm: CostModel, *, max_flag_rate: float | None = None
                   ) -> dict:
    """Threshold minimising expected cost, optionally under a capacity limit.

    ``max_flag_rate`` matters in practice: an operations team can expedite some
    share of orders, not any share, and the unconstrained optimum often flags
    far more than anyone could handle.
    """
    sweep = sweep_thresholds(y_true, p, cm)
    if max_flag_rate is not None:
        sweep = sweep[sweep["flag_rate"] <= max_flag_rate]
    if sweep.empty:
        raise ValueError("no threshold satisfies the capacity constraint")
    return sweep.loc[sweep["cost_policy"].idxmin()].to_dict()


def required_effectiveness(y_true, p, threshold: float, cm: CostModel) -> float:
    """Minimum intervention effectiveness for the policy to break even.

    This is the number to hand an operations team *before* they run a trial:
    the policy is worth testing only if they believe the intervention can beat
    this, and a subsequent A/B test is what would confirm it.
    """
    y = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(p, dtype=float).ravel()
    flagged = p >= threshold
    late_flagged = int(y[flagged].sum())
    n_flag = int(flagged.sum())
    if late_flagged == 0 or cm.cost_late == 0:
        return float("inf")
    return float(cm.cost_action * n_flag / (cm.cost_late * late_flagged))


def sensitivity(y_true, p, *, cost_late=(30, 60, 120), cost_action=(6, 12, 24),
                effectiveness=(0.15, 0.30, 0.50), max_flag_rate: float | None = 0.20
                ) -> pd.DataFrame:
    """Grid over every assumed input.

    Reported instead of a single headline figure. If the sign of the saving
    flips inside a plausible range of the assumptions, the honest conclusion is
    that the data does not yet support the policy -- which is a finding, not a
    failure.
    """
    rows = []
    for cl in cost_late:
        for ca in cost_action:
            for eff in effectiveness:
                cm = CostModel(cost_late=cl, cost_action=ca, effectiveness=eff)
                try:
                    best = best_threshold(y_true, p, cm, max_flag_rate=max_flag_rate)
                except ValueError:
                    continue
                rows.append({
                    "cost_late": cl, "cost_action": ca, "effectiveness": eff,
                    "threshold": round(best["threshold"], 4),
                    "flag_rate": round(best["flag_rate"], 4),
                    "recall": round(best["recall"], 4),
                    "saving_per_order": round(best["saving_per_order"], 3),
                    "saving_pct": round(best["saving_pct"], 2),
                })
    return pd.DataFrame(rows)
