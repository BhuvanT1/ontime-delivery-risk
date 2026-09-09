import numpy as np
import pandas as pd
import pytest

from otd.data import TARGET, time_split
from otd.models import BaseRate, LGBMModel, LogisticModel, PromiseRule


def _frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    ds = pd.Timestamp("2017-01-01") + pd.to_timedelta(np.arange(n) // 4, unit="D")
    dist = rng.uniform(10, 3000, n)
    promise = rng.integers(5, 40, n)
    # Real signal: far away and a short promise means trouble.
    logit = -3.2 + 0.0011 * dist - 0.05 * promise
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "order_id": [f"o{i}" for i in range(n)],
        "customer_unique_id": "u", "primary_seller_id": rng.choice(["s1", "s2"], n),
        "purchased_at": ds, "delivered_at": ds, "estimated_delivery_at": ds,
        TARGET: y, "days_vs_promise": 0,
        "at_distance_km": dist, "at_promise_days": promise,
        "at_total_price": rng.uniform(10, 500, n),
        "at_n_items": rng.integers(1, 4, n),
        "at_seller_prior_n": rng.integers(0, 200, n),
        "at_route_prior_n": rng.integers(0, 500, n),
        "at_customer_state": rng.choice(["SP", "RJ", "BA"], n),
        "at_payment_type": rng.choice(["credit_card", "boleto"], n),
        "post_transit_d": rng.random(n),
    })


@pytest.fixture(scope="module")
def split():
    return time_split(_frame())


def test_base_rate_predicts_the_training_mean(split):
    m = BaseRate().fit(split.train)
    p = m.predict_proba(split.test)
    assert np.allclose(p, split.train[TARGET].mean())


@pytest.mark.parametrize("Model", [BaseRate, PromiseRule, LogisticModel, LGBMModel])
def test_predictions_are_valid_probabilities(split, Model):
    m = Model()
    m.fit(split.train, split.valid) if isinstance(m, (LGBMModel, LogisticModel)) else m.fit(split.train)
    p = m.predict_proba(split.test)
    assert len(p) == len(split.test)
    assert np.isfinite(p).all()
    assert (p >= 0).all() and (p <= 1).all()


@pytest.mark.parametrize("Model", [PromiseRule, LogisticModel, LGBMModel])
def test_models_beat_chance_when_there_is_signal(split, Model):
    from otd import evaluate as ev
    m = Model()
    m.fit(split.train, split.valid) if isinstance(m, (LGBMModel, LogisticModel)) else m.fit(split.train)
    assert ev.roc_auc(split.test[TARGET], m.predict_proba(split.test)) > 0.6


def test_logistic_fit_and_predict_share_one_transform(split):
    """Regression test for a real bug in this repository.

    An earlier version transformed features inside fit and indexed the raw
    frame inside predict, so training saw log1p(prior_n) and scoring saw the
    raw count. Validation ROC-AUC fell to 0.47 -- worse than chance -- and no
    accuracy metric said why. Scoring the training rows recovers the training
    fit only if both paths agree.
    """
    from otd import evaluate as ev
    m = LogisticModel().fit(split.train)
    assert ev.roc_auc(split.train[TARGET], m.predict_proba(split.train)) > 0.6


def test_predictions_are_deterministic(split):
    m = LGBMModel().fit(split.train, split.valid)
    np.testing.assert_array_equal(m.predict_proba(split.test), m.predict_proba(split.test))


def test_predictions_do_not_depend_on_row_order(split):
    """A model whose output changes when the rows are shuffled has state
    leaking between rows."""
    m = LGBMModel().fit(split.train, split.valid)
    shuffled = split.test.sample(frac=1.0, random_state=1)
    a = pd.Series(m.predict_proba(split.test), index=split.test.index)
    b = pd.Series(m.predict_proba(shuffled), index=shuffled.index)
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())


def test_promise_rule_falls_back_on_unseen_cells(split):
    m = PromiseRule().fit(split.train)
    odd = split.test.copy()
    odd["at_distance_km"] = np.nan
    odd["at_promise_days"] = 9999
    p = m.predict_proba(odd)
    assert np.isfinite(p).all() and (p > 0).all()


def test_leaky_variant_scores_far_higher(split):
    """Sanity check on the contrast the report is built around: with the
    post-delivery column switched on the same model becomes near-perfect."""
    from otd import evaluate as ev
    clean = LGBMModel().fit(split.train, split.valid)
    leaky = LGBMModel(include_post_hoc=True).fit(split.train, split.valid)
    assert (ev.roc_auc(split.test[TARGET], leaky.predict_proba(split.test))
            > ev.roc_auc(split.test[TARGET], clean.predict_proba(split.test)))


def test_logistic_coefficients_are_reported_as_odds_ratios(split):
    c = LogisticModel().fit(split.train).coefficients(top=5)
    assert {"feature", "log_odds", "odds_ratio"} <= set(c.columns)
    np.testing.assert_allclose(c.odds_ratio, np.exp(c.log_odds))
