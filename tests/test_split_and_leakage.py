import numpy as np
import pandas as pd
import pytest

from otd.data import (POST_HOC, TARGET, assert_no_post_hoc, drift_report,
                      feature_columns, feature_frame, time_split)


def _frame(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ds = pd.Timestamp("2017-01-01") + pd.to_timedelta(np.arange(n), unit="D")
    return pd.DataFrame({
        "order_id": [f"o{i}" for i in range(n)],
        "customer_unique_id": "u", "primary_seller_id": "s",
        "purchased_at": ds, "delivered_at": ds, "estimated_delivery_at": ds,
        TARGET: (rng.random(n) < 0.1).astype(int), "days_vs_promise": 0,
        "at_promise_days": rng.integers(5, 40, n),
        "at_days_since_start": np.arange(n),
        "at_seller_prior_n": np.arange(n) * 2,
        "at_route_prior_n": np.arange(n),
        "at_customer_state": rng.choice(["SP", "RJ"], n),
        "post_transit_d": rng.random(n) * 10,
        "post_review_score": rng.integers(1, 6, n),
    })


def test_split_is_chronological_and_disjoint():
    sp = time_split(_frame())
    assert sp.train.purchased_at.max() <= sp.valid.purchased_at.min()
    assert sp.valid.purchased_at.max() <= sp.test.purchased_at.min()
    ids = pd.concat([sp.train.order_id, sp.valid.order_id, sp.test.order_id])
    assert ids.is_unique


def test_split_fractions_are_respected():
    df = _frame(1000)
    sp = time_split(df, valid_frac=0.15, test_frac=0.20)
    assert len(sp.test) == pytest.approx(200, abs=2)
    assert len(sp.valid) == pytest.approx(150, abs=2)
    assert len(sp.train) + len(sp.valid) + len(sp.test) == len(df)


def test_post_hoc_columns_never_reach_the_feature_matrix():
    X, _ = feature_frame(_frame())
    assert not [c for c in X.columns if c.startswith("post_")]
    assert_no_post_hoc(X.columns)


def test_post_hoc_can_be_requested_explicitly_for_the_leak_comparison():
    X, _ = feature_frame(_frame(), include_post_hoc=True)
    assert set(POST_HOC) & set(X.columns)
    with pytest.raises(AssertionError):
        assert_no_post_hoc(X.columns)


def test_raw_time_index_is_excluded_by_default():
    """The column that cost 0.11 of test ROC-AUC. It stays in order_fact for
    cohort reporting and out of the model matrix."""
    assert "at_days_since_start" not in feature_columns(_frame())


def test_cumulative_counts_are_bounded_by_log1p():
    df = _frame()
    X, _ = feature_frame(df)
    assert X["at_seller_prior_n"].max() < 20
    assert X["at_seller_prior_n"].max() == pytest.approx(np.log1p(df["at_seller_prior_n"].max()))


def test_drift_report_catches_a_running_total():
    df = _frame()
    sp = time_split(df)
    rep = drift_report(sp.train, sp.test, ["at_route_prior_n", "at_promise_days"])
    assert "at_route_prior_n" in set(rep.feature)
    assert rep.loc[rep.feature == "at_route_prior_n", "share_outside"].iloc[0] > 0.9


def test_drift_report_is_quiet_on_a_stationary_feature():
    df = _frame()
    sp = time_split(df)
    rep = drift_report(sp.train, sp.test, ["at_promise_days"])
    assert rep.empty
