"""Tests against a real warehouse built from the repository's own SQL."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_fact_table_has_exactly_one_row_per_order(mini_fact):
    """Guards against join fan-out. Orders can carry several line items,
    several payment rows and -- in the real export -- several reviews; any of
    those joined naively multiplies the order and silently reweights every
    rate in the report."""
    assert mini_fact.order_id.is_unique


def test_population_is_delivered_orders_only(mini_db):
    n_fact = mini_db.execute("SELECT count(*) FROM order_fact").fetchone()[0]
    n_ok = mini_db.execute("""
        SELECT count(*) FROM orders
        WHERE order_status = 'delivered' AND delivered_at IS NOT NULL
          AND estimated_delivery_at IS NOT NULL AND purchased_at IS NOT NULL
    """).fetchone()[0]
    # Orders with no line items cannot be attributed to a seller and drop out.
    assert n_fact <= n_ok


def test_label_is_a_calendar_day_comparison(mini_fact):
    expected = (mini_fact.delivered_at.dt.normalize()
                > mini_fact.estimated_delivery_at.dt.normalize()).astype(int)
    pd.testing.assert_series_equal(mini_fact.is_late.astype(int), expected,
                                   check_names=False)


def test_seller_history_counts_orders_DELIVERED_before_purchase(mini_fact):
    """The core leakage guarantee.

    A previous order tells you nothing about a seller until it has actually
    been delivered. Counting orders *placed* before this one would credit the
    model with outcomes that had not happened yet at decision time. The ASOF
    join keys on delivery date; this pins that behaviour exactly.
    """
    f = mini_fact
    dd = f.delivered_at.dt.normalize().to_numpy()
    pp = f.purchased_at.dt.normalize().to_numpy()
    sid = f.primary_seller_id.to_numpy()
    for i in range(0, len(f), 7):
        expected = int(((sid == sid[i]) & (dd < pp[i])).sum())
        assert int(f.at_seller_prior_n.iloc[i]) == expected, f"row {i}"


def test_as_of_history_differs_from_the_naive_version(mini_fact):
    """If the two agreed there would be nothing to protect against. On this
    fixture the naive count is strictly larger for a substantial share of
    orders -- that difference is the leak."""
    f = mini_fact
    dd = f.delivered_at.dt.normalize().to_numpy()
    pp = f.purchased_at.dt.normalize().to_numpy()
    sid = f.primary_seller_id.to_numpy()
    naive, asof = [], []
    for i in range(0, len(f), 5):
        naive.append(int(((sid == sid[i]) & (pp < pp[i])).sum()))
        asof.append(int(f.at_seller_prior_n.iloc[i]))
    naive, asof = np.array(naive), np.array(asof)
    assert (naive >= asof).all()
    assert (naive > asof).mean() > 0.3


def test_smoothed_history_rate_is_a_probability(mini_fact):
    for col in ("at_seller_prior_late_rate", "at_route_prior_late_rate"):
        assert mini_fact[col].between(0, 1).all()
        assert mini_fact[col].notna().all()


def test_cold_start_sellers_fall_back_to_the_prior(mini_fact):
    """A seller with no delivered history must not get a rate of 0, which
    would read as 'never late' and is the opposite of what is known."""
    cold = mini_fact[mini_fact.at_seller_prior_n == 0]
    assert len(cold) > 0
    assert (cold.at_seller_prior_late_rate > 0).all()


def test_distance_is_null_not_zero_when_geolocation_is_missing(mini_db):
    n_zero = mini_db.execute(
        "SELECT count(*) FROM order_fact WHERE at_distance_km = 0 AND at_same_state = 0"
    ).fetchone()[0]
    assert n_zero == 0


def test_data_quality_suite_runs_and_reports(mini_db):
    dq = mini_db.execute("SELECT * FROM dq_checks").fetchdf()
    assert len(dq) >= 25
    assert set(dq.severity) <= {"error", "warn", "info"}
    assert (dq.n_fail >= 0).all()
    assert (dq.n_fail <= dq.n_total).all()


def test_referential_integrity_checks_pass_on_clean_input(mini_db):
    orphans = mini_db.execute("""
        SELECT sum(n_fail) FROM dq_checks
        WHERE check_name IN ('orders.customer_id_orphans', 'order_items.product_id_orphans')
    """).fetchone()[0]
    assert orphans == 0


def test_wilson_bounds_bracket_the_point_estimate(mini_db):
    d = mini_db.execute("SELECT * FROM v_segment_distance").fetchdf()
    assert (d.late_rate_lo <= d.late_rate).all()
    assert (d.late_rate <= d.late_rate_hi).all()
    assert (d.late_rate_lo >= 0).all() and (d.late_rate_hi <= 1).all()


def test_smr_equals_observed_over_expected(mini_db):
    d = mini_db.execute("SELECT * FROM v_state_smr").fetchdf()
    if d.empty:
        pytest.skip("fixture too small for the 100-order state threshold")
    np.testing.assert_allclose(d.smr, d.observed / d.expected, rtol=0.02)
    assert (d.smr_lo <= d.smr).all() and (d.smr <= d.smr_hi).all()


def test_unreliable_seller_is_detected_by_the_segmentation(mini_db):
    """s3 is generated at a 60% late rate against 10% for the others, so the
    segmentation has to find it. A test that only checks the SQL runs would
    not notice the logic being inverted."""
    d = mini_db.execute("SELECT * FROM v_segment_seller ORDER BY late_rate DESC").fetchdf()
    assert d.iloc[0].seller_id == "s3"
    assert d.iloc[0].late_rate > 0.4


def test_zip_centroid_is_one_row_per_prefix(mini_db):
    n, d = mini_db.execute(
        "SELECT count(*), count(DISTINCT zip_prefix) FROM zip_centroid").fetchone()
    assert n == d
