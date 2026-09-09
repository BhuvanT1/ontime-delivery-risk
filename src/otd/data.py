"""Feature assembly and the train/test split.

Two rules are enforced here, and both exist because breaking either produces a
model that scores beautifully and is worthless.

1. **Only at-purchase columns.** ``order_fact`` carries ``post_*`` columns --
   when the parcel reached the carrier, how long it was in transit, what the
   customer scored it. All of them are observed after the decision point. They
   are dropped by ``feature_frame`` and can only be obtained by asking for the
   leaky variant explicitly, which exists so the inflation can be measured.

2. **Split on time, not at random.** These orders span two years during which
   the promise policy and the network both changed. A random split lets the
   model learn from August to predict March, which is not a thing anyone can do
   in production, and it also lets near-duplicate orders from the same seller on
   the same day land on both sides of the split.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET = "is_late"
ID_COLS = ["order_id", "customer_unique_id", "primary_seller_id"]
TIME_COLS = ["purchased_at", "delivered_at", "estimated_delivery_at"]

# A raw calendar index. It is the single most useful column in training and
# worthless in production, which is exactly the trap: every future order has a
# value beyond anything in the training range, so a tree sends all of them into
# its terminal bin and a linear model extrapolates off the end of the fitted
# line. Including it cost roughly 0.11 of test ROC-AUC when this repository was
# built -- the model looked fine on validation and fell apart out of time.
# ``drift_report`` below is the check that found it and is run by the training
# script on every fit.
TIME_INDEX = ["at_days_since_start"]

# Cumulative counts. Legitimate signal -- how much history exists for this
# seller or route, which is what separates a cold start from a known quantity --
# but they only ever grow, so in raw form they carry the same unbounded calendar
# drift as the column above. log1p keeps the cold-start contrast, which lives in
# the first few orders, and flattens the runaway tail.
LOG1P_COLS = ["at_seller_prior_n", "at_route_prior_n"]

EXCLUDE = set(ID_COLS + TIME_COLS + TIME_INDEX + [TARGET, "days_vs_promise"])

CATEGORICAL = [
    "at_customer_state", "at_seller_state", "at_primary_category", "at_payment_type",
]

# Deliberately not features. Kept as a named list so the leaky comparison in
# scripts/03_model.py can switch exactly these on and nothing else.
POST_HOC = [
    "post_approval_lag_h", "post_handoff_lag_d", "post_transit_d", "post_total_d",
    "post_seller_missed_deadline", "post_review_score", "post_n_reviews",
]


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    cut_valid: pd.Timestamp
    cut_test: pd.Timestamp

    def describe(self) -> pd.DataFrame:
        rows = []
        for name, part in (("train", self.train), ("valid", self.valid), ("test", self.test)):
            rows.append({
                "split": name, "n": len(part),
                "from": part["purchased_at"].min(), "to": part["purchased_at"].max(),
                "late_rate": round(float(part[TARGET].mean()), 4),
            })
        return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame, *, include_post_hoc: bool = False) -> list[str]:
    cols = [c for c in df.columns if c not in EXCLUDE and not c.startswith("post_")]
    if include_post_hoc:
        cols += [c for c in POST_HOC if c in df.columns]
    return cols


def feature_frame(df: pd.DataFrame, *, include_post_hoc: bool = False
                  ) -> tuple[pd.DataFrame, pd.Series]:
    """Model matrix and label.

    Categoricals become pandas ``category`` dtype rather than one-hot columns:
    LightGBM splits on them natively, and one-hot on a 73-level category column
    would add 73 mostly-empty columns for no gain. The linear model one-hots
    them separately, where it is required.
    """
    cols = feature_columns(df, include_post_hoc=include_post_hoc)
    X = df[cols].copy()
    for c in LOG1P_COLS:
        if c in X.columns:
            X[c] = np.log1p(X[c].astype(float).clip(lower=0))
    for c in CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X, df[TARGET].astype(int)


def time_split(
    df: pd.DataFrame,
    *,
    valid_frac: float = 0.15,
    test_frac: float = 0.20,
    time_col: str = "purchased_at",
) -> Split:
    """Chronological train / validation / test split.

    Validation sits between train and test in time and is used for calibration
    and threshold selection, so the test period is touched exactly once, at the
    end. Choosing the operating threshold on the test set would make the
    reported cost saving an in-sample number.
    """
    if not 0 < valid_frac + test_frac < 1:
        raise ValueError("valid_frac + test_frac must be in (0, 1)")
    d = df.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    n = len(d)
    i_valid = int(n * (1 - valid_frac - test_frac))
    i_test = int(n * (1 - test_frac))
    cut_valid = d[time_col].iloc[i_valid]
    cut_test = d[time_col].iloc[i_test]
    return Split(
        train=d.iloc[:i_valid].copy(),
        valid=d.iloc[i_valid:i_test].copy(),
        test=d.iloc[i_test:].copy(),
        cut_valid=cut_valid,
        cut_test=cut_test,
    )


def assert_no_post_hoc(cols) -> None:
    """Guard used by the training scripts and the test suite."""
    leaked = [c for c in cols if c.startswith("post_")]
    if leaked:
        raise AssertionError(f"post-hoc columns reached the feature matrix: {leaked}")


def drift_report(train: pd.DataFrame, test: pd.DataFrame, cols: list[str],
                 *, tol: float = 0.01) -> pd.DataFrame:
    """Flag numeric features whose test values fall outside the training range.

    Written after a raw day-counter reached the model matrix and quietly cost
    0.11 of test ROC-AUC. The failure is invisible to every accuracy metric on
    a validation split drawn from the same period, so it needs its own check.

    ``share_outside`` is the fraction of test rows beyond the training minimum
    or maximum. Anything approaching 1.0 means the feature is a calendar index
    or an unbounded running total, and no model can do anything sensible with
    it out of time: trees cannot extrapolate at all, and linear models
    extrapolate confidently in a straight line. Either exclude the column or
    transform it onto a bounded scale.
    """
    rows = []
    for c in cols:
        if c not in train.columns or c not in test.columns:
            continue
        if not pd.api.types.is_numeric_dtype(train[c]):
            continue
        lo, hi = train[c].min(), train[c].max()
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        te = test[c].dropna()
        if te.empty:
            continue
        outside = float(((te < lo) | (te > hi)).mean())
        if outside > tol:
            rows.append({
                "feature": c, "train_min": lo, "train_max": hi,
                "test_min": te.min(), "test_max": te.max(),
                "share_outside": round(outside, 4),
                "verdict": "calendar index or running total -- exclude or bound"
                           if outside > 0.5 else "partial drift -- check",
            })
    return (pd.DataFrame(rows).sort_values("share_outside", ascending=False)
            .reset_index(drop=True) if rows else
            pd.DataFrame(columns=["feature", "train_min", "train_max", "test_min",
                                  "test_max", "share_outside", "verdict"]))
