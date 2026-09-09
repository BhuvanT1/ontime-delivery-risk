"""Model ladder.

As with any prediction problem, the interesting quantity is not how good the
best model is but how much better it is than something trivial. The ladder:

  base_rate     everyone gets the training late rate. Any metric that looks
                good here is a metric that cannot see anything.
  promise_rule  a lookup table on distance band x promise length, which is
                roughly what an analyst would build in a spreadsheet in an
                afternoon. This is the real bar to clear.
  logistic      regularised logistic regression on one-hot features. Gives
                coefficients that can be read, and calibrates well by
                construction.
  lgbm          gradient boosting on the same at-purchase features.
  lgbm_leaky    the same model plus post-hoc columns. Not a candidate --
                it exists to quantify how large the usual leak is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import CATEGORICAL, TARGET, feature_frame


class BaseRate:
    """Constant prediction at the training base rate."""

    name = "base_rate"

    def fit(self, train: pd.DataFrame):
        self.p_ = float(train[TARGET].mean())
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.p_, dtype=float)


class PromiseRule:
    """Smoothed lookup on (distance band, promise-length band).

    A deliberately simple, explainable rule of the kind an operations analyst
    would actually ship. Empty and thin cells fall back to the global rate via
    additive smoothing, so the rule degrades gracefully instead of producing a
    0% or 100% cell on three orders.
    """

    name = "promise_rule"

    def __init__(self, smoothing: float = 50.0):
        self.smoothing = smoothing

    @staticmethod
    def _bands(df: pd.DataFrame) -> pd.DataFrame:
        d = pd.DataFrame(index=df.index)
        d["dist"] = pd.cut(df["at_distance_km"], [-1, 200, 500, 1000, 2000, 1e9],
                           labels=["<200", "200-500", "500-1k", "1k-2k", "2k+"])
        d["dist"] = d["dist"].cat.add_categories("unknown").fillna("unknown")
        d["promise"] = pd.cut(df["at_promise_days"], [-1, 10, 17, 25, 35, 1e9],
                              labels=["<=10", "11-17", "18-25", "26-35", "36+"])
        d["promise"] = d["promise"].cat.add_categories("unknown").fillna("unknown")
        return d

    def fit(self, train: pd.DataFrame):
        b = self._bands(train)
        b[TARGET] = train[TARGET].to_numpy()
        self.global_ = float(train[TARGET].mean())
        g = b.groupby(["dist", "promise"], observed=True)[TARGET].agg(["sum", "count"])
        self.table_ = ((g["sum"] + self.smoothing * self.global_)
                       / (g["count"] + self.smoothing))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        b = self._bands(df)
        idx = pd.MultiIndex.from_arrays([b["dist"], b["promise"]])
        return self.table_.reindex(idx).fillna(self.global_).to_numpy(dtype=float)


class LogisticModel:
    """L2 logistic regression in a scikit-learn pipeline.

    Median imputation and standardisation for the numerics, one-hot for the
    categoricals with unseen levels ignored at scoring time. ``class_weight``
    is left at None on purpose: rebalancing would distort the predicted
    probabilities, and probabilities are what the decision layer needs. The
    imbalance is handled by choosing a threshold, not by reweighting the fit.
    """

    name = "logistic"

    def __init__(self, C: float = 0.5, max_iter: int = 2000, seed: int = 7):
        self.C, self.max_iter, self.seed = C, max_iter, seed
        self.include_post_hoc = False

    def _build(self, X: pd.DataFrame):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        cats = [c for c in CATEGORICAL if c in X.columns]
        nums = [c for c in X.columns if c not in cats]
        num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler())])
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=30, sparse_output=False)),
        ])
        pre = ColumnTransformer([("num", num_pipe, nums), ("cat", cat_pipe, cats)])
        return Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(C=self.C, max_iter=self.max_iter,
                                       random_state=self.seed)),
        ]), nums, cats

    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the design matrix.

        Used by BOTH fit and predict, which is the whole point of it existing.
        An earlier version transformed the features inside fit and then indexed
        the raw frame inside predict; training saw log1p(prior_n) and scoring
        saw the raw count, and validation ROC-AUC fell to 0.47 -- worse than
        chance, from a model that was fine. Nothing in the accuracy metrics
        says "your two code paths disagree", so the paths are shared instead.
        """
        X, _ = feature_frame(df, include_post_hoc=self.include_post_hoc)
        for c in CATEGORICAL:
            if c in X.columns:
                X[c] = X[c].astype(object)
        return X

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None,
            include_post_hoc: bool = False):
        self.include_post_hoc = include_post_hoc
        X = self._design(train)
        y = train[TARGET].astype(int)
        self.pipe_, self._nums, self._cats = self._build(X)
        self.pipe_.fit(X, y)
        self.columns_ = list(X.columns)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self._design(df)[self.columns_]
        return self.pipe_.predict_proba(X)[:, 1]

    def coefficients(self, top: int = 25) -> pd.DataFrame:
        """Standardised coefficients as log-odds.

        Read as association, never as effect: the features are correlated with
        one another (distance with destination state, freight with weight), so
        a coefficient is the association of that feature holding the others
        fixed *in this sample*, not what would happen if it were changed.
        """
        names = self.pipe_.named_steps["pre"].get_feature_names_out()
        coefs = self.pipe_.named_steps["clf"].coef_.ravel()
        out = pd.DataFrame({"feature": names, "log_odds": coefs})
        out["abs"] = out["log_odds"].abs()
        out["odds_ratio"] = np.exp(out["log_odds"])
        return out.sort_values("abs", ascending=False).head(top).drop(columns="abs").reset_index(drop=True)


class LGBMModel:
    """LightGBM binary classifier with early stopping on a chronological
    validation slice."""

    def __init__(self, name: str = "lgbm", include_post_hoc: bool = False,
                 params: dict | None = None, num_boost_round: int = 1200, seed: int = 7):
        self.name = name
        self.include_post_hoc = include_post_hoc
        self.num_boost_round = num_boost_round
        self.params = {
            "objective": "binary", "metric": "average_precision",
            # Deliberately low capacity. At 63 leaves the model fits the
            # training period's regime and loses to plain logistic regression
            # out of time; the late rate itself falls from 7.9% to 3.5% across
            # this window, so there is far less stable signal here than the
            # row count suggests. Chosen on the validation split.
            "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 300,
            "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 1,
            "lambda_l2": 20.0, "verbosity": -1, "seed": seed,
            **(params or {}),
        }
        self.best_iteration_: int | None = None

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None):
        import lightgbm as lgb

        Xtr, ytr = feature_frame(train, include_post_hoc=self.include_post_hoc)
        self.columns_ = list(Xtr.columns)
        dtr = lgb.Dataset(Xtr, label=ytr)
        callbacks, valid_sets = [], None
        if valid is not None and len(valid):
            Xva, yva = feature_frame(valid, include_post_hoc=self.include_post_hoc)
            valid_sets = [lgb.Dataset(Xva[self.columns_], label=yva, reference=dtr)]
            callbacks = [lgb.early_stopping(80, verbose=False)]
        self.booster_ = lgb.train(self.params, dtr, num_boost_round=self.num_boost_round,
                                  valid_sets=valid_sets, callbacks=callbacks)
        self.best_iteration_ = self.booster_.best_iteration or self.num_boost_round
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = feature_frame(df, include_post_hoc=self.include_post_hoc)
        return self.booster_.predict(X[self.columns_], num_iteration=self.best_iteration_)

    def importance(self, top: int = 25) -> pd.DataFrame:
        imp = pd.DataFrame({
            "feature": self.columns_,
            "gain": self.booster_.feature_importance("gain"),
            "splits": self.booster_.feature_importance("split"),
        })
        imp["gain_pct"] = 100 * imp["gain"] / imp["gain"].sum()
        return imp.sort_values("gain", ascending=False).head(top).reset_index(drop=True)


def permutation_importance(model, df: pd.DataFrame, metric, *, n_repeats: int = 3,
                           seed: int = 7, top: int | None = None) -> pd.DataFrame:
    """Permutation importance on a held-out frame.

    Preferred to split-gain for the write-up: gain measures how often the
    training procedure found a column useful, permutation importance measures
    how much held-out performance actually depends on it. Correlated features
    still share credit -- distance and destination state carry much the same
    information here -- so the ranking is read as a group, not a league table.
    """
    rng = np.random.default_rng(seed)
    y = df[TARGET].to_numpy(int)
    base = metric(y, model.predict_proba(df))
    cols = [c for c in model.columns_ if c in df.columns]
    rows = []
    for c in cols:
        drops = []
        for _ in range(n_repeats):
            shuffled = df.copy()
            shuffled[c] = rng.permutation(shuffled[c].to_numpy())
            drops.append(base - metric(y, model.predict_proba(shuffled)))
        rows.append({"feature": c, "drop_mean": float(np.mean(drops)),
                     "drop_std": float(np.std(drops))})
    out = pd.DataFrame(rows).sort_values("drop_mean", ascending=False).reset_index(drop=True)
    return out.head(top) if top else out
