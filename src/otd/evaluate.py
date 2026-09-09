"""Classification metrics, calibration and lift.

The label is imbalanced -- around 7% of delivered orders are late, and under 4%
in the final months -- and that shapes which numbers are worth reporting.

ROC-AUC is nearly uninformative about usefulness under this much imbalance: a
model can hold an AUC of 0.75 and still put mostly on-time orders at the top of
its ranking, because the false-positive rate is measured against a huge
negative class. PR-AUC (average precision) is reported alongside it and has a
meaningful floor -- the base rate -- so "0.21" can be read against "0.035".

Calibration is reported separately from discrimination because they fail
independently, and this dataset makes the point: the late rate falls from 7.9%
in the training window to 3.5% in the test window, so a model that ranks orders
perfectly will still be badly over-confident in absolute terms. Ranking metrics
cannot see that; Brier score, ECE and the reliability curve can.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "roc_auc", "pr_auc", "brier", "log_loss", "ece", "reliability_table",
    "lift_table", "recall_at_k", "summarise", "IsotonicCalibrator",
]


def _prep(y, p):
    y = np.asarray(y, dtype=int).ravel()
    p = np.clip(np.asarray(p, dtype=float).ravel(), 1e-9, 1 - 1e-9)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {p.shape}")
    return y, p


def roc_auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    y, p = _prep(y, p)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def pr_auc(y, p) -> float:
    """Average precision. Its floor is the positive rate, so always read it
    next to the base rate rather than against 0.5."""
    from sklearn.metrics import average_precision_score
    y, p = _prep(y, p)
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def brier(y, p) -> float:
    y, p = _prep(y, p)
    return float(np.mean((p - y) ** 2))


def log_loss(y, p) -> float:
    y, p = _prep(y, p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, n_bins: int = 10, strategy: str = "quantile") -> float:
    """Expected calibration error: mean gap between predicted and observed
    frequency, weighted by bin size. Quantile bins by default -- uniform bins
    put almost every order in the first bin at this base rate and report a
    flattering number."""
    y, p = _prep(y, p)
    tab = reliability_table(y, p, n_bins=n_bins, strategy=strategy)
    return float(np.sum(tab["n"] / tab["n"].sum() * (tab["mean_pred"] - tab["frac_pos"]).abs()))


def reliability_table(y, p, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    y, p = _prep(y, p)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    df = pd.DataFrame({"bin": idx, "y": y, "p": p})
    tab = df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"),
                                frac_pos=("y", "mean")).reset_index()
    # Wilson interval, so a bin of 40 orders is not read as precisely as one of
    # 4000 when judging whether the curve really departs from the diagonal.
    k = tab["frac_pos"] * tab["n"]
    n = tab["n"]
    z = 1.96
    centre = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt((k / n * (1 - k / n) + z * z / (4 * n)) / n) / (1 + z * z / n)
    tab["lo"] = (centre - half).clip(lower=0)
    tab["hi"] = (centre + half).clip(upper=1)
    return tab


def lift_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Decile table, ordered by predicted risk.

    This is the table an operations reader will actually use: if we can only
    act on 10% of orders, what share of the late ones do we catch?
    """
    y, p = _prep(y, p)
    order = np.argsort(-p)
    y, p = y[order], p[order]
    n = len(y)
    base = y.mean()
    edges = [int(round(n * i / n_bins)) for i in range(n_bins + 1)]
    rows = []
    for i in range(n_bins):
        s, e = edges[i], edges[i + 1]
        if e <= s:
            continue
        seg = y[s:e]
        rows.append({
            "decile": i + 1, "n": e - s,
            "mean_pred": float(p[s:e].mean()),
            "late_rate": float(seg.mean()),
            "lift": float(seg.mean() / base) if base > 0 else np.nan,
            "n_late": int(seg.sum()),
            "cum_recall": float(y[:e].sum() / y.sum()) if y.sum() else np.nan,
            "cum_flag_rate": (e) / n,
        })
    return pd.DataFrame(rows)


def recall_at_k(y, p, k: float) -> float:
    """Share of late orders caught by flagging the riskiest ``k`` of volume."""
    y, p = _prep(y, p)
    n_flag = max(1, int(round(len(y) * k)))
    idx = np.argsort(-p)[:n_flag]
    return float(y[idx].sum() / y.sum()) if y.sum() else float("nan")


def summarise(y, p, *, name: str = "") -> dict:
    y, p = _prep(y, p)
    return {
        "model": name, "n": len(y), "base_rate": float(y.mean()),
        "roc_auc": roc_auc(y, p), "pr_auc": pr_auc(y, p),
        "brier": brier(y, p), "log_loss": log_loss(y, p), "ece": ece(y, p),
        "mean_pred": float(p.mean()),
        "recall@5%": recall_at_k(y, p, 0.05),
        "recall@10%": recall_at_k(y, p, 0.10),
        "recall@20%": recall_at_k(y, p, 0.20),
    }


class IsotonicCalibrator:
    """Isotonic regression fitted on a held-out, chronologically later slice.

    Isotonic rather than Platt scaling because the miscalibration here is a
    shift in the base rate rather than a smooth distortion of the log-odds, and
    isotonic can absorb that without assuming a shape. Being non-decreasing it
    can never reverse two scores, so discrimination is preserved in the sense
    that matters -- but it is not strictly increasing: it maps whole runs of
    scores onto one plateau value, and those new ties move ROC-AUC by a small
    amount, because a tied pair scores half a concordance instead of one. So
    the ranking is intact, the ranking metrics shift slightly, and the
    probability-quality metrics move a great deal. Calibration and
    discrimination remain separate problems.
    """

    def __init__(self):
        self.iso_ = None

    def fit(self, y, p):
        from sklearn.isotonic import IsotonicRegression
        y, p = _prep(y, p)
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.iso_.fit(p, y)
        return self

    def transform(self, p) -> np.ndarray:
        if self.iso_ is None:
            raise RuntimeError("call fit() first")
        return np.clip(self.iso_.predict(np.asarray(p, dtype=float).ravel()), 1e-9, 1 - 1e-9)
