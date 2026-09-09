"""Render the figures for the report.

Chart choices follow one rule: show the uncertainty wherever a reader might
otherwise rank things that are not distinguishable. Segment rates carry Wilson
intervals, the SMR panel is a forest plot with a reference line at 1.0, and the
reliability curve carries per-bin intervals -- without them a wobbly curve on
40 orders reads as miscalibration when it is just noise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

INK, ACCENT, MUTED, BAD = "#1b2430", "#2f6f9f", "#9aa7b4", "#b4462f"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
    "axes.titlesize": 10, "axes.titleweight": "bold",
})


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def fig_cohort(rep: Path, fig_dir: Path) -> None:
    d = pd.read_csv(rep / "cohort_month.csv", parse_dates=["cohort_month"])
    d = d[d.n_orders >= 100]
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.6, 5.2), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1.4]})
    a.fill_between(d.cohort_month, d.late_rate_lo, d.late_rate_hi, color=ACCENT, alpha=0.18)
    a.plot(d.cohort_month, d.late_rate, color=ACCENT, lw=1.8, marker="o", ms=3)
    a.set_ylabel("late rate")
    a.set_title("Late rate by purchase month (95% Wilson interval)")
    a.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    b.plot(d.cohort_month, d.avg_promise_days, color=MUTED, lw=1.6, label="promised days")
    b.plot(d.cohort_month, d.avg_actual_days, color=INK, lw=1.6, label="actual days")
    b.fill_between(d.cohort_month, d.avg_actual_days, d.avg_promise_days,
                   color=MUTED, alpha=0.20)
    b.set_ylabel("days"); b.legend(frameon=False, fontsize=8, ncol=2)
    b.set_title("The promise is padded: the gap is the buffer Olist quotes")
    _save(fig, fig_dir / "01_cohort.png")


def fig_distance(rep: Path, fig_dir: Path) -> None:
    d = pd.read_csv(rep / "segment_distance.csv")
    d = d[d.distance_band != "unknown"]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = np.arange(len(d))
    err = np.vstack([d.late_rate - d.late_rate_lo, d.late_rate_hi - d.late_rate])
    ax.bar(x, d.late_rate, color=ACCENT, width=0.62)
    ax.errorbar(x, d.late_rate, yerr=err, fmt="none", ecolor=INK, capsize=3, lw=1)
    ax.set_xticks(x, [s.split(". ")[-1] for s in d.distance_band])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("late rate"); ax.set_xlabel("seller to customer distance")
    ax.set_title("Late rate rises monotonically with distance")
    for xi, (r, n) in enumerate(zip(d.late_rate, d.n_orders)):
        ax.text(xi, r + 0.006, f"n={n:,}", ha="center", fontsize=7, color=MUTED)
    _save(fig, fig_dir / "02_distance.png")


def fig_smr(rep: Path, fig_dir: Path) -> None:
    d = pd.read_csv(rep / "state_smr.csv").sort_values("smr")
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    y = np.arange(len(d))
    colors = [BAD if lo > 1 else (ACCENT if hi < 1 else MUTED)
              for lo, hi in zip(d.smr_lo, d.smr_hi)]
    ax.hlines(y, d.smr_lo, d.smr_hi, color=colors, lw=2.2)
    ax.scatter(d.smr, y, color=colors, s=18, zorder=3)
    ax.axvline(1.0, color=INK, lw=1, ls="--")
    ax.set_yticks(y, d.customer_state, fontsize=8)
    ax.set_xlabel("SMR (observed / expected late orders, 95% Byar interval)")
    ax.set_title("Destination performance, adjusted for distance and period\n"
                 "right of the line = worse than comparable orders elsewhere")
    _save(fig, fig_dir / "03_state_smr.png")


def fig_attribution(rep: Path, fig_dir: Path) -> None:
    d = pd.read_csv(rep / "delay_attribution.csv")
    d = d[d.distance_band != "unknown"]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    x = np.arange(len(d))
    ax.bar(x, d.pct_seller_side, color=BAD, width=0.62, label="seller side (order to handoff)")
    ax.bar(x, d.pct_carrier_side, bottom=d.pct_seller_side, color=ACCENT, width=0.62,
           label="carrier side (handoff to door)")
    ax.set_xticks(x, [s.split(". ")[-1] for s in d.distance_band])
    ax.set_ylabel("share of excess days"); ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2)
    ax.set_title("Where the lost days sit, against the on-time median of the same band")
    for xi, v in enumerate(d.pct_carrier_side):
        ax.text(xi, 100 - v / 2, f"{v:.0f}%", ha="center", color="white", fontsize=8)
    _save(fig, fig_dir / "04_attribution.png")


def fig_calibration(rep: Path, fig_dir: Path) -> None:
    raw = pd.read_csv(rep / "reliability_raw.csv")
    cal = pd.read_csv(rep / "reliability_calibrated.csv")
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    lim = max(raw.mean_pred.max(), cal.mean_pred.max(), raw.frac_pos.max()) * 1.08
    ax.plot([0, lim], [0, lim], color=MUTED, ls="--", lw=1, label="perfect")
    for d, c, lbl in ((raw, BAD, "raw"), (cal, ACCENT, "isotonic, fitted on validation")):
        ax.errorbar(d.mean_pred, d.frac_pos,
                    yerr=np.vstack([d.frac_pos - d.lo, d.hi - d.frac_pos]),
                    fmt="o-", color=c, ms=4, lw=1.5, capsize=2, label=lbl)
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Reliability on the test period\nthe raw curve sits below the "
                 "diagonal: the base rate fell")
    _save(fig, fig_dir / "05_calibration.png")


def fig_lift(rep: Path, fig_dir: Path) -> None:
    d = pd.read_csv(rep / "lift_table.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    a.bar(d.decile, d.lift, color=[ACCENT if v >= 1 else MUTED for v in d.lift], width=0.68)
    a.axhline(1.0, color=INK, lw=1, ls="--")
    a.set_xlabel("risk decile (1 = riskiest)"); a.set_ylabel("lift vs base rate")
    a.set_title("Decile lift")
    b.plot([0] + list(d.cum_flag_rate), [0] + list(d.cum_recall), color=ACCENT, lw=2, marker="o", ms=3)
    b.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    b.set_xlabel("share of orders flagged"); b.set_ylabel("share of late orders caught")
    b.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    b.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    b.set_title("Capture curve")
    _save(fig, fig_dir / "06_lift.png")


def fig_models(rep: Path, fig_dir: Path) -> None:
    m = pd.read_csv(rep / "metrics_models.csv")
    t = m[m.split == "test"].copy()
    base = float(t.base_rate.iloc[0])
    t = t.sort_values("pr_auc")
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    colors = [BAD if "LEAKY" in n else ACCENT for n in t.model]
    ax.barh(t.model, t.pr_auc, color=colors, height=0.62)
    ax.axvline(base, color=INK, lw=1.2, ls="--")
    ax.text(base, -0.7, f" base rate {base:.3f}", fontsize=8, color=INK, va="top")
    ax.set_xlabel("PR-AUC on the test period")
    ax.set_title("The leaky model is the point of the chart, not the winner")
    for i, v in enumerate(t.pr_auc):
        ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=8)
    _save(fig, fig_dir / "07_models.png")


def fig_severity(rep: Path, fig_dir: Path) -> None:
    d = pd.read_csv(rep / "late_severity.csv")
    d = d[d.severity != "on time"]
    on_time = pd.read_csv(rep / "late_severity.csv").query("severity == 'on time'")
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = np.arange(len(d))
    ax.bar(x, d.pct_1_or_2_star, color=BAD, width=0.62)
    ax.axhline(float(on_time.pct_1_or_2_star.iloc[0]), color=INK, ls="--", lw=1.2)
    ax.text(len(d) - 0.4, float(on_time.pct_1_or_2_star.iloc[0]) + 0.02,
            "on-time orders", ha="right", fontsize=8)
    ax.set_xticks(x, [s.split(". ")[-1] for s in d.severity], fontsize=8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("share scored 1 or 2 stars")
    ax.set_title("Severity matters: lateness and low scores are strongly associated\n"
                 "(observational -- association, not a measured causal effect)")
    _save(fig, fig_dir / "08_severity.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()
    rep = Path(args.reports)
    fig_dir = rep / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("writing figures:")
    for fn in (fig_cohort, fig_distance, fig_smr, fig_attribution,
               fig_calibration, fig_lift, fig_models, fig_severity):
        try:
            fn(rep, fig_dir)
        except FileNotFoundError as e:
            print(f"  skipped {fn.__name__}: {e}")
