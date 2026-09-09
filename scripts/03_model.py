"""Fit the model ladder, calibrate the champion, and choose an operating point.

Protocol, in order, because the order is what makes the final number mean
anything:

  1. Split chronologically into train / validation / test.
  2. Run a range-drift check on the features before fitting anything.
  3. Fit the ladder on train, with early stopping on validation.
  4. Pick the champion by VALIDATION PR-AUC.
  5. Fit the calibrator on validation.
  6. Score the test period once, and report it.

Steps 4 and 5 exist to keep the test window untouched until the end. Choosing a
model or a threshold by test performance turns the reported figure into a
training metric wearing a disguise.

Run:  python scripts/03_model.py [--db PATH] [--out reports]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from otd import db, decision, evaluate as ev  # noqa: E402
from otd.data import (TARGET, assert_no_post_hoc, drift_report,  # noqa: E402
                      feature_columns, feature_frame, time_split)
from otd.models import (BaseRate, LGBMModel, LogisticModel, PromiseRule,  # noqa: E402
                        permutation_importance)

CANDIDATES = ["promise_rule", "logistic", "lgbm"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--skip-permutation", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    con = db.connect(args.db, read_only=True)
    df = db.load_order_fact(con)
    con.close()
    print(f"order_fact: {len(df):,} orders, late rate {df[TARGET].mean():.4f}")

    sp = time_split(df)
    print(sp.describe().to_string(index=False))
    sp.describe().to_csv(out / "split_summary.csv", index=False)

    Xtr, _ = feature_frame(sp.train)
    assert_no_post_hoc(Xtr.columns)

    drift = drift_report(sp.train, sp.test, feature_columns(df))
    drift.to_csv(out / "feature_drift.csv", index=False)
    if len(drift):
        print("\n=== feature range drift (train -> test) ===")
        print(drift.to_string(index=False))

    # ---- 3. the ladder -----------------------------------------------------
    models = {
        "base_rate": BaseRate().fit(sp.train),
        "promise_rule": PromiseRule().fit(sp.train),
        "logistic": LogisticModel().fit(sp.train),
        "lgbm": LGBMModel(name="lgbm").fit(sp.train, sp.valid),
        # Not a candidate. Fitted only to size the leak that comes from using
        # post-delivery columns, which most public notebooks on this dataset do
        # without saying so.
        "lgbm_LEAKY": LGBMModel(name="lgbm_LEAKY", include_post_hoc=True).fit(sp.train, sp.valid),
    }

    rows, preds_valid, preds_test = [], {}, {}
    for name, m in models.items():
        preds_valid[name] = m.predict_proba(sp.valid)
        preds_test[name] = m.predict_proba(sp.test)
        rows.append({"split": "valid", **ev.summarise(sp.valid[TARGET], preds_valid[name], name=name)})
        rows.append({"split": "test", **ev.summarise(sp.test[TARGET], preds_test[name], name=name)})
    metrics = pd.DataFrame(rows)

    # ---- 4. champion, chosen on validation ---------------------------------
    va = metrics[(metrics.split == "valid") & (metrics.model.isin(CANDIDATES))]
    champion = str(va.loc[va["pr_auc"].idxmax(), "model"])
    print("\n=== validation (selection happens here) ===")
    print(va[["model", "roc_auc", "pr_auc", "brier"]].to_string(index=False))
    print(f"champion: {champion}")

    # ---- 5. calibrate on validation ----------------------------------------
    cal = ev.IsotonicCalibrator().fit(sp.valid[TARGET], preds_valid[champion])
    p_te_cal = cal.transform(preds_test[champion])
    metrics = pd.concat([metrics, pd.DataFrame([
        {"split": "test", **ev.summarise(sp.test[TARGET], p_te_cal, name=f"{champion}_calibrated")}
    ])], ignore_index=True)
    metrics.to_csv(out / "metrics_models.csv", index=False)

    print("\n=== test (scored once) ===")
    print(metrics[metrics.split == "test"].drop(columns="split").to_string(index=False))

    # ---- 6. diagnostics ----------------------------------------------------
    y_te = sp.test[TARGET].to_numpy(int)
    ev.reliability_table(y_te, preds_test[champion]).assign(variant="raw").to_csv(
        out / "reliability_raw.csv", index=False)
    ev.reliability_table(y_te, p_te_cal).assign(variant="calibrated").to_csv(
        out / "reliability_calibrated.csv", index=False)
    lift = ev.lift_table(y_te, p_te_cal)
    lift.to_csv(out / "lift_table.csv", index=False)
    print("\n=== lift, calibrated champion, test ===")
    print(lift[["decile", "n", "mean_pred", "late_rate", "lift", "cum_recall"]].to_string(index=False))

    models["lgbm"].importance(top=30).to_csv(out / "importance_gain.csv", index=False)
    models["logistic"].coefficients(top=30).to_csv(out / "logistic_coefficients.csv", index=False)
    if not args.skip_permutation:
        perm = permutation_importance(models[champion], sp.test, ev.pr_auc, n_repeats=3)
        perm.to_csv(out / "importance_permutation.csv", index=False)
        print("\n=== permutation importance, top 10 (PR-AUC drop) ===")
        print(perm.head(10).to_string(index=False))

    # ---- decision layer ----------------------------------------------------
    cm = decision.CostModel()
    decision.sweep_thresholds(y_te, p_te_cal, cm).to_csv(out / "cost_sweep.csv", index=False)
    best = decision.best_threshold(y_te, p_te_cal, cm, max_flag_rate=0.20)
    req = decision.required_effectiveness(y_te, p_te_cal, best["threshold"], cm)
    decision.sensitivity(y_te, p_te_cal).to_csv(out / "sensitivity.csv", index=False)

    print("\n=== operating point, capacity <= 20% of orders ===")
    for k in ("threshold", "flag_rate", "recall", "precision", "saving_per_order", "saving_pct"):
        print(f"  {k:20s} {best[k]:.4f}")
    print(f"  break-even effectiveness the intervention must beat: {req:.3f}")
    print(f"  naive per-order threshold under these costs:         {cm.naive_threshold:.3f}")

    test_rows = metrics[metrics.split == "test"]
    summary = {
        "n_orders": int(len(df)),
        "late_rate_overall": float(df[TARGET].mean()),
        "champion": champion,
        "champion_selected_on": "validation PR-AUC",
        "split": {
            "train": int(len(sp.train)), "valid": int(len(sp.valid)), "test": int(len(sp.test)),
            "train_late_rate": float(sp.train[TARGET].mean()),
            "valid_late_rate": float(sp.valid[TARGET].mean()),
            "test_late_rate": float(sp.test[TARGET].mean()),
        },
        "test_metrics": {
            r["model"]: {k: float(r[k]) for k in
                         ("roc_auc", "pr_auc", "brier", "ece", "recall@10%", "recall@20%")}
            for _, r in test_rows.iterrows()
        },
        "operating_point": {k: (None if pd.isna(v) else float(v)) for k, v in best.items()},
        "required_effectiveness": float(req),
        "cost_model": {"cost_late": cm.cost_late, "cost_action": cm.cost_action,
                       "effectiveness": cm.effectiveness},
    }
    (out / "model_summary.json").write_text(json.dumps(summary, indent=2))
    np.save(out / "test_predictions.npy", p_te_cal)
    print(f"\nwrote {out}/model_summary.json")


if __name__ == "__main__":
    main()
