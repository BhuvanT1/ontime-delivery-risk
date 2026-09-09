"""Run the cohort, segment and root-cause views and write them to reports/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from otd import db  # noqa: E402

VIEWS = {
    "cohort_month": "v_cohort_month",
    "segment_state": "v_segment_state",
    "segment_distance": "v_segment_distance",
    "segment_category": "v_segment_category",
    "segment_seller": "v_segment_seller",
    "customer_repeat": "v_customer_repeat",
    "stage_timing": "v_stage_timing",
    "stage_timing_by_distance": "v_stage_timing_by_distance",
    "delay_attribution": "v_delay_attribution",
    "state_smr": "v_state_smr",
    "late_severity": "v_late_severity",
    "late_review_assoc": "v_late_review_assoc",
}
SHOW = ["segment_distance", "state_smr", "delay_attribution", "late_severity",
        "late_review_assoc", "customer_repeat"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    con = db.connect(args.db, read_only=True)
    for name, view in VIEWS.items():
        d = db.q(con, f"SELECT * FROM {view}")
        d.to_csv(out / f"{name}.csv", index=False)
        if name in SHOW:
            print(f"\n=== {name} ===")
            print(d.head(12).to_string(index=False))

    print("\n=== seller segments ===")
    print(db.q(con, """
        SELECT seller_segment, count(*) AS n_sellers, sum(n_orders) AS n_orders,
               round(sum(n_late) * 1.0 / sum(n_orders), 4) AS late_rate,
               round(sum(gmv), 0) AS gmv
        FROM v_segment_seller GROUP BY 1 ORDER BY n_orders DESC
    """).to_string(index=False))
    con.close()
    print(f"\nwrote {len(VIEWS)} tables to {out}/")
