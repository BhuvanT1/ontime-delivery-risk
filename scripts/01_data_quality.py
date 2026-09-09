"""Print the data-quality report and write it to reports/data_quality.csv."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from otd import db  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    con = db.connect(args.db, read_only=True)
    dq = db.q(con, "SELECT * FROM dq_checks")
    con.close()

    dq.to_csv(out / "data_quality.csv", index=False)
    for sev in ("error", "warn", "info"):
        part = dq[dq.severity == sev]
        if part.empty:
            continue
        print(f"\n=== {sev.upper()} ({len(part)} checks) ===")
        print(part[["check_name", "n_fail", "n_total", "fail_pct", "note"]].to_string(index=False))

    failing = dq[(dq.severity == "error") & (dq.n_fail > 0)]
    print(f"\n{len(failing)} error-level check(s) with failures.")
