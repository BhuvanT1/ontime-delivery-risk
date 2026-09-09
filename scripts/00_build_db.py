"""Build the DuckDB warehouse from the CSV export.

Run:  python scripts/00_build_db.py [--db PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from otd import db  # noqa: E402

RAW = Path("data/raw")
EXPECTED = [
    "olist_customers_dataset.csv", "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv", "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv", "olist_orders_dataset.csv",
    "olist_products_dataset.csv", "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    args = ap.parse_args()

    missing = [f for f in EXPECTED if not (RAW / f).exists()]
    if missing:
        raise SystemExit(
            f"missing {len(missing)} file(s) in {RAW}: {missing}\n"
            "See data/README.md for where to get them."
        )
    path = db.build(args.db)
    print(f"built {path}")
