"""Shared fixtures.

The SQL is the deliverable in this project, so the fixtures build a real -- if
tiny -- warehouse from the repository's own DDL and run the real
20_order_fact.sql against it. Testing the Python while stubbing the SQL would
leave the part that carries the analysis unverified.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write_csvs(raw: Path, seed: int = 3) -> None:
    """A miniature Olist export: 3 sellers, 2 states, 400 orders."""
    rng = np.random.default_rng(seed)
    n = 400
    start = pd.Timestamp("2017-01-01")
    purchased = start + pd.to_timedelta(rng.integers(0, 400, n), unit="D") \
        + pd.to_timedelta(rng.integers(0, 86400, n), unit="s")
    purchased = pd.Series(sorted(purchased))

    sellers = pd.DataFrame({
        "seller_id": ["s1", "s2", "s3"],
        "seller_zip_code_prefix": ["01001", "20001", "40001"],
        "seller_city": ["sao paulo", "rio", "salvador"],
        "seller_state": ["SP", "RJ", "BA"],
    })
    # State follows the zip prefix. Drawing them independently would put a
    # customer in RJ at Sao Paulo's coordinates, and then "different state,
    # zero distance" rows appear that no real export contains.
    zip_to_state = {"01001": "SP", "20001": "RJ", "40001": "BA"}
    cust_zip = rng.choice(list(zip_to_state), n)
    customers = pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(n)],
        "customer_unique_id": [f"u{i % 300}" for i in range(n)],
        "customer_zip_code_prefix": cust_zip,
        "customer_city": "x",
        "customer_state": [zip_to_state[z] for z in cust_zip],
    })
    products = pd.DataFrame({
        "product_id": ["p1", "p2"], "product_category_name": ["cama_mesa_banho", "beleza_saude"],
        "product_name_lenght": [40, 45], "product_description_lenght": [200, 300],
        "product_photos_qty": [2, 3], "product_weight_g": [500.0, 1200.0],
        "product_length_cm": [20.0, 30.0], "product_height_cm": [10.0, 12.0],
        "product_width_cm": [15.0, 18.0],
    })
    seller_of = rng.choice(["s1", "s2", "s3"], n, p=[0.5, 0.3, 0.2])
    # s3 is deliberately unreliable, so the as-of history test has a signal.
    late_p = np.where(seller_of == "s3", 0.6, 0.1)
    is_late = rng.random(n) < late_p
    promise = 20
    delivered = purchased + pd.to_timedelta(
        np.where(is_late, promise + rng.integers(1, 8, n), rng.integers(3, promise - 2, n)),
        unit="D")
    orders = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(n)],
        "customer_id": customers.customer_id,
        "order_status": "delivered",
        "order_purchase_timestamp": purchased.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
        "order_approved_at": (purchased + pd.Timedelta(hours=2)).dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
        "order_delivered_carrier_date": (purchased + pd.Timedelta(days=2)).dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
        "order_delivered_customer_date": delivered.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
        "order_estimated_delivery_date": (purchased.dt.normalize() + pd.Timedelta(days=promise)).dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
    })
    items = pd.DataFrame({
        "order_id": orders.order_id, "order_item_id": 1,
        "product_id": rng.choice(["p1", "p2"], n), "seller_id": seller_of,
        "shipping_limit_date": (purchased + pd.Timedelta(days=5)).dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
        "price": np.round(rng.uniform(10, 300, n), 2),
        "freight_value": np.round(rng.uniform(5, 40, n), 2),
    })
    payments = pd.DataFrame({
        "order_id": orders.order_id, "payment_sequential": 1,
        "payment_type": rng.choice(["credit_card", "boleto"], n),
        "payment_installments": rng.integers(1, 8, n),
        "payment_value": items.price + items.freight_value,
    })
    reviews = pd.DataFrame({
        "review_id": [f"r{i}" for i in range(n)], "order_id": orders.order_id,
        "review_score": np.where(is_late, rng.integers(1, 3, n), rng.integers(4, 6, n)),
        "review_comment_title": "", "review_comment_message": "",
        "review_creation_date": delivered.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
        "review_answer_timestamp": delivered.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
    })
    geo = pd.DataFrame({
        "geolocation_zip_code_prefix": ["01001", "20001", "40001"] * 4,
        "geolocation_lat": [-23.55, -22.91, -12.97] * 4,
        "geolocation_lng": [-46.63, -43.17, -38.50] * 4,
        "geolocation_city": ["sao paulo", "rio", "salvador"] * 4,
        "geolocation_state": ["SP", "RJ", "BA"] * 4,
    })
    trans = pd.DataFrame({
        "product_category_name": ["cama_mesa_banho", "beleza_saude"],
        "product_category_name_english": ["bed_bath_table", "health_beauty"],
    })

    raw.mkdir(parents=True, exist_ok=True)
    for name, d in [
        ("olist_customers_dataset", customers), ("olist_sellers_dataset", sellers),
        ("olist_products_dataset", products), ("olist_orders_dataset", orders),
        ("olist_order_items_dataset", items), ("olist_order_payments_dataset", payments),
        ("olist_order_reviews_dataset", reviews), ("olist_geolocation_dataset", geo),
        ("product_category_name_translation", trans),
    ]:
        d.to_csv(raw / f"{name}.csv", index=False)


@pytest.fixture(scope="session")
def mini_db(tmp_path_factory):
    """A real DuckDB built from the repo's own SQL over a synthetic export."""
    import duckdb

    work = tmp_path_factory.mktemp("mini")
    _write_csvs(work / "data" / "raw")
    con = duckdb.connect(str(work / "mini.duckdb"))
    cwd = Path.cwd()
    import os
    os.chdir(work)
    try:
        for f in ("00_schema.sql", "01_load.sql", "10_data_quality.sql",
                  "20_order_fact.sql", "30_segments.sql", "40_root_cause.sql"):
            con.execute((ROOT / "sql" / f).read_text(encoding="utf-8"))
    finally:
        os.chdir(cwd)
    yield con
    con.close()


@pytest.fixture(scope="session")
def mini_fact(mini_db) -> pd.DataFrame:
    return mini_db.execute("SELECT * FROM order_fact ORDER BY purchased_at, order_id").fetchdf()
