"""Database access.

DuckDB is used as the warehouse: it is a single file, needs no server, and
speaks standard SQL, so the same statements run against PostgreSQL with only
the read_csv_auto loader swapped for COPY. Keeping every query in .sql files
under sql/ rather than inlining strings in Python is deliberate -- the SQL is
the deliverable in this project, and it should be readable without a Python
interpreter in the way.
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = REPO_ROOT / "sql"
DEFAULT_DB = os.environ.get("OTD_DB", str(REPO_ROOT / "data" / "olist.duckdb"))


def connect(db_path: str | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path = db_path or DEFAULT_DB
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(path, read_only=read_only)


def run_sql_file(con: duckdb.DuckDBPyConnection, name: str) -> None:
    """Execute a .sql file from sql/ as a script."""
    path = SQL_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    con.execute(path.read_text(encoding="utf-8"))


def q(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def build(db_path: str | None = None, verbose: bool = True) -> str:
    """Create the warehouse from the CSV export, end to end."""
    con = connect(db_path)
    try:
        for f in ("00_schema.sql", "01_load.sql", "10_data_quality.sql",
                  "20_order_fact.sql", "30_segments.sql", "40_root_cause.sql"):
            if verbose:
                print(f"  running {f}")
            run_sql_file(con, f)
        n = q(con, "SELECT count(*) AS n FROM order_fact")["n"].iloc[0]
        if verbose:
            print(f"  order_fact: {n:,} rows")
    finally:
        con.close()
    return db_path or DEFAULT_DB


def load_order_fact(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return q(con, "SELECT * FROM order_fact ORDER BY purchased_at, order_id")
