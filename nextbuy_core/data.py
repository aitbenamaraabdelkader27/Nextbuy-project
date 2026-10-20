"""Portable, bounded data loading with explicit relational validation."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE_KEYS = {
    "orders": ["order_id"],
    "products": ["product_id"],
    "aisles": ["aisle_id"],
    "departments": ["department_id"],
    "order_products": ["order_id", "product_id"],
}
REQUIRED = {
    "orders": [
        "order_id",
        "user_id",
        "order_number",
        "order_dow",
        "order_hour_of_day",
        "days_since_prior_order",
    ],
    "products": ["product_id", "product_name", "aisle_id", "department_id"],
    "aisles": ["aisle_id", "aisle"],
    "departments": ["department_id", "department"],
    "order_products": ["order_id", "product_id", "add_to_cart_order", "reordered"],
}


def available_sources() -> dict[str, Path]:
    sources = {"Synthetic demo · generated data": ROOT / "data/demo"}
    if (ROOT / "data/local/orders.csv").exists():
        sources = {"Instacart · local cohort": ROOT / "data/local", **sources}
    return sources


def load_tables(directory: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name, columns in REQUIRED.items():
        path = directory / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name} in {directory}. See README data preparation."
            )
        frame = pd.read_csv(path)
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{name}: empty table")
        for col in columns:
            if col not in {"product_name", "aisle", "department"}:
                frame[col] = pd.to_numeric(frame[col], errors="raise")
                if col != "days_since_prior_order":
                    if frame[col].isna().any() or (frame[col] % 1 != 0).any():
                        raise ValueError(f"{name}.{col}: expected non-null integers")
                    frame[col] = frame[col].astype("int32")
        if frame[TABLE_KEYS[name]].isna().any().any():
            raise ValueError(f"{name}: null primary key")
        if frame.duplicated(TABLE_KEYS[name]).any():
            raise ValueError(f"{name}: duplicate primary key")
        tables[name] = frame
    validate_tables(tables)
    return tables


def validate_tables(tables: dict[str, pd.DataFrame]) -> None:
    relations = [
        ("order_products", "orders", "order_id"),
        ("order_products", "products", "product_id"),
        ("products", "aisles", "aisle_id"),
        ("products", "departments", "department_id"),
    ]
    for child, parent, key in relations:
        if not tables[child][key].isin(tables[parent][key]).all():
            raise ValueError(f"Orphan {key} in {child}")
    orders, lines = tables["orders"], tables["order_products"]
    for col, low, high in [
        ("order_hour_of_day", 0, 23),
        ("order_dow", 0, 6),
        ("order_number", 1, 100),
    ]:
        if not orders[col].between(low, high).all():
            raise ValueError(f"Invalid orders.{col}")
    days = orders["days_since_prior_order"]
    if not days.dropna().between(0, 30).all():
        raise ValueError("days_since_prior_order must be between 0 and 30 (capped)")
    if (days.isna() & orders["order_number"].ne(1)).any():
        raise ValueError("Missing elapsed days outside a first order")
    if not lines["reordered"].isin([0, 1]).all():
        raise ValueError("reordered must be binary")
    if not lines["add_to_cart_order"].ge(1).all():
        raise ValueError("Invalid cart position")
    positions = lines.groupby("order_id").add_to_cart_order.agg(["min", "max", "size", "nunique"])
    if (
        (positions["min"] != 1)
        | (positions["max"] != positions["size"])
        | (positions["nunique"] != positions["size"])
    ).any():
        raise ValueError("Non-contiguous cart positions: incomplete or duplicate basket lines")
    if orders.duplicated(["user_id", "order_number"]).any():
        raise ValueError("Duplicate user order sequence")


def merge_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    products = tables["products"].merge(tables["aisles"], on="aisle_id", validate="many_to_one")
    products = products.merge(tables["departments"], on="department_id", validate="many_to_one")
    frame = tables["order_products"].merge(tables["orders"], on="order_id", validate="many_to_one")
    frame = frame.merge(products, on="product_id", validate="many_to_one")
    for col in ["product_name", "department", "aisle"]:
        if frame[col].isna().any():
            raise ValueError(f"Missing labels: {col}")
        frame[col] = frame[col].astype("category")
    return frame


def filter_orders(frame: pd.DataFrame, days: list[int], hours: tuple[int, int]) -> pd.DataFrame:
    """Order-level filters retain every observed line of selected baskets."""
    return frame.loc[frame.order_dow.isin(days) & frame.order_hour_of_day.between(*hours)].copy()


def quality_report(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "table": name,
                "rows": len(df),
                "columns": len(df.columns),
                "null_cells": int(df.isna().sum().sum()),
                "duplicate_keys": int(df.duplicated(TABLE_KEYS[name]).sum()),
            }
            for name, df in tables.items()
        ]
    )
