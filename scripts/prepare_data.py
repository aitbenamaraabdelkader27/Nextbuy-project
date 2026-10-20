"""Generate redistributable synthetic data or extract a seeded local Instacart cohort."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nextbuy_core.data import ROOT


def generate_demo(destination: Path, users: int = 650) -> None:
    rng = np.random.default_rng(42)
    departments = pd.DataFrame(
        {
            "department_id": [1, 2, 3, 4],
            "department": ["produce", "dairy & eggs", "pantry", "beverages"],
        }
    )
    aisles = pd.DataFrame(
        {
            "aisle_id": [1, 2, 3, 4],
            "aisle": ["fresh produce", "chilled essentials", "cupboard", "drinks"],
        }
    )
    labels = [
        "Bananas",
        "Apples",
        "Avocados",
        "Tomatoes",
        "Spinach",
        "Carrots",
        "Milk",
        "Eggs",
        "Yogurt",
        "Butter",
        "Cheese",
        "Oat milk",
        "Bread",
        "Pasta",
        "Rice",
        "Olive oil",
        "Oats",
        "Peanut butter",
        "Coffee",
        "Tea",
        "Sparkling water",
        "Orange juice",
        "Cocoa",
        "Lemonade",
    ]
    products = pd.DataFrame(
        {
            "product_id": range(1, 25),
            "product_name": labels,
            "aisle_id": np.repeat(range(1, 5), 6),
            "department_id": np.repeat(range(1, 5), 6),
        }
    )
    orders, lines = [], []
    order_id = 0
    for user in range(1, users + 1):
        seen = set()
        preference = rng.dirichlet(np.ones(24) * 0.7)
        for sequence in range(1, int(rng.integers(5, 17))):
            order_id += 1
            orders.append(
                [
                    order_id,
                    user,
                    sequence,
                    int(rng.integers(7)),
                    int(np.clip(rng.normal(13, 4), 0, 23)),
                    np.nan if sequence == 1 else int(rng.integers(1, 31)),
                ]
            )
            basket = rng.choice(
                np.arange(1, 25), size=int(rng.integers(3, 12)), replace=False, p=preference
            ).tolist()
            # Explicit synthetic association; no claims about real customers.
            if 1 in basket and 7 not in basket and rng.random() < 0.6:
                basket.append(7)
            for position, product in enumerate(basket, 1):
                lines.append([order_id, product, position, int(product in seen)])
            seen.update(basket)
    tables = {
        "orders": pd.DataFrame(
            orders,
            columns=[
                "order_id",
                "user_id",
                "order_number",
                "order_dow",
                "order_hour_of_day",
                "days_since_prior_order",
            ],
        ),
        "order_products": pd.DataFrame(
            lines, columns=["order_id", "product_id", "add_to_cart_order", "reordered"]
        ),
        "products": products,
        "aisles": aisles,
        "departments": departments,
    }
    write_tables(tables, destination)
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "synthetic",
                "seed": 42,
                "users": users,
                "description": "Entirely generated; no real customer records.",
            },
            indent=2,
        )
    )


def write_tables(tables: dict, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(destination / f"{name}.csv", index=False)


def prepare_local(source: Path, destination: Path, users: int) -> None:
    orders = pd.read_csv(source / "orders.csv")
    # Remove exact duplicated records only; conflicting keys must fail downstream validation.
    orders = orders.drop_duplicates()
    invalid_order = (
        orders[["user_id", "order_number", "order_dow", "order_hour_of_day"]].isna().any(axis=1)
        | ~orders.order_hour_of_day.between(0, 23)
        | ~orders.order_dow.between(0, 6)
        | ~orders.order_number.between(1, 100)
        | (orders.days_since_prior_order.isna() & orders.order_number.ne(1))
    )
    excluded_source_orders = int(invalid_order.sum())
    orders = orders.loc[~invalid_order].copy()
    chosen = np.random.default_rng(42).choice(
        sorted(orders.user_id.dropna().unique()),
        min(users, orders.user_id.nunique()),
        replace=False,
    )
    orders = orders.loc[orders.user_id.isin(chosen)].copy()
    selected = set(orders.order_id)
    pieces = []
    for chunk in pd.read_csv(source / "order_products.csv", chunksize=250_000):
        pieces.append(chunk.loc[chunk.order_id.isin(selected)])
    lines = pd.concat(pieces, ignore_index=True).drop_duplicates()
    invalid_lines = (
        lines.isna().any(axis=1) | ~lines.reordered.isin([0, 1]) | ~lines.add_to_cart_order.ge(1)
    )
    bad_orders = set(lines.loc[invalid_lines, "order_id"])
    # Exclude entire affected baskets; never silently shorten a basket.
    lines = lines.loc[~lines.order_id.isin(bad_orders)].copy()
    positions = lines.groupby("order_id").add_to_cart_order.agg(["min", "max", "size", "nunique"])
    incomplete = positions.loc[
        (positions["min"] != 1)
        | (positions["max"] != positions["size"])
        | (positions["nunique"] != positions["size"])
    ].index
    bad_orders.update(incomplete)
    lines = lines.loc[~lines.order_id.isin(bad_orders)].copy()
    # Retain complete observed baskets, not orders without line coverage.
    orders = orders.loc[orders.order_id.isin(lines.order_id)]
    products = pd.read_csv(source / "products.csv").drop_duplicates()
    products = products.loc[products.product_id.isin(lines.product_id)]
    tables = {
        "orders": orders,
        "order_products": lines,
        "products": products,
        "aisles": pd.read_csv(source / "aisles.csv").drop_duplicates(),
        "departments": pd.read_csv(source / "departments.csv").drop_duplicates(),
    }
    write_tables(tables, destination)
    manifest = {
        "kind": "instacart_local_cohort",
        "seed": 42,
        "requested_users": users,
        "observed_users": orders.user_id.nunique(),
        "invalid_source_orders_excluded": excluded_source_orders,
        "invalid_cohort_baskets_excluded": len(bad_orders),
        "sampling": "Uniform users from orders, all available lines, exact duplicates removed",
        "files": {},
    }
    for path in source.glob("*.csv"):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest["files"][path.name] = digest.hexdigest()
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--users", type=int, default=3000)
    args = parser.parse_args()
    if args.users < 1:
        parser.error("--users must be positive")
    if args.local:
        prepare_local(ROOT / "nextbuy/datasets", ROOT / "data/local", args.users)
    else:
        generate_demo(ROOT / "data/demo")
