"""Statistics use explicit units and complete observed baskets."""

from collections import Counter
from itertools import combinations

import pandas as pd


def kpis(frame: pd.DataFrame) -> dict[str, float]:
    orders = frame.order_id.nunique()
    return {
        "orders": orders,
        "customers": frame.user_id.nunique(),
        "items": len(frame),
        "repeat_rate": float(frame.reordered.mean()) if len(frame) else 0.0,
        "basket_size": len(frame) / orders if orders else 0.0,
    }


def product_statistics(frame: pd.DataFrame, group: str = "product_name") -> pd.DataFrame:
    return (
        frame.assign(first=frame.add_to_cart_order.eq(1))
        .groupby(group, observed=True)
        .agg(
            items=("order_id", "size"),
            repeat_rate=("reordered", "mean"),
            first_count=("first", "sum"),
            first_ratio=("first", "mean"),
        )
        .reset_index()
    )


def order_activity(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        frame.drop_duplicates("order_id")
        .groupby(column, observed=True)
        .size()
        .rename("orders")
        .reset_index()
    )


def association_pairs(frame: pd.DataFrame, top_n: int = 30, min_count: int = 5) -> pd.DataFrame:
    """Exact pair mining among top-N products; denominator includes ALL selected orders."""
    columns = ["product_a", "product_b", "count", "support", "confidence_a_to_b", "lift"]
    n_orders = frame.order_id.nunique()
    if not n_orders:
        return pd.DataFrame(columns=columns)
    top = frame.product_id.value_counts().head(top_n).index
    selected = frame.loc[frame.product_id.isin(top), ["order_id", "product_id"]].drop_duplicates()
    counts = selected.product_id.value_counts()
    pairs = Counter()
    for basket in selected.groupby("order_id", observed=True).product_id:
        pairs.update(combinations(sorted(basket[1].tolist()), 2))
    names = frame.drop_duplicates("product_id").set_index("product_id").product_name.to_dict()
    rows = []
    for (a, b), count in pairs.items():
        if count >= min_count:
            rows.append(
                {
                    "product_a": names[a],
                    "product_b": names[b],
                    "count": count,
                    "support": count / n_orders,
                    "confidence_a_to_b": count / counts[a],
                    "lift": count * n_orders / (counts[a] * counts[b]),
                }
            )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["count", "product_a", "product_b"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
