import pandas as pd
import pytest

from nextbuy_core.analytics import association_pairs, kpis, order_activity
from nextbuy_core.data import ROOT, filter_orders, load_tables, merge_tables


@pytest.fixture(scope="module")
def frame():
    return merge_tables(load_tables(ROOT / "data/demo"))


def test_loader_is_independent_of_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert len(load_tables(ROOT / "data/demo")["orders"]) > 0


def test_missing_file_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="orders.csv"):
        load_tables(tmp_path)


def test_order_counts_are_not_item_counts(frame):
    activity = order_activity(frame, "order_hour_of_day")
    assert activity.orders.sum() == frame.order_id.nunique()
    assert activity.orders.sum() < len(frame)


def test_temporal_filters_preserve_baskets(frame):
    filtered = filter_orders(frame, [0, 2], (9, 15))
    assert set(filtered.order_dow).issubset({0, 2})
    assert filtered.order_hour_of_day.between(9, 15).all()
    expected = frame.loc[frame.order_id.isin(filtered.order_id)].groupby("order_id").size()
    pd.testing.assert_series_equal(filtered.groupby("order_id").size(), expected)
    assert filter_orders(frame, [], (0, 23)).empty
    assert kpis(filtered)["items"] == len(filtered)


def test_pair_denominator_includes_baskets_without_top_products():
    frame = pd.DataFrame(
        {
            "order_id": [1, 1, 2, 2, 3, 4],
            "product_id": [1, 2, 1, 2, 1, 3],
            "product_name": ["A", "B", "A", "B", "A", "C"],
        }
    )
    pair = association_pairs(frame, top_n=2, min_count=1).iloc[0]
    assert pair["count"] == 2
    assert pair.support == 0.5
    assert pair.confidence_a_to_b == pytest.approx(2 / 3)
    assert pair.lift == pytest.approx(4 / 3)
    assert association_pairs(frame, min_count=10).empty


def test_schema_ranges_duplicates_and_foreign_keys(tmp_path):
    tables = load_tables(ROOT / "data/demo")

    def write():
        for name, table in tables.items():
            table.to_csv(tmp_path / f"{name}.csv", index=False)

    tables["order_products"].loc[0, "reordered"] = 3
    write()
    with pytest.raises(ValueError, match="binary"):
        load_tables(tmp_path)
    tables["order_products"].loc[0, "reordered"] = 0
    tables["order_products"].loc[0, "product_id"] = 9999
    write()
    with pytest.raises(ValueError, match="Orphan"):
        load_tables(tmp_path)
    tables = load_tables(ROOT / "data/demo")
    tables["orders"] = pd.concat([tables["orders"], tables["orders"].head(1)])
    write()
    with pytest.raises(ValueError, match="duplicate primary key"):
        load_tables(tmp_path)


def test_incomplete_cart_is_rejected(tmp_path):
    tables = load_tables(ROOT / "data/demo")
    tables["order_products"].loc[0, "add_to_cart_order"] = 1000
    for name, table in tables.items():
        table.to_csv(tmp_path / f"{name}.csv", index=False)
    with pytest.raises(ValueError, match="Non-contiguous"):
        load_tables(tmp_path)
