import json

import pandas as pd

from nextbuy_core.data import load_tables
from scripts.prepare_data import generate_demo, prepare_local


def test_corrupt_line_excludes_whole_basket_and_records_manifest(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "cohort"
    generate_demo(source, users=8)
    lines = pd.read_csv(source / "order_products.csv")
    bad_order = lines.order_id.iloc[0]
    lines["reordered"] = lines.reordered.astype(float)
    lines.loc[0, "reordered"] = float("nan")
    lines.to_csv(source / "order_products.csv", index=False)
    prepare_local(source, destination, users=8)
    cleaned = load_tables(destination)
    assert bad_order not in set(cleaned["order_products"].order_id)
    assert bad_order not in set(cleaned["orders"].order_id)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["invalid_cohort_baskets_excluded"] == 1
    assert len(manifest["files"]) == 5
