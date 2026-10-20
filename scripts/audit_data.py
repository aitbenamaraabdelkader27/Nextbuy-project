"""Full local source audit; no joins or random row sampling."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nextbuy_core.data import ROOT, TABLE_KEYS

report = {}
frames = {}
for name in TABLE_KEYS:
    frame = pd.read_csv(ROOT / "nextbuy/datasets" / f"{name}.csv")
    for col in frame.select_dtypes("integer"):
        frame[col] = frame[col].astype("int32")
    report[name] = {
        "rows": len(frame),
        "columns": list(frame.columns),
        "missing": frame.isna().sum().to_dict(),
        "duplicates": int(frame.duplicated().sum()),
        "duplicate_keys": int(frame.duplicated(TABLE_KEYS[name]).sum()),
        "dtypes": frame.dtypes.astype(str).to_dict(),
    }
    frames[name] = frame
lines, orders = frames["order_products"], frames["orders"]
report["checks"] = {
    "orphan_order_lines": int((~lines.order_id.isin(orders.order_id)).sum()),
    "orphan_product_lines": int((~lines.product_id.isin(frames["products"].product_id)).sum()),
    "invalid_target": int((~lines.reordered.isin([0, 1])).sum()),
    "invalid_hour": int((~orders.order_hour_of_day.between(0, 23)).sum()),
    "invalid_dow": int((~orders.order_dow.between(0, 6)).sum()),
    "missing_days_after_first": int(
        (orders.days_since_prior_order.isna() & orders.order_number.ne(1)).sum()
    ),
    "orders_without_lines": int((~orders.order_id.isin(lines.order_id)).sum()),
    "cart_positions_missing": int(lines.add_to_cart_order.isna().sum()),
    "cart_positions_max": int(lines.add_to_cart_order.max()),
    "invalid_days": int((~orders.days_since_prior_order.dropna().between(0, 30)).sum()),
    "repeat_rate": float(lines.reordered.mean()),
    "unique_users": orders.user_id.nunique(),
}
path = ROOT / "docs/source_audit.json"
path.write_text(json.dumps(report, indent=2, default=int))
print(path.read_text())
