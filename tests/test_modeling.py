import numpy as np
import pandas as pd
import pytest

from nextbuy_core.data import ROOT, load_tables, merge_tables
from nextbuy_core.modeling import history_features, run_experiment, temporal_split


@pytest.fixture(scope="module")
def frame():
    f = merge_tables(load_tables(ROOT / "data/demo"))
    return f.loc[f.user_id.le(100)].copy()


def test_temporal_partitions_are_disjoint_and_ordered(frame):
    split = temporal_split(frame)
    a, b, c = [set(part.order_id) for part in [split.history, split.train, split.test]]
    assert not (a & b or a & c or b & c)
    history_last = split.history.groupby("user_id").order_number.max()
    train = split.train.groupby("user_id").order_number.min()
    test = split.test.groupby("user_id").order_number.min()
    assert (history_last < train).all() and (train < test).all()


def test_no_training_or_test_label_enters_features(frame):
    split = temporal_split(frame)
    for part in [split.train, split.test]:
        original = history_features(split.history, part)
        changed = part.copy()
        changed["reordered"] = 1 - changed.reordered
        pd.testing.assert_frame_equal(original, history_features(split.history, changed))
        assert "reordered" not in original


def test_history_fallback_and_unseen_categories(frame):
    split = temporal_split(frame)
    unseen = split.test.head(1).copy()
    unseen["user_id"] = -1
    unseen["product_id"] = -1
    transformed = history_features(split.history, unseen)
    assert transformed.user_items.iloc[0] == 0
    assert transformed.prod_reorder_rate.iloc[0] == pytest.approx(split.history.reordered.mean())
    assert not transformed.isna().any().any()


def test_models_baseline_and_reproducibility(frame):
    result = run_experiment(frame)
    again = run_experiment(frame)
    pd.testing.assert_frame_equal(result["scores"], again["scores"])
    scores = result["scores"].set_index("model")
    assert scores.loc["Prevalence baseline", "ROC_AUC"] == 0.5
    assert scores.loc["Prevalence baseline", "Average_precision"] == pytest.approx(
        result["test_prevalence"]
    )
    assert scores[["ROC_AUC", "Average_precision", "Brier_score"]].ge(0).all().all()
    assert scores[["ROC_AUC", "Average_precision", "Brier_score"]].le(1).all().all()
    for detail in result["details"].values():
        assert detail["confusion"].sum() == len(result["y_test"])
        assert np.isfinite(detail["probabilities"]).all()
        example = result["X_test"].head(1).copy()
        example["aisle"] = "unseen aisle"
        assert detail["model"].predict_proba(example).shape == (1, 2)


def test_insufficient_history_is_explicit(frame):
    with pytest.raises(ValueError, match="three observed"):
        temporal_split(frame.loc[frame.order_number.eq(1)])
