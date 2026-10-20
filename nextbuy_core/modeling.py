"""Fixed-protocol diagnostic classification, not next-basket prediction."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = [
    "add_to_cart_order",
    "order_hour_of_day",
    "order_dow",
    "days_since_prior_order",
    "order_number",
    "prod_reorder_rate",
    "prod_items",
    "user_reorder_rate",
    "user_items",
]
CATEGORICAL = ["department", "aisle"]


@dataclass
class TemporalSplit:
    history: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    excluded_users: int


def temporal_split(frame: pd.DataFrame) -> TemporalSplit:
    """Last observed basket = test; penultimate = training; older baskets = history.

    Requires >=3 observed orders. No global calendar ordering is possible in these data.
    """
    orders = frame[["user_id", "order_id", "order_number"]].drop_duplicates()
    orders = orders.sort_values(["user_id", "order_number"])
    eligible = orders.groupby("user_id").size().loc[lambda x: x >= 3].index
    orders = orders.loc[orders.user_id.isin(eligible)].copy()
    rank = orders.groupby("user_id").cumcount(ascending=False)
    sets = [
        set(orders.loc[rank >= 2, "order_id"]),
        set(orders.loc[rank == 1, "order_id"]),
        set(orders.loc[rank == 0, "order_id"]),
    ]
    parts = [frame.loc[frame.order_id.isin(ids)].copy() for ids in sets]
    if any(part.empty for part in parts):
        raise ValueError("Need at least one customer with three observed baskets.")
    return TemporalSplit(*parts, frame.user_id.nunique() - len(eligible))


def history_features(history: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Fit aggregates on a separate history set. frame.reordered is never read."""
    out = frame.copy()
    prior = float(history.reordered.mean())
    for key, prefix in [("product_id", "prod"), ("user_id", "user")]:
        stats = history.groupby(key, observed=True).reordered.agg(["sum", "count"])
        # Fixed smoothing strength, never selected against test outcomes.
        rates = (stats["sum"] + 10 * prior) / (stats["count"] + 10)
        out[f"{prefix}_reorder_rate"] = out[key].map(rates).fillna(prior)
        out[f"{prefix}_items"] = out[key].map(stats["count"]).fillna(0)
    out["days_since_prior_order"] = out.days_since_prior_order.fillna(0)
    return out[NUMERIC + CATEGORICAL].copy()


def evaluate(y: pd.Series, proba: np.ndarray) -> dict[str, float]:
    return {
        "ROC_AUC": float(roc_auc_score(y, proba)) if y.nunique() == 2 else float("nan"),
        "Average_precision": float(average_precision_score(y, proba))
        if y.nunique() == 2
        else float("nan"),
        "Brier_score": float(brier_score_loss(y, proba)),
    }


def run_experiment(frame: pd.DataFrame) -> dict[str, Any]:
    split = temporal_split(frame)
    X_train = history_features(split.history, split.train)
    X_test = history_features(split.history, split.test)
    y_train, y_test = split.train.reordered, split.test.reordered
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Both train and test must contain both target classes.")

    def preprocess() -> ColumnTransformer:
        return ColumnTransformer(
            [
                ("numeric", StandardScaler(), NUMERIC),
                ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ]
        )

    estimators = {
        "Prevalence baseline": DummyClassifier(strategy="prior"),
        "Logistic regression": Pipeline(
            [
                ("prep", preprocess()),
                ("classifier", LogisticRegression(C=1.0, max_iter=1500, random_state=42)),
            ]
        ),
        "Random forest": Pipeline(
            [
                ("prep", preprocess()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=80, max_depth=10, min_samples_leaf=8, random_state=42, n_jobs=1
                    ),
                ),
            ]
        ),
    }
    rows, details = [], {}
    for name, model in estimators.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        scores = evaluate(y_test, proba)
        pred = (proba >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_test, proba)
        precision, recall, _ = precision_recall_curve(y_test, proba)
        rows.append({"model": name, **scores})
        details[name] = {
            "model": model,
            "probabilities": proba,
            "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
            "confusion": confusion_matrix(y_test, pred, labels=[0, 1]),
            "roc": (fpr, tpr),
            "pr": (recall, precision),
        }
    return {
        "scores": pd.DataFrame(rows),
        "details": details,
        "X_test": X_test,
        "test_rows": split.test.reset_index(drop=True),
        "y_test": y_test.to_numpy(),
        "sizes": {
            "history_items": len(split.history),
            "train_items": len(split.train),
            "test_items": len(split.test),
            "excluded_users": split.excluded_users,
        },
        "test_prevalence": float(y_test.mean()),
    }
