# src/modeling.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def validate_ml_columns(df: pd.DataFrame) -> None:
    """
    Validate that the dataframe contains all required columns for the ML pipeline.
    """
    required = {
        "user_id",
        "order_id",
        "order_number",
        "product_id",
        "reordered",
        "add_to_cart_order",
        "order_hour_of_day",
        "order_dow",
        "days_since_prior_order",
        "department",
        "aisle",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required ML columns: {sorted(missing)}")


def prepare_ml_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only columns needed for ML and optimize dtypes.
    """
    validate_ml_columns(df)

    ml_cols = [
        "user_id",
        "order_id",
        "order_number",
        "product_id",
        "reordered",
        "add_to_cart_order",
        "order_hour_of_day",
        "order_dow",
        "days_since_prior_order",
        "department",
        "aisle",
    ]

    out = df[ml_cols].copy()

    int_cols = [
        "user_id",
        "order_id",
        "order_number",
        "product_id",
        "reordered",
        "add_to_cart_order",
        "order_hour_of_day",
        "order_dow",
    ]
    for col in int_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce", downcast="integer")

    out["days_since_prior_order"] = pd.to_numeric(
        out["days_since_prior_order"], errors="coerce", downcast="float"
    ).fillna(0)

    out["department"] = out["department"].astype("category")
    out["aisle"] = out["aisle"].astype("category")

    return out


def build_test_order_ids_from_orders(orders: pd.DataFrame) -> set:
    """
    Build the test set as the last order of each user from the lightweight orders table.
    """
    required = {"user_id", "order_id", "order_number"}
    missing = required - set(orders.columns)
    if missing:
        raise ValueError(f"Missing required columns in orders: {sorted(missing)}")

    tmp = orders[["user_id", "order_id", "order_number"]].dropna().copy()
    tmp["order_number"] = pd.to_numeric(tmp["order_number"], errors="coerce")
    tmp = tmp.dropna(subset=["order_number"])
    tmp = tmp.drop_duplicates(subset=["order_id"])

    last_orders = (
        tmp.sort_values(["user_id", "order_number"])
        .groupby("user_id", as_index=False)
        .tail(1)
    )

    test_order_ids = set(last_orders["order_id"].astype(int).tolist())
    return test_order_ids


def split_train_test_from_order_ids(
    df_ml: pd.DataFrame,
    test_order_ids: set,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the ML dataframe into train/test using a set of test order ids.
    """
    mask_test = df_ml["order_id"].isin(test_order_ids)
    train = df_ml.loc[~mask_test].copy()
    test = df_ml.loc[mask_test].copy()
    return train, test


def add_stats_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    product_col: str = "product_id",
    user_col: str = "user_id",
    target_col: str = "reordered",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add product/user aggregate statistics using train only.
    Memory-efficient implementation using map().
    """
    train = train.copy()
    test = test.copy()

    train[target_col] = pd.to_numeric(train[target_col], errors="coerce").fillna(0)

    prod_rate = train.groupby(product_col, observed=True)[target_col].mean()
    prod_count = train.groupby(product_col, observed=True)[target_col].count()

    user_rate = train.groupby(user_col, observed=True)[target_col].mean()
    user_count = train.groupby(user_col, observed=True)[target_col].count()

    global_rate = float(train[target_col].mean())
    if np.isnan(global_rate):
        global_rate = 0.0

    for frame in (train, test):
        frame["prod_reorder_rate"] = frame[product_col].map(prod_rate).fillna(global_rate)
        frame["prod_orders"] = frame[product_col].map(prod_count).fillna(0)
        frame["user_reorder_rate"] = frame[user_col].map(user_rate).fillna(global_rate)
        frame["user_orders"] = frame[user_col].map(user_count).fillna(0)
        frame["days_since_prior_order"] = pd.to_numeric(
            frame["days_since_prior_order"], errors="coerce"
        ).fillna(0)

    return train, test


def build_feature_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "reordered",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Build train/test feature matrices for the ML models.
    """
    needed_cols = [
        "user_id",
        "product_id",
        "reordered",
        "add_to_cart_order",
        "order_hour_of_day",
        "order_dow",
        "days_since_prior_order",
        "order_number",
        "department",
        "aisle",
    ]

    missing_train = set(needed_cols) - set(train.columns)
    missing_test = set(needed_cols) - set(test.columns)
    if missing_train:
        raise ValueError(f"Missing columns in train: {sorted(missing_train)}")
    if missing_test:
        raise ValueError(f"Missing columns in test: {sorted(missing_test)}")

    train = train[needed_cols].copy()
    test = test[needed_cols].copy()

    train_fe, test_fe = add_stats_features(train, test)

    numeric_features = [
        "add_to_cart_order",
        "order_hour_of_day",
        "order_dow",
        "days_since_prior_order",
        "order_number",
        "prod_reorder_rate",
        "user_reorder_rate",
        "prod_orders",
        "user_orders",
    ]
    categorical_features = ["department", "aisle"]

    for col in numeric_features:
        train_fe[col] = pd.to_numeric(train_fe[col], errors="coerce").fillna(0)
        test_fe[col] = pd.to_numeric(test_fe[col], errors="coerce").fillna(0)

    X_train = train_fe[numeric_features + categorical_features].copy()
    y_train = pd.to_numeric(train_fe[target_col], errors="coerce").fillna(0).astype(int)

    X_test = test_fe[numeric_features + categorical_features].copy()
    y_test = pd.to_numeric(test_fe[target_col], errors="coerce").fillna(0).astype(int)

    return X_train, y_train, X_test, y_test, numeric_features, categorical_features


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_features: List[str],
    categorical_features: List[str],
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Train and evaluate:
    - Logistic Regression
    - Random Forest

    Returns:
    - results_df: summary table
    - details: detailed metrics per model
    """
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    models: Dict[str, Pipeline] = {
        "LogisticRegression": Pipeline(
            steps=[
                ("prep", preprocess),
                ("clf", LogisticRegression(max_iter=400)),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("prep", preprocess),
                ("clf", RandomForestClassifier(
                    n_estimators=50,
                    max_depth=12,
                    min_samples_leaf=5,
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                )),
            ]
        ),
    }

    results_rows: List[Dict[str, Any]] = []
    details: Dict[str, Dict[str, Any]] = {}

    unique_labels = np.unique(y_test)

    for name, model in models.items():
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        if len(unique_labels) > 1:
            roc_auc = float(roc_auc_score(y_test, proba))
            avg_precision = float(average_precision_score(y_test, proba))
        else:
            roc_auc = float("nan")
            avg_precision = float("nan")

        results_rows.append({
            "model": name,
            "ROC_AUC": roc_auc,
            "AvgPrecision": avg_precision,
        })

        details[name] = {
            "model": model,
            "roc_auc": roc_auc,
            "avg_precision": avg_precision,
            "classification_report": classification_report(y_test, pred, digits=3),
        }

    results_df = (
        pd.DataFrame(results_rows)
        .sort_values("ROC_AUC", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    return results_df, details