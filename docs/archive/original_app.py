import gc
from pathlib import Path
from itertools import combinations
from collections import Counter

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.modeling import train_and_evaluate_models


st.set_page_config(
    page_title="NextBuy Dashboard",
    page_icon=None,
    layout="wide",
)


DATA_DIR = Path("datasets")
REQUIRED_FILES = {
    "aisles": DATA_DIR / "aisles.csv",
    "departments": DATA_DIR / "departments.csv",
    "products": DATA_DIR / "products.csv",
    "orders": DATA_DIR / "orders.csv",
    "order_products": DATA_DIR / "order_products.csv",
}


def check_files():
    missing = [str(path) for path in REQUIRED_FILES.values() if not path.exists()]
    return len(missing) == 0, missing


@st.cache_data(show_spinner=False)
def load_raw_data():
    return {
        "aisles": pd.read_csv(REQUIRED_FILES["aisles"]),
        "departments": pd.read_csv(REQUIRED_FILES["departments"]),
        "products": pd.read_csv(REQUIRED_FILES["products"]),
        "orders": pd.read_csv(REQUIRED_FILES["orders"]),
        "order_products": pd.read_csv(REQUIRED_FILES["order_products"]),
    }


@st.cache_data(show_spinner=False)
def get_raw_shapes():
    raw = load_raw_data()
    return pd.DataFrame(
        {
            "table": list(raw.keys()),
            "rows": [len(v) for v in raw.values()],
            "columns": [v.shape[1] for v in raw.values()],
        }
    )


@st.cache_data(show_spinner=False)
def build_merged_dataframe(sample_size: int = 100000):
    raw = load_raw_data()

    prod = (
        raw["products"]
        .merge(raw["aisles"], on="aisle_id", how="left")
        .merge(raw["departments"], on="department_id", how="left")
    )

    df = (
        raw["order_products"]
        .merge(raw["orders"], on="order_id", how="left")
        .merge(prod, on="product_id", how="left")
    )

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    return df


@st.cache_data(show_spinner=False)
def build_test_order_ids():
    raw = load_raw_data()
    orders = raw["orders"][["user_id", "order_id", "order_number"]].dropna().copy()
    orders["order_number"] = pd.to_numeric(orders["order_number"], errors="coerce")
    orders = orders.dropna(subset=["order_number"]).drop_duplicates(subset=["order_id"])

    last_orders = (
        orders.sort_values(["user_id", "order_number"])
        .groupby("user_id", as_index=False)
        .tail(1)
    )

    return set(last_orders["order_id"].astype(int).tolist())


@st.cache_data(show_spinner=False)
def build_ml_dataframe(sample_size: int = 200000):
    raw = load_raw_data()

    prod = (
        raw["products"][["product_id", "aisle_id", "department_id"]]
        .merge(raw["aisles"], on="aisle_id", how="left")
        .merge(raw["departments"], on="department_id", how="left")
    )

    df_ml = (
        raw["order_products"][["order_id", "product_id", "add_to_cart_order", "reordered"]]
        .merge(
            raw["orders"][
                ["order_id", "user_id", "order_number", "order_dow", "order_hour_of_day", "days_since_prior_order"]
            ],
            on="order_id",
            how="left",
        )
        .merge(prod[["product_id", "aisle", "department"]], on="product_id", how="left")
    )

    if len(df_ml) > sample_size:
        df_ml = df_ml.sample(sample_size, random_state=42)

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
        df_ml[col] = pd.to_numeric(df_ml[col], errors="coerce", downcast="integer")

    df_ml["days_since_prior_order"] = pd.to_numeric(
        df_ml["days_since_prior_order"], errors="coerce", downcast="float"
    ).fillna(0)

    df_ml["department"] = df_ml["department"].astype("category")
    df_ml["aisle"] = df_ml["aisle"].astype("category")

    return df_ml


@st.cache_data(show_spinner=False)
def build_ml_split(max_rows: int = 150000):
    df_ml = build_ml_dataframe(sample_size=max_rows)
    test_order_ids = build_test_order_ids()

    mask_test = df_ml["order_id"].isin(test_order_ids)
    train = df_ml.loc[~mask_test]
    test = df_ml.loc[mask_test]

    return train, test


@st.cache_data(show_spinner=False)
def build_features(max_rows: int = 150000):
    train, test = build_ml_split(max_rows=max_rows)

    prod_reorder_rate = train.groupby("product_id", observed=True)["reordered"].mean()
    prod_orders = train.groupby("product_id", observed=True)["reordered"].count()

    user_reorder_rate = train.groupby("user_id", observed=True)["reordered"].mean()
    user_orders = train.groupby("user_id", observed=True)["reordered"].count()

    global_reorder_rate = float(train["reordered"].mean())

    keep_cols = [
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

    train_fe = train[keep_cols]
    test_fe = test[keep_cols]

    train_fe = train_fe.assign(
        prod_reorder_rate=train_fe["product_id"].map(prod_reorder_rate).fillna(global_reorder_rate),
        prod_orders=train_fe["product_id"].map(prod_orders).fillna(0),
        user_reorder_rate=train_fe["user_id"].map(user_reorder_rate).fillna(global_reorder_rate),
        user_orders=train_fe["user_id"].map(user_orders).fillna(0),
    )

    test_fe = test_fe.assign(
        prod_reorder_rate=test_fe["product_id"].map(prod_reorder_rate).fillna(global_reorder_rate),
        prod_orders=test_fe["product_id"].map(prod_orders).fillna(0),
        user_reorder_rate=test_fe["user_id"].map(user_reorder_rate).fillna(global_reorder_rate),
        user_orders=test_fe["user_id"].map(user_orders).fillna(0),
    )

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
        train_fe[col] = pd.to_numeric(train_fe[col], errors="coerce", downcast="float").fillna(0)
        test_fe[col] = pd.to_numeric(test_fe[col], errors="coerce", downcast="float").fillna(0)

    train_fe["department"] = train_fe["department"].astype("category")
    train_fe["aisle"] = train_fe["aisle"].astype("category")
    test_fe["department"] = test_fe["department"].astype("category")
    test_fe["aisle"] = test_fe["aisle"].astype("category")

    X_train = train_fe[numeric_features + categorical_features]
    y_train = pd.to_numeric(train_fe["reordered"], errors="coerce").fillna(0).astype("int8")

    X_test = test_fe[numeric_features + categorical_features]
    y_test = pd.to_numeric(test_fe["reordered"], errors="coerce").fillna(0).astype("int8")

    return X_train, y_train, X_test, y_test, numeric_features, categorical_features


@st.cache_data(show_spinner=False)
def run_models(max_rows: int = 150000):
    X_train, y_train, X_test, y_test, numeric_features, categorical_features = build_features(max_rows=max_rows)

    results_df, details = train_and_evaluate_models(
        X_train,
        y_train,
        X_test,
        y_test,
        numeric_features,
        categorical_features,
    )

    return results_df, details, X_train.shape, X_test.shape


@st.cache_data(show_spinner=False)
def compute_top_pairs(sample_size: int = 20000, top_n_products: int = 50):
    df = build_merged_dataframe(sample_size=sample_size)

    df_pairs = df.dropna(subset=["product_name"]).copy()
    df_pairs["product_name"] = df_pairs["product_name"].astype(str)

    top_products = df_pairs["product_name"].value_counts().head(top_n_products).index
    df_pairs = df_pairs[df_pairs["product_name"].isin(top_products)]

    order_groups = df_pairs.groupby("order_id")["product_name"].apply(list)

    pair_counter = Counter()
    for products in order_groups:
        unique_products = sorted(set(products))
        if len(unique_products) >= 2:
            for pair in combinations(unique_products, 2):
                pair_counter[pair] += 1

    top_pairs = pd.DataFrame(
        [
            {"product_1": p1, "product_2": p2, "count": count}
            for (p1, p2), count in pair_counter.most_common(10)
        ]
    )

    if not top_pairs.empty:
        top_pairs["pair"] = top_pairs["product_1"] + " + " + top_pairs["product_2"]

    return top_pairs


# Sidebar
st.sidebar.title("NextBuy Dashboard")
st.sidebar.caption("Version légère")

ok, missing = check_files()
if not ok:
    st.error("Missing dataset files:")
    for m in missing:
        st.write(f"- {m}")
    st.stop()

eda_sample_size = st.sidebar.slider("EDA sample size", 20000, 150000, 100000, step=10000)
pairs_sample_size = st.sidebar.slider("Co-purchase sample size", 5000, 50000, 20000, step=5000)
top_n_products = st.sidebar.slider("Top products for pairs", 20, 100, 50, step=10)
ml_max_rows = st.sidebar.slider("ML max rows", 50000, 200000, 150000, step=25000)


st.title("NextBuy Dashboard")
st.markdown(
    """
Ce dashboard présente le projet **NextBuy** :
- analyse exploratoire des comportements d’achat,
- insights business,
- et modélisation prédictive sur la variable **`reordered`**.
"""
)

raw_shapes = get_raw_shapes()
col1, col2, col3 = st.columns(3)
col1.metric("Tables", len(raw_shapes))
col2.metric("Lignes orders.csv", f"{int(raw_shapes.loc[raw_shapes['table']=='orders', 'rows'].iloc[0]):,}")
col3.metric("Lignes order_products.csv", f"{int(raw_shapes.loc[raw_shapes['table']=='order_products', 'rows'].iloc[0]):,}")

tabs = st.tabs(["Overview", "EDA", "Business Insights", "Machine Learning", "Data Quality"])

with tabs[0]:
    st.subheader("Overview")
    st.dataframe(raw_shapes, use_container_width=True)

    with st.spinner("Loading sample merged dataframe..."):
        df_preview = build_merged_dataframe(sample_size=eda_sample_size)

    st.write(f"Sample merged dataframe shape: {df_preview.shape}")
    st.dataframe(df_preview.head(20), use_container_width=True)

with tabs[1]:
    st.subheader("EDA")

    with st.spinner("Loading EDA sample..."):
        df_eda = build_merged_dataframe(sample_size=eda_sample_size)

    c1, c2 = st.columns(2)

    with c1:
        st.write("**Top 15 products**")
        top_products = df_eda["product_name"].value_counts().head(15).sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        top_products.plot(kind="barh", ax=ax)
        ax.set_xlabel("Number of orders")
        ax.set_ylabel("")
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.write("**Top 15 departments**")
        top_depts = df_eda["department"].value_counts().head(15).sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        top_depts.plot(kind="barh", ax=ax)
        ax.set_xlabel("Items ordered")
        ax.set_ylabel("")
        st.pyplot(fig)
        plt.close(fig)

    c3, c4 = st.columns(2)

    with c3:
        st.write("**Orders by hour**")
        hourly = df_eda["order_hour_of_day"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        hourly.plot(kind="bar", ax=ax)
        ax.set_xlabel("Hour")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)

    with c4:
        st.write("**Orders by day of week**")
        dow = df_eda["order_dow"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        dow.plot(kind="bar", ax=ax)
        ax.set_xlabel("Day of week")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)

    del df_eda
    gc.collect()

with tabs[2]:
    st.subheader("Business Insights")

    with st.spinner("Loading insight sample..."):
        df_insights = build_merged_dataframe(sample_size=eda_sample_size)

    st.write("**Most frequently first-added products**")
    first_items = (
        df_insights.loc[df_insights["add_to_cart_order"] == 1, "product_name"]
        .value_counts()
        .head(10)
        .reset_index()
    )
    first_items.columns = ["product_name", "count"]
    st.dataframe(first_items, use_container_width=True)

    st.write("**Top co-purchase pairs**")
    with st.spinner("Computing co-purchase pairs..."):
        top_pairs = compute_top_pairs(
            sample_size=pairs_sample_size,
            top_n_products=top_n_products,
        )

    st.dataframe(top_pairs, use_container_width=True)

    if not top_pairs.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(top_pairs["pair"], top_pairs["count"])
        ax.set_xlabel("Times bought together")
        ax.set_title("Top co-purchase pairs")
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown(
        """
**Bundle idea**
- Offer a discount when both products are bought together
- Place them close to each other on the website
- Recommend one product when the other is added to cart
- Reuse top pairs in weekly promotional campaigns
"""
    )

    del df_insights
    gc.collect()

with tabs[3]:
    st.subheader("Machine Learning")
    st.markdown(
        """
Target: **`reordered`**

Models compared:
- Logistic Regression
- Random Forest

Split strategy:
- the **last order of each user** is used as the test set.
"""
    )

    if st.button("Run ML pipeline"):
        with st.spinner("Training models..."):
            try:
                results_df, details, xtr_shape, xte_shape = run_models(max_rows=ml_max_rows)

                st.success("ML pipeline completed.")
                st.write(f"X_train shape: {xtr_shape}")
                st.write(f"X_test shape: {xte_shape}")
                st.dataframe(results_df, use_container_width=True)

                fig, ax = plt.subplots(figsize=(8, 4))
                results_df.plot(
                    x="model",
                    y=["ROC_AUC", "AvgPrecision"],
                    kind="bar",
                    ax=ax,
                )
                ax.set_title("Model comparison")
                ax.set_ylabel("Score")
                ax.tick_params(axis="x", rotation=0)
                st.pyplot(fig)
                plt.close(fig)

                for name, info in details.items():
                    with st.expander(f"Details — {name}"):
                        st.write(f"ROC_AUC: {info['roc_auc']:.4f}")
                        st.write(f"AvgPrecision: {info['avg_precision']:.4f}")
                        st.code(info["classification_report"])

                st.markdown(
                    """
### Interpretation
The best model is the one with the highest **ROC-AUC** and **Average Precision**.

This model can help NextBuy improve:
- recommendations,
- promotion targeting,
- and customer retention strategies.
"""
                )
            except Exception as e:
                st.error(f"ML pipeline failed: {e}")

        gc.collect()

with tabs[4]:
    st.subheader("Data Quality")

    with st.spinner("Loading quality sample..."):
        df_quality = build_merged_dataframe(sample_size=eda_sample_size)

    missing_df = df_quality.isna().sum().sort_values(ascending=False).head(10).reset_index()
    missing_df.columns = ["column", "missing_count"]
    st.dataframe(missing_df, use_container_width=True)

    dq1, dq2 = st.columns(2)
    dq1.metric("Duplicate rows in sample", int(df_quality.duplicated().sum()))
    dq2.metric("Sample rows", f"{len(df_quality):,}")

    st.write("**Target distribution**")
    target_dist = df_quality["reordered"].value_counts(dropna=False).reset_index()
    target_dist.columns = ["reordered", "count"]
    st.dataframe(target_dist, use_container_width=True)

    del df_quality
    gc.collect()