"""Presentation only: cached data, explicit navigation and bounded on-demand ML."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from nextbuy_core.analytics import association_pairs, kpis, order_activity, product_statistics
from nextbuy_core.charts import COLORS, style
from nextbuy_core.data import (
    available_sources,
    filter_orders,
    load_tables,
    merge_tables,
    quality_report,
)
from nextbuy_core.modeling import run_experiment

px.defaults.color_discrete_sequence = COLORS

PAGES = ["Overview", "Data explorer", "Basket insights", "Model lab", "Methodology & quality"]


@st.cache_data(show_spinner=False, max_entries=2)
def get_data(directory: str, signature: tuple) -> tuple[dict, pd.DataFrame]:
    tables = load_tables(Path(directory))
    return tables, merge_tables(tables)


@st.cache_resource(show_spinner=False, max_entries=2)
def experiment(directory: str, signature: tuple) -> dict:
    _, frame = get_data(directory, signature)
    return run_experiment(frame)


def chart(fig: go.Figure, height: int = 360) -> None:
    st.plotly_chart(style(fig, height), width="stretch", theme=None, config={"displaylogo": False})


def metrics(frame: pd.DataFrame) -> None:
    values = kpis(frame)
    columns = st.columns(4)
    for col, label, value, help_text in [
        (
            columns[0],
            "Observed orders",
            f"{values['orders']:,}",
            "Distinct order IDs with available articles.",
        ),
        (
            columns[1],
            "Customers",
            f"{values['customers']:,}",
            "Anonymous customer IDs in this scope.",
        ),
        (
            columns[2],
            "Repeat article share",
            f"{values['repeat_rate']:.1%}",
            "Share of purchased lines flagged as bought previously; not future retention.",
        ),
        (
            columns[3],
            "Articles / basket",
            f"{values['basket_size']:.1f}",
            "Observed product lines per distinct order; not quantity or revenue.",
        ),
    ]:
        with col, st.container(border=True):
            st.metric(label, value, help=help_text)


def overview(frame: pd.DataFrame) -> None:
    st.caption("01 / THE BIG PICTURE")
    st.title("From grocery baskets to clear insights.")
    st.write(
        "Explore purchase patterns, discover product associations and evaluate repeat-status classifiers."
    )
    metrics(frame)
    left, right = st.columns([1.3, 1])
    with left, st.container(border=True):
        stats = product_statistics(frame, "department").sort_values("items", ascending=True)
        chart(
            px.bar(
                stats,
                x="items",
                y="department",
                orientation="h",
                title="Where articles are purchased",
                hover_data=["repeat_rate"],
                labels={"items": "Purchased article lines", "department": ""},
            )
        )
    with right, st.container(border=True):
        chart(
            px.area(
                order_activity(frame, "order_hour_of_day"),
                x="order_hour_of_day",
                y="orders",
                title="Daily rhythm",
                labels={
                    "order_hour_of_day": "Hour (dataset local clock)",
                    "orders": "Distinct orders",
                },
            )
        )
    with st.container(border=True):
        st.subheader("Read the data with the right lens")
        st.write(
            "An article is a product line, not a monetary sale. `reordered = 1` means this customer bought the product before. The supplied files contain no prices, calendar dates or future non-purchases."
        )
        st.caption(
            "Start with Data explorer, investigate Basket insights, then inspect the evaluation protocol in Model lab."
        )


def explorer(frame: pd.DataFrame) -> None:
    st.caption("02 / EXPLORE THE EVIDENCE")
    st.title("A closer look at every basket.")
    st.write(
        "Temporal filters retain complete observed baskets. Department selection below applies to article analyses only."
    )
    metrics(frame)
    department = st.selectbox(
        "Article department", ["All departments"] + sorted(frame.department.unique().tolist())
    )
    articles = (
        frame if department == "All departments" else frame.loc[frame.department.eq(department)]
    )
    a, b = st.columns(2)
    with a:
        view = st.selectbox(
            "Distribution", ["Basket size", "Order hour", "Day index", "Days since prior order"]
        )
        orders = frame.drop_duplicates("order_id")
        if view == "Basket size":
            sizes = frame.groupby("order_id").size().rename("articles").reset_index()
            fig = px.histogram(
                sizes, x="articles", nbins=35, labels={"articles": "Observed articles / basket"}
            )
        elif view == "Days since prior order":
            fig = px.histogram(
                orders,
                x="days_since_prior_order",
                nbins=31,
                labels={"days_since_prior_order": "Elapsed days (30 = capped)"},
            )
            st.caption(
                f"{orders.days_since_prior_order.isna().sum():,} first orders with no elapsed interval are excluded."
            )
        else:
            key = "order_hour_of_day" if view == "Order hour" else "order_dow"
            fig = px.bar(
                order_activity(frame, key),
                x=key,
                y="orders",
                labels={key: view, "orders": "Distinct orders"},
            )
        chart(fig)
        st.caption(
            "Order distributions use all departments in the temporal scope. Day codes 0–6 are not assigned weekday names."
        )
    with b:
        stats = product_statistics(articles).nlargest(12, "items").sort_values("items")
        chart(
            px.bar(
                stats,
                x="items",
                y="product_name",
                orientation="h",
                title="Most purchased products",
                labels={"items": "Article lines", "product_name": ""},
            )
        )
    st.subheader("Inspect & export")
    query = st.text_input("Search product name", placeholder="e.g. banana")
    matches = articles.loc[
        articles.product_name.astype(str).str.contains(query, case=False, regex=False)
    ]
    st.caption(f"{len(matches):,} matching article lines · preview limited to 500 rows")
    st.dataframe(matches.head(500), hide_index=True, width="stretch")
    st.download_button(
        "Download matching articles (CSV)",
        matches.to_csv(index=False).encode(),
        "nextbuy_articles.csv",
        "text/csv",
    )


def insights(frame: pd.DataFrame) -> None:
    st.caption("03 / BASKET INTELLIGENCE")
    st.title("What travels together?")
    st.write(
        "Associations describe observed behaviour. Promotion ideas require an experiment to establish business impact."
    )
    left, right = st.columns(2)
    minimum = st.slider("Minimum product observations", 1, 200, 20)
    stats = product_statistics(frame)
    stats = stats.loc[stats["items"] >= minimum]
    with left:
        chart(
            px.scatter(
                stats,
                x="items",
                y="repeat_rate",
                hover_name="product_name",
                title="Volume & repeat behaviour",
                labels={"items": "Purchased article lines", "repeat_rate": "Repeat article share"},
            )
        )
    with right:
        first = stats.nlargest(10, "first_ratio").sort_values("first_ratio")
        chart(
            px.bar(
                first,
                x="first_ratio",
                y="product_name",
                orientation="h",
                title="First in the basket",
                hover_data=["items", "first_count"],
                labels={"first_ratio": "First position / product purchases", "product_name": ""},
            )
        )
    st.subheader("Co-purchase explorer")
    a, b, c = st.columns(3)
    top = a.slider("Candidate products (by frequency)", 10, 100, 30, step=10)
    count = b.slider("Minimum co-purchases", 1, 100, 5)
    ranking = c.selectbox("Rank pairs by", ["count", "lift", "support", "confidence_a_to_b"])
    with st.spinner("Counting pairs in complete observed baskets…"):
        pairs = association_pairs(frame, top, count).sort_values(ranking, ascending=False)
    st.caption(
        f"Denominator: {frame.order_id.nunique():,} selected orders, including baskets without candidate products. Confidence is directional: A → B."
    )
    if pairs.empty:
        st.info(
            "No pair meets the selected count. Lower the minimum or broaden the temporal filters."
        )
    else:
        plot = pairs.head(10).copy()
        plot["pair"] = plot.product_a.astype(str) + " + " + plot.product_b.astype(str)
        chart(
            px.bar(
                plot.sort_values(ranking),
                x=ranking,
                y="pair",
                orientation="h",
                labels={"pair": ""},
                title="Leading observed pairs",
            ),
            420,
        )
        st.dataframe(pairs.head(100), hide_index=True, width="stretch")
        st.download_button(
            "Download all qualifying pairs",
            pairs.to_csv(index=False).encode(),
            "nextbuy_pairs.csv",
            "text/csv",
        )
    with st.expander("How to interpret these measures"):
        st.write(
            "Support = orders containing A and B / all selected orders. Confidence A → B = co-purchases / orders containing A. Lift = support / (share containing A × share containing B). Lift above 1 indicates a positive association, not causation. Rare pairs have unstable estimates; inspect counts."
        )


def model_lab(directory: str, signature: tuple) -> None:
    st.caption("04 / MODEL LAB")
    st.title("Measure before making claims.")
    st.info(
        "Diagnostic task: classify whether an already purchased article was previously bought. Cart position is observed after selection. These probabilities are not forecasts of a future purchase."
    )
    st.write(
        "Fixed protocol: customers with at least three observed orders; older orders supply history, the penultimate supplies training and the last supplies test. History statistics are frozen for both sets. The experiment always uses the full selected data source; exploration filters do not affect it."
    )
    if st.button("Run fixed evaluation", type="primary"):
        with st.spinner("Fitting a prevalence baseline, logistic regression and random forest…"):
            try:
                result = experiment(directory, signature)
                st.session_state["experiment_result"] = (directory, signature, result)
            except ValueError as exc:
                st.error(str(exc))
    saved = st.session_state.get("experiment_result")
    if not saved or saved[:2] != (directory, signature):
        st.caption(
            "Evaluation runs on demand and remains available while navigating. No hyperparameter search uses the test set."
        )
        return
    result = saved[2]
    st.success("Fixed evaluation complete · threshold 0.50 · seed 42")
    sizes = result["sizes"]
    st.caption(
        f"History: {sizes['history_items']:,} articles · Training: {sizes['train_items']:,} · Test: {sizes['test_items']:,} · Customers excluded for insufficient history: {sizes['excluded_users']:,}"
    )
    st.dataframe(result["scores"], hide_index=True, width="stretch")
    st.caption(
        f"ROC-AUC: ranking quality; 0.5 = chance. Average precision: precision–recall summary; constant baseline = test prevalence ({result['test_prevalence']:.3f}). Brier score: probability error; lower is better. No automatic winner is selected on test."
    )
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        for name, detail in result["details"].items():
            fig.add_scatter(x=detail["roc"][0], y=detail["roc"][1], name=name, mode="lines")
        fig.update_layout(
            title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate"
        )
        chart(fig)
    with b:
        fig = go.Figure()
        for name, detail in result["details"].items():
            fig.add_scatter(
                x=detail["pr"][0], y=detail["pr"][1], name=name, mode="lines", line_shape="hv"
            )
        fig.update_layout(
            title="Precision–recall curves", xaxis_title="Recall", yaxis_title="Precision"
        )
        chart(fig)
    name = st.selectbox("Inspect classifier", list(result["details"]))
    detail = result["details"][name]
    with st.expander("Classification report & confusion matrix"):
        st.dataframe(pd.DataFrame(detail["report"]).T, width="stretch")
        st.dataframe(
            pd.DataFrame(
                detail["confusion"],
                index=["Actual 0", "Actual 1"],
                columns=["Predicted 0", "Predicted 1"],
            )
        )
    st.subheader("Inspect an actual held-out article")
    index = int(st.number_input("Test article index", 0, len(result["test_rows"]) - 1, 0))
    row = result["test_rows"].iloc[index]
    st.write(f"{row.product_name} · order {row.order_id} · observed target {int(row.reordered)}")
    probability = float(detail["model"].predict_proba(result["X_test"].iloc[[index]])[0, 1])
    st.metric("Estimated repeat-status probability", f"{probability:.1%}")
    st.caption(
        "Model output for an observed test article; probability calibration has not been independently validated."
    )
    st.download_button(
        "Download evaluation metrics",
        result["scores"].to_csv(index=False).encode(),
        "nextbuy_evaluation.csv",
        "text/csv",
    )


def methodology(tables: dict, frame: pd.DataFrame, source: str) -> None:
    st.caption("05 / TRUST & METHODOLOGY")
    st.title("Know what the evidence can say.")
    st.subheader("Data provenance")
    st.write(
        f"Active source: **{source}**. This page audits the entire active cohort, independently of exploration filters."
    )
    st.write(
        "The original academic files identify the Instacart 2017 grocery dataset. They contain a partial extract, with no supplied licence, extraction recipe, calendar dates or prices. Real customer IDs are anonymous identifiers. The bundled synthetic demo is generated from scratch and must never be presented as customer evidence."
    )
    st.dataframe(quality_report(tables), hide_index=True, width="stretch")
    nulls = frame.isna().sum().rename("missing_cells").rename_axis("column").reset_index()
    st.dataframe(nulls.loc[nulls.missing_cells.gt(0)], hide_index=True, width="stretch")
    st.caption(
        "Missing elapsed days on first orders are structural. They stay missing for exploration; model preprocessing substitutes zero. Duplicated keys, invalid ranges and orphan references cause a clear loading error."
    )
    with st.container(border=True):
        st.subheader("Scientific boundaries")
        st.markdown("""- Basket completeness means all available source lines are preserved; the upstream extract may itself be incomplete.
- Last observed orders are not guaranteed to be the customer's actual last orders.
- Sequence ordering is per customer. Without calendar dates, this is not a global chronological backtest.
- History aggregates exclude training/test labels and stay frozen. No test-driven tuning or model selection is performed.
- Cart position and purchased-product selection make this a retrospective classification task, with no negative examples of unpurchased products.
- No confidence intervals, independent external validation or causal promotion effects are claimed.
- Day codes remain 0–6; elapsed days are capped at 30. No calendar period or currency is inferred.""")
    st.subheader("Source references")
    st.markdown(
        "[Instacart competition data](https://www.kaggle.com/competitions/instacart-market-basket-analysis/data) · [Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)"
    )


def main() -> None:
    # Only our own markup is styled; no generated Streamlit class names or hidden chrome.
    st.html("""<style>
    .nb-brand {font-size: 1.75rem; font-weight: 800; letter-spacing: -.06em; color: #087F8C;}
    .nb-kicker {font-size: .68rem; letter-spacing: .15em; color: #65758E; margin-bottom: 1.8rem;}
    .nb-footer {border-top: 1px solid #DDE5EF; padding-top: 1rem; color: #65758E; font-size: .8rem; margin-top: 2rem;}
    </style>""")
    sources = available_sources()
    with st.sidebar:
        st.html(
            '<div class="nb-brand">◈ NextBuy</div><div class="nb-kicker">GROCERY INTELLIGENCE / LAB</div>'
        )
        page = st.radio("Workspace", PAGES, label_visibility="collapsed")
        st.divider()
        source = st.selectbox("Data source", list(sources))
        st.caption("REAL LOCAL COHORT" if "Instacart" in source else "SYNTHETIC DEMONSTRATION")
    directory = sources[source]
    signature = tuple(
        (p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(directory.glob("*.csv"))
    )
    try:
        with st.spinner("Validating and loading grocery data…"):
            tables, frame = get_data(str(directory), signature)
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        st.error(f"Data unavailable: {exc}")
        st.info(
            "Restore the five CSV tables or run `python scripts/prepare_data.py`. See the README."
        )
        st.stop()
    with st.sidebar:
        st.caption(f"{frame.order_id.nunique():,} orders · {len(frame):,} article lines")
        if page in PAGES[:3]:
            st.divider()
            st.caption("ORDER SCOPE")
            days = st.multiselect("Day indices", list(range(7)), default=list(range(7)))
            hours = st.slider("Order hour", 0, 23, (0, 23))
        else:
            days, hours = list(range(7)), (0, 23)
        st.divider()
        st.caption("EPITECH research project\n\nRestored for reproducible exploration.")
    if "Synthetic" in source:
        st.warning(
            "Synthetic demo · generated grocery records. Results illustrate the software and are not Instacart findings."
        )
    else:
        st.caption("INSTACART / LOCAL SAMPLE · seed 42 · all available lines for sampled customers")
    selected = filter_orders(frame, days, hours)
    if page in PAGES[:3] and selected.empty:
        st.info(
            "No orders match these filters. Select at least one day and broaden the hour range."
        )
    elif page == PAGES[0]:
        overview(selected)
    elif page == PAGES[1]:
        explorer(selected)
    elif page == PAGES[2]:
        insights(selected)
    elif page == PAGES[3]:
        model_lab(str(directory), signature)
    else:
        methodology(tables, frame, source)
    st.html(
        '<div class="nb-footer">NextBuy / Grocery Intelligence &nbsp; · &nbsp; Observations, associations and model estimates are labelled separately.</div>'
    )
