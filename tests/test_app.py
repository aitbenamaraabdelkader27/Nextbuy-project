from streamlit.testing.v1 import AppTest

from nextbuy_core.data import ROOT
from nextbuy_core.ui import PAGES


def test_all_pages_sources_filters_and_model_interaction():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not app.exception
    # Exercise the redistributable cloud path even when the real cohort exists locally.
    app.sidebar.selectbox[0].select("Synthetic demo · generated data").run()
    for page in PAGES:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
        if page == "Data explorer":
            next(w for w in app.selectbox if w.label == "Distribution").select("Order hour").run()
            app.text_input[0].set_value("Bananas").run()
            assert not app.exception
        if page == "Basket insights":
            next(w for w in app.selectbox if w.label == "Rank pairs by").select("lift").run()
            assert not app.exception
        if page == "Model lab":
            app.button[0].click().run()
            assert not app.exception
            assert len(app.success) == 1
            next(w for w in app.selectbox if w.label == "Inspect classifier").select(
                "Random forest"
            ).run()
            app.number_input[0].set_value(5).run()
            assert not app.exception
    app.sidebar.radio[0].set_value("Overview").run()
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any("No orders match" in x.value for x in app.info)


def test_local_cohort_pages_if_available():
    if not (ROOT / "data/local/orders.csv").exists():
        return
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    for page in PAGES:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
