"""Exercise the running dashboard and save screenshots plus browser evidence."""

import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/screenshots"
OUT.mkdir(parents=True, exist_ok=True)


def check() -> None:
    with sync_playwright() as p:
        cached = sorted(
            (Path.home() / ".cache/ms-playwright").glob("chromium-*/chrome-linux64/chrome")
        )
        executable = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE") or (
            str(cached[-1]) if cached else None
        )
        browser = p.chromium.launch(
            executable_path=executable, headless=True, args=["--no-sandbox"]
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1200}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto("http://127.0.0.1:8501", wait_until="networkidle")
        page.get_by_role("heading", name="From grocery baskets to clear insights.").wait_for()
        page.locator(".nb-footer").wait_for(state="attached")
        page.locator(".js-plotly-plot").nth(1).wait_for()
        page.wait_for_timeout(700)
        page.screenshot(path=str(OUT / "overview-desktop.png"), full_page=True)
        # Exercise an actual hour-range filter in the running browser.
        first_slider = page.get_by_role("slider").first
        first_slider.focus()
        first_slider.press("ArrowRight")
        page.locator(".nb-footer").wait_for(state="attached")
        page.wait_for_timeout(700)
        assert page.locator('[data-testid="stException"]').count() == 0
        first_slider.press("ArrowLeft")
        page.wait_for_timeout(700)
        pages = {
            "Data explorer": "A closer look at every basket.",
            "Basket insights": "What travels together?",
            "Model lab": "Measure before making claims.",
            "Methodology & quality": "Know what the evidence can say.",
        }
        evidence = []
        for name, heading in pages.items():
            page.get_by_text(name, exact=True).click()
            page.get_by_role("heading", name=heading, exact=True).wait_for()
            if name == "Data explorer":
                page.get_by_role("textbox", name="Search product name").fill("banana")
                page.get_by_role("textbox", name="Search product name").press("Enter")
                page.get_by_text("matching article lines", exact=False).wait_for()
                with page.expect_download() as download:
                    page.get_by_role("button", name="Download matching articles (CSV)").click()
                saved = ROOT / "test-results/export.csv"
                saved.parent.mkdir(exist_ok=True)
                download.value.save_as(str(saved))
                assert saved.stat().st_size > 0
            if name == "Model lab":
                page.get_by_role("button", name="Run fixed evaluation").click()
                page.get_by_text("Fixed evaluation complete", exact=False).wait_for(timeout=90000)
                page.get_by_role("spinbutton", name="Test article index").fill("4")
                page.get_by_role("spinbutton", name="Test article index").press("Enter")
                page.get_by_text("Estimated repeat-status probability", exact=True).wait_for()
            page.locator(".nb-footer").wait_for(state="attached")
            page.wait_for_timeout(700)
            assert page.locator('[data-testid="stException"]').count() == 0, name
            evidence.append({"page": name, "charts": page.locator(".js-plotly-plot").count()})
            page.evaluate(
                "document.querySelectorAll('*').forEach(el => { if (el.scrollTop) el.scrollTop = 0; });"
            )
            page.screenshot(
                path=str(OUT / (name.lower().replace(" ", "-").replace("&", "and") + ".png")),
                full_page=True,
            )
        page.get_by_text("Overview", exact=True).click()
        page.get_by_role("heading", name="From grocery baskets to clear insights.").wait_for()
        for width, height, label in [(768, 1024, "tablet"), (390, 844, "mobile")]:
            page.set_viewport_size({"width": width, "height": height})
            page.evaluate(
                "document.querySelectorAll('*').forEach(el => { if (el.scrollTop) el.scrollTop = 0; });"
            )
            page.wait_for_timeout(700)
            assert page.locator('[data-testid="stException"]').count() == 0
            page.screenshot(path=str(OUT / f"overview-{label}.png"), full_page=True)
            evidence.append(
                {
                    "viewport": label,
                    "horizontal_overflow": page.evaluate(
                        "document.documentElement.scrollWidth > innerWidth"
                    ),
                }
            )
        browser.close()
        (ROOT / "docs/browser-check.json").write_text(
            json.dumps({"pages": evidence, "page_errors": errors}, indent=2)
        )
        assert not errors, errors
        print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    check()
