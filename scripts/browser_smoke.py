"""Chromium checks of the real dashboard and authenticated API, without mocks."""

import json

from playwright.sync_api import expect, sync_playwright
from validation_server import ROOT, server

if __name__ == "__main__":
    output = ROOT / "reports/browser"
    output.mkdir(parents=True, exist_ok=True)
    with server() as (base, key), sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(base + "/ui/")
        page.get_by_label("API key", exact=True).fill(key)
        page.get_by_role("button", name="Connect", exact=True).click()
        expect(page.get_by_text("Synthetic environment.", exact=True)).to_be_visible()
        page.get_by_role("button", name="New experiment", exact=True).click()
        expect(page.get_by_role("dialog")).to_be_visible()
        page.keyboard.press("Escape")
        expect(page.get_by_role("dialog")).to_have_count(0)
        page.get_by_role("button", name="New experiment", exact=True).click()
        page.get_by_label("Requests per trial").fill("8")
        page.get_by_role("button", name="Run experiment", exact=True).click()
        expect(page.locator("tbody .badge.completed").first).to_be_visible(timeout=30000)
        page.screenshot(path=str(output / "experiments-desktop.png"), full_page=True)
        for name in ["Overview", "Optimization", "Models", "Infrastructure"]:
            page.locator("nav").get_by_role("button", name=name, exact=True).click()
            expect(page.locator("h1")).to_have_text(
                "Measure. Compare. Optimize." if name == "Overview" else name
            )
            page.screenshot(path=str(output / f"{name.lower()}-desktop.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.locator("nav").get_by_role("button", name="Overview", exact=True).click()
        expect(page.get_by_role("button", name="New experiment", exact=True)).to_be_visible()
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), (
            "Mobile page overflows viewport"
        )
        page.screenshot(path=str(output / "overview-mobile.png"), full_page=True)
        assert not errors, errors
        assert page.evaluate("localStorage.length") == 0
        assert page.evaluate("sessionStorage.length") == 0
        browser.close()
    (output / "result.json").write_text(
        json.dumps(
            {
                "passed": True,
                "browser": "Chromium",
                "checks": [
                    "real API authentication",
                    "experiment submission and completion",
                    "five views",
                    "Escape closes dialog",
                    "390px mobile overflow",
                    "no uncaught JS errors",
                    "no credentials in web storage",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print("Real browser validation passed")
