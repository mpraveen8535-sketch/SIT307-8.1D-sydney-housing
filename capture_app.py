"""Capture the running app for the report after notebook execution."""

import os
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1365, "height": 900}, device_scale_factor=1.5, color_scheme="light")
    page.goto(os.environ.get("STREAMLIT_URL", "http://localhost:8501"), wait_until="networkidle")
    page.screenshot(path=str(FIG / "06_app_form.png"))
    page.get_by_role("button", name="Estimate sold price").click()
    page.get_by_text("Point estimate").wait_for()
    page.mouse.wheel(0, 900)
    page.wait_for_timeout(250)
    page.screenshot(path=str(FIG / "07_app_result.png"))

    page.get_by_role("tab", name="Batch CSV").click()
    page.locator('input[type="file"]').set_input_files(str(ROOT / "data" / "example_input.csv"))
    page.get_by_role("button", name="Download estimates").wait_for()
    page.mouse.wheel(0, 900)
    page.wait_for_timeout(250)
    page.screenshot(path=str(FIG / "08_app_batch.png"))
    browser.close()
print("Captured app screenshots")
