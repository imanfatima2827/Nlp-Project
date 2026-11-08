"""
Take screenshots of the running web app for the manual and report.

    1. In one terminal:      python app.py
    2. In a second terminal: pip install playwright
                             python take_screenshots.py
Screenshots are saved to docs/screenshots/. Uses Microsoft Edge (already on Windows) or Chrome,
so no extra browser download is needed.
"""
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:7860"
OUT = Path(__file__).parent / "docs" / "screenshots"


def launch(p):
    for channel in ("msedge", "chrome", None):
        try:
            return p.chromium.launch(channel=channel) if channel else p.chromium.launch()
        except Exception:  # noqa: BLE001 - try the next browser
            continue
    raise SystemExit("No browser found. Install Edge or Chrome, or run: playwright install chromium")


def wait_until_stable(page, before: str, timeout_s: int = 60):
    """Wait until the chat text has changed and then stopped changing (answer finished)."""
    chat = page.locator("#chat")
    start, last, stable_since = time.time(), before, None
    while time.time() - start < timeout_s:
        now = chat.inner_text()
        if now != before and now == last:
            stable_since = stable_since or time.time()
            if time.time() - stable_since > 2.5:
                return
        else:
            stable_since = None
        last = now
        time.sleep(0.5)
    print("  (answer took too long, taking the screenshot anyway)")


def ask(page, question: str):
    before = page.locator("#chat").inner_text()
    box = page.locator("#q textarea")
    box.fill(question)
    box.press("Enter")
    wait_until_stable(page, before)


def new_chat(page):
    page.get_by_role("button", name="نئی گفتگو").click()
    time.sleep(1)


def shot(page, name: str):
    path = OUT / name
    page.screenshot(path=str(path), full_page=True)
    print(f"saved {path}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = launch(p)

        # ---- desktop
        page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
        page.goto(URL, wait_until="networkidle")
        page.wait_for_selector("#q textarea")
        time.sleep(3)                                   # let the Urdu web font load
        shot(page, "01_home.png")

        ask(page, "کریکٹر سرٹیفیکیٹ کی فیس کتنی ہے؟")
        ask(page, "اس کے لیے کون سے کاغذات چاہییں؟")
        shot(page, "02_service_and_followup.png")

        new_chat(page)
        ask(page, "lahore khidmat markaz ka address kya hai")
        shot(page, "03_roman_urdu_district.png")

        new_chat(page)
        ask(page, "لائسنس گم ہو گیا ہے نقل کیسے بنے گی")
        shot(page, "04_lost_licence.png")

        new_chat(page)
        ask(page, "آج موسم کیسا ہے؟")
        shot(page, "05_out_of_scope.png")
        page.close()

        # ---- phone
        mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=3,
                                  is_mobile=True, has_touch=True)
        mobile.goto(URL, wait_until="networkidle")
        mobile.wait_for_selector("#q textarea")
        time.sleep(3)
        ask(mobile, "پولیس ویریفیکیشن کی فیس")
        shot(mobile, "06_mobile.png")
        browser.close()
    print(f"\nDone. Send the files in {OUT} back to Claude.")


if __name__ == "__main__":
    main()
