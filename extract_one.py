"""
Scrape every district's Khidmat Markaz centres (address, email, phone, hours) from the PKM contact page.

    python extract_one.py            # headless
    python extract_one.py --show     # watch the browser
Output: data/raw/extracted_data.json   (then run: python clean_data.py)
"""
import argparse
import time

from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from scraper_utils import BASE_URL, URDU_SWITCH_URL, clean_text, element_text, make_driver, save_json

CONTACT_MAP_URL = f"{BASE_URL}/public/home/contact_map_urdu"

EXCLUDED_HEADINGS = {"خدمت مراکز", "تعارف", "ہماری سروسز", "ہم سے رابطہ کریں", "Contact Details", "اہم خصوصیات"}


def wait_for_tab(driver, tab):
    """Wait until the clicked tab's pane is visible instead of a fixed 6-second sleep."""
    href = tab.get_attribute("href") or ""
    if "#" in href and href.split("#", 1)[1]:
        try:
            WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.ID, href.split("#", 1)[1])))
            time.sleep(0.5)
            return
        except Exception:  # noqa: BLE001 - fall back to a short sleep below
            pass
    time.sleep(2)


def extract_page(driver) -> dict:
    soup = BeautifulSoup(driver.page_source, "html.parser")
    data, heading = {}, None
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "li", "p"]):
        if el.name.startswith("h"):
            text = element_text(el)
            heading = None if text in EXCLUDED_HEADINGS else text
            if heading:
                data.setdefault(heading, [])
        elif heading:
            text = element_text(el)
            if text and text not in data[heading]:
                data[heading].append(text)
    return data


def main(headless: bool):
    driver = make_driver(headless)
    try:
        driver.get(URDU_SWITCH_URL)
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, "container")))
        driver.get(CONTACT_MAP_URL)
        tabs = WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((By.XPATH, "//ul[@id='myUL']/li/a")))

        result = {}
        for tab in tabs:
            name = clean_text(tab.text)
            print(f"Extracting: {name}")
            try:
                driver.execute_script("arguments[0].scrollIntoView(true); arguments[0].click();", tab)
                wait_for_tab(driver, tab)
                data = extract_page(driver)
                if data:
                    result[name] = data
            except Exception as e:  # noqa: BLE001
                print(f"  ! error on {name}: {e}")
        save_json(result, "extracted_data.json")
    finally:
        driver.quit()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--show", action="store_true", help="show the browser window")
    main(headless=not p.parse_args().show)
