"""
Scrape the 14 services (documents, fee, time, delivery) from the PKM Urdu services page.

    python extract_services.py            # headless
    python extract_services.py --show     # watch the browser
Output: data/raw/extracted_services.json   (then run: python clean_data.py)
"""
import argparse
import time

from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from scraper_utils import BASE_URL, URDU_SWITCH_URL, clean_text, element_text, make_driver, save_json

SERVICES_PAGE_URL = f"{BASE_URL}/public/home/services_urdu"


def extract_active_tab(driver) -> dict:
    soup = BeautifulSoup(driver.page_source, "html.parser")
    container = soup.find("div", class_="tab-pane active")
    if not container:
        return {}
    data = {}
    for title in container.find_all("p", class_="services-title"):
        bullets = title.find_next_sibling("ul", class_="custom-bullets")
        if bullets:
            data[element_text(title)] = [element_text(li) for li in bullets.find_all("li") if element_text(li)]
    return data


def main(headless: bool):
    driver = make_driver(headless)
    try:
        driver.get(URDU_SWITCH_URL)   # sets the Urdu language cookie
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, "container")))
        driver.get(SERVICES_PAGE_URL)
        tabs = WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((By.XPATH, "//ul[@id='myUL']/li/a")))

        result = {}
        for tab in tabs:
            name = clean_text(tab.text)
            print(f"Extracting: {name}")
            try:
                driver.execute_script("arguments[0].scrollIntoView(true); arguments[0].click();", tab)
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".tab-pane.active")))
                time.sleep(0.5)   # let the pane finish swapping content
                data = extract_active_tab(driver)
                if data:
                    result[name] = data
                else:
                    print(f"  ! no content found for {name}")
            except Exception as e:  # noqa: BLE001 - keep going with the other tabs
                print(f"  ! error on {name}: {e}")
        save_json(result, "extracted_services.json")
    finally:
        driver.quit()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--show", action="store_true", help="show the browser window")
    main(headless=not p.parse_args().show)
