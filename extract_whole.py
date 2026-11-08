"""
Scrape the PKM home page text (intro, IG message, contact details).

    python extract_whole.py
Output: data/raw/extracted_data_pkm.json

Note: the chatbot does NOT index this file directly - the home page mixes real content with menus,
and the verify/track pages are only form fields and a country list. Read the output and update
general_info() in clean_data.py by hand if the website's facts change.
"""
import argparse

from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from scraper_utils import URDU_SWITCH_URL, element_text, make_driver, save_json


def extract_home(driver) -> list:
    soup = BeautifulSoup(driver.page_source, "html.parser")
    for tag in soup(["nav", "header", "footer", "script", "style", "select", "option", "form"]):
        tag.decompose()   # drop menus, dropdowns (district / country lists) and scripts
    seen, out = set(), []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = element_text(el)
        if len(text) > 15 and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def main(headless: bool):
    driver = make_driver(headless)
    try:
        driver.get(URDU_SWITCH_URL)
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        save_json({URDU_SWITCH_URL: {"Content List": extract_home(driver)}}, "extracted_data_pkm.json")
    finally:
        driver.quit()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--show", action="store_true", help="show the browser window")
    main(headless=not p.parse_args().show)
