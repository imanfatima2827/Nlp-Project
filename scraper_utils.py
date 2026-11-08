"""Helpers shared by the three PKM scrapers."""
import json
import re
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

BASE_URL = "https://pkm.punjab.gov.pk"
URDU_SWITCH_URL = f"{BASE_URL}/public/langswitch/switchLanguage/urdu"
RAW_DIR = Path(__file__).parent / "data" / "raw"


def make_driver(headless: bool = True) -> webdriver.Chrome:
    """Selenium 4.6+ downloads the matching chromedriver itself - no hard-coded driver path needed."""
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1400,1000")
    return webdriver.Chrome(options=options)


def clean_text(text: str) -> str:
    """Collapse whitespace only. The old version deleted ':' '-' '/' which glued labels to values
    ("ایڈریسجنرل بس اسٹینڈ") and broke dates and phone numbers."""
    text = re.sub(r"[​-‏‪-‮﻿]", "", text)   # invisible direction marks
    return " ".join(text.split())


def element_text(el) -> str:
    """BeautifulSoup text with a space between child elements (get_text(strip=True) glued words)."""
    return clean_text(el.get_text(" ", strip=True))


def save_json(data, filename: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / filename
    path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"Saved {path}")
    return path
