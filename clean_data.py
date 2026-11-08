"""
Turn the raw scraped JSON (data/raw) into clean, structured JSON (data/clean).

    python clean_data.py

Outputs:
    data/clean/services.json     14 services, each with documents / fee / time / delivery
    data/clean/centers.json      Khidmat Markaz centres per district (address, phone, email, hours)
    data/clean/general_info.json Short facts about PKM taken from the home page

What it fixes in the raw data:
    * labels glued to values  ("ایڈریسجنرل بس اسٹینڈ" -> address: "جنرل بس اسٹینڈ")
    * misspellings from the website ("ڈرایئونگ" -> "ڈرائیونگ", "ٹایئم" -> "ٹائم")
    * wrong district spellings ("رحیم یا خان" -> "رحیم یار خان")
    * trailing colons / duplicate colons in section names
    * multiple phone numbers in one field, "92" country prefix, dots in numbers
    * opening dates whose separators were stripped ("4112017"); ambiguous ones are left as null
    * drops navigation / form / country-list noise from extracted_data_pkm.json
"""
import json
import re
from datetime import date
from pathlib import Path

from text_utils import fix_spelling

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"

# Urdu section heading -> stable English key used by the chatbot.
SECTION_KEYS = {
    "ضروری دستاویزات": "required_documents",
    "پراسسنگ فی": "fee",
    "درخواست مکمل ہونے کا ٹائم": "processing_time",
    "ڈلیوری کا ذریعہ": "delivery",
    "لرنرز پرمٹ کے لیے عمر کی حد": "age_limit",
}


def clean_heading(text: str) -> str:
    return fix_spelling(text).replace(":", "").strip()


# --------------------------------------------------------------------------- services
def clean_services(raw: dict) -> list:
    services = []
    for name, sections in raw.items():
        service = {"name": fix_spelling(name), "sections": {}}
        for heading, items in sections.items():
            heading = clean_heading(heading)
            key = SECTION_KEYS.get(heading)
            if key is None:   # the website added a new section: keep it, but tell the developer
                print(f"warning: new section {heading!r} in {name!r} - add it to SECTION_KEYS / ASPECT_WORDS")
                key = heading
            values = [fix_spelling(x) for x in items if x.strip()]
            service["sections"][key] = {"title": heading, "items": values}
        services.append(service)
    return services


# --------------------------------------------------------------------------- centres
LABELS = [("email", "ای میل ایڈریس"), ("address", "ایڈریس"), ("phone", "فون")]  # email first: it contains "ایڈریس"


def split_label(line: str):
    for key, label in LABELS:
        if line.startswith(label):
            return key, line[len(label):].strip(" :")
    return None, line


def clean_phones(value: str) -> list:
    phones = []
    for part in re.split(r"[\s,/]+", value):
        digits = re.sub(r"\D", "", part)
        if not digits:
            continue
        if digits.startswith("92") and len(digits) >= 11:   # +92 country code -> 0
            digits = "0" + digits[2:]
        phones.append(digits)
    return phones


def parse_opening_date(raw: str):
    """Return an ISO date, or None if it can't be read unambiguously.
    New scrapes keep separators ("4-11-2017", read as day-month-year). Old scrapes lost them
    ("4112017"), so those are only accepted when exactly one day/month split is a valid date."""
    m = re.search(r"(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{4})", raw)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1))).isoformat()
        except ValueError:
            return None
    digits = re.sub(r"\D", "", raw)
    if len(digits) < 6:
        return None
    day_month, year = digits[:-4], int(digits[-4:])
    candidates = set()
    for cut in range(1, len(day_month)):
        d, m = day_month[:cut], day_month[cut:]
        if len(d) > 2 or len(m) > 2:
            continue
        try:
            candidates.add(date(year, int(m), int(d)).isoformat())
        except ValueError:
            pass
    return candidates.pop() if len(candidates) == 1 else None


def clean_centers(raw: dict):
    districts, head_office = [], None
    for district_raw, blocks in raw.items():
        district = fix_spelling(district_raw)
        info = {"district": district, "centers": [], "hours": [], "opening_date": None}

        for block_name, lines in blocks.items():
            if block_name == "آپریشنل معلومات":
                for line in lines:
                    line = fix_spelling(re.sub(r"\s+", " ", line))
                    if line.startswith("افتتاح کی تاریخ"):
                        info["opening_date"] = parse_opening_date(line)
                    else:
                        info["hours"].append(line)
                continue
            if block_name == "نقشہ پر لوکیشن" or not lines:
                continue

            center = {"name": fix_spelling(block_name), "address": None, "email": None, "phones": []}
            for line in lines:
                key, value = split_label(line)
                if key == "phone":
                    center["phones"] = clean_phones(value)
                elif key and value:
                    center[key] = fix_spelling(value) if key == "address" else value.strip()
            info["centers"].append(center)

        if district == "پنجاب":          # not a district: it is the provincial head office entry
            head_office = info
        else:
            districts.append(info)
    return districts, head_office


# --------------------------------------------------------------------------- general info
# The home page text was scraped with words glued together and mixed with menus, so it can't be
# split automatically. These entries are the factual sentences from that page, with spacing restored.
# Keep them in sync with the website when you re-scrape.
def general_info(head_office) -> list:
    hours = head_office["hours"] if head_office else []
    return [
        {
            "topic": "پولیس خدمت مرکز کیا ہے",
            "text": "پنجاب پولیس نے پنجاب کے 36 اضلاع میں عوامی سہولت کے مراکز قائم کیے ہیں جنہیں پولیس خدمت مرکز کہا جاتا ہے۔ "
                    "ون ونڈو آپریشن کے ذریعے شہری آسانی سے اپنی مطلوبہ سروسز حاصل کر سکتے ہیں۔ "
                    "پنجاب انفارمیشن ٹیکنالوجی بورڈ (پی آئی ٹی بی) پنجاب پولیس کا ٹیکنالوجی پارٹنر ہے۔",
        },
        {
            "topic": "کسی بھی ضلع سے سروس",
            "text": "انٹیگریٹڈ خدمت مرکز سسٹم کے تحت شہری پنجاب کے کسی بھی ضلع کے پولیس خدمت مرکز سے خدمات حاصل کر سکتے ہیں۔ "
                    "کریکٹر سرٹیفیکیٹ یا لائسنس کی تجدید کے لیے اپنے آبائی ضلع جانا ضروری نہیں۔",
        },
        {
            "topic": "پولیس کے خلاف شکایت",
            "text": "محکمہ پولیس کے خلاف شکایات بھی پولیس خدمت مراکز میں جمع کرائی جا سکتی ہیں۔",
        },
        {
            "topic": "میڈیکو لیگل سرٹیفیکیٹ",
            "text": "تمام ڈسٹرکٹ ہیڈ کوارٹر ہسپتالوں میں پولیس خدمت کاؤنٹر بنائے جا رہے ہیں تاکہ شہری میڈیکو لیگل سرٹیفیکیٹ "
                    "پولیس اسٹیشن جائے بغیر وہیں سے حاصل کر سکیں۔",
        },
        {
            "topic": "رابطہ اور ہیڈ آفس",
            "text": "پولیس خدمت مرکز ہیڈ آفس: سینٹرل پولیس آفس (سی پی او)، لاہور۔ فون: 04299211356۔ "
                    "ای میل: pkm.punjab@gmail.com، pkm.punjab@punjabpolice.gov.pk۔ ویب سائٹ: punjabpolice.gov.pk"
                    + (f"۔ اوقات کار: {'، '.join(hours)}" if hours else ""),
        },
        {
            "topic": "دستاویز کی تصدیق",
            "text": "خدمت مرکز سے جاری کردہ دستاویز (کریکٹر سرٹیفیکیٹ یا جنرل پولیس ویریفیکیشن) کی تصدیق ویب سائٹ "
                    "pkm.punjab.gov.pk/verify پر کی جا سکتی ہے۔ اس کے لیے ضلع، شناختی کارڈ نمبر، دستاویز کی قسم، "
                    "پاسپورٹ نمبر، ملک، تاریخ اجرا اور درخواست نمبر درکار ہیں۔",
        },
        {
            "topic": "درخواست کی ٹریکنگ",
            "text": "اپنی درخواست کا اسٹیٹس ویب سائٹ pkm.punjab.gov.pk/track پر دیکھا جا سکتا ہے۔ "
                    "اس کے لیے ضلع، شناختی کارڈ نمبر، دستاویز کی قسم، پاسپورٹ نمبر اور ملک درج کریں۔",
        },
    ]


def main():
    CLEAN.mkdir(parents=True, exist_ok=True)
    services = clean_services(json.loads((RAW / "extracted_services.json").read_text(encoding="utf-8")))
    districts, head_office = clean_centers(json.loads((RAW / "extracted_data.json").read_text(encoding="utf-8")))
    info = general_info(head_office)

    for name, obj in [("services.json", services), ("centers.json", districts), ("general_info.json", info)]:
        (CLEAN / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    n_centers = sum(len(d["centers"]) for d in districts)
    print(f"services: {len(services)} | districts: {len(districts)} | centres: {n_centers} | general facts: {len(info)}")
    missing = [c["name"] for d in districts for c in d["centers"] if not (c["address"] and c["phones"])]
    if missing:
        print("centres missing address or phone:", missing)
    no_date = [d["district"] for d in districts if not d["opening_date"]]
    if no_date:
        print("opening date ambiguous/unknown (left null):", no_date)


if __name__ == "__main__":
    main()
