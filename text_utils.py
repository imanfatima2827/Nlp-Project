"""Shared Urdu text helpers used by the cleaner and the chatbot."""
import re
import unicodedata

# Misspellings that come from the PKM website itself. Fixing them matters because
# users type the correct spelling (e.g. "ڈرائیونگ"), and the old data said "ڈرایئونگ".
# Explicit fixes run first; the generic rules below handle the rest.
SPELLING_FIXES = {
    "ڈرایئونگ": "ڈرائیونگ",
    "سایئکل": "سائیکل",
    "سرایئکی": "سرائیکی",
    "کمپیوڑرایزڈ": "کمپیوٹرائزڈ",
    "کوریئر": "کورئیر",
    "قوایئد": "قواعد",
    "ایف۔آئ۔آر۔": "ایف آئی آر ",
    "ذریئے": "ذریعے",
    "رحیم یا خان": "رحیم یار خان",
    "وہاری": "وہاڑی",
    "اوکارہ": "اوکاڑہ",
}

# Generic patterns: the site writes "ئی/ئے/ؤ" as a letter followed by a stray zabar,
# and "ائ" as "ایئ" (ٹایئم -> ٹائم, فایئل -> فائل).
GENERIC_FIXES = [
    ("ےَ", "ئے"),
    ("یَ", "ئی"),
    ("وَ", "ؤ"),
    ("یئ", "ئ"),
]

# Arabic-keyboard letters -> Urdu letters (users on Arabic layouts type these).
CHAR_MAP = str.maketrans({
    "ي": "ی", "ى": "ی", "ك": "ک", "ه": "ہ", "ة": "ہ", "ۀ": "ۂ",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
})

# Zabar, zer, pesh, tanween, shadda, sukun, khari zabar, etc.
DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭ]")

URDU_LETTERS = re.compile(r"[؀-ۿ]")


def fix_spelling(text: str) -> str:
    """Fix display text (used when cleaning data)."""
    text = unicodedata.normalize("NFC", text)
    for wrong, right in SPELLING_FIXES.items():
        text = text.replace(wrong, right)
    for wrong, right in GENERIC_FIXES:
        text = text.replace(wrong, right)
    return re.sub(r"[ \t]+", " ", text).strip()


# Common ways users spell the same word. Applied to queries AND documents, for matching only.
SEARCH_VARIANTS = {
    "سرٹیفکیٹ": "سرٹیفیکیٹ",
    "سرٹیفیکٹ": "سرٹیفیکیٹ",
    "سرٹفکیٹ": "سرٹیفیکیٹ",
    "ویری فیکیشن": "ویریفیکیشن",
    "ویریفکیشن": "ویریفیکیشن",
    "لائیسنس": "لائسنس",
    "لایسنس": "لائسنس",
    "ڈرائونگ": "ڈرائیونگ",
    "ڈرایونگ": "ڈرائیونگ",
    "کرایہ دار": "کرائے دار",
    "کرایے دار": "کرائے دار",
    "ایف ائی ار": "ایف آئی آر",
    # English loanwords people use instead of the website's Urdu terms
    "رینیو": "تجدید",
    "رینیول": "تجدید",
    "انٹرنیشنل": "بین الاقوامی",
    "ڈپلیکیٹ": "نقل",
}


def normalize_for_search(text: str) -> str:
    """Normalize text for matching only (never shown to the user)."""
    text = fix_spelling(text).translate(CHAR_MAP)
    text = DIACRITICS.sub("", text)
    for variant, canonical in SEARCH_VARIANTS.items():
        text = text.replace(variant, canonical)
    return re.sub(r"\s+", " ", text).strip().lower()


def has_urdu(text: str) -> bool:
    return bool(URDU_LETTERS.search(text))


# District aliases: alternate spellings, short forms and English/Roman names.
# Keys must match the cleaned district names in data/clean/centers.json.
DISTRICT_ALIASES = {
    "اٹک": ["attock", "campbellpur"],
    "بہاولنگر": ["بہاول نگر", "bahawalnagar"],
    "بہاولپور": ["بہاول پور", "bahawalpur"],
    "بھکر": ["bhakkar"],
    "چکوال": ["chakwal"],
    "چنیوٹ": ["chiniot"],
    "ڈی جی خان": ["ڈیرہ غازی خان", "dg khan", "d.g khan", "dera ghazi khan"],
    "فیصل آباد": ["فیصل اباد", "لائلپور", "faisalabad", "lyallpur"],
    "گجرانوالہ": ["گوجرانوالہ", "gujranwala"],
    "گجرات": ["gujrat"],
    "حافظ آباد": ["حافظ اباد", "hafizabad"],
    "جھنگ": ["jhang"],
    "جھلم": ["jhelum"],
    "قصور": ["kasur", "qasur"],
    "خانیوال": ["khanewal"],
    "خوشاب": ["khushab"],
    "لاہور": ["lahore"],
    "لیہ": ["layyah", "leiah"],
    "لودھران": ["لودھراں", "lodhran"],
    "منڈی بہاؤالدین": ["منڈی بہاؤ الدین", "منڈی بہاوالدین", "mandi bahauddin", "mandi bahaudin"],
    "میانوالی": ["mianwali"],
    "ملتان": ["multan"],
    "مظفر گڑھ": ["مظفرگڑھ", "muzaffargarh"],
    "ننکانہ صاحب": ["ننکانہ", "nankana"],
    "نارووال": ["narowal"],
    "اوکاڑہ": ["اوکارہ", "okara"],
    "پاکپتن": ["پاک پتن", "pakpattan"],
    "رحیم یار خان": ["رحیم یا خان", "rahim yar khan", "ryk"],
    "راجن پور": ["راجنپور", "rajanpur"],
    "راولپنڈی": ["پنڈی", "rawalpindi", "pindi"],
    "ساہیوال": ["sahiwal"],
    "سرگودھا": ["sargodha"],
    "شیخوپورہ": ["شیخو پورہ", "sheikhupura"],
    "سیالکوٹ": ["sialkot"],
    "ٹوبہ ٹیک سنگھ": ["ٹوبہ", "toba tek singh"],
    "وہاڑی": ["وہاری", "vehari"],
}
