"""
Retrieval logic for the Punjab Police Khidmat Markaz Urdu assistant (no GUI code here).

How a question is answered:
  1. "What was my last question?"   -> answered from chat history.
  2. A district is named and the user asks where/phone/address/hours (or asks nothing service-specific)
                                    -> that district's centres, read straight from centers.json.
  3. Otherwise semantic search (cosine similarity) over services, service sections, districts and
     general facts. Below the similarity threshold the bot says it doesn't know.
  4. If the question is about one aspect of a service (fee, documents, time, delivery, age), only that
     section is returned; otherwise the whole service card.

The embedding model is wrapped in a tiny `Encoder` interface so it can be swapped (or faked in tests).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Protocol

import numpy as np

from text_utils import DISTRICT_ALIASES, has_urdu, normalize_for_search

ROOT = Path(__file__).parent
CLEAN_DIR = ROOT / "data" / "clean"
CACHE_DIR = ROOT / "data" / "cache"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Cosine similarity needed to answer. Tune it with evaluate.py on your machine.
DEFAULT_THRESHOLD = 0.45

MSG_UNKNOWN = ("معاف کیجیے، اس سوال کا جواب میرے پاس موجود نہیں۔\n"
               "آپ ان سروسز کے بارے میں پوچھ سکتے ہیں: {services}\n"
               "یا کسی ضلع کے خدمت مرکز کا پتہ اور فون نمبر پوچھیں، مثلاً: \"لاہور خدمت مرکز کا پتہ\"")
MSG_NOT_URDU = "براہ کرم اپنا سوال اردو میں لکھیں۔ مثال: \"کریکٹر سرٹیفیکیٹ کی فیس کتنی ہے؟\""
MSG_ASK_DISTRICT = ("پنجاب کے تمام 36 اضلاع میں پولیس خدمت مراکز موجود ہیں۔ "
                    "براہ کرم اپنے ضلع کا نام بتائیں، مثلاً: \"ملتان خدمت مرکز کا پتہ\"")
MSG_ANY_CENTRE = "نوٹ: یہ سروس پنجاب کے کسی بھی پولیس خدمت مرکز سے حاصل کی جا سکتی ہے۔"

# Words that usually mean the question depends on the previous one ("اس کی فیس؟").
FOLLOWUP_WORDS = ["اس", "اسکی", "اسکا", "اسکے", "وہ", "ان", "انکی", "انکا", "ایسے میں"]

# Strong keywords that meaning-based search can miss (found by evaluate.py). Checked in order.
# Each rule: (groups, kind, target). Every group must match; a group matches if ANY of its words appears.
KEYWORD_ROUTES = [
    ([["میڈیکو", "ایم ایل سی"]], "general", "میڈیکو لیگل سرٹیفیکیٹ"),
    ([["اسٹیٹس", "سٹیٹس", "ٹریک", "ٹریکنگ", "status", "track"]], "general", "درخواست کی ٹریکنگ"),
    ([["اصلی", "جعلی", "ویریفائی"]], "general", "دستاویز کی تصدیق"),
    ([["دوسرے ضلع", "کسی بھی ضلع", "دوسرے شہر", "کسی اور ضلع"]], "general", "کسی بھی ضلع سے سروس"),
    ([["لائسنس"], ["گم", "گمشدہ", "نقل", "خراب", "پھٹ"]], "service", "نقل ڈرائیونگ لائسنس"),
    ([["گم", "گمشدہ"]], "service", "گمشدگی رپورٹ"),
]

HISTORY_PHRASES = ["پچھلا سوال", "پہلے پوچھا", "آخری سوال", "پچھلے سوال", "میں نے کیا پوچھا"]

# Words that signal which part of a service the user wants.
ASPECT_WORDS = {
    "fee": ["فیس", "فی", "خرچ", "خرچہ", "قیمت", "روپے", "پیسے", "چارجز", "کتنے کا"],
    "required_documents": ["دستاویزات", "دستاویز", "کاغذات", "کاغذ", "ڈاکومنٹ", "ضروری چیزیں", "کیا چاہیے", "کیا درکار", "کیا لانا"],
    "processing_time": ["کتنے دن", "کتنا وقت", "کتنے وقت", "کب ملے", "کب تک", "ٹائم", "مدت", "دن لگتے", "دن میں"],
    "delivery": ["ڈلیوری", "ترسیل", "کیسے ملے", "کورئیر", "گھر پر"],
    "age_limit": ["عمر", "سال کی عمر", "عمر کی حد"],
}
# Words that signal the user wants a centre's location / contact / opening hours.
LOCATION_WORDS = ["پتہ", "پتا", "ایڈریس", "کہاں", "جگہ", "لوکیشن", "فون", "نمبر", "رابطہ", "ای میل",
                  "اوقات", "کھلتا", "کھلا", "بند", "ٹائمنگ", "مرکز", "مراکز", "address", "phone", "location"]


# --------------------------------------------------------------------------- encoder
class Encoder(Protocol):
    name: str

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalised vectors, shape (len(texts), dim)."""


class SentenceTransformerEncoder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer  # imported here so tests don't need it
        self.name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False).astype("float32")


# --------------------------------------------------------------------------- documents
@dataclass
class Doc:
    kind: str            # "service" | "section" | "district" | "general"
    key: str             # service name / district name / topic
    search_text: str     # what gets embedded (kept short: the model only reads ~128 tokens)
    answer: str          # what is shown to the user
    section: Optional[str] = None


def _bullets(items):
    return "\n".join(f"• {x}" for x in items)


def format_service(service: dict, only: Optional[str] = None) -> str:
    sections = service["sections"]
    keys = [only] if only else list(sections)
    parts = [service["name"]]
    for k in keys:
        sec = sections[k]
        parts.append(f"{sec['title']}:\n{_bullets(sec['items'])}")
    return "\n\n".join(parts)


def format_district(d: dict) -> str:
    n = len(d["centers"])
    lines = [f"{d['district']} میں {n} پولیس خدمت {'مرکز' if n == 1 else 'مراکز'}:"]
    for i, c in enumerate(d["centers"], 1):
        lines.append("")
        lines.append(f"{i}. {c['name']}" if n > 1 else c["name"])
        if c.get("address"):
            lines.append(f"   پتہ: {c['address']}")
        if c.get("phones"):
            lines.append(f"   فون: {'، '.join(c['phones'])}")
        if c.get("email"):
            lines.append(f"   ای میل: {c['email']}")
    if d.get("hours"):
        lines.append("")
        lines.append(f"اوقات کار: {'، '.join(d['hours'])}")
    return "\n".join(lines)


def build_documents(clean_dir: Path = CLEAN_DIR):
    services = json.loads((clean_dir / "services.json").read_text(encoding="utf-8"))
    districts = json.loads((clean_dir / "centers.json").read_text(encoding="utf-8"))
    general = json.loads((clean_dir / "general_info.json").read_text(encoding="utf-8"))

    docs: List[Doc] = []
    for s in services:
        titles = "، ".join(sec["title"] for sec in s["sections"].values())
        docs.append(Doc("service", s["name"], f"{s['name']} سروس: {titles}", format_service(s)))
        for key, sec in s["sections"].items():
            text = f"{s['name']} - {sec['title']}: {' ، '.join(sec['items'])}"
            docs.append(Doc("section", s["name"], text[:400], format_service(s, key), section=key))
    for d in districts:
        addresses = " ، ".join(c["address"] or "" for c in d["centers"])
        docs.append(Doc("district", d["district"],
                        f"{d['district']} پولیس خدمت مرکز کا پتہ، فون نمبر، ای میل: {addresses}"[:400],
                        format_district(d)))
    for g in general:
        docs.append(Doc("general", g["topic"], f"{g['topic']}: {g['text']}"[:400], g["text"]))
    return docs, {s["name"]: s for s in services}, {d["district"]: d for d in districts}


# --------------------------------------------------------------------------- history
@dataclass
class Turn:
    timestamp: datetime
    query: str
    response: str
    sources: List[str] = field(default_factory=list)
    is_meta: bool = False


class ChatHistory:
    def __init__(self, log_path: Optional[Path] = None):
        self.turns: List[Turn] = []
        self.log_path = log_path

    def add(self, turn: Turn):
        self.turns.append(turn)
        if self.log_path:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"time": turn.timestamp.isoformat(timespec="seconds"),
                                        "query": turn.query, "response": turn.response,
                                        "sources": turn.sources}, ensure_ascii=False) + "\n")
            except OSError:
                pass  # logging must never break the chat

    def recent(self, limit: int = 5) -> List[Turn]:
        return self.turns[-limit:]

    def last_user_query(self) -> str:
        for t in reversed(self.turns):
            if not t.is_meta:
                return t.query
        return ""


# --------------------------------------------------------------------------- bot
@dataclass
class Answer:
    text: str
    sources: List[str]
    score: Optional[float] = None
    context: List[str] = field(default_factory=list)   # facts handed to the LLM (empty = no LLM call)


def _contains_any(text: str, words) -> bool:
    """Whole-word match, so "فی" (fee) does not fire inside "فیصل آباد"."""
    return any(re.search(r"(?<!\w)" + re.escape(normalize_for_search(w)) + r"(?!\w)", text) for w in words)


class UrduChatbot:
    def __init__(self, encoder: Encoder, clean_dir: Path = CLEAN_DIR,
                 threshold: float = DEFAULT_THRESHOLD, cache_dir: Optional[Path] = CACHE_DIR,
                 history: Optional[ChatHistory] = None, llm=None):
        self.encoder = encoder
        self.llm = llm          # optional: llm.GeminiLLM, or anything with rewrite_query() and answer()
        self.threshold = threshold
        self.history = history or ChatHistory()
        self.docs, self.services, self.districts = build_documents(clean_dir)
        self.general_docs = {d.key: d for d in self.docs if d.kind == "general"}
        for _, kind, target in KEYWORD_ROUTES:   # catch typos in the rules at start-up, not mid-chat
            if target not in (self.services if kind == "service" else self.general_docs):
                raise ValueError(f"KEYWORD_ROUTES target not found in data: {target!r}")
        self.matrix = self._load_or_build_index(cache_dir)
        self._district_patterns = self._compile_districts()

    # ---- index -----------------------------------------------------------------------------
    def _load_or_build_index(self, cache_dir: Optional[Path]) -> np.ndarray:
        texts = [normalize_for_search(d.search_text) for d in self.docs]
        digest = hashlib.sha256((self.encoder.name + "\n" + "\n".join(texts)).encode("utf-8")).hexdigest()[:16]
        # One file per embedding model, so the local and Gemini indexes don't delete each other.
        # These files are small and committed to git: the server then never has to re-embed the documents.
        slug = re.sub(r"[^A-Za-z0-9]+", "-", self.encoder.name).strip("-")
        cache_file = cache_dir / f"index_{slug}_{digest}.npy" if cache_dir else None
        if cache_file and cache_file.exists():
            return np.load(cache_file)
        print(f"Building search index with {self.encoder.name} ({len(texts)} texts)...")
        matrix = self.encoder.encode(texts)
        if cache_file:
            cache_dir.mkdir(parents=True, exist_ok=True)
            for old in cache_dir.glob(f"index_{slug}_*.npy"):   # same model, old data -> stale
                old.unlink()
            for legacy in cache_dir.glob("index_????????????????.npy"):   # files from older versions
                legacy.unlink()
            np.save(cache_file, matrix)
        return matrix

    def _compile_districts(self):
        pats = []
        for name in self.districts:
            for alias in [name, *DISTRICT_ALIASES.get(name, [])]:
                a = normalize_for_search(alias)
                pats.append((len(a), name, re.compile(r"(?<!\w)" + re.escape(a) + r"(?!\w)")))
        return sorted(pats, key=lambda p: -p[0])  # longest first ("ڈیرہ غازی خان" before "خان")

    # ---- helpers ---------------------------------------------------------------------------
    def detect_district(self, qn: str) -> Optional[str]:
        for _, name, pat in self._district_patterns:
            if pat.search(qn):
                return name
        return None

    @staticmethod
    def keyword_route(qn: str):
        for groups, kind, target in KEYWORD_ROUTES:
            if all(_contains_any(qn, group) for group in groups):
                return kind, target
        return None

    @staticmethod
    def detect_aspect(qn: str) -> Optional[str]:
        for key, words in ASPECT_WORDS.items():
            if _contains_any(qn, words):
                return key
        return None

    def search(self, qn: str, kinds=None, k: int = 5):
        encode_query = getattr(self.encoder, "encode_query", self.encoder.encode)
        scores = self.matrix @ encode_query([qn])[0]
        order = np.argsort(-scores)
        hits = [(self.docs[i], float(scores[i])) for i in order if kinds is None or self.docs[i].kind in kinds]
        return hits[:k]

    def _service_names(self):
        return "، ".join(self.services)

    # ---- main entry ------------------------------------------------------------------------
    def answer(self, query: str, history: Optional[ChatHistory] = None) -> Answer:
        """history: pass one ChatHistory per user (web app). Defaults to the bot's own (desktop app)."""
        history = history if history is not None else self.history
        query = query.strip()
        if not query:
            return Answer("براہ کرم اپنا سوال لکھیں۔", [])
        qn = normalize_for_search(query)

        if _contains_any(qn, HISTORY_PHRASES):
            last = history.last_user_query()
            text = f"آپ کا پچھلا سوال تھا: {last}" if last else "آپ نے ابھی تک کوئی سوال نہیں پوچھا۔"
            ans = Answer(text, ["history"])
            history.add(Turn(datetime.now(), query, text, ans.sources, is_meta=True))
            return ans

        search_query, search_qn = query, qn
        past = [(t.query, t.response) for t in history.turns if not t.is_meta][-3:]
        if self.llm and (not has_urdu(query) or (past and _contains_any(qn, FOLLOWUP_WORDS))):
            try:   # Roman Urdu / English / "اس کی فیس؟" -> full Urdu question, so retrieval can find it
                search_query = self.llm.rewrite_query(query, past)
                search_qn = normalize_for_search(search_query)
            except Exception as e:  # noqa: BLE001 - fall back to the original wording
                print(f"[gemini] rewrite failed: {type(e).__name__}: {e}")

        ans = self._answer(search_query, search_qn)
        if self.llm and ans.context:
            try:
                ans.text = self.llm.answer(query, ans.context, past)
                ans.sources.append("llm")
            except Exception as e:  # noqa: BLE001 - quota hit, no internet, etc.: show the retrieved facts
                print(f"[gemini] answer failed: {type(e).__name__}: {e}")
                ans.text += "\n\n(AI جواب اس وقت دستیاب نہیں، اس لیے براہ راست معلومات دکھائی گئی ہیں۔)"
        history.add(Turn(datetime.now(), query, ans.text, ans.sources))
        return ans

    def _extra_context(self, hits, main_keys, limit: int = 2) -> List[str]:
        """Other relevant facts besides the main answer, so the LLM can handle mixed questions."""
        out, seen = [], set(main_keys)
        for doc, score in hits:
            if len(out) >= limit or score < self.threshold:
                break
            if doc.key in seen:
                continue
            seen.add(doc.key)
            out.append(format_service(self.services[doc.key]) if doc.kind in ("service", "section") else doc.answer)
        return out

    def _answer(self, query: str, qn: str) -> Answer:
        district = self.detect_district(qn)
        if not has_urdu(query) and not district:
            return Answer(MSG_NOT_URDU, [])

        aspect = self.detect_aspect(qn)
        wants_location = _contains_any(qn, LOCATION_WORDS)
        asks_for_centre = wants_location and _contains_any(qn, ["مرکز", "مراکز", "khidmat markaz", "pkm"])
        hits = self.search(qn, k=8)
        best_doc, best_score = hits[0]

        # Keyword rules first: they fix cases the similarity search gets wrong.
        route = self.keyword_route(qn)
        if route and route[0] == "general":
            doc = self.general_docs[route[1]]
            return Answer(doc.answer, [f"general:{doc.key}"], None,
                          [doc.answer, *self._extra_context(hits, [doc.key])])

        # A service named word-for-word in the question (or picked by a keyword rule) beats the ranking.
        named_service = route[1] if route else next(
            (n for n in sorted(self.services, key=len, reverse=True) if normalize_for_search(n) in qn), None)
        best_service = next(((d, s) for d, s in hits if d.kind in ("service", "section")), (None, -1.0))
        service_name = named_service or (best_service[0].key if best_service[0] else None)
        service_score = max(best_service[1], self.threshold if named_service else -1.0)
        service_confident = service_score >= self.threshold and (aspect is not None or named_service is not None)

        # 1) district named: give its centres unless the question is clearly about a service
        if district and not (service_confident and (aspect or not wants_location)):
            text = format_district(self.districts[district])
            return Answer(text, [f"centers:{district}"], best_score, [text])

        # 2) "where is the khidmat markaz?" without a district -> ask which district
        if asks_for_centre and not service_confident and not district:
            return Answer(MSG_ASK_DISTRICT, [], best_score)

        if not service_confident:
            if best_score < self.threshold:
                return Answer(MSG_UNKNOWN.format(services=self._service_names()), [], best_score)
            if best_doc.kind == "district":
                return Answer(MSG_ASK_DISTRICT, [], best_score)
            # 3) general fact
            if best_doc.kind == "general":
                return Answer(best_doc.answer, [f"general:{best_doc.key}"], best_score,
                              [best_doc.answer, *self._extra_context(hits, [best_doc.key])])
        elif best_doc.kind == "general" and aspect is None and best_score >= best_service[1] + 0.05:
            # e.g. "کریکٹر سرٹیفیکیٹ کی تصدیق کیسے کریں" names a service but asks about verification
            return Answer(best_doc.answer, [f"general:{best_doc.key}"], best_score,
                          [best_doc.answer, *self._extra_context(hits, [best_doc.key])])

        # 4) service: one section if a specific aspect was asked, else the whole card
        service = self.services[service_name]
        if aspect and aspect in service["sections"]:
            text = format_service(service, aspect)
            src = f"service:{service['name']}:{aspect}"
        else:
            text = format_service(service)
            if aspect:
                text += "\n\n(اس سروس کے لیے یہ معلومات ویب سائٹ پر دستیاب نہیں، اوپر مکمل تفصیل دی گئی ہے۔)"
            src = f"service:{service['name']}"
        if district:
            text += "\n\n" + MSG_ANY_CENTRE
        # the LLM gets the full service card, so it can also answer parts the keywords missed
        context = [format_service(service) + ("\n\n" + MSG_ANY_CENTRE if district else ""),
                   *self._extra_context(hits, [service["name"]])]
        return Answer(text, [src], service_score, context)


# Similarity scales differ between embedding models, so each has its own default threshold.
# The Gemini value is provisional: confirm it with  EMBEDDINGS=gemini python evaluate.py
THRESHOLDS = {"local": DEFAULT_THRESHOLD, "gemini": 0.65}


def embeddings_mode() -> str:
    from llm import load_env
    load_env()
    return os.environ.get("EMBEDDINGS", "local").strip().lower()


def make_encoder():
    """EMBEDDINGS=local (default): sentence-transformers on this machine.
    EMBEDDINGS=gemini: Google's embedding API, for small free servers."""
    if embeddings_mode() == "gemini":
        from llm import GeminiEncoder
        return GeminiEncoder()
    return SentenceTransformerEncoder()


def default_threshold() -> float:
    if os.environ.get("THRESHOLD"):
        return float(os.environ["THRESHOLD"])
    return THRESHOLDS.get(embeddings_mode(), DEFAULT_THRESHOLD)


def create_default_bot(threshold: Optional[float] = None, use_llm: bool = True,
                       history_log: Optional[Path] = None):
    """Embedding model (local or Gemini) + Gemini answers if configured. Returns (bot, status text)."""
    if threshold is None:
        threshold = default_threshold()
    llm, status = None, "AI جواب بند (--no-llm)"
    if use_llm:
        from llm import try_create_llm
        llm, status = try_create_llm()
    bot = UrduChatbot(make_encoder(), threshold=threshold,
                      history=ChatHistory(history_log), llm=llm)
    return bot, status
