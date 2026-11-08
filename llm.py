"""
Gemini answer writer (free tier from Google AI Studio).

Setup:
  1. Get a free key: https://aistudio.google.com/apikey
  2. Create a file named .env next to this file containing:
         GEMINI_API_KEY=your-key-here
         # optional, default is Google's alias for the latest Flash model:
         # GEMINI_MODEL=gemini-flash-latest
  3. pip install google-genai

The retrieval step (chatbot_core.py) still decides WHAT information is relevant. Gemini only rewrites
that information into a natural Urdu answer, and is told never to add facts that aren't in it.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

DEFAULT_MODEL = "gemini-flash-latest"

SYSTEM_PROMPT = """آپ پنجاب پولیس خدمت مرکز (PKM) کے اردو معلوماتی اسسٹنٹ ہیں۔
قواعد:
1. صرف نیچے دی گئی "معلومات" سے جواب دیں۔ کوئی فیس، دستاویز، پتہ، فون نمبر یا مدت اپنی طرف سے نہ بتائیں۔
2. اگر معلومات میں جواب موجود نہیں تو صاف کہیں کہ یہ معلومات دستیاب نہیں، اور قریبی خدمت مرکز یا ہیلپ لائن 04299211356 سے رابطے کا مشورہ دیں۔
3. جواب ہمیشہ اردو رسم الخط میں دیں، چاہے سوال رومن اردو یا انگریزی میں ہو۔
4. مختصر اور واضح رہیں۔ فہرست ہو تو ہر نکتہ نئی لائن پر "•" سے شروع کریں۔ مارک ڈاؤن (** یا #) استعمال نہ کریں۔
5. فیس اور نمبرز بالکل ویسے ہی لکھیں جیسے معلومات میں ہیں۔"""

REWRITE_PROMPT = """نیچے ایک صارف کا سوال ہے جو رومن اردو، انگریزی یا اردو میں ہو سکتا ہے، اور پچھلی گفتگو بھی ہو سکتی ہے۔
اسے ایک مکمل، خود مختار سوال کی شکل میں اردو رسم الخط میں دوبارہ لکھیں تاکہ تلاش میں استعمال ہو سکے۔
اگر سوال پچھلی گفتگو کا حوالہ دیتا ہے (مثلاً "اس کی فیس؟") تو سروس یا ضلع کا نام شامل کریں۔
سروسز اور اضلاع کے نام ویب سائٹ والی اردو میں لکھیں (مثلاً کریکٹر سرٹیفیکیٹ، ڈرائیونگ لائسنس کی تجدید، راولپنڈی)۔
صرف دوبارہ لکھا گیا سوال لوٹائیں، کچھ اور نہیں۔"""


def load_env(path: Path = Path(__file__).parent / ".env"):
    """Minimal .env reader so python-dotenv isn't required. Real env vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _format_history(history: List[Tuple[str, str]]) -> str:
    return "\n".join(f"صارف: {q}\nاسسٹنٹ: {a[:300]}" for q, a in history)


# Tried in order. The first that works becomes the default for the rest of the session.
FALLBACK_MODELS = ["gemini-flash-lite-latest", "gemini-2.5-flash"]
RETRY_CODES = {429, 500, 502, 503, 504}     # busy / overloaded / deadline: worth one retry


def _code(e) -> Optional[int]:
    code = getattr(e, "code", None)
    return code if isinstance(code, int) else None


class GeminiLLM:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        from google import genai            # pip install google-genai
        from google.genai import types
        load_env()
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set (put it in a .env file next to llm.py)")
        first = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self.models = [first] + [m for m in FALLBACK_MODELS if m != first]
        self.model = first
        self.types = types
        self.no_thinking = {m: True for m in self.models}   # becomes False if a model rejects the setting
        # Per-request limit. google-genai also sends it to Google as the server deadline, so it must not be
        # too short: "thinking" models were hitting 504 DEADLINE_EXCEEDED with 30 s.
        self.timeout_s = int(os.environ.get("GEMINI_TIMEOUT", "60"))
        self.total_budget_s = int(os.environ.get("GEMINI_TOTAL_BUDGET", "75"))   # all retries together
        # google-genai 3.x switched the default to the "v1" API; the model alias and system instructions
        # used here need "v1beta" (the 1.x/2.x default). Override with GEMINI_API_VERSION if ever needed.
        api_version = os.environ.get("GEMINI_API_VERSION", "v1beta")
        self.client = genai.Client(api_key=api_key,
                                   http_options=types.HttpOptions(api_version=api_version,
                                                                  timeout=self.timeout_s * 1000))

    def health_check(self):
        """One tiny real request, so start-up reports the truth instead of 'active' when calls will fail."""
        self._generate("Reply with the single word OK.", "OK?", temperature=0.0)

    def _config(self, model: str, system: str, temperature: float):
        kwargs = dict(system_instruction=system, temperature=temperature)
        if self.no_thinking.get(model):
            # Our answers only rephrase given facts, so "thinking" just adds delay (and timeouts).
            kwargs["thinking_config"] = self.types.ThinkingConfig(thinking_budget=0)
        return self.types.GenerateContentConfig(**kwargs)

    def _generate(self, system: str, prompt: str, temperature: float) -> str:
        start, last_error = time.monotonic(), None
        for model in list(self.models):
            attempt = 0
            while attempt < 2 and time.monotonic() - start < self.total_budget_s:
                try:
                    resp = self.client.models.generate_content(
                        model=model, contents=prompt, config=self._config(model, system, temperature))
                    text = (resp.text or "").strip()
                    if not text:
                        raise RuntimeError("empty response from Gemini (possibly blocked by safety filters)")
                    if model != self.model:
                        print(f"[gemini] using {model} (instead of {self.model})")
                        self.model = model
                        self.models.remove(model)
                        self.models.insert(0, model)
                    return text.replace("**", "")
                except Exception as e:  # noqa: BLE001
                    last_error, code = e, _code(e)
                    if code == 400 and self.no_thinking.get(model):
                        self.no_thinking[model] = False      # model doesn't accept thinking_budget=0
                        continue                              # retry at once without it (not counted)
                    if code in RETRY_CODES and attempt == 0:
                        print(f"[gemini] {model}: {code}, retrying once")
                        time.sleep(2)
                        attempt += 1
                        continue
                    print(f"[gemini] {model} failed: {type(e).__name__}: {str(e)[:200]}")
                    break                                     # try the next model
        raise last_error or RuntimeError("Gemini: time budget used up")

    def rewrite_query(self, query: str, history: List[Tuple[str, str]]) -> str:
        prompt = (f"پچھلی گفتگو:\n{_format_history(history)}\n\n" if history else "") + f"سوال: {query}"
        return self._generate(REWRITE_PROMPT, prompt, temperature=0.0).splitlines()[0].strip()

    def answer(self, query: str, contexts: List[str], history: List[Tuple[str, str]]) -> str:
        info = "\n\n---\n\n".join(contexts)
        prompt = ((f"پچھلی گفتگو:\n{_format_history(history)}\n\n" if history else "")
                  + f"معلومات:\n{info}\n\nسوال: {query}")
        return self._generate(SYSTEM_PROMPT, prompt, temperature=0.2)


def try_create_llm():
    """Return (llm, status message). llm is None when Gemini isn't set up; the bot then works without it."""
    try:
        llm = GeminiLLM()
    except ImportError:
        return None, "Gemini بند: google-genai انسٹال نہیں (pip install google-genai)"
    except Exception as e:  # noqa: BLE001
        print(f"[gemini] {e}")
        return None, f"Gemini بند: {e}"
    try:
        llm.health_check()
        return llm, f"Gemini ({llm.model}) فعال ہے"
    except Exception as e:  # noqa: BLE001
        if _code(e) in RETRY_CODES:
            # Google was busy/slow just now. Keep Gemini on: every answer falls back safely if it fails.
            print(f"[gemini] start-up check: Google busy ({_code(e)}); keeping Gemini on")
            return llm, f"Gemini ({llm.model}) فعال ہے (ابتدائی جانچ میں سرور مصروف تھا)"
        print(f"[gemini] start-up check failed: {type(e).__name__}: {e}")
        return None, f"Gemini بند: {type(e).__name__}: {str(e)[:150]}"


class GeminiEncoder:
    """Embeddings from Google's API instead of the local model. Used for small free servers (512 MB RAM):
    no PyTorch and no 1 GB model download. Turn on with EMBEDDINGS=gemini."""

    def __init__(self, model: Optional[str] = None, dims: int = 768):
        from google import genai
        from google.genai import types
        load_env()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("EMBEDDINGS=gemini needs GEMINI_API_KEY")
        self.types = types
        self.model = model or os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
        self.dims = dims
        self.name = f"gemini:{self.model}:{dims}"      # part of the index cache key
        self.per_minute = int(os.environ.get("GEMINI_EMBED_PER_MINUTE", "90"))
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            api_version=os.environ.get("GEMINI_API_VERSION", "v1beta"), timeout=60_000))

    def _embed(self, texts, task: str):
        import numpy as np
        kwargs = {"output_dimensionality": self.dims}
        if self.model == "gemini-embedding-001":       # task types exist only for this model
            kwargs["task_type"] = task
        vectors = []
        for i in range(0, len(texts), 10):              # small batches: the model has an input token limit
            batch = texts[i:i + 10]
            self._wait_for_quota(len(batch))
            for attempt in range(5):
                try:
                    resp = self.client.models.embed_content(
                        model=self.model, contents=batch, config=self.types.EmbedContentConfig(**kwargs))
                    vectors.extend(e.values for e in resp.embeddings)
                    break
                except Exception as e:  # noqa: BLE001
                    if _code(e) in RETRY_CODES and attempt < 4:
                        # Google says how long to wait ("Please retry in 49.48s"); use it when present.
                        m = re.search(r"retry in ([\d.]+)s", str(e))
                        delay = float(m.group(1)) + 1 if m else 5 * (attempt + 1)
                        print(f"[gemini] embedding quota reached, waiting {delay:.0f} s...")
                        time.sleep(delay)
                        continue
                    raise
        m = np.asarray(vectors, dtype="float32")
        return m / np.linalg.norm(m, axis=1, keepdims=True)   # needed below 3072 dimensions

    def _wait_for_quota(self, n: int):
        """Free tier: 100 texts per minute, and every text in a batch counts. Stay under 90."""
        now = time.monotonic()
        self._sent = [(t, k) for t, k in getattr(self, "_sent", []) if now - t < 60]
        while sum(k for _, k in self._sent) + n > self.per_minute:
            wait = 60 - (now - self._sent[0][0]) + 0.5
            print(f"[gemini] pacing embeddings for the free-tier limit, waiting {wait:.0f} s...")
            time.sleep(max(wait, 0.5))
            now = time.monotonic()
            self._sent = [(t, k) for t, k in self._sent if now - t < 60]
        self._sent.append((now, n))

    def encode(self, texts):
        return self._embed(list(texts), "RETRIEVAL_DOCUMENT")

    def encode_query(self, texts):
        return self._embed(list(texts), "RETRIEVAL_QUERY")

