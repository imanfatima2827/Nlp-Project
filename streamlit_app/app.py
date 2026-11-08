"""
Streamlit version of the web app, for free hosting on Streamlit Community Cloud (no card needed).

Deploy:  share.streamlit.io -> Create app -> this repo, branch main, main file: streamlit_app/app.py
         Advanced settings -> Secrets:   GEMINI_API_KEY = "your-key"
Local:   pip install -r streamlit_app/requirements.txt
         streamlit run streamlit_app/app.py

Uses Gemini embeddings (EMBEDDINGS=gemini), so no PyTorch is needed and it fits in 1 GB RAM.
The same chatbot_core.py powers this, the Gradio app (app.py) and the desktop app.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Streamlit secrets -> environment variables (read by llm.py / chatbot_core.py)
for _key in ("GEMINI_API_KEY", "GEMINI_MODEL", "THRESHOLD", "EMBEDDINGS"):
    try:
        if _key in st.secrets:
            os.environ.setdefault(_key, str(st.secrets[_key]))
    except Exception:  # noqa: BLE001 - no secrets file when running locally; .env is used instead
        pass
os.environ.setdefault("EMBEDDINGS", "gemini")

from chatbot_core import ChatHistory, create_default_bot  # noqa: E402

DATA = ROOT / "data"
FEEDBACK_LOG = DATA / "feedback.jsonl"
EXAMPLES = [
    "کریکٹر سرٹیفیکیٹ کی فیس کتنی ہے؟",
    "ڈرائیونگ لائسنس کی تجدید کے لیے کون سے کاغذات چاہییں؟",
    "لاہور خدمت مرکز کا پتہ",
    "پولیس کے خلاف شکایت کہاں کریں؟",
    "character certificate kitne din mein milta hai",
]
ERROR_MSG = "معاف کیجیے، کوئی خرابی پیش آئی۔ براہ کرم دوبارہ کوشش کریں۔"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
.urdu, [data-testid="stChatMessage"], [data-testid="stChatMessage"] *, [data-testid="stChatInput"] textarea,
.stButton button p {
    font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', serif;
}
[data-testid="stChatMessage"] { direction: rtl; text-align: right; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { line-height: 2.3; font-size: 1.05rem; }
[data-testid="stChatInput"] textarea { direction: rtl; text-align: right; line-height: 2.1; }
.title { text-align: center; line-height: 2.2; margin: 0; }
.title h1 { font-family: 'Noto Nastaliq Urdu', serif; font-size: 2.1rem; margin: 0; padding: 0; line-height: 2.2; }
.title p { font-family: 'Noto Nastaliq Urdu', serif; font-size: 1.15rem; opacity: .8; margin: 0; }
.intro { direction: rtl; text-align: center; line-height: 2.4; padding-bottom: .5rem; }
.note { direction: rtl; text-align: center; font-size: .85rem; opacity: .7; line-height: 2.2; }
.stButton button p { font-size: .85rem; }
"""


@st.cache_resource(show_spinner="ماڈل لوڈ ہو رہا ہے، براہ کرم انتظار کریں...")
def get_bot():
    """One shared bot (index + Gemini client) for all visitors; each visitor has their own history."""
    return create_default_bot(history_log=DATA / "chat_history.jsonl")


def save_feedback(index: int, liked: bool, message: str):
    record = {"time": datetime.now().isoformat(timespec="seconds"), "liked": liked, "index": index,
              "message": message[:2000]}
    try:
        DATA.mkdir(exist_ok=True)
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def show(text: str):
    # Markdown joins single line breaks; keep the answer's lines and bullets as written.
    st.markdown(text.replace("\n", "  \n"))


st.set_page_config(page_title="PKM Urdu Assistant", page_icon="🚓", layout="centered")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
st.markdown('<div class="title"><h1>پنجاب پولیس خدمت مرکز</h1><p>اردو معلوماتی اسسٹنٹ</p></div>',
            unsafe_allow_html=True)
st.markdown('<div class="urdu intro">خدمت مرکز کی سروسز، فیس، ضروری کاغذات اور مراکز کے پتے کے بارے میں پوچھیں۔ '
            'اردو یا رومن اردو دونوں میں سوال کر سکتے ہیں۔</div>', unsafe_allow_html=True)

bot, llm_status = get_bot()
state = st.session_state
if "messages" not in state:
    state.messages = []                       # what the chat shows
    state.history = ChatHistory(DATA / "chat_history.jsonl")   # this visitor's memory for follow-ups
    state.rated = set()

# example questions (only before the first question, to keep the page clean)
if not state.messages:
    st.caption("مثال سوالات")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        if cols[i % 2].button(q, key=f"ex{i}", use_container_width=True):
            state.pending = q

# conversation so far
for i, m in enumerate(state.messages):
    with st.chat_message(m["role"]):
        show(m["content"])
        if m["role"] == "assistant":
            rating = st.feedback("thumbs", key=f"fb{i}")
            if rating is not None and i not in state.rated:
                state.rated.add(i)
                save_feedback(i, rating == 1, m["content"])

prompt = st.chat_input("اپنا سوال یہاں لکھیں...") or state.pop("pending", None)
if prompt:
    state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        show(prompt)
    with st.chat_message("assistant"):
        with st.spinner("جواب تلاش کیا جا رہا ہے..."):
            try:
                answer = bot.answer(prompt, state.history).text
            except Exception as e:  # noqa: BLE001 - never crash the page
                print(f"error answering {prompt!r}: {e}")
                answer = ERROR_MSG
        show(answer)
    state.messages.append({"role": "assistant", "content": answer})
    st.rerun()                                 # redraw so the new answer gets its feedback buttons

if state.messages and st.button("نئی گفتگو", key="new_chat"):
    state.messages = []
    state.history = ChatHistory(DATA / "chat_history.jsonl")
    state.rated = set()
    st.rerun()

st.markdown(f'<div class="note">یہ ایک غیر سرکاری طلبہ پروجیکٹ ہے۔ معلومات pkm.punjab.gov.pk سے لی گئی ہیں؛ '
            f'حتمی تصدیق کے لیے ہیلپ لائن 04299211356 پر رابطہ کریں۔<br>{llm_status}</div>',
            unsafe_allow_html=True)
