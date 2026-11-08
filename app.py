"""
Web chat for the PKM Urdu assistant (Gradio).

    python app.py              # open http://127.0.0.1:7860
    python app.py --share      # also get a temporary public link to send to friends
    python app.py --no-llm     # without Gemini

On Hugging Face Spaces this file runs automatically (see README "Deploy").
Gemini key: .env locally, or a Space secret named GEMINI_API_KEY.
"""
import argparse
import inspect
import json
import os
from datetime import datetime
from pathlib import Path

import gradio as gr

from chatbot_core import ChatHistory, create_default_bot

DATA = Path(__file__).parent / "data"
HISTORY_LOG = DATA / "chat_history.jsonl"
FEEDBACK_LOG = DATA / "feedback.jsonl"

BOT = None          # created in main(); one shared model for all visitors
LLM_STATUS = ""

EXAMPLES = [
    "کریکٹر سرٹیفیکیٹ کی فیس کتنی ہے؟",
    "ڈرائیونگ لائسنس کی تجدید کے لیے کون سے کاغذات چاہییں؟",
    "لاہور خدمت مرکز کا پتہ",
    "پولیس کے خلاف شکایت کہاں کریں؟",
    "character certificate kitne din mein milta hai",
]

HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap" '
        'rel="stylesheet">')
CSS = """
/* Nastaliq is much taller than Latin text. Gradio sizes its boxes for Latin text, so without extra
   line-height and overflow:visible the Urdu lines get clipped and a scrollbar (up/down arrows) appears. */
.urdu, #title, #title *, #intro, #intro *, #footer, #footer *, #chat, #chat *, #q textarea {
    font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', 'Urdu Typesetting', serif;
}
#title, #intro, #footer, #title *, #intro *, #footer * {
    overflow: visible !important;
    max-height: none !important;
    height: auto !important;
}
#title { text-align: center; }
#title h1 { font-size: 30px; line-height: 2.1; margin: 4px 0 0; }
#title h3 { font-size: 20px; line-height: 2.1; margin: 0; font-weight: 400; opacity: 0.85; }
#intro .urdu { direction: rtl; text-align: center; font-size: 17px; line-height: 2.5; padding: 6px 0 10px; }
#footer .note { direction: rtl; text-align: center; font-size: 14px; line-height: 2.4; opacity: 0.75; padding: 6px 0; }
#chat { font-size: 17px; line-height: 2.3; }
#q textarea { direction: rtl; text-align: right; font-size: 17px; line-height: 2.2; padding-top: 6px; padding-bottom: 6px; }
"""

ERROR_MSG = "معاف کیجیے، کوئی خرابی پیش آئی۔ براہ کرم دوبارہ کوشش کریں۔"


# --------------------------------------------------------------------------- version-proof helpers
# Gradio 6 moved css/theme/head from Blocks() to launch() and dropped Chatbot(type=...).
# These helpers pass each option only where the installed Gradio version accepts it.
def _params(fn):
    return inspect.signature(fn).parameters


def make(component, **kwargs):
    accepted = _params(component.__init__)
    return component(**{k: v for k, v in kwargs.items() if k in accepted})


def split_style_kwargs(**style):
    blocks_kw, launch_kw = {}, {}
    for key, value in style.items():
        if key in _params(gr.Blocks.launch):
            launch_kw[key] = value
        elif key in _params(gr.Blocks.__init__):
            blocks_kw[key] = value
    return blocks_kw, launch_kw


# --------------------------------------------------------------------------- event handlers
def add_user_message(text, messages):
    """Runs instantly: show the question and clear the box. The answer comes in respond()."""
    text = (text or "").strip()
    if not text:
        return gr.update(), messages, messages, None
    messages = messages + [{"role": "user", "content": text}]
    return "", messages, messages, text


def respond(pending, messages, history):
    if not pending:
        return messages, messages, history
    history = history if history is not None else ChatHistory(HISTORY_LOG)   # one history per visitor
    try:
        text = BOT.answer(pending, history).text
    except Exception as e:  # noqa: BLE001 - never crash the page
        print(f"error answering {pending!r}: {e}")
        text = ERROR_MSG
    messages = messages + [{"role": "assistant", "content": text}]
    return messages, messages, history


def clear_chat():
    return [], [], None, None


def save_feedback(data: gr.LikeData):
    """👍/👎 on an answer -> data/feedback.jsonl. Review the 👎 ones and add them to evaluate.py."""
    record = {"time": datetime.now().isoformat(timespec="seconds"), "liked": data.liked,
              "index": data.index, "message": str(data.value)[:2000]}
    try:
        DATA.mkdir(exist_ok=True)
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as e:
        print(f"could not save feedback: {e}")


# --------------------------------------------------------------------------- layout
def build_ui():
    blocks_kw, launch_kw = split_style_kwargs(css=CSS, head=HEAD, theme=gr.themes.Soft(primary_hue="green"))
    with gr.Blocks(title="PKM Urdu Assistant", **blocks_kw) as demo:
        gr.Markdown("# پنجاب پولیس خدمت مرکز\n### اردو معلوماتی اسسٹنٹ", elem_id="title")
        gr.Markdown(
            elem_id="intro", value=
            "<div class='urdu'>خدمت مرکز کی سروسز، فیس، ضروری کاغذات اور مراکز کے پتے کے بارے میں پوچھیں۔ "
            "اردو یا رومن اردو دونوں میں سوال کر سکتے ہیں۔</div>")

        chatbot = make(gr.Chatbot, elem_id="chat", height=520, rtl=True, type="messages",
                       label="گفتگو", show_label=False, show_copy_button=True)
        messages = gr.State([])     # what the chat window shows
        history = gr.State(None)    # ChatHistory for this visitor (follow-up questions)
        pending = gr.State(None)    # question waiting for an answer

        with gr.Row():
            box = make(gr.Textbox, elem_id="q", placeholder="اپنا سوال یہاں لکھیں اور Enter دبائیں...",
                       show_label=False, rtl=True, lines=1, max_lines=4, scale=8, autofocus=True)
            send = make(gr.Button, value="بھیجیں", variant="primary", scale=1, min_width=90)
        gr.Examples(EXAMPLES, inputs=box, label="مثال سوالات")
        new_chat = make(gr.Button, value="نئی گفتگو", variant="secondary", size="sm")

        gr.Markdown(
            elem_id="footer", value=
            f"<div class='note'>یہ ایک غیر سرکاری طلبہ پروجیکٹ ہے۔ معلومات pkm.punjab.gov.pk سے لی گئی ہیں؛ "
            f"حتمی تصدیق کے لیے ہیلپ لائن 04299211356 پر رابطہ کریں۔<br>{LLM_STATUS}</div>")

        for trigger in (box.submit, send.click):
            trigger(add_user_message, [box, messages], [box, chatbot, messages, pending], queue=False) \
                .then(respond, [pending, messages, history], [chatbot, messages, history])
        new_chat.click(clear_chat, None, [chatbot, messages, history, pending], queue=False)
        chatbot.like(save_feedback, None, None)
    return demo, launch_kw


def main():
    global BOT, LLM_STATUS
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true", help="create a temporary public link")
    p.add_argument("--no-llm", action="store_true", help="don't use Gemini")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    args = p.parse_args()

    print("Loading model...")
    BOT, LLM_STATUS = create_default_bot(use_llm=not args.no_llm)
    print(f"Ready. {LLM_STATUS}")

    demo, launch_kw = build_ui()
    demo.queue(default_concurrency_limit=4)
    on_server = bool(os.environ.get("SPACE_ID") or os.environ.get("RENDER"))   # Hugging Face / Render
    demo.launch(server_name="0.0.0.0" if on_server else "127.0.0.1", server_port=args.port,
                share=args.share and not on_server, **launch_kw)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:      # Ctrl+C: stop quietly instead of printing a traceback
        print("\nStopped.")
