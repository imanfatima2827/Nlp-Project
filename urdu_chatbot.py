"""
Punjab Police Khidmat Markaz - Urdu Query Assistant (desktop GUI).

    python clean_data.py          # once, or after re-scraping
    python urdu_chatbot.py        # GUI
    python urdu_chatbot.py --cli  # same bot in the terminal (handy for quick testing)
"""
import argparse
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import scrolledtext, ttk

from chatbot_core import DEFAULT_THRESHOLD, create_default_bot

HISTORY_LOG = Path(__file__).parent / "data" / "chat_history.jsonl"
URDU_FONTS = ["Jameel Noori Nastaleeq", "Noto Nastaliq Urdu", "Urdu Typesetting", "Noto Naskh Arabic",
              "Segoe UI", "Arial"]


def create_bot(threshold: float, use_llm: bool = True):
    return create_default_bot(threshold, use_llm, HISTORY_LOG)


# --------------------------------------------------------------------------- GUI
class ChatApp:
    def __init__(self, root: tk.Tk, threshold: float, use_llm: bool = True):
        self.root = root
        self.bot = None
        self.llm_status = ""
        self.results = queue.Queue()   # worker threads -> GUI thread (Tk is not thread-safe)

        root.title("Punjab Police Khidmat Markaz - Urdu Query Assistant")
        root.geometry("820x640")
        family = next((f for f in URDU_FONTS if f in tkfont.families()), "TkDefaultFont")
        self.urdu_font = (family, 14)
        self.urdu_bold = (family, 14, "bold")

        ttk.Label(root, text="پنجاب پولیس خدمت مرکز\nاردو معلوماتی اسسٹنٹ", font=(family, 18),
                  justify="center").pack(pady=10)

        self.output = scrolledtext.ScrolledText(root, wrap="word", font=self.urdu_font, height=20)
        self.output.tag_configure("q", justify="right", font=self.urdu_bold, foreground="#1a4d8f",
                                  spacing1=10)
        self.output.tag_configure("a", justify="right", spacing3=10, rmargin=8)
        self.output.tag_configure("info", justify="center", foreground="#777777")
        self.output.configure(state="disabled")
        self.output.pack(fill="both", expand=True, padx=10, pady=5)

        row = ttk.Frame(root)
        row.pack(fill="x", padx=10, pady=8)
        self.ask_btn = ttk.Button(row, text="پوچھیں", command=self.submit, state="disabled")
        self.ask_btn.pack(side="left")
        ttk.Button(row, text="ہسٹری", command=self.show_history).pack(side="left", padx=5)
        ttk.Button(row, text="صاف کریں", command=self.clear).pack(side="left")
        self.entry = ttk.Entry(row, font=self.urdu_font, justify="right")
        self.entry.pack(side="right", fill="x", expand=True, padx=(10, 0))
        self.entry.bind("<Return>", lambda _e: self.submit())   # Enter key sends the question
        self.entry.focus_set()

        self.status = ttk.Label(root, text="ماڈل لوڈ ہو رہا ہے، براہ کرم انتظار کریں...", anchor="e")
        self.status.pack(fill="x", padx=10, pady=(0, 8))

        threading.Thread(target=self._load, args=(threshold, use_llm), daemon=True).start()
        root.after(100, self._poll)

    # loading and answering run off the GUI thread so the window never freezes
    def _load(self, threshold, use_llm):
        try:
            self.results.put(("ready", create_bot(threshold, use_llm)))
        except Exception as e:  # noqa: BLE001 - show any startup error in the window
            self.results.put(("error", f"ماڈل لوڈ نہیں ہو سکا: {e}"))

    def _ask(self, query):
        try:
            self.results.put(("answer", self.bot.answer(query).text))
        except Exception as e:  # noqa: BLE001
            self.results.put(("error", f"خرابی: {e}"))

    def _poll(self):
        try:
            while True:
                kind, payload = self.results.get_nowait()
                if kind == "ready":
                    self.bot, self.llm_status = payload
                    self.status.configure(text=f"تیار ہے۔ اپنا سوال لکھیں اور Enter دبائیں۔  |  {self.llm_status}")
                    self.ask_btn.configure(state="normal")
                    self._write("مثال: کریکٹر سرٹیفیکیٹ کی فیس کتنی ہے؟  |  لاہور خدمت مرکز کا پتہ\n", "info")
                elif kind == "answer":
                    self._write(payload + "\n", "a")
                    self._set_busy(False)
                else:
                    self._write(payload + "\n", "info")
                    self.status.configure(text=payload)
                    self._set_busy(False)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _set_busy(self, busy):
        if self.bot:
            self.ask_btn.configure(state="disabled" if busy else "normal")
            self.status.configure(text="جواب تلاش کیا جا رہا ہے..." if busy else f"تیار ہے۔  |  {self.llm_status}")

    def _write(self, text, tag):
        self.output.configure(state="normal")
        self.output.insert(tk.END, text, tag)
        self.output.configure(state="disabled")
        self.output.see(tk.END)

    def submit(self):
        query = self.entry.get().strip()
        if not query or not self.bot or str(self.ask_btn["state"]) == "disabled":
            return
        self.entry.delete(0, tk.END)
        self._write(f"سوال: {query}\n", "q")
        self._set_busy(True)
        threading.Thread(target=self._ask, args=(query,), daemon=True).start()

    def clear(self):
        self.output.configure(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.configure(state="disabled")

    def show_history(self):
        win = tk.Toplevel(self.root)
        win.title("Chat History")
        box = scrolledtext.ScrolledText(win, width=80, height=24, wrap="word", font=self.urdu_font)
        box.tag_configure("r", justify="right")
        box.pack(fill="both", expand=True, padx=5, pady=5)
        turns = self.bot.history.recent(20) if self.bot else []
        if not turns:
            box.insert(tk.END, "ابھی تک کوئی گفتگو نہیں ہوئی۔", "r")
        for t in turns:
            box.insert(tk.END, f"{t.timestamp:%Y-%m-%d %H:%M:%S}\nسوال: {t.query}\nجواب: {t.response}\n{'-' * 50}\n", "r")
        box.configure(state="disabled")


# --------------------------------------------------------------------------- CLI
def run_cli(threshold: float, use_llm: bool = True):
    print("Loading model...")
    bot, status = create_bot(threshold, use_llm)
    print(f"Ready. {status}\nType a question (empty line to quit).")
    while True:
        q = input("\nسوال: ").strip()
        if not q:
            break
        a = bot.answer(q)
        print(a.text)
        print(f"  [sources={a.sources} score={a.score if a.score is None else round(a.score, 3)}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="run in the terminal instead of the GUI")
    parser.add_argument("--threshold", type=float, default=None,
                        help="minimum cosine similarity needed to answer (tune with evaluate.py)")
    parser.add_argument("--no-llm", action="store_true", help="don't use Gemini, show retrieved facts only")
    args = parser.parse_args()
    if args.cli:
        run_cli(args.threshold, not args.no_llm)
    else:
        root = tk.Tk()
        ChatApp(root, args.threshold, not args.no_llm)
        root.mainloop()


if __name__ == "__main__":
    main()
