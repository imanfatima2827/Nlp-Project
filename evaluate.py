"""
Check the bot against known questions and help pick the similarity threshold.

    python evaluate.py
    python evaluate.py --threshold 0.5

Each case is (question, expected source prefix). Use None for questions the bot should NOT answer.
Add real questions from users here as you collect them - this is how you prove an upgrade helped.
"""
import argparse
import tempfile
from pathlib import Path

from chatbot_core import DEFAULT_THRESHOLD, UrduChatbot

CASES = [
    # services - specific aspects
    ("کریکٹر سرٹیفیکیٹ کی فیس کتنی ہے؟", "service:کریکٹر سرٹیفیکیٹ:fee"),
    ("کریکٹر سرٹیفکیٹ بنوانے کے لیے کون سے کاغذات چاہیے؟", "service:کریکٹر سرٹیفیکیٹ:required_documents"),
    ("کریکٹر سرٹیفیکیٹ کتنے دن میں ملتا ہے", "service:کریکٹر سرٹیفیکیٹ:processing_time"),
    ("پولیس ویریفیکیشن کی فیس", "service:پولیس ویریفیکیشن:fee"),
    ("لائسنس کی تجدید میں کتنے دن لگتے ہیں", "service:ڈرائیونگ لائسنس کی تجدید:processing_time"),
    ("لائسنس رینیو کروانے کی فیس کتنی ہے", "service:ڈرائیونگ لائسنس کی تجدید:fee"),
    ("لرنر پرمٹ کے لیے عمر کی حد کیا ہے", "service:لرنرز ڈرائیونگ لائسنس:age_limit"),
    ("انٹرنیشنل ڈرائیونگ لائسنس کے لیے کیا دستاویزات چاہییں", "service:بین الاقوامی ڈرائیونگ لائسنس:required_documents"),
    ("لائسنس گم ہو گیا ہے نقل کیسے بنے گی", "service:نقل ڈرائیونگ لائسنس"),
    ("گاڑی کی تصدیق کی فیس", "service:گاڑی کی تصدیق:fee"),
    ("ملازم کی رجسٹریشن کے لیے کیا چاہیے", "service:ملازم کی رجسٹریشن:required_documents"),
    ("کرایہ دار کی رجسٹریشن کیسے ہوتی ہے", "service:کرائے دار کی رجسٹریشن"),
    ("موبائل گم ہو گیا رپورٹ کیسے درج کراؤں", "service:گمشدگی رپورٹ"),
    ("ایف آئی آر کی کاپی کیسے ملے گی", "service:ایف آئی آر کی کاپی"),
    ("خواتین پر تشدد کی رپورٹ", "service:خواتین پر تشدد رپورٹ"),
    # centres
    ("لاہور خدمت مرکز کا پتہ", "centers:لاہور"),
    ("فیصل آباد میں خدمت مرکز کہاں ہے", "centers:فیصل آباد"),
    ("ڈیرہ غازی خان خدمت مرکز کا فون نمبر", "centers:ڈی جی خان"),
    ("رحیم یار خان کا خدمت مرکز", "centers:رحیم یار خان"),
    ("ملتان کا خدمت مرکز کب کھلتا ہے", "centers:ملتان"),
    ("پنڈی میں خدمت مرکز", "centers:راولپنڈی"),
    # general
    ("پولیس کے خلاف شکایت کہاں کریں", "general:پولیس کے خلاف شکایت"),
    ("اپنی درخواست کا اسٹیٹس کیسے چیک کروں", "general:درخواست کی ٹریکنگ"),
    ("کریکٹر سرٹیفیکیٹ اصلی ہے یا نہیں کیسے تصدیق کریں", "general:دستاویز کی تصدیق"),
    ("کیا میں دوسرے ضلع سے کریکٹر سرٹیفیکیٹ بنوا سکتا ہوں", "general:کسی بھی ضلع سے سروس"),
    ("میڈیکو لیگل سرٹیفیکیٹ کہاں سے ملے گا", "general:میڈیکو لیگل سرٹیفیکیٹ"),
    # different wording of the cases above (added after the first evaluation run)
    ("ڈپلیکیٹ لائسنس کیسے بنے گا", "service:نقل ڈرائیونگ لائسنس"),
    ("میری درخواست کہاں تک پہنچی ٹریک کیسے کروں", "general:درخواست کی ٹریکنگ"),
    ("جعلی کریکٹر سرٹیفیکیٹ کی پہچان کیسے کریں", "general:دستاویز کی تصدیق"),
    # out of scope: should NOT answer
    ("آج موسم کیسا ہے", None),
    ("پاکستان کا دارالحکومت کیا ہے", None),
    ("بریانی کیسے بناتے ہیں", None),
    ("کرکٹ میچ کا اسکور کیا ہے", None),
]


def run(bot: UrduChatbot):
    ok, rows = 0, []
    for q, expected in CASES:
        a = bot.answer(q)
        got = a.sources[0] if a.sources else None
        passed = (got is None) if expected is None else (got is not None and got.startswith(expected))
        ok += passed
        rows.append((passed, a.score, q, expected, got))
    for passed, score, q, expected, got in rows:
        s = "  -  " if score is None else f"{score:.3f}"
        print(f"{'PASS' if passed else 'FAIL'}  {s}  {q}\n        expected={expected}  got={got}")
    print(f"\n{ok}/{len(CASES)} correct")

    # district and keyword answers don't depend on the threshold, so leave them out of this check
    inside = [r[1] for r in rows if r[3] is not None and r[1] is not None
              and not (r[4] or "").startswith("centers:")]
    outside = [r[1] for r in rows if r[3] is None and r[1] is not None]
    if inside and outside:
        print(f"lowest in-scope score: {min(inside):.3f} | highest out-of-scope score: {max(outside):.3f}")
        print("A good threshold sits between those two numbers.")
    return ok


def main():
    from chatbot_core import ChatHistory, default_threshold, embeddings_mode, make_encoder
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=None)
    args = p.parse_args()
    log = Path(tempfile.gettempdir()) / "pkm_eval_history.jsonl"   # keep eval out of the real chat log
    threshold = args.threshold if args.threshold is not None else default_threshold()
    print(f"embeddings: {embeddings_mode()} | threshold: {threshold}")
    run(UrduChatbot(make_encoder(), threshold=threshold, history=ChatHistory(log)))


if __name__ == "__main__":
    main()
