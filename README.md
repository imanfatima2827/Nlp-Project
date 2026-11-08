---
title: PKM Urdu Assistant
emoji: 🚓
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Urdu Q&A on Punjab Police Khidmat Markaz services
---

# Punjab Police Khidmat Markaz – Urdu Query Assistant

Answers Urdu questions about Punjab Police Khidmat Markaz (PKM) services and centres:
fees, required documents, processing time, delivery, centre addresses, phone numbers and timings.

## Setup

```bash
pip install -r requirements.txt
python clean_data.py        # builds data/clean/ from data/raw/
python app.py               # web app  -> http://127.0.0.1:7860   (add --share for a public link)
python urdu_chatbot.py      # desktop app   (or: python urdu_chatbot.py --cli)
python evaluate.py          # accuracy check + threshold tuning
```

The first run downloads the embedding model (~1 GB) and builds the search index. After that the index
is cached in `data/cache/`, so startup is fast. The cache rebuilds itself when the data or model changes.

## Gemini (free AI answers, optional)

1. Get a free API key at https://aistudio.google.com/apikey
2. Copy `.env.example` to `.env` and paste your key: `GEMINI_API_KEY=...`
3. Run the app. The status bar shows whether Gemini is active.

With Gemini the bot:
- writes natural Urdu answers from the retrieved facts (it is told not to add any facts of its own),
- understands Roman Urdu / English questions ("character certificate ki fees kitni hai"),
- handles follow-ups ("اس کی فیس کتنی ہے؟").

Without a key, or if the free quota runs out or there's no internet, it falls back to showing the
retrieved facts directly. Use `--no-llm` to turn Gemini off. Out-of-scope questions are refused before
Gemini is called, so it can't invent answers about unrelated topics.

Free tier note: limits are shown in AI Studio, and Google may use free-tier prompts to improve its
products, so don't send personal data (CNIC numbers etc.) through it.

## Deploy free on Render

Hugging Face now requires a paid PRO plan for Gradio Spaces, so the free option is **Render** (no card needed).
Render's free plan has 512 MB RAM, too little for the local embedding model, so the server version uses
Google's embedding API instead (`EMBEDDINGS=gemini`, already set in `render.yaml`).

1. Test server mode locally (PowerShell):
   `$env:EMBEDDINGS="gemini"; python evaluate.py` and check the threshold line; then `python app.py`.
2. Push the project to a GitHub repository (`.env` is excluded by `.gitignore`).
3. On render.com: sign up with GitHub, **New → Blueprint**, pick the repository, paste your `GEMINI_API_KEY`, **Apply**.
4. After the build (3–5 min) the app is live at `https://pkm-urdu-assistant.onrender.com` (or similar).

Free Render services sleep after 15 minutes without visitors; the next visit wakes them in about a minute.
`deploy_hf.py` still works if you have a Hugging Face PRO account.

## Project layout

| File | Purpose |
|---|---|
| `extract_services.py`, `extract_one.py`, `extract_whole.py` | Scrape pkm.punjab.gov.pk into `data/raw/` (services, centres, home page) |
| `scraper_utils.py` | Shared scraper helpers (Chrome driver, text cleaning, saving) |
| `clean_data.py` | Turns raw JSON into clean structured JSON in `data/clean/` |
| `text_utils.py` | Urdu spelling fixes, search normalisation, district aliases |
| `chatbot_core.py` | Retrieval and answer logic (no GUI) |
| `llm.py` | Gemini: query rewriting and answer writing |
| `app.py` | Gradio web app (also runs on Hugging Face Spaces) |
| `urdu_chatbot.py` | Tkinter desktop app + terminal mode |
| `evaluate.py` | 30 test questions; reports accuracy and suggests a threshold |
| `render.yaml`, `requirements-server.txt` | Free deployment on Render (Gemini embeddings, no PyTorch) |
| `deploy_hf.py` | Deploy to Hugging Face Spaces (needs a PRO account) |
| `take_screenshots.py` | Takes screenshots of the running web app for the docs |
| `docs/` | User manual and project report (PDF) |

## Updating the data

```bash
python extract_services.py && python extract_one.py && python extract_whole.py
python clean_data.py
```
Chromedriver is downloaded automatically by Selenium. Add `--show` to watch the browser.
`clean_data.py` prints warnings for anything it couldn't parse (new sections, ambiguous dates).

## Tuning the threshold

`evaluate.py` prints the lowest similarity score among questions the bot should answer and the highest
among questions it should refuse. Set `DEFAULT_THRESHOLD` in `chatbot_core.py` (or `--threshold`)
between those two numbers.
