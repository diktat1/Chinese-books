---
title: Chinese Graded Reader
emoji: "\U0001F4DA"
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
---

# Chinese Graded Reader Converter

Turn any Chinese EPUB into a **learning kit**: graded reader, Anki flashcards, or audiobook.

## Features

- **Pinyin annotations** above each Chinese character (inline `<ruby>` tags)
- **Translations** via Google Translate (free) or Claude via OpenRouter (premium)
- **Parallel text** mode: side-by-side Chinese + translation columns, sentence-aligned
- **HSK 4 simplification**: rewrite vocabulary to HSK 4 level using Claude
- **Anki flashcard decks** with sentence cards (Chinese + pinyin + translation + audio)
- **Audiobooks** with bilingual narration (edge-tts, M4B with chapter markers)
- **Kindle AZW3** output (requires Calibre)
- **Real-time progress** via SSE in the web UI

## Quick Start

```bash
pip install -r requirements.txt

# Web UI (drag & drop, real-time progress bar)
python web_app.py   # open http://localhost:5000

# CLI examples
python convert.py book.epub                              # pinyin + Google Translate
python convert.py book.epub --pinyin-only                # fast, no API calls
python convert.py book.epub --parallel-text              # side-by-side columns
python convert.py book.epub --parallel-text --use-claude # Claude translation
python convert.py book.epub --simplify-hsk4 --use-claude # HSK 4 + Claude
python convert.py book.epub --anki --target fr           # Anki deck, French
python convert.py book.epub --audio                      # bilingual audiobook
python convert.py book.epub --kindle                     # AZW3 for Kindle
```

## Layout Modes

### Ruby mode (default)
Pinyin appears inline above each character. Translation block before each paragraph.

### Parallel text (`--parallel-text`)
Two-column table: Chinese sentences (with pinyin) on the left, translations on the right. Each sentence gets its own row. Uses `<table>` HTML for maximum e-reader compatibility.

## Translation Options

| Option | Flag | Cost |
|--------|------|------|
| Google Translate | (default) | Free |
| Claude Sonnet | `--use-claude` | ~$3/M tokens |
| Claude Opus | `--use-claude --use-opus` | ~$15/M tokens |

Set `OPENROUTER_API_KEY` env var for Claude features. Get a key at https://openrouter.ai/

## Deployment (HF Spaces)

The app deploys to Hugging Face Spaces via Docker. Set `OPENROUTER_API_KEY` as a secret in Space settings.

## Project Structure

```
Chinese-books/
├── convert.py                      # CLI entry point
├── web_app.py                      # Web UI (Flask + SSE progress)
├── Dockerfile                      # HF Spaces deployment
├── requirements.txt
└── graded_reader/
    ├── __init__.py
    ├── calibre.py                  # EPUB-to-AZW3 conversion
    ├── chinese_processing.py       # Jieba segmentation, pinyin, ruby HTML, sentence splitting
    ├── translator.py               # Google Translate (free, per-sentence + paragraph)
    ├── claude_translator.py        # Claude via OpenRouter (batch sentence + paragraph)
    ├── claude_simplifier.py        # HSK 4 vocabulary simplification via Claude
    ├── epub_processor.py           # EPUB processing: ruby mode + parallel text mode
    ├── anki_generator.py           # Anki .apkg deck generation
    └── audio_generator.py          # Audiobook generation (edge-tts, M4B)
```
