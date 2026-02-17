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
- **Translations** via Google Translate (free) or any LLM via OpenRouter (12+ models)
- **Parallel text** mode: side-by-side Chinese + translation columns, sentence-aligned
- **HSK 4 simplification**: rewrite vocabulary to HSK 4 level using an LLM
- **Anki flashcard decks** with sentence cards (Chinese + pinyin + translation + audio)
- **Audiobooks** with bilingual narration (edge-tts, M4B with chapter markers)
- **Real-time progress** via SSE in the web UI

## Quick Start

```bash
pip install -r requirements.txt

# Web UI (drag & drop, real-time progress bar)
python web_app.py   # open http://localhost:5000

# CLI examples
python convert.py book.epub                                 # pinyin + Google Translate
python convert.py book.epub --pinyin-only                   # fast, no API calls
python convert.py book.epub --parallel-text                 # side-by-side columns
python convert.py book.epub --tier standard                 # DeepSeek V3 (~$0.10/book)
python convert.py book.epub --tier premium                  # Claude Sonnet 4.5 (~$1.65/book)
python convert.py book.epub --model deepseek/deepseek-chat  # specific model
python convert.py book.epub --simplify-hsk4                 # HSK 4 simplification
python convert.py book.epub --anki --target fr              # Anki deck, French
python convert.py book.epub --audio                         # bilingual audiobook
python convert.py --list-models                             # show all available models
```

## Translation Engines

| Tier | Default Model | Cost/book | Chinese Quality |
|------|--------------|-----------|-----------------|
| Free | Google Translate | $0 (no API key) | Good |
| Free (OpenRouter) | DeepSeek R1 | $0 (rate limited) | Excellent |
| Standard | DeepSeek V3 | ~$0.10 | Excellent |
| Standard | Qwen3 235B Instruct | ~$0.01 | Excellent |
| Standard | Gemini 2.5 Flash | ~$0.27 | Very good |
| Premium | Claude Sonnet 4.5 | ~$1.65 | Excellent |
| Premium | GPT-4o | ~$1.13 | Excellent |
| Premium | Claude Opus 4.5 | ~$2.75 | Excellent |

Set `OPENROUTER_API_KEY` env var for LLM models. Get a key at https://openrouter.ai/

Run `python convert.py --list-models` to see the full catalog.

## Layout Modes

### Ruby mode (default)
Pinyin appears inline above each character. Translation block before each paragraph.

### Parallel text (`--parallel-text`)
Two-column table: Chinese sentences (with pinyin) on the left, translations on the right. Each sentence gets its own row. Uses `<table>` HTML for maximum e-reader compatibility.

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
    ├── models.py                   # OpenRouter model catalog (12+ models, 3 tiers)
    ├── chinese_processing.py       # Jieba segmentation, pinyin, ruby HTML, sentence splitting
    ├── translator.py               # Google Translate (free, per-sentence + paragraph)
    ├── llm_translator.py           # LLM translation via OpenRouter (any model)
    ├── llm_simplifier.py           # HSK 4 vocabulary simplification via LLM
    ├── epub_processor.py           # EPUB processing: ruby mode + parallel text mode
    ├── anki_generator.py           # Anki .apkg deck generation
    └── audio_generator.py          # Audiobook generation (edge-tts, M4B)
```
