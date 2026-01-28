# Chinese Graded Reader Converter

Converts a Chinese EPUB ebook into a **graded reader** with:

- **Pinyin** annotations above each Chinese character (using HTML `<ruby>` tags)
- **English translations** after each paragraph

## How It Works

```
Input EPUB (Chinese) ──> Segment words (jieba)
                           ──> Add pinyin (pypinyin + <ruby> tags)
                           ──> Translate paragraphs (Google Translate)
                           ──> Output EPUB (graded reader)
```

1. **Read** the input EPUB and parse each chapter's HTML
2. **Segment** Chinese text into words using jieba
3. **Annotate** each Chinese character with pinyin using `<ruby>` tags
4. **Translate** each Chinese paragraph to English and insert it below
5. **Style** with CSS optimized for EPUB/Kindle ruby rendering
6. **Write** the new EPUB file

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Full conversion: pinyin + English translation
python convert.py book.epub

# Specify output filename
python convert.py book.epub -o my_reader.epub

# Pinyin only (no English translation — much faster)
python convert.py book.epub --pinyin-only

# Translation only (no pinyin)
python convert.py book.epub --translation-only

# Translate to a different language (e.g. Japanese)
python convert.py book.epub --target ja

# Verbose logging
python convert.py book.epub -v
```

## Output Format

The output EPUB contains:

```html
<p>
  <ruby>你<rp>(</rp><rt>nǐ</rt><rp>)</rp></ruby>
  <ruby>好<rp>(</rp><rt>hǎo</rt><rp>)</rp></ruby>
</p>
<p class="translation">Hello</p>
```

- Pinyin appears **above** each character via ruby annotations
- English translation appears below each paragraph in italic gray text
- `<rp>` tags provide fallback for readers without ruby support: `你(nǐ)好(hǎo)`

## Kindle Compatibility

For best results on Kindle:

1. **Convert to AZW3** (not MOBI) using [Calibre](https://calibre-ebook.com/) — MOBI does not support `<ruby>` tags
2. The output EPUB uses vendor-prefixed CSS (`-epub-ruby-position`, `-webkit-ruby-position`) for maximum device compatibility
3. Line height is set to 2.0 to prevent pinyin from overlapping the line above
4. If ruby doesn't render, Kindle Paperwhite+ has a built-in pinyin mode: Settings > Language & Dictionaries > add Chinese keyboard > tap "Pinyin" while reading

## Project Structure

```
Chinese-books/
├── convert.py                      # CLI entry point
├── requirements.txt                # Python dependencies
├── README.md
└── graded_reader/
    ├── __init__.py
    ├── chinese_processing.py       # Word segmentation + pinyin + ruby HTML
    ├── translator.py               # Chinese-to-English translation
    └── epub_processor.py           # EPUB read/write + HTML processing + CSS
```

## Dependencies

| Library | Purpose |
|---------|---------|
| [ebooklib](https://github.com/aerkalov/ebooklib) | Read and write EPUB files |
| [jieba](https://github.com/fxsjy/jieba) | Chinese word segmentation |
| [pypinyin](https://github.com/mozillazg/python-pinyin) | Pinyin conversion with tone marks |
| [deep-translator](https://github.com/nidhaloff/deep-translator) | Translation via Google Translate (free) |
| [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/) | HTML parsing and manipulation |
| [lxml](https://lxml.de/) | Fast HTML/XML parser backend |
