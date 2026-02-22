#!/usr/bin/env python3
"""
Generate a beginner Japanese EPUB with furigana, romaji, and parallel Chinese.

Reads the N5 script from beginner_japanese_script.json and produces an EPUB
where each sentence shows:
  1. Japanese with furigana (ruby annotations above kanji)
  2. Romaji transliteration
  3. Corresponding Chinese translation
"""
import json
import logging
from pathlib import Path

import pykakasi
from ebooklib import epub

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

kks = pykakasi.kakasi()

# Section titles matching generate_script.py organization
SECTIONS = [
    (0, 10, "Introduction"),
    (10, 19, "The Problem Appears"),
    (19, 27, "Intel Background"),
    (27, 34, "New Product (Pentium)"),
    (34, 41, "The Bug Discovered"),
    (41, 48, "Intel's Initial Reaction"),
    (48, 55, "CNN Arrives, Crisis Escalates"),
    (55, 62, "Customer Complaints"),
    (62, 71, "IBM Stops Shipments"),
    (71, 81, "Internal Crisis"),
    (81, 87, "Decision to Change"),
    (87, 92, "The Cost"),
    (92, 100, "What Changed?"),
    (100, 108, "Brand and Size"),
    (108, 116, "The Boss Is Last to Know"),
    (116, 127, "Lesson Learned"),
]


def add_furigana_html(text: str) -> tuple[str, str]:
    """Convert Japanese text to HTML with ruby annotations and romaji.
    Adds spaces between words for readability."""
    items = kks.convert(text)
    ruby_parts = []
    romaji_parts = []
    _PUNCT = set('。、！？「」（）．，.!?,')

    for item in items:
        orig = item['orig']
        hira = item['hira']
        hepburn = item['hepburn']

        has_kanji = any('\u4e00' <= c <= '\u9fff' for c in orig)
        is_punct = all(c in _PUNCT or c.isspace() for c in orig)

        if has_kanji and hira != orig:
            ruby_parts.append(
                f'<ruby>{orig}<rp>(</rp><rt>{hira}</rt><rp>)</rp></ruby>'
            )
        else:
            ruby_parts.append(orig)

        # Add word boundary space for readability
        if not is_punct:
            ruby_parts.append(' ')

        if hepburn and hepburn not in ('.', ',', '!', '?'):
            romaji_parts.append(hepburn)

    return ''.join(ruby_parts).strip(), ' '.join(romaji_parts)


EPUB_CSS = '''
/* Base styling for Japanese learning EPUB */
body {
    font-family: "Hiragino Kaku Gothic ProN", "Noto Sans CJK JP", "Yu Gothic", sans-serif;
    line-height: 2.2;
    margin: 1em;
    color: #333;
}

h1 {
    font-size: 1.4em;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    color: #222;
    border-bottom: 2px solid #c0392b;
    padding-bottom: 0.3em;
}

h2 {
    font-size: 1.15em;
    margin-top: 1.8em;
    margin-bottom: 0.4em;
    color: #555;
    border-bottom: 1px solid #ddd;
    padding-bottom: 0.2em;
}

/* Ruby annotation styling (furigana) */
ruby {
    ruby-align: center;
    -epub-ruby-position: over;
    -webkit-ruby-position: before;
    ruby-position: over;
}

rt {
    font-size: 0.5em;
    font-style: normal;
    font-weight: normal;
    color: #c0392b;
    ruby-align: center;
}

rp {
    display: none;
}

/* Sentence block */
.sentence-block {
    margin-bottom: 1.8em;
    padding: 0.6em 0.8em;
    border-left: 3px solid #3498db;
    background-color: #f9f9f9;
    border-radius: 0 4px 4px 0;
}

/* Sentence number */
.sentence-num {
    font-size: 0.75em;
    color: #999;
    font-weight: bold;
    margin-right: 0.3em;
}

/* Japanese line with furigana */
.jp-line {
    font-size: 1.15em;
    line-height: 2.4;
    margin-bottom: 0.3em;
    color: #222;
}

/* Romaji line */
.romaji-line {
    font-size: 0.8em;
    color: #888;
    font-style: italic;
    line-height: 1.4;
    margin-bottom: 0.3em;
    letter-spacing: 0.02em;
}

/* Chinese translation line */
.cn-line {
    font-size: 0.95em;
    color: #555;
    line-height: 1.6;
    padding-left: 0.3em;
    border-left: 2px solid #e74c3c;
    margin-left: 0.2em;
}

/* Title page */
.title-page {
    text-align: center;
    margin-top: 3em;
}

.title-page h1 {
    font-size: 1.6em;
    border: none;
    color: #c0392b;
}

.title-page .subtitle {
    font-size: 1.0em;
    color: #666;
    margin-top: 0.5em;
}

.title-page .meta {
    font-size: 0.85em;
    color: #999;
    margin-top: 2em;
    line-height: 1.8;
}

/* How to use this book */
.how-to-use {
    margin: 1em 0;
    padding: 0.8em;
    background-color: #f0f7ff;
    border: 1px solid #b3d4fc;
    border-radius: 4px;
    font-size: 0.9em;
    line-height: 1.8;
}

.how-to-use h2 {
    border: none;
    color: #2c3e50;
    margin-top: 0;
}
'''


def build_title_page(title: str, source_book: str, pair_count: int) -> str:
    """Build HTML for the title page."""
    return f'''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="ja">
<head><title>{title}</title></head>
<body>
<div class="title-page">
    <h1>初級日本語 ドリル</h1>
    <p class="subtitle">Beginner Japanese Vocabulary Drill</p>
    <p class="subtitle">初级日语词汇练习</p>
    <p class="subtitle" style="margin-top:1.5em; font-size:0.9em;">
        Chapter: {title}
    </p>
    <div class="meta">
        <p>Source: {source_book}</p>
        <p>Level: JLPT N5 (A1-A2)</p>
        <p>{pair_count} sentence pairs with furigana &amp; romaji</p>
    </div>
</div>

<div class="how-to-use">
    <h2>How to Use This Book / この本の使い方</h2>
    <p>Each sentence is shown in three forms:</p>
    <p><strong>1.</strong> Japanese with furigana (reading above kanji) — 日本語（ふりがな付き）</p>
    <p><strong>2.</strong> Romaji (Latin alphabet reading) — ローマ字</p>
    <p><strong>3.</strong> Chinese translation — 中文翻译</p>
    <p style="margin-top:0.5em;">All vocabulary is JLPT N5 level. Read the Japanese first,
    check the furigana if needed, then confirm meaning with the Chinese.</p>
    <p>所有词汇都是JLPT N5级别。先读日语，需要时查看假名注音，然后用中文确认意思。</p>
</div>
</body>
</html>'''


def build_chapter_html(pairs: list, section_title: str,
                       start_idx: int) -> str:
    """Build HTML for one chapter/section."""
    blocks = []
    for i, pair in enumerate(pairs):
        jp = pair['japanese']
        cn = pair['chinese']
        num = start_idx + i + 1

        ruby_html, romaji = add_furigana_html(jp)

        blocks.append(f'''<div class="sentence-block">
    <p class="jp-line"><span class="sentence-num">{num}.</span> {ruby_html}</p>
    <p class="romaji-line">{romaji}</p>
    <p class="cn-line">{cn}</p>
</div>''')

    body = '\n'.join(blocks)

    return f'''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="ja">
<head><title>{section_title}</title></head>
<body>
<h2>{section_title}</h2>
{body}
</body>
</html>'''


def main():
    # Load script
    with open('beginner_japanese_script.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    pairs = data['sentence_pairs']
    chapter_title = data['chapter_title']
    source_book = data.get('source_book', '')
    logger.info(f'Loaded {len(pairs)} sentence pairs')

    # Create EPUB
    book = epub.EpubBook()
    book.set_identifier('beginner-japanese-n5-drill-ch1')
    book.set_title(f'初級日本語ドリル — {chapter_title}')
    book.set_language('ja')
    book.add_author('Generated from: ' + source_book)

    # Add metadata
    book.add_metadata('DC', 'subject', 'JLPT N5')
    book.add_metadata('DC', 'subject', 'Japanese Language Learning')
    book.add_metadata('DC', 'description',
                      'Beginner Japanese vocabulary drill with furigana, '
                      'romaji, and parallel Chinese translation. '
                      'All vocabulary is JLPT N5 level.')

    # CSS
    css_item = epub.EpubItem(
        uid='style', file_name='style/default.css',
        media_type='text/css', content=EPUB_CSS.encode('utf-8'),
    )
    book.add_item(css_item)

    # Title page
    title_html = build_title_page(chapter_title, source_book, len(pairs))
    title_page = epub.EpubHtml(
        title='Title Page', file_name='title.xhtml', lang='ja',
    )
    title_page.content = title_html.encode('utf-8')
    title_page.add_item(css_item)
    book.add_item(title_page)

    # Build section chapters
    spine = ['nav', title_page]
    toc = []

    for sec_idx, (start, end, sec_name) in enumerate(SECTIONS):
        section_pairs = pairs[start:end]
        if not section_pairs:
            continue

        chapter_html = build_chapter_html(section_pairs, sec_name, start)

        chapter = epub.EpubHtml(
            title=sec_name,
            file_name=f'section_{sec_idx:02d}.xhtml',
            lang='ja',
        )
        chapter.content = chapter_html.encode('utf-8')
        chapter.add_item(css_item)
        book.add_item(chapter)

        spine.append(chapter)
        toc.append(chapter)

        logger.info(f'  Section {sec_idx + 1}: {sec_name} ({len(section_pairs)} pairs)')

    # Set TOC and spine
    book.toc = toc
    book.spine = spine

    # Add navigation
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # Write EPUB
    output_path = 'beginner_japanese_ch1.epub'
    epub.write_epub(output_path, book)

    size_kb = Path(output_path).stat().st_size / 1024
    logger.info(f'\nDone! Output: {output_path} ({size_kb:.0f} KB)')
    logger.info(f'  {len(pairs)} sentences across {len(SECTIONS)} sections')
    logger.info(f'  Furigana + Romaji + Chinese translation')


if __name__ == '__main__':
    main()
