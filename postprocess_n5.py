#!/usr/bin/env python3
"""
Post-process beginner Japanese script for strict N5 compliance + minimal EPUB.

Three phases:
  1. Audit: Run tightened N5 validator on all chapters, collect violations
  2. Correct: Batch LLM correction of violating sentences (up to 5 rounds)
  3. Build: Generate minimal-styling EPUB with furigana + romaji + Chinese

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python postprocess_n5.py

    # Audit only (no LLM calls):
    python postprocess_n5.py --audit-only

    # EPUB only (from existing corrected JSON):
    python postprocess_n5.py --epub-only
"""
import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INPUT_JSON = Path('beginner_japanese_full_script.json')
OUTPUT_JSON = Path('beginner_japanese_full_script_n5.json')
OUTPUT_EPUB = Path('beginner_japanese_full_n5.epub')

LLM_MODEL = 'deepseek/deepseek-chat'
BATCH_SIZE = 30
MAX_CORRECTION_ROUNDS = 5


# ---------------------------------------------------------------------------
# Phase 1: Audit
# ---------------------------------------------------------------------------

def audit_chapter(pairs: list[dict]) -> list[dict]:
    """
    Validate each sentence pair against the tightened N5 validator.

    Returns list of dicts for violating pairs:
      {'index': int, 'japanese': str, 'chinese': str,
       'vocab_violations': [str], 'grammar_violations': [str]}
    """
    from graded_reader.beginner_japanese import validate_n5

    violating = []
    for i, pair in enumerate(pairs):
        jp = pair['japanese']
        report = validate_n5(jp)

        vocab_words = [v['word'] for v in report['violations']]
        grammar_labels = [g['label'] for g in report.get('grammar_violations', [])]

        if vocab_words or grammar_labels:
            violating.append({
                'index': i,
                'japanese': jp,
                'chinese': pair['chinese'],
                'vocab_violations': vocab_words,
                'grammar_violations': grammar_labels,
            })

    return violating


def audit_all(chapters: list[dict]) -> dict:
    """Audit all chapters and print summary."""
    total_pairs = 0
    total_violating = 0
    chapter_reports = []

    for ch_idx, ch in enumerate(chapters):
        pairs = ch['sentence_pairs']
        total_pairs += len(pairs)

        violations = audit_chapter(pairs)
        total_violating += len(violations)

        chapter_reports.append({
            'chapter_title': ch.get('chapter_title', f'Chapter {ch_idx}'),
            'total': len(pairs),
            'violating': len(violations),
            'violations': violations,
        })

        pct = len(violations) / len(pairs) * 100 if pairs else 0
        logger.info(
            f'  Ch.{ch_idx + 1} "{ch.get("chapter_title", "?")[:30]}": '
            f'{len(violations)}/{len(pairs)} violating ({pct:.0f}%)'
        )

    pct = total_violating / total_pairs * 100 if total_pairs else 0
    logger.info(f'\nTotal: {total_violating}/{total_pairs} violating ({pct:.0f}%)')

    return {
        'total_pairs': total_pairs,
        'total_violating': total_violating,
        'chapters': chapter_reports,
    }


# ---------------------------------------------------------------------------
# Phase 2: Batch LLM correction
# ---------------------------------------------------------------------------

_CORRECTION_SYSTEM_PROMPT = """You are a Japanese language teaching expert. You MUST rewrite sentences using ONLY JLPT N5 vocabulary and grammar.

ALLOWED grammar (use ONLY these):
- ～は～です (X is Y)
- ～があります/います (there is X)
- ～を～ます (do X)
- ～に行きます/来ます (go/come to X)
- ～で～ます (do X at/with Y)
- ～から～まで (from X to Y)
- ～たいです (want to do X)
- ～てください (please do X)
- ～ましょう (let's do X)
- ～と思います (I think X)
- ～とき (when X)

FORBIDDEN (do NOT use any of these):
- なければなりません → use てください instead
- ことができます → use simple verb ます form
- について → use の こと or は
- passive ～られる → use active voice
- causative ～させる → use simple てください
- たら/ば/なら conditionals → use と or とき
- しかし → use でも
- ために → use simple から
- かもしれません → omit or use と思います
- でしょう → use と思います
- より → use もっと or omit
- ので → use から
- のに → omit or rephrase
- ながら → split into two sentences
- である → use です

WORD SUBSTITUTIONS for business/technical terms:
- 競争相手 → ほかの会社
- 戦略 → 大切なけいかく
- 製品 → もの/品物
- 管理者 → 上の人
- 市場 → お店
- 利益 → お金
- 技術 → しごとのやりかた
- 会議 → みんなで話すこと
- 変化 → かわること
- 機械 → きかい (if N5) or もの
- 社員 → 会社の人
- 社長 → 会社で一番上の人
- 目標 → やりたいこと
- 計画 → けいかく
- 世界 → せかい
- 心配 → しんぱい

Keep sentences SHORT (5-10 words). Simplify complex ideas.

OUTPUT FORMAT — output ONLY numbered JP/CN pairs:
1. JP: [corrected Japanese]
   CN: [corresponding Chinese — simplified to match]
2. JP: ...
   CN: ..."""


def _call_llm_for_correction(violating_batch: list[dict]) -> list[tuple[str, str]]:
    """Send a batch of violating pairs to LLM for correction."""
    from graded_reader.beginner_japanese import _call_llm, _parse_sentence_pairs

    parts = []
    for item in violating_batch:
        violations = ', '.join(item['vocab_violations'][:8])
        grammar = ', '.join(item['grammar_violations'][:4])
        issue_str = violations
        if grammar:
            issue_str += f'; GRAMMAR: {grammar}'

        parts.append(
            f"JP: {item['japanese']}\n"
            f"CN: {item['chinese']}\n"
            f"Problems: {issue_str}"
        )

    user_prompt = (
        "REWRITE these Japanese sentences using ONLY JLPT N5 vocabulary and grammar.\n"
        "Replace ALL non-N5 words. Keep sentences SHORT.\n\n"
        + '\n\n'.join(f'{i + 1}. {p}' for i, p in enumerate(parts))
        + "\n\nOutput ONLY numbered JP/CN pairs. One pair per input sentence."
    )

    response = _call_llm(
        _CORRECTION_SYSTEM_PROMPT, user_prompt, LLM_MODEL,
        max_tokens=4000, max_retries=3,
    )

    return _parse_sentence_pairs(response)


def correct_chapter(pairs: list[dict], chapter_label: str) -> list[dict]:
    """
    Run up to MAX_CORRECTION_ROUNDS of LLM correction on a chapter's pairs.
    Returns the corrected pairs list (modified in place).
    """
    for round_num in range(1, MAX_CORRECTION_ROUNDS + 1):
        violations = audit_chapter(pairs)

        if not violations:
            logger.info(f'    Round {round_num}: 0 violations — done!')
            break

        logger.info(
            f'    Round {round_num}: {len(violations)} violating sentences, '
            f'sending to LLM in batches of {BATCH_SIZE}...'
        )

        corrected_count = 0

        # Process in batches
        for batch_start in range(0, len(violations), BATCH_SIZE):
            batch = violations[batch_start:batch_start + BATCH_SIZE]

            try:
                corrections = _call_llm_for_correction(batch)

                # Apply corrections back to pairs
                for j, (new_jp, new_cn) in enumerate(corrections):
                    if j < len(batch):
                        idx = batch[j]['index']
                        pairs[idx] = {'japanese': new_jp, 'chinese': new_cn}
                        corrected_count += 1

            except Exception as e:
                logger.warning(f'    Batch {batch_start // BATCH_SIZE + 1} failed: {e}')

            # Rate limit between batches
            time.sleep(1)

        logger.info(f'    Round {round_num}: applied {corrected_count} corrections')

    return pairs


# ---------------------------------------------------------------------------
# Phase 3: Minimal EPUB generation
# ---------------------------------------------------------------------------

MINIMAL_CSS = '''\
body { font-family: sans-serif; line-height: 2; margin: 1em; }
ruby { ruby-position: over; }
rt { font-size: 0.55em; color: #888; }
rp { display: none; }
.n { font-size: 0.75em; color: #aaa; }
.r { font-size: 0.85em; font-style: italic; color: #666; }
.c { font-size: 0.9em; color: #444; }
'''


def _add_furigana_html(text: str, kks) -> tuple[str, str]:
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


def build_minimal_epub(chapters_data: list[dict], source_book: str, output_path: str):
    """Build a minimal-styling EPUB with furigana, romaji, and Chinese."""
    import pykakasi
    from ebooklib import epub

    kks = pykakasi.kakasi()

    total_pairs = sum(len(ch['sentence_pairs']) for ch in chapters_data)

    book = epub.EpubBook()
    book.set_identifier('beginner-japanese-n5-full-book-v2')
    book.set_title('初級日本語ドリル N5')
    book.set_language('ja')
    book.add_author(f'Generated from: {source_book}')

    # Minimal CSS
    css_item = epub.EpubItem(
        uid='style', file_name='style/default.css',
        media_type='text/css', content=MINIMAL_CSS.encode('utf-8'),
    )
    book.add_item(css_item)

    # Title page — plain text, no instruction box
    title_html = f'''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="ja">
<head><title>初級日本語ドリル N5</title></head>
<body>
<h1>初級日本語ドリル N5</h1>
<p>Beginner Japanese Vocabulary Drill</p>
<p>初级日语词汇练习</p>
<p>Source: {source_book}</p>
<p>Level: JLPT N5</p>
<p>{total_pairs} sentence pairs across {len(chapters_data)} chapters</p>
</body>
</html>'''

    title_page = epub.EpubHtml(
        title='Title Page', file_name='title.xhtml', lang='ja',
    )
    title_page.content = title_html.encode('utf-8')
    title_page.add_item(css_item)
    book.add_item(title_page)

    spine = ['nav', title_page]
    toc = []
    global_num = 0

    for ch_idx, ch_data in enumerate(chapters_data):
        ch_title = ch_data.get('chapter_title', f'Chapter {ch_idx + 1}')
        pairs = ch_data['sentence_pairs']

        if not pairs:
            continue

        blocks = []
        for pair in pairs:
            jp = pair['japanese']
            cn = pair['chinese']
            global_num += 1

            ruby_html, romaji = _add_furigana_html(jp, kks)

            blocks.append(
                f'<p><span class="n">{global_num}.</span> {ruby_html}</p>\n'
                f'<p class="r">{romaji}</p>\n'
                f'<p class="c">{cn}</p>\n'
                f'<br/>'
            )

        body = '\n'.join(blocks)
        chapter_html = f'''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="ja">
<head><title>Ch.{ch_idx + 1}: {ch_title}</title></head>
<body>
<h1>Ch.{ch_idx + 1}: {ch_title}</h1>
{body}
</body>
</html>'''

        chapter = epub.EpubHtml(
            title=f'Ch.{ch_idx + 1}: {ch_title}',
            file_name=f'chapter_{ch_idx:02d}.xhtml',
            lang='ja',
        )
        chapter.content = chapter_html.encode('utf-8')
        chapter.add_item(css_item)
        book.add_item(chapter)

        spine.append(chapter)
        toc.append(chapter)

    book.toc = toc
    book.spine = spine
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    epub.write_epub(output_path, book)
    size_kb = Path(output_path).stat().st_size / 1024
    logger.info(f'EPUB written: {output_path} ({size_kb:.0f} KB)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Post-process N5 compliance + minimal EPUB rebuild',
    )
    parser.add_argument(
        '--audit-only', action='store_true',
        help='Only audit violations, no LLM correction or EPUB',
    )
    parser.add_argument(
        '--epub-only', action='store_true',
        help='Only build EPUB from existing corrected JSON',
    )
    parser.add_argument(
        '--input', default=str(INPUT_JSON),
        help=f'Input JSON path (default: {INPUT_JSON})',
    )
    parser.add_argument(
        '--output-json', default=str(OUTPUT_JSON),
        help=f'Output JSON path (default: {OUTPUT_JSON})',
    )
    parser.add_argument(
        '--output-epub', default=str(OUTPUT_EPUB),
        help=f'Output EPUB path (default: {OUTPUT_EPUB})',
    )
    parser.add_argument(
        '--model', default=LLM_MODEL,
        help=f'LLM model for corrections (default: {LLM_MODEL})',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    global LLM_MODEL
    LLM_MODEL = args.model

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if args.epub_only:
        # Load corrected JSON
        json_path = Path(args.output_json)
        if not json_path.exists():
            logger.error(f'Corrected JSON not found: {json_path}')
            sys.exit(1)
        logger.info(f'Loading corrected JSON: {json_path}')
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        json_path = Path(args.input)
        if not json_path.exists():
            logger.error(f'Input JSON not found: {json_path}')
            sys.exit(1)
        logger.info(f'Loading: {json_path}')
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    chapters = data['chapters']
    source_book = data.get('source_book', 'Unknown')
    total_pairs = sum(len(ch['sentence_pairs']) for ch in chapters)
    logger.info(f'Loaded {total_pairs} pairs across {len(chapters)} chapters')

    # ------------------------------------------------------------------
    # Phase 1: Audit
    # ------------------------------------------------------------------
    logger.info('')
    logger.info('=' * 60)
    logger.info('PHASE 1: Audit N5 compliance (tightened validator)')
    logger.info('=' * 60)

    audit = audit_all(chapters)

    if args.audit_only:
        logger.info('\n--audit-only: stopping here.')
        return

    if args.epub_only:
        # Skip correction, go straight to EPUB
        logger.info('\n--epub-only: skipping correction, building EPUB...')
        logger.info('')
        logger.info('=' * 60)
        logger.info('PHASE 3: Building minimal EPUB')
        logger.info('=' * 60)
        build_minimal_epub(chapters, source_book, args.output_epub)
        return

    # ------------------------------------------------------------------
    # Phase 2: Batch LLM correction
    # ------------------------------------------------------------------
    if audit['total_violating'] == 0:
        logger.info('\nNo violations found — skipping correction.')
    else:
        # Check API key
        api_key = os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            logger.error(
                'OPENROUTER_API_KEY not set.\n'
                'Set it with: export OPENROUTER_API_KEY=sk-or-...\n'
                'Get a key at: https://openrouter.ai/keys'
            )
            sys.exit(1)

        logger.info('')
        logger.info('=' * 60)
        logger.info(f'PHASE 2: Batch LLM correction (model={LLM_MODEL})')
        logger.info('=' * 60)

        for ch_idx, ch in enumerate(chapters):
            pairs = ch['sentence_pairs']
            ch_title = ch.get('chapter_title', f'Chapter {ch_idx}')

            # Quick check if this chapter has violations
            violations = audit_chapter(pairs)
            if not violations:
                logger.info(f'\n  Ch.{ch_idx + 1} "{ch_title[:30]}": 0 violations — skip')
                continue

            logger.info(
                f'\n  Ch.{ch_idx + 1} "{ch_title[:30]}": '
                f'{len(violations)} violations — correcting...'
            )

            corrected_pairs = correct_chapter(pairs, ch_title)
            ch['sentence_pairs'] = corrected_pairs

        # Final audit
        logger.info('')
        logger.info('--- Final audit after correction ---')
        final_audit = audit_all(chapters)

        logger.info(
            f'\nBefore: {audit["total_violating"]}/{audit["total_pairs"]} violating'
        )
        logger.info(
            f'After:  {final_audit["total_violating"]}/{final_audit["total_pairs"]} violating'
        )

    # Save corrected JSON
    total_pairs = sum(len(ch['sentence_pairs']) for ch in chapters)
    output_data = {
        'source_book': source_book,
        'model': LLM_MODEL,
        'total_pairs': total_pairs,
        'chapters': chapters,
    }
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info(f'\nCorrected JSON saved: {args.output_json}')

    # ------------------------------------------------------------------
    # Phase 3: Minimal EPUB
    # ------------------------------------------------------------------
    logger.info('')
    logger.info('=' * 60)
    logger.info('PHASE 3: Building minimal EPUB')
    logger.info('=' * 60)

    build_minimal_epub(chapters, source_book, args.output_epub)

    logger.info('')
    logger.info('Done!')
    logger.info(f'  JSON: {args.output_json}')
    logger.info(f'  EPUB: {args.output_epub}')


if __name__ == '__main__':
    main()
