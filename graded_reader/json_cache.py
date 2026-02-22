"""
Intermediate JSON cache for EPUB graded reader generation.

Separates the expensive LLM work (extract) from the cheap formatting work (build).
  - extract_to_json: source EPUB -> JSON (runs all LLM calls, done once)
  - build_epub_from_json: JSON + source EPUB -> output EPUB (no LLM calls, instant)
  - enrich_json: add translations for new languages to existing JSON
"""

import json
import logging
import posixpath
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString
from ebooklib import epub

from .chinese_processing import (
    annotate_text,
    contains_chinese,
    split_sentences,
)
from .llm_simplifier import (
    add_word_spacing_llm,
    simplify_to_hsk4,
    verify_simplification,
)
from .llm_translator import translate_text_llm
from .epub_processor import (
    RUBY_CSS,
    _build_toc_page,
    _deduplicate_items,
    _ensure_nav_document,
    _fix_toc_uids,
    _split_oversized_items,
)

logger = logging.getLogger(__name__)

# Same block tags used in epub_processor.py for consistency
_ALL_BLOCKS = [
    'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th',
    'caption', 'blockquote', 'dt', 'dd', 'figcaption', 'pre',
    'div', 'section', 'article', 'aside', 'header', 'footer',
]

# Subset used for content chapter detection (same as epub_processor.py)
_CONTENT_BLOCKS = [
    'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th',
    'blockquote', 'dt', 'dd', 'figcaption', 'div',
]


@dataclass
class BuildConfig:
    """All formatting options for build_epub_from_json (no LLM involvement)."""
    mode: str = "interlinear"          # interlinear, dual_ruby, parallel_text
    target_languages: list[str] = field(default_factory=lambda: ["pt", "it", "fr", "de", "es", "tr"])
    lang_start_index: int = 0
    use_simplified: bool = False
    word_spacing: bool = True
    css_overrides: dict | None = None  # e.g. {"zh_font_size": "1.1em"}


# ---------------------------------------------------------------------------
# Extraction: source EPUB -> JSON
# ---------------------------------------------------------------------------

def _get_spine_items(book: epub.EpubBook) -> list[epub.EpubItem]:
    """Get XHTML items in spine (reading) order."""
    items_by_id = {}
    for item in book.get_items():
        items_by_id[item.get_id()] = item
        items_by_id[item.get_name()] = item

    spine_items = []
    for entry in book.spine:
        item_id = entry[0] if isinstance(entry, tuple) else entry
        item = items_by_id.get(item_id)
        if item and item.get_type() == 9:  # ITEM_DOCUMENT
            spine_items.append(item)
    return spine_items


def _identify_content_chapters(spine_items: list[epub.EpubItem]) -> list[epub.EpubItem]:
    """Filter spine items to those containing Chinese text blocks."""
    content_items = []
    for item in spine_items:
        html_raw = item.get_content().decode('utf-8', errors='replace')
        soup = BeautifulSoup(html_raw, 'lxml')
        for block in soup.find_all(_CONTENT_BLOCKS):
            if block.find(_CONTENT_BLOCKS):
                continue
            text = block.get_text().strip()
            if text and contains_chinese(text):
                content_items.append(item)
                break
    return content_items


def _extract_chinese_blocks(html_content: str) -> list[tuple[str, str]]:
    """Extract (tag_name, plain_text) for each leaf Chinese block in HTML."""
    soup = BeautifulSoup(html_content, 'lxml')
    blocks = []
    for block in soup.find_all(_ALL_BLOCKS):
        if block.find(_ALL_BLOCKS):
            continue
        plain_text = block.get_text()
        if not contains_chinese(plain_text):
            continue
        blocks.append((block.name, plain_text))
    return blocks


def _translate_toc_recursive(
    toc,
    target_languages: list[str],
    model: str,
) -> list[dict]:
    """Translate TOC entries to all target languages, returning JSON-serializable list."""
    result = []
    for item in toc:
        if isinstance(item, tuple):
            section, children = item
            title = section.title if hasattr(section, 'title') else str(section)
            href = section.href if hasattr(section, 'href') else ''
            entry = {
                'original': title or '',
                'href': href or '',
                'translations': {},
                'subsections': _translate_toc_recursive(children, target_languages, model),
            }
        else:
            title = item.title if hasattr(item, 'title') else str(item)
            href = item.href if hasattr(item, 'href') else ''
            entry = {
                'original': title or '',
                'href': href or '',
                'translations': {},
                'subsections': [],
            }

        # Translate the title to each target language
        if entry['original'] and contains_chinese(entry['original']):
            for lang in target_languages:
                try:
                    tr = translate_text_llm(
                        entry['original'], source='zh-CN', target=lang, model=model,
                    )
                    if tr and not tr.startswith('['):
                        entry['translations'][lang] = tr
                except Exception as e:
                    logger.warning(f'TOC translation to {lang} failed: {e}')
        else:
            # Already non-Chinese — use as-is for all languages
            for lang in target_languages:
                entry['translations'][lang] = entry['original']

        result.append(entry)
    return result


def extract_to_json(
    epub_path: str,
    output_json: str,
    target_languages: list[str] | None = None,
    simplify_hsk4: bool = False,
    model: str | None = None,
    chapter_start: int = 0,
    chapter_count: int = 0,
    max_workers: int = 8,
) -> str:
    """
    Extract all LLM-dependent data from a source EPUB into a JSON cache.

    If output_json already exists, performs incremental update: skips sentences
    that already have translations for all requested languages.

    Returns the path to the written JSON file.
    """
    if target_languages is None:
        target_languages = ["pt", "it", "fr", "de", "es", "tr"]

    if model is None:
        from .models import TIER_DEFAULTS
        model = TIER_DEFAULTS["standard"]

    # Load existing JSON for incremental update
    existing_data = None
    if Path(output_json).exists():
        with open(output_json, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        logger.info(f'Loaded existing JSON cache: {output_json}')

    # Read source EPUB
    logger.info(f'Reading EPUB: {epub_path}')
    book = epub.read_epub(epub_path, options={'ignore_ncx': True})

    # Walk spine to find content chapters
    spine_items = _get_spine_items(book)
    content_items = _identify_content_chapters(spine_items)

    # Apply chapter range
    if chapter_start > 0 or chapter_count > 0:
        end = chapter_start + chapter_count if chapter_count > 0 else len(content_items)
        content_items = content_items[chapter_start:end]
        logger.info(f'Processing chapters [{chapter_start}:{end}]')

    logger.info(f'Found {len(content_items)} content chapters to process')

    # Build existing chapter lookup for incremental mode
    existing_chapters = {}
    if existing_data:
        for ch in existing_data.get('chapters', []):
            existing_chapters[ch['source_file']] = ch

    # Process each chapter
    chapters = []
    for ch_idx, item in enumerate(content_items):
        source_file = item.get_name()
        logger.info(f'Processing chapter {ch_idx + 1}/{len(content_items)}: {source_file}')

        html_content = item.get_content().decode('utf-8', errors='replace')
        raw_blocks = _extract_chinese_blocks(html_content)

        # Check existing chapter data for incremental update
        existing_ch = existing_chapters.get(source_file, {})
        existing_blocks = {b['index']: b for b in existing_ch.get('blocks', [])}

        blocks = []
        # Collect all sentences across all blocks for batch translation
        all_sentences_info = []  # (block_idx, sent_idx, chinese_text, existing_translations)

        for blk_idx, (tag, plain_text) in enumerate(raw_blocks):
            existing_blk = existing_blocks.get(blk_idx, {})

            # Determine simplified and word_spaced text
            simplified = existing_blk.get('simplified', '')
            word_spaced = existing_blk.get('word_spaced', '')

            if simplify_hsk4 and not simplified:
                try:
                    s = simplify_to_hsk4(plain_text, model=model)
                    if s and not s.startswith('['):
                        s = verify_simplification(plain_text, s, model=model)
                        simplified = s
                except Exception as e:
                    logger.warning(f'Simplification failed for block {blk_idx}: {e}')

            if not word_spaced:
                try:
                    ws = add_word_spacing_llm(plain_text, model=model)
                    if ws:
                        word_spaced = ws
                except Exception as e:
                    logger.warning(f'Word spacing failed for block {blk_idx}: {e}')

            # Split into sentences (from simplified if available, else word_spaced, else original)
            text_for_splitting = simplified or word_spaced or plain_text
            sents = split_sentences(text_for_splitting)
            if not sents:
                sents = [text_for_splitting.strip()]

            # Also split original for the chinese field
            orig_sents = split_sentences(plain_text)
            if not orig_sents:
                orig_sents = [plain_text.strip()]

            # Build sentence data, reusing existing translations
            existing_sents = existing_blk.get('sentences', [])
            sentences = []
            for s_idx, sent_text in enumerate(sents):
                # Match existing sentence by index
                existing_sent = existing_sents[s_idx] if s_idx < len(existing_sents) else {}
                existing_translations = existing_sent.get('translations', {})

                orig_sent = orig_sents[s_idx] if s_idx < len(orig_sents) else sent_text

                sent_data = {
                    'chinese': orig_sent,
                    'chinese_simplified': sent_text if simplified else '',
                    'chinese_spaced': '',  # will be set below
                    'translations': dict(existing_translations),
                }

                # Set spaced version
                if word_spaced:
                    spaced_sents = split_sentences(word_spaced)
                    if not spaced_sents:
                        spaced_sents = [word_spaced.strip()]
                    sent_data['chinese_spaced'] = spaced_sents[s_idx] if s_idx < len(spaced_sents) else ''

                # Track which languages need translation
                for lang in target_languages:
                    if lang not in sent_data['translations']:
                        all_sentences_info.append((blk_idx, s_idx, orig_sent, lang))

                sentences.append(sent_data)

            block_data = {
                'index': blk_idx,
                'tag': tag,
                'original': plain_text,
                'simplified': simplified,
                'word_spaced': word_spaced,
                'sentences': sentences,
            }
            blocks.append(block_data)

        # Batch translate all missing sentences (parallel)
        if all_sentences_info:
            logger.info(f'  Translating {len(all_sentences_info)} sentences...')

            # Group by language for better batching
            def _translate(info):
                blk_idx, s_idx, text, lang = info
                try:
                    tr = translate_text_llm(text, source='zh-CN', target=lang, model=model)
                    if tr and not tr.startswith('['):
                        return blk_idx, s_idx, lang, tr
                except Exception as e:
                    logger.warning(f'Translation failed ({lang}): {e}')
                return blk_idx, s_idx, lang, ''

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_translate, info) for info in all_sentences_info]
                for future in as_completed(futures):
                    blk_idx, s_idx, lang, tr = future.result()
                    if tr:
                        blocks[blk_idx]['sentences'][s_idx]['translations'][lang] = tr

        chapters.append({
            'source_file': source_file,
            'chapter_index': ch_idx,
            'blocks': blocks,
        })

    # Translate TOC
    logger.info('Translating TOC entries...')
    toc_data = _translate_toc_recursive(book.toc, target_languages, model)

    # Get metadata from Dublin Core namespace
    DC = 'http://purl.org/dc/elements/1.1/'
    dc_meta = book.metadata.get(DC, {})

    title = ''
    title_entries = dc_meta.get('title', [])
    if title_entries:
        title = title_entries[0][0] if isinstance(title_entries[0], tuple) else str(title_entries[0])

    author = ''
    creator_entries = dc_meta.get('creator', [])
    if creator_entries:
        author = creator_entries[0][0] if isinstance(creator_entries[0], tuple) else str(creator_entries[0])

    data = {
        'metadata': {
            'title': title,
            'author': author,
            'source_epub': epub_path,
            'created': datetime.now(timezone.utc).isoformat(),
            'models': {
                'simplify': model if simplify_hsk4 else None,
                'translate': model,
                'word_spacing': model,
            },
            'target_languages': target_languages,
        },
        'toc': toc_data,
        'chapters': chapters,
    }

    # Write JSON
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f'JSON cache written to: {output_json}')
    return output_json


# ---------------------------------------------------------------------------
# Extract from existing output EPUBs (no LLM calls)
# ---------------------------------------------------------------------------

def _extract_zh_tr_pairs(z, chapter_files: list[str]) -> list[tuple[str, str]]:
    """
    Extract (chinese_text, translation_text) pairs from .zh/.tr paragraphs
    across one or more XHTML files (handling _p1, _p2 splits).
    """
    pairs = []
    for fname in chapter_files:
        content = z.read(fname).decode('utf-8', errors='replace')
        soup = BeautifulSoup(content, 'lxml')

        zh_tags = soup.find_all('p', class_='zh')
        tr_tags = soup.find_all('p', class_='tr')

        # Build a map: for each .zh, find the next .tr sibling
        for zh_tag in zh_tags:
            # Strip ruby annotations to get plain Chinese text
            zh_copy = BeautifulSoup(str(zh_tag), 'html.parser')
            for rt in zh_copy.find_all(['rt', 'rp']):
                rt.decompose()
            zh_text = zh_copy.get_text().strip()

            # Find paired .tr (should be the next sibling)
            tr_text = ''
            next_sib = zh_tag.find_next_sibling()
            if next_sib and next_sib.get('class') and 'tr' in next_sib.get('class', []):
                tr_text = next_sib.get_text().strip()

            pairs.append((zh_text, tr_text))

    return pairs


def _detect_chapter_language(translations: list[str]) -> str:
    """Detect the language of a list of translation strings."""
    sample = ' '.join(t for t in translations[:10] if t)
    if not sample.strip():
        return 'unknown'
    try:
        from langdetect import detect
        return detect(sample)
    except Exception:
        return 'unknown'


def extract_from_existing_epubs(
    source_epub: str,
    original_epub: str | None = None,
    simplified_epub: str | None = None,
    output_json: str = 'book_data/book.json',
    target_languages: list[str] | None = None,
    lang_start_index: int = 0,
) -> str:
    """
    Build a JSON cache by extracting data from existing output EPUBs.

    No LLM calls are made. Reads the source EPUB for block structure,
    then matches sentences to .zh/.tr pairs in the output EPUBs.

    Args:
        source_epub: Path to the original Chinese EPUB (unprocessed).
        original_epub: Path to the interlinear EPUB with original (word-spaced) Chinese.
        simplified_epub: Path to the interlinear EPUB with simplified Chinese.
        output_json: Path for the output JSON cache file.
        target_languages: Language rotation order used when building the EPUBs.
        lang_start_index: The --lang-start value used when building the EPUBs.

    Returns the path to the written JSON file.
    """
    import re as _re
    import zipfile

    if target_languages is None:
        target_languages = ["pt", "it", "fr", "de", "es", "tr"]

    # Read source EPUB for block structure
    logger.info(f'Reading source EPUB: {source_epub}')
    book = epub.read_epub(source_epub, options={'ignore_ncx': True})
    spine_items = _get_spine_items(book)
    content_items = _identify_content_chapters(spine_items)

    logger.info(f'Found {len(content_items)} content chapters')

    # Open output EPUB zip files
    orig_zip = zipfile.ZipFile(original_epub) if original_epub else None
    simp_zip = zipfile.ZipFile(simplified_epub) if simplified_epub else None

    # Build file lookup for output EPUBs (group split files by base name)
    def _group_output_files(z):
        if z is None:
            return {}
        groups = {}
        for f in sorted(z.namelist()):
            if not f.endswith('.xhtml'):
                continue
            # Strip EPUB/ prefix and _pN suffix to get base name
            short = f.split('/')[-1]
            base = _re.sub(r'_p\d+\.xhtml$', '.xhtml', short)
            groups.setdefault(base, []).append(f)
        return groups

    orig_files = _group_output_files(orig_zip)
    simp_files = _group_output_files(simp_zip)

    # Process each content chapter
    chapters = []
    for ch_idx, item in enumerate(content_items):
        source_file = item.get_name()
        short_name = source_file.split('/')[-1]

        # Determine target language for this chapter
        lang = target_languages[(lang_start_index + ch_idx) % len(target_languages)]

        logger.info(f'Processing chapter {ch_idx + 1}/{len(content_items)}: {short_name} (lang={lang})')

        # Extract blocks from source EPUB
        html_content = item.get_content().decode('utf-8', errors='replace')
        raw_blocks = _extract_chinese_blocks(html_content)

        # Get .zh/.tr pairs from output EPUBs
        orig_pairs = _extract_zh_tr_pairs(orig_zip, orig_files.get(short_name, [])) if orig_zip else []
        simp_pairs = _extract_zh_tr_pairs(simp_zip, simp_files.get(short_name, [])) if simp_zip else []

        # Walk through blocks, splitting sentences to match output pairs
        pair_cursor_orig = 0
        pair_cursor_simp = 0

        blocks = []
        for blk_idx, (tag, plain_text) in enumerate(raw_blocks):
            # Split source text into sentences (deterministic)
            sents = split_sentences(plain_text)
            if not sents:
                sents = [plain_text.strip()]

            sentences = []
            word_spaced_parts = []
            simplified_parts = []

            for s_idx, orig_sent in enumerate(sents):
                sent_data = {
                    'chinese': orig_sent,
                    'chinese_simplified': '',
                    'chinese_spaced': '',
                    'translations': {},
                }

                # Match with original EPUB (word-spaced Chinese + translation)
                if pair_cursor_orig < len(orig_pairs):
                    spaced_zh, tr_text = orig_pairs[pair_cursor_orig]
                    sent_data['chinese_spaced'] = spaced_zh
                    word_spaced_parts.append(spaced_zh)
                    if tr_text:
                        sent_data['translations'][lang] = tr_text
                    pair_cursor_orig += 1

                # Match with simplified EPUB (simplified Chinese + translation)
                if pair_cursor_simp < len(simp_pairs):
                    simp_zh, simp_tr = simp_pairs[pair_cursor_simp]
                    sent_data['chinese_simplified'] = simp_zh
                    simplified_parts.append(simp_zh)
                    # Simplified EPUB may have different translation (same language)
                    # Keep original's translation as primary
                    if simp_tr and lang not in sent_data['translations']:
                        sent_data['translations'][lang] = simp_tr
                    pair_cursor_simp += 1

                sentences.append(sent_data)

            block_data = {
                'index': blk_idx,
                'tag': tag,
                'original': plain_text,
                'simplified': ' '.join(simplified_parts) if simplified_parts else '',
                'word_spaced': ' '.join(word_spaced_parts) if word_spaced_parts else '',
                'sentences': sentences,
            }
            blocks.append(block_data)

        chapters.append({
            'source_file': source_file,
            'chapter_index': ch_idx,
            'blocks': blocks,
        })

    # Close zip files
    if orig_zip:
        orig_zip.close()
    if simp_zip:
        simp_zip.close()

    # Extract TOC with detected translations
    toc_data = _extract_toc_from_book(book)

    # Get metadata
    DC = 'http://purl.org/dc/elements/1.1/'
    dc_meta = book.metadata.get(DC, {})
    title = ''
    title_entries = dc_meta.get('title', [])
    if title_entries:
        title = title_entries[0][0] if isinstance(title_entries[0], tuple) else str(title_entries[0])
    author = ''
    creator_entries = dc_meta.get('creator', [])
    if creator_entries:
        author = creator_entries[0][0] if isinstance(creator_entries[0], tuple) else str(creator_entries[0])

    data = {
        'metadata': {
            'title': title,
            'author': author,
            'source_epub': source_epub,
            'created': datetime.now(timezone.utc).isoformat(),
            'models': {
                'simplify': 'extracted_from_epub',
                'translate': 'extracted_from_epub',
                'word_spacing': 'extracted_from_epub',
            },
            'target_languages': target_languages,
        },
        'toc': toc_data,
        'chapters': chapters,
    }

    # Write JSON
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Summary stats
    total_sents = sum(len(b['sentences']) for ch in chapters for b in ch['blocks'])
    total_translated = sum(
        1 for ch in chapters for b in ch['blocks'] for s in b['sentences']
        if s['translations']
    )
    logger.info(f'JSON cache written to: {output_json}')
    logger.info(f'  {len(chapters)} chapters, {total_sents} sentences, {total_translated} translated')

    return output_json


def _extract_toc_from_book(book: epub.EpubBook) -> list[dict]:
    """Extract TOC structure from book as JSON-serializable list."""
    result = []
    for item in book.toc:
        if isinstance(item, tuple):
            section, children = item
            title = section.title if hasattr(section, 'title') else str(section)
            href = section.href if hasattr(section, 'href') else ''
            entry = {
                'original': title or '',
                'href': href or '',
                'translations': {},
                'subsections': _extract_toc_from_book_inner(children),
            }
        else:
            title = item.title if hasattr(item, 'title') else str(item)
            href = item.href if hasattr(item, 'href') else ''
            entry = {
                'original': title or '',
                'href': href or '',
                'translations': {},
                'subsections': [],
            }
        result.append(entry)
    return result


def _extract_toc_from_book_inner(toc) -> list[dict]:
    """Recursive helper for TOC extraction."""
    result = []
    for item in toc:
        if isinstance(item, tuple):
            section, children = item
            title = section.title if hasattr(section, 'title') else str(section)
            href = section.href if hasattr(section, 'href') else ''
            entry = {
                'original': title or '',
                'href': href or '',
                'translations': {},
                'subsections': _extract_toc_from_book_inner(children),
            }
        else:
            title = item.title if hasattr(item, 'title') else str(item)
            href = item.href if hasattr(item, 'href') else ''
            entry = {
                'original': title or '',
                'href': href or '',
                'translations': {},
                'subsections': [],
            }
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Enrich: add translations for new languages to existing JSON
# ---------------------------------------------------------------------------

def enrich_json(
    json_path: str,
    new_languages: list[str],
    model: str | None = None,
    max_workers: int = 8,
) -> str:
    """
    Add translations for new languages to an existing JSON cache.

    Skips sentences that already have translations for each language.
    Returns the path to the updated JSON file.
    """
    if model is None:
        from .models import TIER_DEFAULTS
        model = TIER_DEFAULTS["standard"]

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Collect missing translations
    work_items = []  # (ch_idx, blk_idx, s_idx, text, lang)
    for ch_idx, chapter in enumerate(data['chapters']):
        for blk_idx, block in enumerate(chapter['blocks']):
            for s_idx, sent in enumerate(block['sentences']):
                for lang in new_languages:
                    if lang not in sent['translations']:
                        work_items.append((ch_idx, blk_idx, s_idx, sent['chinese'], lang))

    if not work_items:
        logger.info('All sentences already have translations for requested languages.')
        return json_path

    logger.info(f'Translating {len(work_items)} missing sentences...')

    def _translate(item):
        ch_idx, blk_idx, s_idx, text, lang = item
        try:
            tr = translate_text_llm(text, source='zh-CN', target=lang, model=model)
            if tr and not tr.startswith('['):
                return ch_idx, blk_idx, s_idx, lang, tr
        except Exception as e:
            logger.warning(f'Translation failed ({lang}): {e}')
        return ch_idx, blk_idx, s_idx, lang, ''

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_translate, item) for item in work_items]
        for future in as_completed(futures):
            ch_idx, blk_idx, s_idx, lang, tr = future.result()
            if tr:
                data['chapters'][ch_idx]['blocks'][blk_idx]['sentences'][s_idx]['translations'][lang] = tr

    # Update metadata
    existing_langs = set(data['metadata'].get('target_languages', []))
    existing_langs.update(new_languages)
    data['metadata']['target_languages'] = sorted(existing_langs)

    # Also enrich TOC translations
    _enrich_toc_translations(data.get('toc', []), new_languages, model)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f'Updated JSON cache: {json_path}')
    return json_path


def _enrich_toc_translations(toc_entries: list[dict], new_languages: list[str], model: str):
    """Add missing TOC translations for new languages."""
    for entry in toc_entries:
        original = entry.get('original', '')
        if original and contains_chinese(original):
            for lang in new_languages:
                if lang not in entry.get('translations', {}):
                    try:
                        tr = translate_text_llm(original, source='zh-CN', target=lang, model=model)
                        if tr and not tr.startswith('['):
                            entry.setdefault('translations', {})[lang] = tr
                    except Exception as e:
                        logger.warning(f'TOC enrich translation to {lang} failed: {e}')
        _enrich_toc_translations(entry.get('subsections', []), new_languages, model)


# ---------------------------------------------------------------------------
# Build: JSON + source EPUB -> output EPUB (no LLM calls)
# ---------------------------------------------------------------------------

def _apply_css_overrides(css: str, overrides: dict | None) -> str:
    """Apply CSS overrides to the base RUBY_CSS."""
    if not overrides:
        return css

    replacements = {
        'zh_font_size': ('.zh', 'font-size', overrides.get('zh_font_size')),
        'tr_font_size': ('.tr', 'font-size', overrides.get('tr_font_size')),
        'zh_line_height': ('.zh', 'line-height', overrides.get('zh_line_height')),
        'tr_line_height': ('.tr', 'line-height', overrides.get('tr_line_height')),
        'tr_color': ('.tr', 'color', overrides.get('tr_color')),
        'body_line_height': ('body', 'line-height', overrides.get('body_line_height')),
    }

    import re
    for key, (selector, prop, value) in replacements.items():
        if value is None:
            continue
        # Find the selector block and replace the property value
        # This handles simple cases like ".zh { ... font-size: 1.1em; ... }"
        escaped_sel = re.escape(selector)
        pattern = rf'({escaped_sel}\s*\{{[^}}]*?{re.escape(prop)}\s*:\s*)[^;]+(;)'
        replacement = rf'\g<1>{value}\2'
        css = re.sub(pattern, replacement, css, count=1)

    return css


def _apply_translated_toc(book: epub.EpubBook, toc_data: list[dict], lang: str):
    """Replace book.toc titles with translations from JSON for the given language."""
    def _apply_to_toc(toc, toc_json):
        for i, item in enumerate(toc):
            if i >= len(toc_json):
                break
            entry = toc_json[i]
            translated = entry.get('translations', {}).get(lang, entry.get('original', ''))

            if isinstance(item, tuple):
                section, children = item
                if hasattr(section, 'title') and translated:
                    section.title = translated
                _apply_to_toc(children, entry.get('subsections', []))
            elif hasattr(item, 'title') and translated:
                item.title = translated

    _apply_to_toc(book.toc, toc_data)


def build_epub_from_json(
    json_path: str,
    output_path: str,
    config: BuildConfig | None = None,
    source_epub: str | None = None,
) -> str:
    """
    Build an EPUB from a JSON cache and the original source EPUB.

    No LLM calls are made. Pinyin is computed deterministically at build time.

    Args:
        json_path: Path to the JSON cache file.
        output_path: Path for the output EPUB.
        config: BuildConfig with formatting options.
        source_epub: Path to the source EPUB. If None, uses metadata.source_epub from JSON.

    Returns the path to the written EPUB.
    """
    if config is None:
        config = BuildConfig()

    # Load JSON cache
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Determine source EPUB path
    epub_path = source_epub or data['metadata']['source_epub']
    logger.info(f'Reading source EPUB: {epub_path}')
    book = epub.read_epub(epub_path, options={'ignore_ncx': True})

    # Add CSS (with optional overrides)
    css_content = _apply_css_overrides(RUBY_CSS, config.css_overrides)
    css_item = epub.EpubItem(
        uid='graded_reader_style',
        file_name='style/graded_reader.css',
        media_type='text/css',
        content=css_content.encode('utf-8'),
    )
    book.add_item(css_item)

    # Build lookup from source_file -> chapter JSON data
    chapter_lookup = {ch['source_file']: ch for ch in data['chapters']}

    # Process each HTML document in the book
    all_doc_items = list(book.get_items_of_type(9))

    for item in all_doc_items:
        source_file = item.get_name()
        ch_data = chapter_lookup.get(source_file)

        if not ch_data:
            # Non-content chapter — leave unchanged, just add CSS link
            item_dir = posixpath.dirname(item.get_name())
            css_rel = posixpath.relpath('style/graded_reader.css', item_dir)
            item.add_link(href=css_rel, rel='stylesheet', type='text/css')
            continue

        # Use chapter_index from JSON for language rotation (matches extract order)
        ch_idx = ch_data['chapter_index']
        if config.target_languages:
            lang = config.target_languages[
                (config.lang_start_index + ch_idx) % len(config.target_languages)
            ]
        else:
            lang = 'en'

        logger.info(f'Building chapter {ch_idx + 1}: {source_file} (lang={lang})')

        # Parse HTML and extract blocks (same logic as extraction)
        html_content = item.get_content().decode('utf-8', errors='replace')
        soup = BeautifulSoup(html_content, 'lxml')

        # Find Chinese blocks (same filtering as epub_processor)
        block_tags = soup.find_all(_ALL_BLOCKS)
        chinese_blocks = []
        for block in block_tags:
            if block.find(_ALL_BLOCKS):
                continue
            plain_text = block.get_text()
            if not contains_chinese(plain_text):
                continue
            chinese_blocks.append(block)

        # Match blocks by index with JSON chapter data
        json_blocks = {b['index']: b for b in ch_data['blocks']}

        for blk_idx, block in enumerate(chinese_blocks):
            blk_data = json_blocks.get(blk_idx)
            if not blk_data:
                continue

            if config.mode == 'interlinear':
                _build_interlinear_block(
                    block, soup, blk_data, lang, config,
                )
            elif config.mode == 'parallel_text':
                _build_parallel_text_block(
                    block, soup, blk_data, lang, config,
                )
            else:
                # Default: interlinear
                _build_interlinear_block(
                    block, soup, blk_data, lang, config,
                )

        # Set processed content back
        result = str(soup)
        if 'xmlns' in html_content and 'xmlns' not in result:
            result = result.replace('<html', '<html xmlns="http://www.w3.org/1999/xhtml"', 1)
        item.set_content(result.encode('utf-8'))

        # Link CSS
        item_dir = posixpath.dirname(item.get_name())
        css_rel = posixpath.relpath('style/graded_reader.css', item_dir)
        item.add_link(href=css_rel, rel='stylesheet', type='text/css')

    # Apply translated TOC (use first language for TOC)
    toc_lang = config.target_languages[0] if config.target_languages else 'en'
    if data.get('toc'):
        _apply_translated_toc(book, data['toc'], toc_lang)

    # Standard EPUB post-processing (same as epub_processor.py)
    _fix_toc_uids(book.toc)
    _deduplicate_items(book)

    for item in book.get_items():
        if item.get_id() is None:
            item_name = item.get_name().replace('/', '_').replace('.', '_')
            item.set_id(f'item_{item_name}')

    _split_oversized_items(book)
    _build_toc_page(book)
    _ensure_nav_document(book)

    logger.info(f'Writing output EPUB: {output_path}')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(output_path, book)
    logger.info('Done!')
    return output_path


def _build_interlinear_block(
    block,
    soup: BeautifulSoup,
    blk_data: dict,
    lang: str,
    config: BuildConfig,
):
    """Build interlinear (alternating .zh/.tr) HTML for a single block."""
    sentences = blk_data.get('sentences', [])
    if not sentences:
        return

    new_elements = []
    for sent in sentences:
        # Choose which Chinese text variant to use
        if config.use_simplified and sent.get('chinese_simplified'):
            zh_text = sent['chinese_simplified']
        elif config.word_spacing and sent.get('chinese_spaced'):
            zh_text = sent['chinese_spaced']
        else:
            zh_text = sent['chinese']

        # Build Chinese paragraph with pinyin ruby (computed at build time)
        zh_html = annotate_text(zh_text, word_spacing=config.word_spacing)
        zh_p = soup.new_tag('p')
        zh_p['class'] = 'zh'
        zh_content = BeautifulSoup(zh_html, 'html.parser')
        for child in list(zh_content.children):
            zh_p.append(child.extract() if hasattr(child, 'extract') else NavigableString(str(child)))
        new_elements.append(zh_p)

        # Translation paragraph
        tr_text = sent.get('translations', {}).get(lang, '')
        if tr_text:
            tr_p = soup.new_tag('p')
            tr_p['class'] = 'tr'
            tr_p.string = tr_text
            new_elements.append(tr_p)

    # Replace the original block
    if new_elements:
        for elem in new_elements:
            block.insert_before(elem)
        block.decompose()


def _build_parallel_text_block(
    block,
    soup: BeautifulSoup,
    blk_data: dict,
    lang: str,
    config: BuildConfig,
):
    """Build parallel text (two-column table) HTML for a single block."""
    sentences = blk_data.get('sentences', [])
    if not sentences:
        return

    is_heading = block.name in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6')
    table_class = 'parallel-heading' if is_heading else 'parallel-table'

    table = soup.new_tag('table')
    table['class'] = table_class

    for sent in sentences:
        if config.use_simplified and sent.get('chinese_simplified'):
            zh_text = sent['chinese_simplified']
        elif config.word_spacing and sent.get('chinese_spaced'):
            zh_text = sent['chinese_spaced']
        else:
            zh_text = sent['chinese']

        tr_row = soup.new_tag('tr')

        # Chinese cell with pinyin
        td_zh = soup.new_tag('td')
        td_zh['class'] = 'zh-col'
        zh_html = annotate_text(zh_text, word_spacing=config.word_spacing)
        zh_content = BeautifulSoup(zh_html, 'html.parser')
        for child in list(zh_content.children):
            td_zh.append(child.extract() if hasattr(child, 'extract') else NavigableString(str(child)))

        # Translation cell
        td_en = soup.new_tag('td')
        td_en['class'] = 'en-col'
        td_en.string = sent.get('translations', {}).get(lang, '')

        tr_row.append(td_zh)
        tr_row.append(td_en)
        table.append(tr_row)

    if block.name == 'li':
        block.clear()
        block.append(table)
    elif block.name in ('td', 'th'):
        # Avoid nested tables — fall back to stacked
        _build_interlinear_block(block, soup, blk_data, lang, config)
    else:
        block.replace_with(table)
