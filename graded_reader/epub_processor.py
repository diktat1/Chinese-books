"""
EPUB processing: read an input EPUB, annotate Chinese text with pinyin
and English translations, and write a new EPUB.
"""

import logging
import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning
from ebooklib import epub

warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)

from .chinese_processing import (
    annotate_text,
    contains_chinese,
    split_sentences,
    text_to_spaced_chinese,
    text_to_pinyin,
)
from .translator import translate_text, translate_sentences
from .llm_simplifier import simplify_to_hsk4, is_openrouter_available as is_simplifier_available
from .llm_translator import translate_text_llm, translate_sentences_llm

logger = logging.getLogger(__name__)

RUBY_CSS = '''
/* Ruby annotation styling for pinyin */
ruby {
    ruby-align: center;
    -epub-ruby-position: over;
    -webkit-ruby-position: before;
    ruby-position: over;
}

rt {
    font-size: 0.45em;
    font-style: normal;
    font-weight: normal;
    color: #888;
    ruby-align: center;
    letter-spacing: 0.02em;
}

rp {
    display: none;
}

/* Base styling */
body {
    line-height: 2.0;
    font-family: "Songti SC", "Noto Serif CJK SC", "Source Han Serif SC", serif;
}

p {
    line-height: 2.0;
    margin-bottom: 1em;
    text-align: justify;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    line-height: 2.0;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
}

/* Lists */
ul, ol {
    margin-bottom: 1em;
    padding-left: 1.5em;
}

li {
    line-height: 2.0;
    margin-bottom: 0.3em;
}

/* Blockquotes */
blockquote {
    line-height: 2.0;
    margin: 0.8em 0;
    padding-left: 1em;
    border-left: 3px solid #ddd;
}

/* Translation styling - consistent for all element types */
.translation {
    font-size: 0.75em;
    color: #666;
    font-style: italic;
    line-height: 1.4;
    margin-top: 0.2em;
    margin-bottom: 1em;
    padding: 0.4em 0.8em;
    background-color: #f8f8f8;
    border-left: 3px solid #ccc;
    border-radius: 0 4px 4px 0;
    font-family: Georgia, "Times New Roman", serif;
}

/* Inline translation for list items and blockquotes */
.translation-inline {
    display: block;
    font-size: 0.75em;
    color: #666;
    font-style: italic;
    line-height: 1.4;
    margin-top: 0.2em;
    padding: 0.2em 0;
    font-family: Georgia, "Times New Roman", serif;
}

/* Parallel text (two-column) table layout */
.parallel-table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
    margin-bottom: 1.2em;
}

.parallel-table td {
    width: 50%;
    vertical-align: top;
    padding: 0.4em 0.6em;
    border-bottom: 1px solid #eee;
}

.parallel-table td.zh-col {
    font-size: 1.1em;
    line-height: 2.0;
}

.parallel-table td.en-col {
    font-size: 0.9em;
    color: #444;
    font-style: italic;
    line-height: 1.6;
    font-family: Georgia, "Times New Roman", serif;
}

.parallel-heading {
    width: 100%;
    border-collapse: collapse;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
}

.parallel-heading td {
    vertical-align: middle;
    padding: 0.3em 0.6em;
    font-weight: bold;
}

/* Word-level spans for Kindle dictionary lookup */
.cw {
    display: inline;
}

/* Dark mode support for e-readers that support it */
@media (prefers-color-scheme: dark) {
    rt {
        color: #aaa;
    }
    .translation, .translation-inline {
        background-color: #2a2a2a;
        color: #bbb;
        border-left-color: #555;
    }
    .parallel-table td {
        border-bottom-color: #333;
    }
    .parallel-table td.en-col {
        color: #bbb;
    }
}
'''


def process_epub(
    input_path: str,
    output_path: str,
    add_pinyin: bool = True,
    add_translation: bool = True,
    translation_source: str = 'zh-CN',
    translation_target: str = 'en',
    word_spacing: bool = False,
    parallel_text: bool = False,
    simplify_hsk4: bool = False,
    llm_model: str | None = None,
    progress_callback=None,
    target_languages: list[str] | None = None,
) -> None:
    """
    Read an EPUB file, add pinyin annotations and/or English translations,
    and write the result to a new EPUB file.

    Args:
        input_path: Path to the input EPUB file.
        output_path: Path for the output EPUB file.
        add_pinyin: Whether to add pinyin ruby annotations.
        add_translation: Whether to add English translations after paragraphs.
        translation_source: Source language code for translation.
        translation_target: Target language code for translation.
        word_spacing: Whether to add spaces between Chinese words for dictionary lookup.
        parallel_text: If True, use two-column table layout with Chinese sentences
                       on the left and translations on the right, aligned row-by-row.
        simplify_hsk4: If True, simplify Chinese vocabulary to HSK 4 level.
        llm_model: OpenRouter model ID for LLM translation/simplification,
                   or None to use Google Translate (free).
    """
    logger.info(f'Reading EPUB: {input_path}')
    book = epub.read_epub(input_path, options={'ignore_ncx': True})

    # Log all items found in the EPUB for debugging
    all_items = list(book.get_items())
    logger.debug(f'Found {len(all_items)} items in EPUB')
    for item in all_items:
        logger.debug(f'  - {item.get_name()} (type: {item.get_type()})')

    # Add our custom CSS stylesheet
    css_item = epub.EpubItem(
        uid='graded_reader_style',
        file_name='style/graded_reader.css',
        media_type='text/css',
        content=RUBY_CSS.encode('utf-8'),
    )
    book.add_item(css_item)

    # Process each HTML document in the book
    items = list(book.get_items_of_type(9))  # 9 = ITEM_DOCUMENT
    total = len(items)

    for idx, item in enumerate(items, 1):
        # Rotate target language per chapter if target_languages is set
        if target_languages:
            chapter_target = target_languages[(idx - 1) % len(target_languages)]
        else:
            chapter_target = translation_target

        logger.info(f'Processing chapter {idx}/{total}: {item.get_name()} (target={chapter_target})')
        if progress_callback:
            progress_callback(idx, total, f'Processing chapter {idx}/{total}')

        content = item.get_content().decode('utf-8')
        processed = _process_html_content(
            content,
            add_pinyin=add_pinyin,
            add_translation=add_translation,
            translation_source=translation_source,
            translation_target=chapter_target,
            word_spacing=word_spacing,
            parallel_text=parallel_text,
            simplify_hsk4=simplify_hsk4,
            llm_model=llm_model,
        )
        item.set_content(processed.encode('utf-8'))

        # Link our CSS to this chapter
        item.add_item(css_item)

    # Fix TOC entries that may have None uid (ebooklib read/write roundtrip issue)
    _fix_toc_uids(book.toc)

    # Deduplicate items (ebooklib can create duplicates during read/write roundtrip,
    # which causes images and other resources to appear multiple times)
    _deduplicate_items(book)

    # Ensure all items have proper IDs to be included in the manifest
    for item in book.get_items():
        if item.get_id() is None:
            # Generate an ID from the filename
            item_name = item.get_name().replace('/', '_').replace('.', '_')
            item.set_id(f'item_{item_name}')
            logger.debug(f'Fixed missing ID for: {item.get_name()}')

    # Ensure EPUB3 navigation document exists (required for EPUB3 compliance)
    _ensure_nav_document(book)

    logger.info(f'Writing output EPUB: {output_path}')
    epub.write_epub(output_path, book)
    logger.info('Done!')


def _fix_toc_uids(toc, prefix='toc') -> None:
    """Ensure all TOC Link entries have a non-None uid."""
    for i, item in enumerate(toc):
        if isinstance(item, tuple):
            # Section: (Section, [children])
            section, children = item
            if hasattr(section, 'uid') and section.uid is None:
                section.uid = f'{prefix}_{i}'
            _fix_toc_uids(children, prefix=f'{prefix}_{i}')
        elif hasattr(item, 'uid') and item.uid is None:
            item.uid = f'{prefix}_{i}'


def _deduplicate_items(book: epub.EpubBook) -> None:
    """
    Remove duplicate items from the EPUB book.

    ebooklib can create duplicate entries during read/write roundtrip,
    causing images and other resources to appear multiple times in the
    output. This deduplicates by file name, keeping the first occurrence.
    """
    seen_names = set()
    items_to_remove = []

    for item in list(book.get_items()):
        name = item.get_name()
        if name in seen_names:
            items_to_remove.append(item)
            logger.debug(f'Removing duplicate item: {name}')
        else:
            seen_names.add(name)

    for item in items_to_remove:
        # ebooklib stores items internally; remove from the items list
        try:
            book.items.remove(item)
        except (ValueError, AttributeError):
            pass


def _ensure_nav_document(book: epub.EpubBook) -> None:
    """
    Ensure the book has an EPUB3 navigation document.

    EPUB3 requires a nav document with the 'nav' property in the manifest.
    If the book was created from an EPUB2 file (which only has NCX),
    we need to create the nav document.
    """
    # Check if nav document already exists
    for item in book.get_items():
        if isinstance(item, epub.EpubNav):
            logger.debug('Nav document already exists')
            return
        # Also check for nav property in item properties
        if hasattr(item, 'properties') and item.properties and 'nav' in item.properties:
            logger.debug('Item with nav property already exists')
            return

    logger.info('Creating EPUB3 navigation document')

    # Create the nav document
    nav = epub.EpubNav(uid='nav', file_name='nav.xhtml')

    # Add nav to the book
    book.add_item(nav)

    # Add nav to spine (at the beginning, hidden)
    # First, get current spine
    current_spine = list(book.spine) if book.spine else []

    # Insert nav at the beginning with linear='no' to hide it from reading order
    book.spine = [('nav', 'no')] + current_spine


def _deduplicate_html_images(soup: BeautifulSoup) -> None:
    """
    Remove duplicate image references within a single HTML document.

    Some EPUBs have the same image `src` appearing multiple times in a chapter.
    This removes all but the first occurrence of each unique image source.
    """
    seen_srcs = set()
    for img in soup.find_all('img'):
        src = img.get('src', '')
        if not src:
            continue
        if src in seen_srcs:
            # Remove the duplicate image (and its parent if it's an otherwise-empty wrapper)
            parent = img.parent
            img.decompose()
            if parent and parent.name in ('p', 'div', 'span') and not parent.get_text(strip=True) and not parent.find(['img', 'svg']):
                parent.decompose()
            logger.debug(f'Removed duplicate image: {src}')
        else:
            seen_srcs.add(src)

    # Also handle SVG images
    for svg in soup.find_all('image'):
        href = svg.get('xlink:href', svg.get('href', ''))
        if not href:
            continue
        if href in seen_srcs:
            parent = svg.parent
            svg.decompose()
            if parent and parent.name == 'svg' and not parent.find_all():
                wrapper = parent.parent
                parent.decompose()
                if wrapper and wrapper.name in ('p', 'div') and not wrapper.get_text(strip=True) and not wrapper.find(['img', 'svg']):
                    wrapper.decompose()
            logger.debug(f'Removed duplicate SVG image: {href}')
        else:
            seen_srcs.add(href)


def _process_html_content(
    html: str,
    add_pinyin: bool = True,
    add_translation: bool = True,
    translation_source: str = 'zh-CN',
    translation_target: str = 'en',
    word_spacing: bool = False,
    parallel_text: bool = False,
    simplify_hsk4: bool = False,
    llm_model: str | None = None,
) -> str:
    """
    Process a single HTML document: annotate Chinese text nodes with
    pinyin and insert English translations after Chinese paragraphs.

    If parallel_text is True, uses two-column table layout with Chinese
    sentences on the left and translations on the right.

    If simplify_hsk4 is True, simplifies vocabulary to HSK 4 level before processing.
    If llm_model is set, uses that OpenRouter model instead of Google Translate.
    """
    soup = BeautifulSoup(html, 'lxml')

    # Inject our CSS link into <head>
    head = soup.find('head')
    if head:
        link_tag = soup.new_tag(
            'link',
            rel='stylesheet',
            type='text/css',
            href='style/graded_reader.css',
        )
        head.append(link_tag)

    # Deduplicate images within this HTML document
    # (some EPUBs have the same image referenced multiple times)
    _deduplicate_html_images(soup)

    # Find all block-level elements that contain Chinese text.
    # Any block that has nested block children is skipped — those children
    # will be processed on their own pass through the loop. This prevents
    # duplicate translations (e.g. <blockquote><p>text</p></blockquote>
    # would otherwise translate both the blockquote and the p).
    _ALL_BLOCKS = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th',
                   'caption', 'blockquote', 'dt', 'dd', 'figcaption', 'pre',
                   'div', 'section', 'article', 'aside', 'header', 'footer']

    block_tags = soup.find_all(_ALL_BLOCKS)

    for block in block_tags:
        # Skip any block that has nested block children — those children
        # will be processed on their own pass through the loop.
        if block.find(_ALL_BLOCKS):
            continue

        plain_text = block.get_text()
        if not contains_chinese(plain_text):
            continue

        # Apply HSK 4 simplification if enabled
        if simplify_hsk4:
            simplified_text = simplify_to_hsk4(plain_text, model=llm_model)
            if simplified_text and not simplified_text.startswith('['):
                # Update the block content with simplified text
                block.clear()
                block.string = simplified_text
                plain_text = simplified_text

        if parallel_text and add_translation:
            # Two-column parallel text layout
            _process_block_parallel_text(
                block, soup, plain_text,
                add_pinyin=add_pinyin,
                translation_source=translation_source,
                translation_target=translation_target,
                llm_model=llm_model,
                word_spacing=word_spacing,
            )
        else:
            # Standard ruby annotation format
            if add_pinyin:
                _annotate_block(block, soup, word_spacing=word_spacing)

            if add_translation:
                _add_translation_to_block(
                    block, soup, plain_text,
                    translation_source=translation_source,
                    translation_target=translation_target,
                    llm_model=llm_model,
                )

    # Return as string, preserving the XML declaration if present
    result = str(soup)

    # Ensure we have proper XHTML if the original was XHTML
    if 'xmlns' in html and 'xmlns' not in result:
        result = result.replace('<html', '<html xmlns="http://www.w3.org/1999/xhtml"', 1)

    return result


def _annotate_block(block: Tag, soup: BeautifulSoup, word_spacing: bool = False) -> None:
    """
    Walk through the text nodes inside a block element and replace
    Chinese text with ruby-annotated HTML.
    """
    # Collect text nodes (we need to iterate over a copy since we modify in place)
    text_nodes = []
    for descendant in block.descendants:
        if isinstance(descendant, NavigableString) and not isinstance(
            descendant, (type(soup.new_string('')).__class__,)
        ):
            if descendant.parent.name not in ('rt', 'rp', 'ruby', 'script', 'style'):
                text_nodes.append(descendant)

    for text_node in text_nodes:
        original = str(text_node)
        if not contains_chinese(original):
            continue

        annotated_html = annotate_text(original, word_spacing=word_spacing)
        if annotated_html == original:
            continue

        # Parse the annotated HTML and replace the text node
        new_content = BeautifulSoup(annotated_html, 'html.parser')
        text_node.replace_with(new_content)


def _get_translation(
    plain_text: str,
    translation_source: str,
    translation_target: str,
    llm_model: str | None,
) -> str | None:
    """Get translation for text, returning None on failure."""
    if llm_model:
        translation = translate_text_llm(
            plain_text, source=translation_source, target=translation_target,
            model=llm_model
        )
    else:
        translation = translate_text(
            plain_text, source=translation_source, target=translation_target
        )
    if translation and not translation.startswith('[Translation failed') and not translation.startswith('[Translation'):
        return translation
    return None


def _add_translation_to_block(
    block: Tag,
    soup: BeautifulSoup,
    plain_text: str,
    translation_source: str = 'zh-CN',
    translation_target: str = 'en',
    llm_model: str | None = None,
) -> None:
    """
    Add translation to any block element type.
    Translation is placed BEFORE the Chinese text so the reader sees the
    target language first, then the Chinese with pinyin.

    For <p>, <h*>, <blockquote>, <div>, <section>, etc.: inserts a translation <div> before the block.
    For <li>, <dt>, <dd>: prepends inline translation span (siblings would break list structure).
    For <td>, <th>: prepends inline translation inside the cell.
    """
    translation = _get_translation(
        plain_text, translation_source, translation_target,
        llm_model,
    )
    if not translation:
        return

    if block.name in ('li', 'td', 'th', 'dt', 'dd'):
        # For list items, table cells, and definition list entries,
        # prepend translation inline (inserting siblings would break structure)
        trans_span = soup.new_tag('span')
        trans_span['class'] = 'translation-inline'
        trans_span.string = translation
        block.insert(0, trans_span)
    else:
        # For p, h*, blockquote, caption, div, section, etc.: insert before block
        trans_div = soup.new_tag('div')
        trans_div['class'] = 'translation'
        trans_div.string = translation
        block.insert_before(trans_div)


def _get_parallel_translations(
    plain_text: str,
    translation_source: str,
    translation_target: str,
    llm_model: str | None,
) -> list[tuple[str, str]]:
    """
    Split Chinese text into sentences and get aligned translations.

    Returns a list of (chinese_sentence, translation) tuples.
    Falls back to a single paragraph pair on any failure.
    """
    sentences = split_sentences(plain_text)

    if not sentences:
        # No sentence boundaries found — treat whole text as one unit
        sentences = [plain_text.strip()]

    if len(sentences) == 1:
        translation = _get_translation(
            sentences[0], translation_source, translation_target,
            llm_model,
        )
        return [(sentences[0], translation or '')]

    # Batch translate
    try:
        if llm_model:
            translations = translate_sentences_llm(
                sentences,
                source=translation_source,
                target=translation_target,
                model=llm_model,
            )
        else:
            translations = translate_sentences(
                sentences,
                source=translation_source,
                target=translation_target,
            )
    except Exception as e:
        logger.warning(f"Batch sentence translation failed: {e}. Falling back to paragraph.")
        translation = _get_translation(
            plain_text, translation_source, translation_target,
            llm_model,
        )
        return [(plain_text.strip(), translation or '')]

    # Pair them up
    pairs = []
    for i, zh in enumerate(sentences):
        en = translations[i] if i < len(translations) else ''
        # Skip error markers
        if en.startswith('[Translation'):
            en = ''
        pairs.append((zh, en))
    return pairs


def _process_block_parallel_text(
    block: Tag,
    soup: BeautifulSoup,
    plain_text: str,
    add_pinyin: bool = True,
    translation_source: str = 'zh-CN',
    translation_target: str = 'en',
    llm_model: str | None = None,
    word_spacing: bool = False,
) -> None:
    """
    Process a block element using two-column parallel text layout.

    Builds an HTML table with Chinese sentences (with optional ruby pinyin)
    on the left and translations on the right, one row per sentence.

    For <td>/<th> blocks, falls back to stacked layout to avoid nested tables.
    For <li> blocks, nests a table inside the list item.
    """
    sentence_pairs = _get_parallel_translations(
        plain_text, translation_source, translation_target,
        llm_model,
    )

    # For table cells, fall back to standard stacked layout
    if block.name in ('td', 'th'):
        if add_pinyin:
            _annotate_block(block, soup, word_spacing=word_spacing)
        # Add inline translation
        full_translation = ' '.join(en for _, en in sentence_pairs if en)
        if full_translation:
            trans_span = soup.new_tag('span')
            trans_span['class'] = 'translation-inline'
            trans_span.string = full_translation
            block.insert(0, trans_span)
        return

    is_heading = block.name in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6')
    table_class = 'parallel-heading' if is_heading else 'parallel-table'

    table = soup.new_tag('table')
    table['class'] = table_class

    for zh_sentence, en_sentence in sentence_pairs:
        tr = soup.new_tag('tr')

        # Chinese cell
        td_zh = soup.new_tag('td')
        td_zh['class'] = 'zh-col'

        if add_pinyin:
            annotated = annotate_text(zh_sentence, word_spacing=word_spacing)
            annotated_soup = BeautifulSoup(annotated, 'html.parser')
            for child in list(annotated_soup.children):
                td_zh.append(child.extract() if hasattr(child, 'extract') else NavigableString(str(child)))
        else:
            if word_spacing:
                td_zh.string = text_to_spaced_chinese(zh_sentence)
            else:
                td_zh.string = zh_sentence

        # Translation cell
        td_en = soup.new_tag('td')
        td_en['class'] = 'en-col'
        td_en.string = en_sentence

        tr.append(td_zh)
        tr.append(td_en)
        table.append(tr)

    if block.name == 'li':
        # Nest table inside the list item to preserve list structure
        block.clear()
        block.append(table)
    else:
        block.replace_with(table)
