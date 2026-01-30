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

from .chinese_processing import annotate_text, contains_chinese
from .translator import translate_text

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

/* Extra line height to accommodate pinyin above characters */
body {
    line-height: 2.2;
    font-family: "Songti SC", "Noto Serif CJK SC", "Source Han Serif SC", serif;
}

p {
    line-height: 2.2;
    margin-bottom: 1em;
    text-align: justify;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    line-height: 2.2;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
}

/* English translation styling */
.translation {
    font-size: 0.8em;
    color: #666;
    font-style: italic;
    margin-top: 0.3em;
    margin-bottom: 1.2em;
    padding: 0.5em 0.8em;
    background-color: #f8f8f8;
    border-left: 3px solid #ccc;
    border-radius: 0 4px 4px 0;
    font-family: Georgia, "Times New Roman", serif;
}

/* Dark mode support for e-readers that support it */
@media (prefers-color-scheme: dark) {
    rt {
        color: #aaa;
    }
    .translation {
        background-color: #2a2a2a;
        color: #bbb;
        border-left-color: #555;
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
        logger.info(f'Processing chapter {idx}/{total}: {item.get_name()}')

        content = item.get_content().decode('utf-8')
        processed = _process_html_content(
            content,
            add_pinyin=add_pinyin,
            add_translation=add_translation,
            translation_source=translation_source,
            translation_target=translation_target,
            word_spacing=word_spacing,
        )
        item.set_content(processed.encode('utf-8'))

        # Link our CSS to this chapter
        item.add_item(css_item)

    # Fix TOC entries that may have None uid (ebooklib read/write roundtrip issue)
    _fix_toc_uids(book.toc)

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


def _process_html_content(
    html: str,
    add_pinyin: bool = True,
    add_translation: bool = True,
    translation_source: str = 'zh-CN',
    translation_target: str = 'en',
    word_spacing: bool = False,
) -> str:
    """
    Process a single HTML document: annotate Chinese text nodes with
    pinyin and insert English translations after Chinese paragraphs.
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

    # Find all paragraph-level elements that contain Chinese text
    block_tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th', 'caption', 'blockquote'])

    for block in block_tags:
        plain_text = block.get_text()
        if not contains_chinese(plain_text):
            continue

        if add_pinyin:
            _annotate_block(block, soup, word_spacing=word_spacing)

        if add_translation and block.name == 'p':
            # Translate the original plain text (before pinyin was added)
            translation = translate_text(
                plain_text, source=translation_source, target=translation_target
            )
            if translation and not translation.startswith('[Translation failed'):
                trans_p = soup.new_tag('p')
                trans_p['class'] = 'translation'
                trans_p.string = translation
                block.insert_after(trans_p)

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
