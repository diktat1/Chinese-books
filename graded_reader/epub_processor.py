"""
EPUB processing: read an input EPUB, annotate Chinese text with pinyin
and English translations, and write a new EPUB.
"""

import copy
import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag
from ebooklib import epub

from .chinese_processing import annotate_text, contains_chinese
from .translator import translate_text

logger = logging.getLogger(__name__)

RUBY_CSS = '''
/* Ruby annotation styling for pinyin */
ruby {
    ruby-align: center;
}

rt {
    font-size: 0.5em;
    font-style: normal;
    font-weight: normal;
    color: #555;
    ruby-align: center;
}

rp {
    display: none;
}

ruby {
    -epub-ruby-position: over;
    -webkit-ruby-position: before;
    ruby-position: over;
}

/* Extra line height to accommodate pinyin above characters */
body {
    line-height: 2.0;
}

p {
    line-height: 2.0;
    margin-bottom: 0.8em;
}

/* English translation styling */
.translation {
    font-size: 0.85em;
    color: #666;
    font-style: italic;
    margin-top: 0.2em;
    margin-bottom: 1em;
    padding-left: 0.5em;
    border-left: 2px solid #ddd;
}
'''


def process_epub(
    input_path: str,
    output_path: str,
    add_pinyin: bool = True,
    add_translation: bool = True,
    translation_source: str = 'zh-CN',
    translation_target: str = 'en',
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
    """
    logger.info(f'Reading EPUB: {input_path}')
    book = epub.read_epub(input_path)

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
        )
        item.set_content(processed.encode('utf-8'))

        # Link our CSS to this chapter
        item.add_item(css_item)

    logger.info(f'Writing output EPUB: {output_path}')
    epub.write_epub(output_path, book, {})
    logger.info('Done!')


def _process_html_content(
    html: str,
    add_pinyin: bool = True,
    add_translation: bool = True,
    translation_source: str = 'zh-CN',
    translation_target: str = 'en',
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
            _annotate_block(block, soup)

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


def _annotate_block(block: Tag, soup: BeautifulSoup) -> None:
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

        annotated_html = annotate_text(original)
        if annotated_html == original:
            continue

        # Parse the annotated HTML and replace the text node
        new_content = BeautifulSoup(annotated_html, 'html.parser')
        text_node.replace_with(new_content)
