"""
Anki deck generator from Chinese EPUB books.

Extracts Chinese sentences from EPUB, generates pinyin + translation + TTS audio,
and packages everything into an Anki .apkg deck.

Card format:
  Front: Chinese sentence + audio
  Back:  Pinyin + translation
"""

import hashlib
import logging
import re
import tempfile
from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import epub
from pypinyin import pinyin, Style

from .chinese_processing import contains_chinese, is_chinese_char, segment_text

logger = logging.getLogger(__name__)

# Chinese sentence-ending punctuation
_SENTENCE_ENDERS = re.compile(r'([。！？；…]+)')


def _split_sentences(text: str) -> list[str]:
    """
    Split Chinese text into sentences at sentence-ending punctuation.
    Keeps the punctuation attached to the sentence.
    """
    parts = _SENTENCE_ENDERS.split(text)
    sentences = []
    i = 0
    while i < len(parts):
        s = parts[i].strip()
        # Attach the punctuation delimiter to the preceding text
        if i + 1 < len(parts):
            s += parts[i + 1]
            i += 2
        else:
            i += 1
        s = s.strip()
        if s and contains_chinese(s):
            sentences.append(s)
    return sentences


def _text_to_pinyin(text: str) -> str:
    """Convert Chinese text to pinyin string with spaces between words."""
    words = segment_text(text)
    result = []
    for word in words:
        if contains_chinese(word):
            word_py = []
            for char in word:
                if is_chinese_char(char):
                    py = pinyin(char, style=Style.TONE, heteronym=False)
                    word_py.append(py[0][0] if py and py[0] else char)
                else:
                    word_py.append(char)
            result.append(''.join(word_py))
        else:
            result.append(word)
    return ' '.join(w for w in result if w.strip())


def _get_spine_items(book):
    """Return document items in spine (reading) order, not manifest order."""
    items_by_id = {}
    for item in book.get_items():
        items_by_id[item.get_id()] = item
        items_by_id[item.get_name()] = item

    spine_items = []
    for entry in book.spine:
        item_id = entry[0] if isinstance(entry, tuple) else entry
        item = items_by_id.get(item_id)
        if item and item.get_type() == 9:
            spine_items.append(item)

    return spine_items or list(book.get_items_of_type(9))


def _extract_sentences_from_epub(epub_path: str) -> list[str]:
    """Extract all Chinese sentences from an EPUB file, in reading order."""
    book = epub.read_epub(epub_path)
    sentences = []

    for item in _get_spine_items(book):
        html = item.get_content().decode('utf-8', errors='replace')
        soup = BeautifulSoup(html, 'lxml')

        # Extract text from block elements
        blocks = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                                'li', 'td', 'blockquote', 'div'])
        for block in blocks:
            text = block.get_text().strip()
            if not text or not contains_chinese(text):
                continue
            for sentence in _split_sentences(text):
                if len(sentence) >= 2:  # Skip single characters
                    sentences.append(sentence)

    return sentences


def _generate_audio(sentence: str, output_path: str) -> bool:
    """Generate TTS audio for a Chinese sentence. Returns True on success."""
    try:
        from gtts import gTTS
        tts = gTTS(text=sentence, lang='zh-CN', slow=False)
        tts.save(output_path)
        return True
    except Exception as e:
        logger.warning(f'TTS generation failed for "{sentence[:20]}...": {e}')
        return False


def _stable_id(name: str) -> int:
    """Generate a stable numeric ID from a string name."""
    return int(hashlib.md5(name.encode()).hexdigest()[:10], 16)


def generate_anki_deck(
    epub_path: str,
    output_path: str,
    translation_target: str = 'fr',
    translation_source: str = 'zh-CN',
    use_claude: bool = False,
    use_opus: bool = False,
    include_audio: bool = True,
    max_sentences: int = 0,
) -> str:
    """
    Generate an Anki deck (.apkg) from a Chinese EPUB.

    Args:
        epub_path: Path to the input EPUB file.
        output_path: Path for the output .apkg file.
        translation_target: Target language code (default: 'fr' for French).
        translation_source: Source language code (default: 'zh-CN').
        use_claude: Use Claude for translation instead of Google Translate.
        use_opus: Use Claude Opus model.
        include_audio: Generate TTS audio for each sentence.
        max_sentences: Maximum number of sentences (0 = all).

    Returns:
        Path to the generated .apkg file.
    """
    import genanki

    book_name = Path(epub_path).stem

    # Extract sentences
    logger.info(f'Extracting sentences from {epub_path}...')
    sentences = _extract_sentences_from_epub(epub_path)
    if max_sentences > 0:
        sentences = sentences[:max_sentences]
    logger.info(f'Found {len(sentences)} sentences')

    if not sentences:
        raise ValueError('No Chinese sentences found in the EPUB')

    # Set up translation function
    if use_claude:
        from .claude_translator import translate_text_claude
        def translate(text):
            return translate_text_claude(
                text, source=translation_source, target=translation_target,
                use_opus=use_opus,
            )
    else:
        from .translator import translate_text
        def translate(text):
            return translate_text(
                text, source=translation_source, target=translation_target,
            )

    # Define Anki model (note type)
    model = genanki.Model(
        _stable_id(f'chinese-graded-reader-{book_name}'),
        'Chinese Graded Reader',
        fields=[
            {'name': 'Chinese'},
            {'name': 'Pinyin'},
            {'name': 'Translation'},
            {'name': 'Audio'},
        ],
        templates=[{
            'name': 'Chinese → Translation',
            'qfmt': '''
                <div class="chinese">{{Chinese}}</div>
                {{Audio}}
            ''',
            'afmt': '''
                {{FrontSide}}
                <hr id="answer">
                <div class="pinyin">{{Pinyin}}</div>
                <div class="translation">{{Translation}}</div>
            ''',
        }],
        css='''
            .card {
                font-family: "Songti SC", "Noto Serif CJK SC", serif;
                text-align: center;
                padding: 20px;
            }
            .chinese {
                font-size: 28px;
                line-height: 1.6;
                margin-bottom: 15px;
            }
            .pinyin {
                font-size: 18px;
                color: #666;
                margin-bottom: 10px;
                font-family: Georgia, serif;
            }
            .translation {
                font-size: 18px;
                color: #333;
                font-style: italic;
            }
        ''',
    )

    deck = genanki.Deck(_stable_id(f'deck-{book_name}'), f'Chinese: {book_name}')
    media_files = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, sentence in enumerate(sentences):
            logger.info(f'Processing sentence {i + 1}/{len(sentences)}: {sentence[:30]}...')

            # Generate pinyin
            py = _text_to_pinyin(sentence)

            # Generate translation
            translation = translate(sentence)
            if not translation or translation.startswith('[Translation'):
                translation = '(translation unavailable)'

            # Generate audio
            audio_field = ''
            if include_audio:
                audio_filename = f'chinese_{book_name}_{i:04d}.mp3'
                audio_path = str(Path(tmpdir) / audio_filename)
                if _generate_audio(sentence, audio_path):
                    media_files.append(audio_path)
                    audio_field = f'[sound:{audio_filename}]'

            note = genanki.Note(
                model=model,
                fields=[sentence, py, translation, audio_field],
            )
            deck.add_note(note)

        # Write the .apkg file
        package = genanki.Package(deck)
        package.media_files = media_files
        package.write_to_file(output_path)

    logger.info(f'Anki deck written to {output_path} ({len(sentences)} cards)')
    return output_path
