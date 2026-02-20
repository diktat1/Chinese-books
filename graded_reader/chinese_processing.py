"""
Chinese text processing: word segmentation and pinyin annotation.

Uses jieba for segmentation and pypinyin for pinyin conversion.
Outputs HTML with <ruby> tags for inline pinyin display.

Word boundaries use U+200B (zero-width space) between Chinese words.
This allows Kindle and other e-readers to identify word boundaries
for dictionary lookup without visible spacing artifacts.
"""

import re
import jieba
from pypinyin import pinyin, Style

# Chinese sentence-ending punctuation
_SENTENCE_ENDERS = re.compile(r'([。！？；…]+)')

# Zero-width space: invisible but creates a word boundary for e-readers
WORD_BOUNDARY = '\u200b'


def split_sentences(text: str) -> list[str]:
    """
    Split Chinese text into sentences at sentence-ending punctuation.
    Keeps the punctuation attached to the sentence.
    Returns only non-empty segments that contain Chinese characters.
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


def is_chinese_char(char: str) -> bool:
    """Check if a character is a CJK unified ideograph."""
    cp = ord(char)
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0x2B820 <= cp <= 0x2CEAF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
    )


def contains_chinese(text: str) -> bool:
    """Check if a string contains any Chinese characters."""
    return any(is_chinese_char(c) for c in text)


def segment_text(text: str) -> list[str]:
    """Segment Chinese text into words using jieba."""
    return list(jieba.cut(text))


def word_to_ruby_html(word: str) -> str:
    """
    Convert a Chinese word to HTML with <ruby> pinyin annotations.

    Each character gets its own ruby annotation (mono ruby style) for
    maximum compatibility. Non-Chinese text is returned unchanged.
    """
    if not contains_chinese(word):
        return word

    result = []
    for char in word:
        if is_chinese_char(char):
            py = pinyin(char, style=Style.TONE, heteronym=False)
            py_str = py[0][0] if py and py[0] else char
            result.append(
                f'<ruby>{char}<rp>(</rp><rt>{py_str}</rt><rp>)</rp></ruby>'
            )
        else:
            result.append(char)
    return ''.join(result)


def word_to_dual_ruby_html(word: str, word_meaning: str = '') -> str:
    """
    Convert a Chinese word to HTML with pinyin ruby on every character
    and the word-level meaning shown above the word using nested ruby.

    Uses nested ruby for cross-platform stacking (works on Kindle + Apple Books):
        <ruby><ruby>char<rt>pinyin</rt></ruby><rt>meaning</rt></ruby>

    Layout (top to bottom): meaning / pinyin / character
    Meaning spans the entire word; pinyin is per-character.

    Args:
        word: The Chinese word to annotate.
        word_meaning: Contextual meaning for the whole word (e.g., 'level' for 水平).
    """
    if not contains_chinese(word):
        return word

    # Build inner ruby: each character with its pinyin
    inner_parts = []
    for char in word:
        if is_chinese_char(char):
            py = pinyin(char, style=Style.TONE, heteronym=False)
            py_str = py[0][0] if py and py[0] else ''
            inner_parts.append(f'<ruby>{char}<rp>(</rp><rt>{py_str}</rt><rp>)</rp></ruby>')
        else:
            inner_parts.append(char)

    inner_html = ''.join(inner_parts)

    if word_meaning:
        # Wrap with outer ruby for word-level meaning
        return f'<ruby>{inner_html}<rp>(</rp><rt>{word_meaning}</rt><rp>)</rp></ruby>'
    else:
        return inner_html


def annotate_text(text: str, word_spacing: bool = False) -> str:
    """
    Take a plain Chinese text string and return HTML with ruby pinyin
    annotations above each Chinese character.

    Non-Chinese text (punctuation, numbers, Latin) is left unchanged.

    Word spacing uses zero-width spaces (U+200B) between words to create
    word boundaries that e-readers can use for dictionary lookup, without
    visible spacing artifacts.

    Args:
        text: The text to annotate.
        word_spacing: If True, adds zero-width spaces between Chinese words
                      to help e-readers recognize word boundaries for lookup.
    """
    if not contains_chinese(text):
        return text

    words = segment_text(text)

    if not word_spacing:
        return ''.join(word_to_ruby_html(w) for w in words)

    # With word spacing: wrap each Chinese word in a <span> and separate
    # with zero-width spaces so Kindle can identify word boundaries
    result = []
    prev_was_chinese = False
    for word in words:
        word_is_chinese = contains_chinese(word)
        if prev_was_chinese and word_is_chinese:
            result.append(WORD_BOUNDARY)
        if word_is_chinese:
            # Wrap entire word's ruby in a span for Kindle word selection
            result.append(f'<span class="cw">{word_to_ruby_html(word)}</span>')
        else:
            result.append(word_to_ruby_html(word))
        prev_was_chinese = word_is_chinese

    return ''.join(result)


def annotate_text_dual_ruby(
    text: str,
    meanings: dict[str, str] | None = None,
    word_spacing: bool = False,
) -> str:
    """
    Annotate Chinese text with dual ruby (pinyin + word-level meaning).

    meanings maps jieba-segmented words → contextual translation.
    Word meaning is shown above the first character of each word;
    remaining characters get pinyin only.
    Always adds spaces between words for readability.
    """
    if not contains_chinese(text):
        return text

    words = segment_text(text)
    meanings = meanings or {}

    result = []
    prev_was_chinese = False
    for word in words:
        word_is_chinese = contains_chinese(word)
        if prev_was_chinese and word_is_chinese:
            # Visible thin space between words for readability
            result.append(' ')
        if word_is_chinese:
            word_meaning = meanings.get(word, '')
            result.append(word_to_dual_ruby_html(word, word_meaning))
        else:
            result.append(word)
        prev_was_chinese = word_is_chinese

    return ''.join(result)


def annotate_paragraph(paragraph_text: str, word_spacing: bool = False) -> str:
    """
    Annotate an entire paragraph. Splits on newlines to preserve
    line structure, annotates each line, then rejoins.

    Args:
        paragraph_text: The paragraph text to annotate.
        word_spacing: If True, adds spaces between Chinese words.
    """
    lines = paragraph_text.split('\n')
    annotated_lines = [annotate_text(line, word_spacing=word_spacing) for line in lines]
    return '\n'.join(annotated_lines)


def _is_punctuation(text: str) -> bool:
    """Check if text is only punctuation (Chinese or ASCII)."""
    import string
    cn_punct = '，。！？；：""''【】《》（）、·…—'
    all_punct = string.punctuation + cn_punct + ' \t\n'
    return all(c in all_punct for c in text)


def text_to_spaced_chinese(text: str) -> str:
    """
    Convert Chinese text to word-spaced format for better readability.

    Uses jieba segmentation to identify word boundaries and inserts
    zero-width spaces (U+200B) between words. This creates invisible
    word boundaries that e-readers use for dictionary lookup and word
    selection, without visible spacing artifacts.

    Example:
        "我今天去北京" -> "我\u200b今天\u200b去\u200b北京"
    """
    if not contains_chinese(text):
        return text

    words = segment_text(text)
    result = []
    prev_was_word = False  # Previous segment was a word (not punctuation)

    for word in words:
        is_punct = _is_punctuation(word)

        if not is_punct:
            # This is a word segment - add zero-width space if previous was also a word
            if prev_was_word and result:
                result.append(WORD_BOUNDARY)
            result.append(word)
            prev_was_word = True
        else:
            # Punctuation - just append, no space logic
            result.append(word)
            prev_was_word = False

    return ''.join(result)


def text_to_pinyin(text: str) -> str:
    """
    Convert Chinese text to pinyin with word spacing.

    Uses jieba segmentation to preserve word boundaries, converts each
    Chinese word to pinyin, and joins with spaces. Non-Chinese text
    (punctuation, numbers, letters) is preserved in place.

    Example:
        "我今天去北京。" -> "wǒ jīntiān qù běijīng。"
        "我在D物流公司" -> "wǒ zài D wùliú gōngsī"
    """
    if not contains_chinese(text):
        return text

    words = segment_text(text)
    result = []
    prev_was_word = False  # Previous segment was a word (not punctuation)

    for word in words:
        is_punct = _is_punctuation(word)
        word_has_chinese = contains_chinese(word)

        if is_punct:
            # Punctuation - just append
            result.append(word)
            prev_was_word = False
        elif word_has_chinese:
            # Chinese word - convert to pinyin
            if prev_was_word and result:
                result.append(' ')

            word_pinyin = []
            for char in word:
                if is_chinese_char(char):
                    py = pinyin(char, style=Style.TONE, heteronym=False)
                    py_str = py[0][0] if py and py[0] else char
                    word_pinyin.append(py_str)
                else:
                    word_pinyin.append(char)
            result.append(''.join(word_pinyin))
            prev_was_word = True
        else:
            # Non-Chinese word (letters, numbers) - keep as-is
            if prev_was_word and result:
                result.append(' ')
            result.append(word)
            prev_was_word = True

    return ''.join(result)
