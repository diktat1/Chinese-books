"""
Chinese text processing: word segmentation and pinyin annotation.

Uses jieba for segmentation and pypinyin for pinyin conversion.
Outputs HTML with <ruby> tags for inline pinyin display.
"""

import re
import jieba
from pypinyin import pinyin, Style


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


def annotate_text(text: str, word_spacing: bool = False) -> str:
    """
    Take a plain Chinese text string and return HTML with ruby pinyin
    annotations above each Chinese character.

    Non-Chinese text (punctuation, numbers, Latin) is left unchanged.

    Args:
        text: The text to annotate.
        word_spacing: If True, adds spaces between Chinese words to help
                      e-readers recognize word boundaries for dictionary lookup.
    """
    if not contains_chinese(text):
        return text

    words = segment_text(text)

    if not word_spacing:
        return ''.join(word_to_ruby_html(w) for w in words)

    # With word spacing: add space between consecutive Chinese words
    result = []
    prev_was_chinese = False
    for word in words:
        word_is_chinese = contains_chinese(word)
        # Add space before this word if both previous and current are Chinese
        if prev_was_chinese and word_is_chinese:
            result.append(' ')
        result.append(word_to_ruby_html(word))
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


def text_to_spaced_chinese(text: str) -> str:
    """
    Convert Chinese text to word-spaced format for better readability.

    Uses jieba segmentation to identify word boundaries and inserts
    spaces between Chinese words. Non-Chinese text is preserved as-is.

    Example:
        "我今天去北京" -> "我 今天 去 北京"
    """
    if not contains_chinese(text):
        return text

    words = segment_text(text)
    result = []
    prev_was_chinese = False

    for word in words:
        word_is_chinese = contains_chinese(word)
        # Add space before this word if both previous and current are Chinese
        if prev_was_chinese and word_is_chinese:
            result.append(' ')
        result.append(word)
        prev_was_chinese = word_is_chinese

    return ''.join(result)


def text_to_pinyin(text: str) -> str:
    """
    Convert Chinese text to pinyin with word spacing.

    Uses jieba segmentation to preserve word boundaries, converts each
    Chinese word to pinyin, and joins with spaces. Non-Chinese text
    (punctuation, numbers) is preserved in place.

    Example:
        "我今天去北京。" -> "wǒ jīntiān qù běijīng。"
    """
    if not contains_chinese(text):
        return text

    words = segment_text(text)
    result = []
    prev_was_chinese = False

    for word in words:
        word_is_chinese = contains_chinese(word)

        if word_is_chinese:
            # Add space before Chinese words (except first)
            if prev_was_chinese:
                result.append(' ')

            # Convert word to pinyin
            word_pinyin = []
            for char in word:
                if is_chinese_char(char):
                    py = pinyin(char, style=Style.TONE, heteronym=False)
                    py_str = py[0][0] if py and py[0] else char
                    word_pinyin.append(py_str)
                else:
                    word_pinyin.append(char)
            result.append(''.join(word_pinyin))
        else:
            # Non-Chinese (punctuation, numbers, spaces)
            result.append(word)

        prev_was_chinese = word_is_chinese

    return ''.join(result)
