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


def annotate_text(text: str) -> str:
    """
    Take a plain Chinese text string and return HTML with ruby pinyin
    annotations above each Chinese character.

    Non-Chinese text (punctuation, numbers, Latin) is left unchanged.
    """
    if not contains_chinese(text):
        return text

    words = segment_text(text)
    return ''.join(word_to_ruby_html(w) for w in words)


def annotate_paragraph(paragraph_text: str) -> str:
    """
    Annotate an entire paragraph. Splits on newlines to preserve
    line structure, annotates each line, then rejoins.
    """
    lines = paragraph_text.split('\n')
    annotated_lines = [annotate_text(line) for line in lines]
    return '\n'.join(annotated_lines)
