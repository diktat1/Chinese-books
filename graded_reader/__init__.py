"""Graded reader conversion utilities for Chinese EPUB ebooks."""

from .calibre import (
    is_calibre_installed,
    get_calibre_version,
    convert_epub_to_azw3,
    CalibreNotFoundError,
)
from .claude_simplifier import (
    simplify_to_hsk4,
    analyze_vocabulary_level,
    is_openrouter_available,
)
from .claude_translator import (
    translate_text_claude,
)
from .audio_generator import (
    generate_audiobook,
)

__all__ = [
    'is_calibre_installed',
    'get_calibre_version',
    'convert_epub_to_azw3',
    'CalibreNotFoundError',
    'simplify_to_hsk4',
    'analyze_vocabulary_level',
    'is_openrouter_available',
    'translate_text_claude',
    'generate_audiobook',
]
