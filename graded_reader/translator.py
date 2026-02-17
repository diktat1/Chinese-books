"""
Translation module: Chinese to English paragraph translation.

Uses deep-translator with Google Translate backend (free, no API key).
Includes rate-limiting and retry logic to avoid being blocked.
"""

import time
import logging

from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)

# Google Translate has a ~5000 character limit per request
MAX_CHUNK_SIZE = 4500


def translate_text(text: str, source: str = 'zh-CN', target: str = 'en') -> str:
    """
    Translate a text string from source to target language.

    Handles long texts by splitting into chunks under the API limit.
    Retries on failure with exponential backoff.
    """
    text = text.strip()
    if not text:
        return ''

    if len(text) <= MAX_CHUNK_SIZE:
        return _translate_with_retry(text, source, target)

    # Split long text into chunks at sentence boundaries
    chunks = _split_text(text, MAX_CHUNK_SIZE)
    translated_chunks = []
    for i, chunk in enumerate(chunks):
        translated = _translate_with_retry(chunk, source, target)
        translated_chunks.append(translated)
        # Rate limit between chunks
        if i < len(chunks) - 1:
            time.sleep(0.5)

    return ' '.join(translated_chunks)


def translate_sentences(
    sentences: list[str],
    source: str = 'zh-CN',
    target: str = 'en',
) -> list[str]:
    """
    Translate a list of sentences individually via Google Translate.

    Returns a list of translations, one per input sentence.
    """
    translations = []
    for i, sentence in enumerate(sentences):
        translation = _translate_with_retry(sentence, source, target)
        translations.append(translation)
        if i < len(sentences) - 1:
            time.sleep(0.3)
    return translations


def _translate_with_retry(
    text: str, source: str, target: str, max_retries: int = 3
) -> str:
    """Translate with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            result = GoogleTranslator(source=source, target=target).translate(text)
            return result or ''
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    f'Translation attempt {attempt + 1} failed: {e}. '
                    f'Retrying in {wait}s...'
                )
                time.sleep(wait)
            else:
                logger.error(f'Translation failed after {max_retries} attempts: {e}')
                return f'[Translation failed: {e}]'


def _split_text(text: str, max_size: int) -> list[str]:
    """
    Split text into chunks, preferring sentence boundaries.
    Falls back to splitting at any boundary if sentences are too long.
    """
    # Chinese sentence-ending punctuation + standard punctuation
    sentence_enders = ('。', '！', '？', '；', '.', '!', '?', ';', '\n')

    chunks = []
    remaining = text

    while len(remaining) > max_size:
        # Find the last sentence boundary within the limit
        split_pos = -1
        for ender in sentence_enders:
            pos = remaining.rfind(ender, 0, max_size)
            if pos > split_pos:
                split_pos = pos

        if split_pos == -1:
            # No sentence boundary found, split at max_size
            split_pos = max_size - 1

        chunks.append(remaining[: split_pos + 1].strip())
        remaining = remaining[split_pos + 1 :].strip()

    if remaining:
        chunks.append(remaining)

    return chunks
