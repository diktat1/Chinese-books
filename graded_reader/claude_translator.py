"""
Translation module using Claude API.

Provides high-quality Chinese to English (or other language) translation
using Claude as an alternative to Google Translate.
"""

import os
import time
import logging
from typing import Optional

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default model for translation
DEFAULT_MODEL = "claude-sonnet-4-20250514"
OPUS_MODEL = "claude-opus-4-0-20250514"

# Maximum text length per request (characters)
MAX_CHUNK_SIZE = 4000


def is_anthropic_available() -> bool:
    """Check if the Anthropic SDK is installed."""
    return ANTHROPIC_AVAILABLE


def get_api_key() -> Optional[str]:
    """Get Anthropic API key from environment variable."""
    return os.environ.get('ANTHROPIC_API_KEY')


def translate_text_claude(
    text: str,
    source: str = 'Chinese',
    target: str = 'English',
    use_opus: bool = False,
    max_retries: int = 3,
) -> str:
    """
    Translate text from source to target language using Claude.

    Provides high-quality, context-aware translations that are often
    more natural and accurate than machine translation services.

    Args:
        text: The text to translate.
        source: Source language name (e.g., 'Chinese', 'zh-CN').
        target: Target language name (e.g., 'English', 'Japanese').
        use_opus: If True, use Claude Opus for highest quality.
        max_retries: Number of retry attempts on API failure.

    Returns:
        Translated text.
        Returns original text with error note if translation fails.
    """
    if not ANTHROPIC_AVAILABLE:
        logger.error("Anthropic SDK not installed. Install with: pip install anthropic")
        return f"[Translation unavailable - anthropic not installed] {text}"

    api_key = get_api_key()
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        return f"[Translation unavailable - API key not set] {text}"

    text = text.strip()
    if not text:
        return ''

    # Normalize language names
    source = _normalize_language(source)
    target = _normalize_language(target)

    model = OPUS_MODEL if use_opus else DEFAULT_MODEL

    # Handle long texts by chunking
    if len(text) <= MAX_CHUNK_SIZE:
        return _translate_with_retry(text, source, target, model, max_retries)

    # Split long text into chunks at sentence boundaries
    chunks = _split_text(text, MAX_CHUNK_SIZE)
    translated_chunks = []

    for i, chunk in enumerate(chunks):
        translated = _translate_with_retry(chunk, source, target, model, max_retries)
        translated_chunks.append(translated)
        # Small delay between chunks to avoid rate limiting
        if i < len(chunks) - 1:
            time.sleep(0.3)

    return ' '.join(translated_chunks)


def _normalize_language(lang: str) -> str:
    """Normalize language codes to human-readable names for Claude."""
    lang_map = {
        'zh-cn': 'Chinese (Simplified)',
        'zh-tw': 'Chinese (Traditional)',
        'zh': 'Chinese',
        'en': 'English',
        'ja': 'Japanese',
        'ko': 'Korean',
        'fr': 'French',
        'de': 'German',
        'es': 'Spanish',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'th': 'Thai',
        'vi': 'Vietnamese',
    }
    return lang_map.get(lang.lower(), lang)


def _translate_with_retry(
    text: str,
    source: str,
    target: str,
    model: str,
    max_retries: int,
) -> str:
    """Translate text with exponential backoff retry."""
    client = anthropic.Anthropic()

    system_prompt = f"""You are an expert translator specializing in {source} to {target} translation.

Instructions:
1. Translate the given text accurately and naturally
2. Preserve the tone and style of the original
3. Keep proper nouns, names, and technical terms appropriate to the target language
4. Maintain paragraph structure and formatting
5. Output ONLY the translation - no explanations, notes, or original text

Your translation should read naturally as if originally written in {target}."""

    user_prompt = f"""Translate this {source} text to {target}:

{text}

Remember: Output only the {target} translation, nothing else."""

    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=len(text) * 4,  # Allow for expansion
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                system=system_prompt,
            )

            result = message.content[0].text.strip()
            return result

        except anthropic.RateLimitError as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    f'Translation rate limited (attempt {attempt + 1}): {e}. '
                    f'Retrying in {wait}s...'
                )
                time.sleep(wait)
            else:
                logger.error(f'Translation rate limited after {max_retries} attempts: {e}')
                return f'[Translation rate limited] {text}'

        except anthropic.APIError as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    f'Translation API error (attempt {attempt + 1}): {e}. '
                    f'Retrying in {wait}s...'
                )
                time.sleep(wait)
            else:
                logger.error(f'Translation failed after {max_retries} attempts: {e}')
                return f'[Translation failed: {e}]'

        except Exception as e:
            logger.error(f'Unexpected error during translation: {e}')
            return f'[Translation error: {e}]'

    return text


def _split_text(text: str, max_size: int) -> list[str]:
    """
    Split text into chunks, preferring sentence boundaries.
    Falls back to splitting at any boundary if sentences are too long.
    """
    # Chinese and standard sentence-ending punctuation
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
