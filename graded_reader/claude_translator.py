"""
Translation module using OpenRouter API.

Provides high-quality Chinese to English (or other language) translation
using Claude (via OpenRouter) as an alternative to Google Translate.
"""

import os
import time
import logging
from typing import Optional

try:
    from openai import OpenAI
    import openai
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default model for translation
DEFAULT_MODEL = "anthropic/claude-sonnet-4"
OPUS_MODEL = "anthropic/claude-opus-4"

# Maximum text length per request (characters)
MAX_CHUNK_SIZE = 4000


def is_openrouter_available() -> bool:
    """Check if the OpenAI SDK (for OpenRouter) is installed."""
    return OPENROUTER_AVAILABLE


def get_api_key() -> Optional[str]:
    """Get OpenRouter API key from environment variable."""
    return os.environ.get('OPENROUTER_API_KEY')


def _create_client() -> 'OpenAI':
    """Create an OpenRouter client."""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "https://github.com/diktat1/Chinese-books",
            "X-Title": "Chinese Graded Reader",
        },
    )


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
    if not OPENROUTER_AVAILABLE:
        logger.error("OpenAI SDK not installed. Install with: pip install openai")
        return f"[Translation unavailable - openai not installed] {text}"

    api_key = get_api_key()
    if not api_key:
        logger.error("OPENROUTER_API_KEY environment variable not set")
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
    client = _create_client()

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
            completion = client.chat.completions.create(
                model=model,
                max_tokens=len(text) * 4,  # Allow for expansion
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            content = completion.choices[0].message.content
            if content is None:
                logger.warning("Empty response from API")
                return text
            return content.strip()

        except openai.RateLimitError as e:
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

        except openai.APIError as e:
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
