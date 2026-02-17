"""
Translation module using OpenRouter API.

Provides high-quality Chinese to English (or other language) translation
using LLMs (via OpenRouter) as an alternative to Google Translate.
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


def _normalize_language(lang: str) -> str:
    """Normalize language codes to human-readable names for LLM prompts."""
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


def translate_text_llm(
    text: str,
    source: str = 'Chinese',
    target: str = 'English',
    model: Optional[str] = None,
    max_retries: int = 3,
) -> str:
    """
    Translate text from source to target language using an LLM via OpenRouter.

    Args:
        text: The text to translate.
        source: Source language name (e.g., 'Chinese', 'zh-CN').
        target: Target language name (e.g., 'English', 'Japanese').
        model: OpenRouter model ID. Defaults to the standard tier default.
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

    if model is None:
        from .models import TIER_DEFAULTS
        model = TIER_DEFAULTS["standard"]

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


def translate_sentences_llm(
    sentences: list[str],
    source: str = 'Chinese',
    target: str = 'English',
    model: Optional[str] = None,
    max_retries: int = 3,
) -> list[str]:
    """
    Translate a list of Chinese sentences, returning one translation per sentence.

    Uses a single API call with numbered format for alignment. If the response
    count doesn't match input count, falls back to per-sentence translation.

    Args:
        sentences: List of Chinese sentences to translate.
        source: Source language name.
        target: Target language name.
        model: OpenRouter model ID. Defaults to the standard tier default.
        max_retries: Number of retry attempts.

    Returns:
        List of translations, one per input sentence.
    """
    if not sentences:
        return []

    if not OPENROUTER_AVAILABLE or not get_api_key():
        return [translate_text_llm(s, source, target, model, max_retries)
                for s in sentences]

    # Single sentence — just translate directly
    if len(sentences) == 1:
        return [translate_text_llm(sentences[0], source, target, model, max_retries)]

    source = _normalize_language(source)
    target = _normalize_language(target)

    if model is None:
        from .models import TIER_DEFAULTS
        model = TIER_DEFAULTS["standard"]

    numbered_input = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))

    system_prompt = (
        f"You are an expert translator from {source} to {target}. "
        f"You will receive numbered sentences. Translate each one separately. "
        f"Return ONLY the translations, numbered to match. One translation per line. "
        f"Do not combine, split, reorder, or skip any sentences."
    )
    user_prompt = numbered_input

    client = _create_client()

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                max_tokens=sum(len(s) for s in sentences) * 4,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            content = completion.choices[0].message.content
            if content is None:
                break

            # Parse numbered lines
            lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
            translations = []
            for line in lines:
                # Strip leading number and punctuation: "1. text" or "1) text"
                import re as _re
                cleaned = _re.sub(r'^\d+[\.\)]\s*', '', line)
                if cleaned:
                    translations.append(cleaned)

            if len(translations) == len(sentences):
                return translations

            # Count mismatch — retry or fall back
            logger.warning(
                f"Sentence count mismatch: expected {len(sentences)}, "
                f"got {len(translations)}. Attempt {attempt + 1}/{max_retries}."
            )
            if attempt < max_retries - 1:
                time.sleep(1)
                continue

        except openai.RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                break
        except openai.APIError:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                break
        except Exception as e:
            logger.error(f"Unexpected error in batch translation: {e}")
            break

    # Fall back: translate each sentence individually
    logger.info("Falling back to per-sentence translation")
    results = []
    for s in sentences:
        results.append(translate_text_llm(s, source, target, model, max_retries))
        time.sleep(0.3)
    return results


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
