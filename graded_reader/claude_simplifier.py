"""
HSK 4 Vocabulary Simplifier using OpenRouter API.

This module uses Claude (via OpenRouter) to simplify Chinese text to HSK 4
vocabulary level, replacing advanced words with simpler equivalents that
HSK 4 learners can understand.
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

# Default model for simplification (can be overridden)
DEFAULT_MODEL = "anthropic/claude-sonnet-4"
OPUS_MODEL = "anthropic/claude-opus-4"


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


def simplify_to_hsk4(
    text: str,
    use_opus: bool = False,
    max_retries: int = 3,
) -> str:
    """
    Simplify Chinese text to HSK 4 vocabulary level using Claude.

    Replaces advanced vocabulary (HSK 5-6 and beyond) with HSK 4 or simpler
    equivalents while preserving the original meaning as much as possible.

    Args:
        text: The Chinese text to simplify.
        use_opus: If True, use Claude Opus for higher quality (slower, more expensive).
        max_retries: Number of retry attempts on API failure.

    Returns:
        Simplified Chinese text at HSK 4 level.
        Returns original text with error note if simplification fails.
    """
    if not OPENROUTER_AVAILABLE:
        logger.error("OpenAI SDK not installed. Install with: pip install openai")
        return f"[Simplification unavailable - openai not installed] {text}"

    api_key = get_api_key()
    if not api_key:
        logger.error("OPENROUTER_API_KEY environment variable not set")
        return f"[Simplification unavailable - API key not set] {text}"

    text = text.strip()
    if not text:
        return ''

    model = OPUS_MODEL if use_opus else DEFAULT_MODEL

    return _simplify_with_retry(text, model, max_retries)


def _simplify_with_retry(text: str, model: str, max_retries: int) -> str:
    """Simplify text with exponential backoff retry."""
    client = _create_client()

    system_prompt = """You are a Chinese language simplification expert. Your task is to simplify Chinese text to HSK 4 vocabulary level.

HSK 4 represents an intermediate level with approximately 1,200 vocabulary words. Students at this level can discuss a wide range of topics and communicate fluently with native Chinese speakers.

Instructions:
1. Identify any words or phrases above HSK 4 level (HSK 5, HSK 6, or non-HSK vocabulary)
2. Replace them with simpler HSK 4 or below equivalents that preserve the meaning
3. Keep the sentence structure natural and grammatically correct
4. Preserve proper nouns, names, and places
5. Preserve numbers and punctuation exactly
6. If a word cannot be simplified without losing essential meaning, keep it but try to add context clues

IMPORTANT: Output ONLY the simplified Chinese text. Do not include any explanations, notes, or English text."""

    user_prompt = f"""Simplify this Chinese text to HSK 4 vocabulary level:

{text}

Remember: Output only the simplified Chinese text, nothing else."""

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                max_tokens=len(text) * 3,  # Allow for expansion
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
                    f'Simplification rate limited (attempt {attempt + 1}): {e}. '
                    f'Retrying in {wait}s...'
                )
                time.sleep(wait)
            else:
                logger.error(f'Simplification rate limited after {max_retries} attempts: {e}')
                return f'[Simplification rate limited] {text}'

        except openai.APIError as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    f'Simplification API error (attempt {attempt + 1}): {e}. '
                    f'Retrying in {wait}s...'
                )
                time.sleep(wait)
            else:
                logger.error(f'Simplification failed after {max_retries} attempts: {e}')
                return f'[Simplification failed: {e}] {text}'

        except Exception as e:
            logger.error(f'Unexpected error during simplification: {e}')
            return f'[Simplification error: {e}] {text}'

    return text


def analyze_vocabulary_level(
    text: str,
    use_opus: bool = False,
) -> dict:
    """
    Analyze a Chinese text and identify words above HSK 4 level.

    Args:
        text: The Chinese text to analyze.
        use_opus: If True, use Claude Opus for more accurate analysis.

    Returns:
        A dictionary containing:
        - 'advanced_words': List of words above HSK 4 with their levels
        - 'simplified_suggestions': Suggested HSK 4 replacements
        - 'difficulty_score': Estimated difficulty (1-10)
    """
    if not OPENROUTER_AVAILABLE:
        return {'error': 'OpenAI SDK not installed'}

    api_key = get_api_key()
    if not api_key:
        return {'error': 'OPENROUTER_API_KEY not set'}

    client = _create_client()
    model = OPUS_MODEL if use_opus else DEFAULT_MODEL

    system_prompt = """You are a Chinese language analysis expert. Analyze the given text and identify vocabulary difficulty.

Output your analysis in this exact format:
ADVANCED_WORDS:
- word1 (HSK level or "beyond HSK") -> suggested_replacement
- word2 (HSK level) -> suggested_replacement

DIFFICULTY_SCORE: X/10

Keep the format exact. If no advanced words found, write "ADVANCED_WORDS: None"."""

    try:
        completion = client.chat.completions.create(
            model=model,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this Chinese text:\n\n{text}"},
            ],
        )

        content = completion.choices[0].message.content
        if content is None:
            return {'error': 'Empty response from API'}

        response = content

        # Parse the response
        result = {
            'advanced_words': [],
            'difficulty_score': 5,
            'raw_analysis': response,
        }

        lines = response.split('\n')
        in_words_section = False

        for line in lines:
            line = line.strip()
            if line.startswith('ADVANCED_WORDS:'):
                in_words_section = True
                if 'None' in line:
                    in_words_section = False
            elif line.startswith('DIFFICULTY_SCORE:'):
                in_words_section = False
                try:
                    score_part = line.split(':')[1].strip()
                    score = int(score_part.split('/')[0])
                    result['difficulty_score'] = score
                except (ValueError, IndexError):
                    pass
            elif in_words_section and line.startswith('-'):
                result['advanced_words'].append(line[1:].strip())

        return result

    except Exception as e:
        logger.error(f'Analysis failed: {e}')
        return {'error': str(e)}
