"""
HSK Vocabulary Simplifier using OpenRouter API or Anthropic API directly.

This module uses LLMs (via OpenRouter or Anthropic) to simplify Chinese text
to a target HSK vocabulary level, replacing advanced words with simpler
equivalents that learners can understand.
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

# Anthropic direct API model name mapping
_ANTHROPIC_MODEL_MAP = {
    'claude-sonnet-4': 'claude-sonnet-4-20250514',
    'claude-sonnet-4.5': 'claude-sonnet-4-5-20250514',
    'claude-haiku-4': 'claude-haiku-4-20250414',
    'claude-opus-4': 'claude-opus-4-20250514',
}


def _is_anthropic_direct(model: str) -> bool:
    """Check if a model ID should use the Anthropic API directly."""
    if not model:
        return False
    return (
        model.startswith('claude-')
        and '/' not in model
        and os.environ.get('ANTHROPIC_API_KEY')
    )


def _call_anthropic_direct(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> str | None:
    """Call the Anthropic API directly. Returns response text or None."""
    import anthropic

    resolved = _ANTHROPIC_MODEL_MAP.get(model, model)
    client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

    message = client.messages.create(
        model=resolved,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    if message.content and message.content[0].text:
        return message.content[0].text.strip()
    return None


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
    model: Optional[str] = None,
    max_retries: int = 3,
    hsk_level: str = '4',
    max_sentence_words: int = 0,
) -> str:
    """
    Simplify Chinese text to a target HSK vocabulary level using an LLM.

    Replaces advanced vocabulary with simpler equivalents that learners
    at the target HSK level can understand.

    Args:
        text: The Chinese text to simplify.
        model: Model ID. Defaults to the premium tier default.
        max_retries: Number of retry attempts on API failure.
        hsk_level: Target HSK level (e.g., '4', '4-5').
        max_sentence_words: Max words per sentence (0=no limit).

    Returns:
        Simplified Chinese text.
        Returns original text with error note if simplification fails.
    """
    use_anthropic = model and _is_anthropic_direct(model)

    if not use_anthropic:
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

    if model is None:
        from .models import TIER_DEFAULTS
        model = TIER_DEFAULTS["premium"]

    return _simplify_with_retry(text, model, max_retries, hsk_level, max_sentence_words)


def _simplify_with_retry(
    text: str,
    model: str,
    max_retries: int,
    hsk_level: str = '4',
    max_sentence_words: int = 0,
) -> str:
    """Simplify text with exponential backoff retry."""
    # Build dynamic system prompt based on HSK level
    level_desc = f'HSK {hsk_level}'
    word_limit_instr = ''
    if max_sentence_words > 0:
        word_limit_instr = (
            f'\n10. Keep each sentence to approximately {max_sentence_words} '
            f'words or fewer. Split longer sentences into shorter ones.'
        )

    system_prompt = f"""You are a Chinese language simplification expert. Your task is to simplify Chinese text to {level_desc} vocabulary level while preserving the author's ideas.

Instructions:
1. Identify any words or phrases above {level_desc} level
2. Replace them with simpler {level_desc} or below equivalents that preserve the meaning
3. Keep the sentence structure natural and grammatically correct
4. Preserve proper nouns, names, and places
5. Preserve numbers and punctuation exactly
6. If a word cannot be simplified without losing essential meaning, keep it but try to add context clues
7. Add a space between each Chinese word (word segmentation). For example: "我 喜欢 学习 中文" instead of "我喜欢学习中文"
8. NEVER remove or alter key concepts, metaphors, analogies, or technical terms from the original. If a concept like "10倍速因素" appears, keep it — simplify the surrounding grammar instead
9. Preserve the author's analytical frameworks and reasoning. Replace difficult vocabulary only when meaning is fully preserved{word_limit_instr}

IMPORTANT: Output ONLY the simplified Chinese text with spaces between words. Do not include any explanations, notes, or English text."""

    user_prompt = f"""Simplify this Chinese text to {level_desc} vocabulary level. Add a space between each word. Preserve all key concepts and technical terms:

{text}

Remember: Output only the simplified Chinese text with word spacing, nothing else."""

    # Route to Anthropic direct API if applicable
    if _is_anthropic_direct(model):
        for attempt in range(max_retries):
            try:
                result = _call_anthropic_direct(
                    model, system_prompt, user_prompt,
                    max_tokens=len(text) * 3,
                )
                if result:
                    return result
                return text
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        f'Anthropic simplification error (attempt {attempt + 1}): {e}. '
                        f'Retrying in {wait}s...'
                    )
                    time.sleep(wait)
                else:
                    logger.error(f'Simplification failed after {max_retries} attempts: {e}')
                    return f'[Simplification failed: {e}] {text}'
        return text

    # OpenRouter path
    client = _create_client()

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


def verify_simplification(
    original: str,
    simplified: str,
    model: Optional[str] = None,
    max_retries: int = 3,
) -> str:
    """
    Verify that a simplified text preserves all key concepts from the original.

    Asks the LLM to compare original and simplified texts, and returns a
    corrected version if key concepts were lost, or the simplified text as-is.

    Args:
        original: The original Chinese text before simplification.
        simplified: The simplified Chinese text to verify.
        model: Model ID for the verification call.
        max_retries: Number of retry attempts on API failure.

    Returns:
        Corrected simplified text (with word spacing), or the original
        simplified text if no issues were found.
    """
    if not original.strip() or not simplified.strip():
        return simplified

    if model is None:
        from .models import TIER_DEFAULTS
        model = TIER_DEFAULTS["premium"]

    system_prompt = """You are a Chinese text quality reviewer. Compare an original Chinese text with its simplified version and check whether ALL key concepts, metaphors, technical terms, and meaning are preserved.

If the simplified version is faithful, output it exactly as-is.
If concepts were lost or distorted, output a CORRECTED version that:
- Restores the lost concepts and terms
- Keeps the grammar simple (HSK 4-5 level)
- Maintains spaces between words (word segmentation)

IMPORTANT: Output ONLY the final Chinese text with spaces between words. No explanations."""

    user_prompt = f"""Original:
{original}

Simplified:
{simplified}

Does the simplified version preserve ALL key concepts? If not, output a corrected version. Output only Chinese text with word spacing."""

    use_anthropic = model and _is_anthropic_direct(model)

    if use_anthropic:
        for attempt in range(max_retries):
            try:
                result = _call_anthropic_direct(
                    model, system_prompt, user_prompt,
                    max_tokens=len(simplified) * 3,
                )
                if result:
                    return result
                return simplified
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f'Verification error (attempt {attempt + 1}): {e}. Retrying in {wait}s...')
                    time.sleep(wait)
                else:
                    logger.error(f'Verification failed after {max_retries} attempts: {e}')
                    return simplified
        return simplified

    if not OPENROUTER_AVAILABLE:
        return simplified

    client = _create_client()

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                max_tokens=len(simplified) * 3,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = completion.choices[0].message.content
            if content:
                return content.strip()
            return simplified
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(f'Verification error (attempt {attempt + 1}): {e}. Retrying in {wait}s...')
                time.sleep(wait)
            else:
                logger.error(f'Verification failed after {max_retries} attempts: {e}')
                return simplified

    return simplified


def add_word_spacing_llm(
    text: str,
    model: Optional[str] = None,
    max_retries: int = 3,
) -> str:
    """
    Add spaces between Chinese words using an LLM for word segmentation.

    Unlike jieba-based segmentation, this uses the LLM's understanding of
    context to produce more accurate word boundaries.

    Args:
        text: Chinese text without word spacing.
        model: Model ID for the LLM call.
        max_retries: Number of retry attempts on API failure.

    Returns:
        The same text with a single space between each Chinese word.
        Returns original text on failure.
    """
    text = text.strip()
    if not text:
        return ''

    if model is None:
        from .models import TIER_DEFAULTS
        model = TIER_DEFAULTS["standard"]

    system_prompt = """You are a Chinese word segmentation expert. Add a single space between each Chinese word in the given text.

Rules:
- Do NOT change, add, or remove any characters
- Only add spaces between words
- Keep all punctuation in place
- Proper nouns should be kept as single units
- Output ONLY the spaced text, nothing else"""

    user_prompt = f"""Add spaces between each Chinese word:

{text}"""

    use_anthropic = model and _is_anthropic_direct(model)

    if use_anthropic:
        for attempt in range(max_retries):
            try:
                result = _call_anthropic_direct(
                    model, system_prompt, user_prompt,
                    max_tokens=len(text) * 3,
                )
                if result:
                    return result
                return text
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f'Word spacing error (attempt {attempt + 1}): {e}. Retrying in {wait}s...')
                    time.sleep(wait)
                else:
                    logger.error(f'Word spacing failed after {max_retries} attempts: {e}')
                    return text
        return text

    if not OPENROUTER_AVAILABLE:
        return text

    client = _create_client()

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                max_tokens=len(text) * 3,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = completion.choices[0].message.content
            if content:
                return content.strip()
            return text
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(f'Word spacing error (attempt {attempt + 1}): {e}. Retrying in {wait}s...')
                time.sleep(wait)
            else:
                logger.error(f'Word spacing failed after {max_retries} attempts: {e}')
                return text

    return text


def analyze_vocabulary_level(
    text: str,
    model: Optional[str] = None,
) -> dict:
    """
    Analyze a Chinese text and identify words above HSK 4 level.

    Args:
        text: The Chinese text to analyze.
        model: OpenRouter model ID. Defaults to the premium tier default.

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

    if model is None:
        from .models import TIER_DEFAULTS
        model = TIER_DEFAULTS["premium"]

    client = _create_client()

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
