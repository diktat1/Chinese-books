"""
Beginner Japanese synthesis from Chinese EPUB chapters.

Takes Chinese source text and generates a vocabulary-drill-style beginner
Japanese synthesis using only JLPT N5 vocabulary, with heavy word repetition.
Outputs paired (Japanese, Chinese) sentences for bilingual audio and EPUB.

Audio pattern per sentence: Japanese → pause → Chinese → pause → Japanese
"""

import asyncio
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
from ebooklib import epub

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# N5 vocabulary set (built once at import time)
# ---------------------------------------------------------------------------

def _build_n5_set() -> set[str]:
    """Build a set of allowed N5 words and readings for validation."""
    from .jlpt_n5_vocab import JLPT_N5_VOCAB

    allowed = set()
    for entry in JLPT_N5_VOCAB:
        word = entry['word'].strip()
        reading = entry['reading'].strip()

        # Add the word itself (kanji form)
        allowed.add(word)
        # Handle entries like "足; 脚" or "いい; よい"
        for part in word.split(';'):
            allowed.add(part.strip())

        # Add the reading (hiragana form)
        # Handle entries like "(〜を) とお" -> "とお"
        clean_reading = re.sub(r'[（(].*?[）)]', '', reading).strip()
        allowed.add(clean_reading)
        for part in clean_reading.split(';'):
            allowed.add(part.strip())

    # Remove empty strings
    allowed.discard('')
    return allowed


_N5_WORDS: set[str] | None = None


def _get_n5_words() -> set[str]:
    global _N5_WORDS
    if _N5_WORDS is None:
        _N5_WORDS = _build_n5_set()
    return _N5_WORDS


# Particles, grammatical elements, and basic forms always allowed
_GRAMMAR_ALLOW = {
    # Particles (N5 only)
    'は', 'が', 'を', 'に', 'へ', 'で', 'と', 'も', 'の', 'か', 'ね', 'よ',
    'から', 'まで',
    'な',
    # Copula and auxiliary forms
    'て', 'た', 'だ', 'です', 'ます', 'ません', 'ました', 'ませんでした',
    'ない', 'なかった', 'でした', 'だった',
    'たい', 'たかった', 'たくない',  # want to (verb+tai)
    'ん',  # contracted の
    # Core verbs (all conjugation forms)
    'する', 'した', 'します', 'しました', 'して', 'し',
    'ある', 'あった', 'あります', 'ありました', 'あって', 'あり',
    'いる', 'いた', 'います', 'いました', 'いて', 'い',
    'なる', 'なった', 'なります', 'なりました', 'なって', 'なり',
    'れる',  # janome uses for basic verb stems
    # Formal nouns (N5 only)
    'こと', 'もの', 'ところ',
    'そう',  # そうです (hearsay) is caught by _FORBIDDEN_GRAMMAR
    # Demonstratives
    'この', 'その', 'あの', 'どの',
    'これ', 'それ', 'あれ', 'どれ',
    'ここ', 'そこ', 'あそこ', 'どこ',
    'こう', 'そう', 'ああ', 'どう',
    # Common adverbs
    'とても', 'たくさん', 'もっと', 'まだ', 'もう', 'すぐ',
    'そして', 'でも', 'だから', 'それから',
    'ちょっと', 'すこし', 'ぜんぶ',
    # Honorific/common suffixes and compound starters
    'さん', 'さま', 'くん', 'ちゃん',
    'お客', 'もらう', 'もらい', 'もらいます', 'もらいました',
    # Punctuation
    '。', '、', '！', '？', '「', '」', '（', '）',
}

# N4+ grammar patterns that should be flagged even if individual tokens pass
_FORBIDDEN_GRAMMAR = [
    (r'なければなりません', 'must (N4)'),
    (r'なければならない', 'must (N4)'),
    (r'なくてはいけません', 'must (N4)'),
    (r'ことができ', 'can do (N4)'),
    (r'について', 'about/regarding (N4)'),
    (r'にとって', 'for/regarding (N3)'),
    (r'ために', 'in order to (N4)'),
    (r'ようにする', 'try to (N4)'),
    (r'ようになる', 'come to (N4)'),
    (r'かもしれません', 'might (N4)'),
    (r'でしょう', 'probably (N4)'),
    (r'ことがあります', 'sometimes (N4)'),
    (r'ほうがいい', 'should (N4)'),
    (r'そうです', 'looks like (N4)'),
    (r'ながら', 'while doing (N4)'),
    (r'ばいい', 'should (N4)'),
    (r'たらいい', 'should (N4)'),
]

# Katakana proper nouns (foreign names, brands) are always allowed
_KATAKANA_PATTERN = re.compile(r'^[ァ-ヶー・]+$')

# Numbers are always allowed
_NUMBER_PATTERN = re.compile(r'^[\d０-９一二三四五六七八九十百千万億]+$')


# ---------------------------------------------------------------------------
# N5 validation using janome tokenizer
# ---------------------------------------------------------------------------

def validate_n5(text: str) -> dict:
    """
    Validate that Japanese text uses only JLPT N5 vocabulary.

    Returns:
        {
            'is_valid': bool,
            'total_tokens': int,
            'violations': list of {'word': str, 'reading': str, 'base_form': str},
            'violation_count': int,
        }
    """
    from janome.tokenizer import Tokenizer

    tokenizer = Tokenizer()
    n5_words = _get_n5_words()
    violations = []
    total_tokens = 0

    for token in tokenizer.tokenize(text):
        surface = token.surface
        base = token.base_form if token.base_form != '*' else surface
        reading = token.reading if token.reading != '*' else ''
        part_of_speech = token.part_of_speech.split(',')[0]
        pos_detail = token.part_of_speech

        # Skip punctuation and symbols
        if part_of_speech in ('記号', '補助記号', 'BOS/EOS'):
            continue

        # Skip numbers
        if _NUMBER_PATTERN.match(surface):
            continue

        # Skip grammar particles, auxiliaries, and common forms
        if surface in _GRAMMAR_ALLOW or base in _GRAMMAR_ALLOW:
            continue

        # Skip particles (助詞) and auxiliary verbs (助動詞)
        if part_of_speech in ('助詞', '助動詞', '接頭詞', '接尾辞', 'フィラー'):
            continue

        # Skip katakana proper nouns (foreign names, brands, etc.)
        if _KATAKANA_PATTERN.match(surface):
            continue

        # Skip ASCII words (English proper nouns like IBM)
        if re.match(r'^[A-Za-z]+$', surface):
            continue

        total_tokens += 1

        # Build list of forms to check against N5 list
        forms_to_check = {surface, base, _to_hiragana(surface), _to_hiragana(base)}
        if reading:
            forms_to_check.add(reading)
            forms_to_check.add(_to_hiragana(reading))

        # Also check with common verb/adjective endings added
        for suffix in ['る', 'う', 'く', 'す', 'む', 'ぶ', 'ぬ', 'つ',
                       'い', 'ます', 'ません']:
            forms_to_check.add(base + suffix)
            forms_to_check.add(_to_hiragana(base) + suffix)

        if any(form in n5_words for form in forms_to_check):
            continue

        violations.append({
            'word': surface,
            'reading': reading,
            'base_form': base,
        })

    # Check for N4+ grammar patterns in the full text
    grammar_violations = []
    for pattern, label in _FORBIDDEN_GRAMMAR:
        if re.search(pattern, text):
            grammar_violations.append({'pattern': pattern, 'label': label})

    return {
        'is_valid': len(violations) == 0 and len(grammar_violations) == 0,
        'total_tokens': total_tokens,
        'violations': violations,
        'violation_count': len(violations),
        'grammar_violations': grammar_violations,
    }


def _to_hiragana(text: str) -> str:
    """Convert katakana to hiragana."""
    return ''.join(
        chr(ord(c) - 0x60) if '\u30A1' <= c <= '\u30F6' else c
        for c in text
    )


# ---------------------------------------------------------------------------
# EPUB chapter extraction
# ---------------------------------------------------------------------------

def _extract_first_content_chapter(
    epub_path: str,
    min_chars: int = 3000,
) -> tuple[str, list[str]]:
    """
    Extract the first real content chapter from an EPUB.

    Skips title pages, copyright, prefaces, and other front matter.
    Returns (title, [paragraphs]).
    """
    from .chinese_processing import contains_chinese

    # Front matter titles to skip
    _FRONT_MATTER = {'前言', '序', '序言', '目录', '版权信息', '版权', '致谢',
                     '引言', '导言', '写在前面', '作者简介', '关于作者',
                     'preface', 'foreword', 'introduction', 'copyright',
                     'titlepage', 'title', 'toc', 'contents'}

    book = epub.read_epub(epub_path)

    # Get spine-ordered items
    items_by_id = {}
    for item in book.get_items():
        items_by_id[item.get_id()] = item
        items_by_id[item.get_name()] = item

    spine_items = []
    for entry in book.spine:
        item_id = entry[0] if isinstance(entry, tuple) else entry
        item = items_by_id.get(item_id)
        if item and item.get_type() == 9:
            spine_items.append(item)

    if not spine_items:
        spine_items = list(book.get_items_of_type(9))

    _ALL_BLOCKS = [
        'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'li', 'td', 'th', 'blockquote', 'dt', 'dd',
        'figcaption', 'pre', 'div', 'section', 'article',
        'aside', 'header', 'footer', 'caption',
    ]

    for item in spine_items:
        html = item.get_content().decode('utf-8', errors='replace')
        soup = BeautifulSoup(html, 'lxml')

        title = None
        for tag in ('h1', 'h2', 'h3'):
            heading = soup.find(tag)
            if heading:
                title = heading.get_text().strip()
                break
        if not title:
            title = Path(item.get_name()).stem

        # Skip front matter
        fname = Path(item.get_name()).stem.lower()
        if (title.lower() in _FRONT_MATTER or fname in _FRONT_MATTER
                or 'preface' in fname or 'copyright' in fname
                or 'titlepage' in fname):
            continue

        paragraphs = []
        for block in soup.find_all(_ALL_BLOCKS):
            if block.find(_ALL_BLOCKS):
                continue
            text = block.get_text().strip()
            if text and contains_chinese(text):
                paragraphs.append(text)

        total_chars = sum(len(p) for p in paragraphs)
        if paragraphs and total_chars >= min_chars:
            logger.info(
                f'First content chapter: "{title}" '
                f'({len(paragraphs)} paragraphs, {total_chars} chars)'
            )
            return (title, paragraphs)

    raise ValueError('No content chapter found with enough Chinese text')


# ---------------------------------------------------------------------------
# LLM synthesis: Chinese -> Beginner Japanese (N5 level)
# ---------------------------------------------------------------------------

def _call_llm(system_prompt: str, user_prompt: str, model: str,
              max_tokens: int = 4000, max_retries: int = 3) -> str:
    """Call LLM via OpenRouter with retry logic."""
    from .llm_translator import (
        _is_anthropic_direct, _call_anthropic_direct,
        _create_client, OPENROUTER_AVAILABLE, get_api_key,
    )
    import openai as openai_module

    if _is_anthropic_direct(model):
        for attempt in range(max_retries):
            try:
                result = _call_anthropic_direct(
                    model, system_prompt, user_prompt, max_tokens,
                )
                if result:
                    return result
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f'LLM error (attempt {attempt + 1}): {e}. Retry in {wait}s')
                    time.sleep(wait)
                else:
                    raise
        return ''

    if not OPENROUTER_AVAILABLE or not get_api_key():
        raise RuntimeError('OPENROUTER_API_KEY required for LLM synthesis')

    client = _create_client()
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = completion.choices[0].message.content
            if content:
                return content.strip()
            return ''
        except openai_module.RateLimitError as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(f'Rate limited (attempt {attempt + 1}): {e}. Retry in {wait}s')
                time.sleep(wait)
            else:
                raise
        except openai_module.APIError as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(f'API error (attempt {attempt + 1}): {e}. Retry in {wait}s')
                time.sleep(wait)
            else:
                raise

    return ''


def _get_n5_word_sample() -> str:
    """Get a sample of N5 words for the LLM prompt."""
    from .jlpt_n5_vocab import JLPT_N5_VOCAB
    # Pick a representative sample to include in the prompt
    sample_words = []
    for entry in JLPT_N5_VOCAB[:200]:
        w = entry['word']
        r = entry['reading']
        m = entry['meaning']
        sample_words.append(f"{w}({r}): {m}")
    return '\n'.join(sample_words)


_SYNTHESIS_SYSTEM_PROMPT = """You are a Japanese language teaching expert creating beginner-level study materials.

Your task: Transform Chinese text into a JLPT N5 level Japanese vocabulary drill that follows the original ideas.

STRICT RULES:
1. Use ONLY JLPT N5 vocabulary (~800 words). This is a HARD requirement.
2. Use ONLY A1-A2 sentence structure: Subject は Object です/ます patterns, simple adjectives, basic verbs.
3. Keep sentences SHORT: 5-12 words maximum per sentence.
4. REPEAT each key word at least 3-4 times across different sentences to reinforce learning.
5. Follow the FLOW of the original Chinese text — cover the same ideas in the same order.
6. Write MULTIPLE sentences per idea — drill the vocabulary by rephrasing and repeating.
7. For concepts that cannot be expressed with N5 vocabulary, simplify to the closest N5-expressible idea.
8. Use these N5 grammar patterns only:
   - ～は～です (X is Y)
   - ～が あります/います (there is X)
   - ～を ～ます (do X)
   - ～に 行きます/来ます (go/come to X)
   - ～で ～ます (do X at/with Y)
   - ～から ～まで (from X to Y)
   - ～たい です (want to do X)
   - ～て ください (please do X)
   - ～ましょう (let's do X)
   - ～と 思います (I think X)

OUTPUT FORMAT — output ONLY numbered sentence pairs, nothing else:
1. JP: [Japanese sentence]
   CN: [Corresponding Chinese — simplified to match the Japanese meaning]
2. JP: [Japanese sentence]
   CN: [Corresponding Chinese]
...

Generate at least 80 sentence pairs for a full chapter. More is better — aim for heavy repetition."""


def synthesize_japanese_from_chinese(
    paragraphs: list[str],
    model: str | None = None,
    max_retries: int = 3,
) -> list[tuple[str, str]]:
    """
    Synthesize beginner Japanese (N5 level) from Chinese paragraphs.

    Processes the chapter in segments of ~3 paragraphs, generating
    vocabulary-drill-style Japanese sentences for each.

    Returns:
        List of (japanese_sentence, chinese_sentence) tuples.
    """
    if model is None:
        from .models import TIER_DEFAULTS
        model = TIER_DEFAULTS['free']

    # Group paragraphs into segments of 3
    segments = []
    for i in range(0, len(paragraphs), 3):
        segment = '\n'.join(paragraphs[i:i+3])
        segments.append(segment)

    all_pairs: list[tuple[str, str]] = []

    for seg_idx, segment in enumerate(segments):
        logger.info(f'Synthesizing segment {seg_idx + 1}/{len(segments)}...')

        user_prompt = f"""Transform this Chinese text into JLPT N5 beginner Japanese vocabulary drill sentences.
Follow the ideas in this text, but express them using ONLY N5 vocabulary and A1-A2 grammar.
Repeat key words many times across sentences.

Chinese text:
{segment}

Remember: Output ONLY numbered JP/CN pairs. At least 15 pairs for this segment. Use only N5 vocabulary."""

        try:
            response = _call_llm(
                _SYNTHESIS_SYSTEM_PROMPT, user_prompt, model,
                max_tokens=4000, max_retries=max_retries,
            )
            pairs = _parse_sentence_pairs(response)
            if pairs:
                all_pairs.extend(pairs)
                logger.info(f'  Segment {seg_idx + 1}: {len(pairs)} sentence pairs generated')
            else:
                logger.warning(f'  Segment {seg_idx + 1}: No pairs parsed from LLM response')
        except Exception as e:
            logger.error(f'  Segment {seg_idx + 1} failed: {e}')

        # Rate limit between segments
        if seg_idx < len(segments) - 1:
            time.sleep(1)

    logger.info(f'Total sentence pairs: {len(all_pairs)}')
    return all_pairs


def _parse_sentence_pairs(response: str) -> list[tuple[str, str]]:
    """Parse LLM response into (japanese, chinese) pairs."""
    pairs = []
    lines = response.strip().split('\n')

    current_jp = None
    current_cn = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Match patterns like:
        # 1. JP: これは会社です。
        # JP: これは会社です。
        # 1. これは会社です。(when JP: prefix is missing)
        jp_match = re.match(
            r'(?:\d+[\.\)]\s*)?(?:JP|日本語|Japanese)\s*[:：]\s*(.+)', line, re.IGNORECASE
        )
        cn_match = re.match(
            r'(?:\d+[\.\)]\s*)?(?:CN|中文|Chinese|ZH)\s*[:：]\s*(.+)', line, re.IGNORECASE
        )

        if jp_match:
            # Save previous pair if complete
            if current_jp and current_cn:
                pairs.append((current_jp.strip(), current_cn.strip()))
            current_jp = jp_match.group(1).strip()
            current_cn = None
        elif cn_match:
            current_cn = cn_match.group(1).strip()

    # Don't forget the last pair
    if current_jp and current_cn:
        pairs.append((current_jp.strip(), current_cn.strip()))

    return pairs


# ---------------------------------------------------------------------------
# N5 validation + correction loop
# ---------------------------------------------------------------------------

def validate_and_correct(
    pairs: list[tuple[str, str]],
    model: str | None = None,
    max_correction_rounds: int = 2,
) -> tuple[list[tuple[str, str]], dict]:
    """
    Validate Japanese sentences against N5 word list and correct violations.

    Returns:
        (corrected_pairs, validation_report)
    """
    if model is None:
        from .models import TIER_DEFAULTS
        model = TIER_DEFAULTS['free']

    all_japanese = '\n'.join(jp for jp, _ in pairs)
    report = validate_n5(all_japanese)

    if report['is_valid']:
        logger.info('N5 validation passed — all words are N5 compliant')
        return pairs, report

    logger.info(
        f'N5 validation: {report["violation_count"]} violations in '
        f'{report["total_tokens"]} tokens'
    )

    # Collect unique violations
    unique_violations = list({v['word'] for v in report['violations']})
    logger.info(f'Unique non-N5 words: {unique_violations[:20]}')

    # Try to correct violations
    for round_num in range(max_correction_rounds):
        # Find which pairs have violations
        pairs_to_fix = []
        for i, (jp, cn) in enumerate(pairs):
            pair_report = validate_n5(jp)
            if not pair_report['is_valid']:
                violation_words = [v['word'] for v in pair_report['violations']]
                pairs_to_fix.append((i, jp, cn, violation_words))

        if not pairs_to_fix:
            break

        logger.info(
            f'Correction round {round_num + 1}: '
            f'fixing {len(pairs_to_fix)} sentences...'
        )

        # Batch correction request
        fix_prompt_parts = []
        for idx, jp, cn, violations in pairs_to_fix[:20]:  # Limit batch size
            fix_prompt_parts.append(
                f'{idx + 1}. JP: {jp}\n'
                f'   CN: {cn}\n'
                f'   Non-N5 words: {", ".join(violations)}'
            )

        fix_prompt = (
            "The following Japanese sentences contain non-JLPT-N5 words. "
            "Rewrite EACH sentence using ONLY N5 vocabulary. "
            "Keep the same meaning as the Chinese. "
            "Keep sentences short and simple.\n\n"
            + '\n'.join(fix_prompt_parts)
            + "\n\nOutput format — ONLY numbered JP/CN pairs:\n"
            "1. JP: [corrected Japanese]\n"
            "   CN: [same Chinese]\n"
        )

        try:
            response = _call_llm(
                _SYNTHESIS_SYSTEM_PROMPT, fix_prompt, model,
                max_tokens=3000,
            )
            corrections = _parse_sentence_pairs(response)

            # Apply corrections
            fix_indices = [idx for idx, _, _, _ in pairs_to_fix[:20]]
            for j, (new_jp, new_cn) in enumerate(corrections):
                if j < len(fix_indices):
                    pairs[fix_indices[j]] = (new_jp, new_cn)

            logger.info(f'  Applied {len(corrections)} corrections')
        except Exception as e:
            logger.warning(f'  Correction round {round_num + 1} failed: {e}')
            break

        time.sleep(1)

    # Final validation
    all_japanese = '\n'.join(jp for jp, _ in pairs)
    final_report = validate_n5(all_japanese)
    unique_remaining = list({v['word'] for v in final_report['violations']})

    logger.info(
        f'Final N5 validation: {final_report["violation_count"]} violations '
        f'({len(unique_remaining)} unique words: {unique_remaining[:15]})'
    )

    return pairs, final_report


# ---------------------------------------------------------------------------
# Audio generation: JP -> pause -> CN -> pause -> JP
# ---------------------------------------------------------------------------

# Edge-TTS voices
_EDGE_VOICES = {
    'ja': 'ja-JP-NanamiNeural',
    'zh-CN': 'zh-CN-XiaoxiaoNeural',
}


async def _synthesize_edge(text: str, voice: str) -> bytes:
    """Synthesize text to MP3 bytes using edge-tts."""
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk['type'] == 'audio':
            buffer.write(chunk['data'])
    result = buffer.getvalue()
    if not result:
        raise RuntimeError('edge-tts returned empty audio')
    return result


def _generate_silence_mp3(duration_ms: int = 500) -> bytes:
    """Generate silence as MP3 using ffmpeg."""
    if not shutil.which('ffmpeg'):
        return b''
    result = subprocess.run(
        [
            'ffmpeg', '-y', '-f', 'lavfi',
            '-i', f'anullsrc=r=24000:cl=mono',
            '-t', str(duration_ms / 1000),
            '-c:a', 'libmp3lame', '-b:a', '32k',
            '-f', 'mp3', 'pipe:1',
        ],
        capture_output=True,
    )
    if result.returncode == 0:
        return result.stdout
    return b''


_silence_short: bytes | None = None  # 0.3s between segments
_silence_long: bytes | None = None   # 0.8s between sentence groups


def _get_silence_short() -> bytes:
    global _silence_short
    if _silence_short is None:
        _silence_short = _generate_silence_mp3(300)
    return _silence_short


def _get_silence_long() -> bytes:
    global _silence_long
    if _silence_long is None:
        _silence_long = _generate_silence_mp3(800)
    return _silence_long


async def _generate_sentence_audio(
    jp_text: str,
    cn_text: str,
    semaphore: asyncio.Semaphore,
) -> bytes:
    """
    Generate audio for one sentence pair: JP -> pause -> CN -> pause -> JP.
    """
    parts = []
    silence_short = _get_silence_short()

    async def synth(text: str, lang: str) -> bytes:
        voice = _EDGE_VOICES.get(lang)
        if not voice:
            raise RuntimeError(f'No voice for {lang}')
        async with semaphore:
            return await _synthesize_edge(text, voice)

    # 1. Japanese (first time)
    try:
        jp_audio = await synth(jp_text, 'ja')
        parts.append(jp_audio)
    except Exception as e:
        logger.warning(f'JP TTS failed: {e}')
        return b''

    # 2. Pause
    if silence_short:
        parts.append(silence_short)

    # 3. Chinese
    try:
        cn_audio = await synth(cn_text, 'zh-CN')
        parts.append(cn_audio)
    except Exception as e:
        logger.warning(f'CN TTS failed: {e}')

    # 4. Pause
    if silence_short:
        parts.append(silence_short)

    # 5. Japanese (repeat)
    try:
        jp_audio_2 = await synth(jp_text, 'ja')
        parts.append(jp_audio_2)
    except Exception as e:
        logger.warning(f'JP TTS repeat failed: {e}')

    return b''.join(parts)


# ---------------------------------------------------------------------------
# M4B assembly
# ---------------------------------------------------------------------------

def _has_ffmpeg() -> bool:
    return shutil.which('ffmpeg') is not None


def _build_m4b(
    chapter_files: list[tuple[str, Path]],
    output_path: str,
    cover_image: bytes | None = None,
) -> str:
    """Combine chapter MP3s into a single M4B with chapter markers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        concat_file = tmpdir / 'concat.txt'
        chapters_meta = []
        cumulative_ms = 0
        concat_lines = []

        for title, audio_path in chapter_files:
            result = subprocess.run(
                [
                    'ffprobe', '-v', 'quiet', '-print_format', 'json',
                    '-show_format', str(audio_path),
                ],
                capture_output=True, text=True,
            )
            duration_s = 0.0
            try:
                info = json.loads(result.stdout)
                duration_s = float(info['format']['duration'])
            except (json.JSONDecodeError, KeyError, ValueError):
                logger.warning(f'Could not probe duration for {audio_path}')

            chapters_meta.append((title, cumulative_ms))
            cumulative_ms += int(duration_s * 1000)

            escaped = str(audio_path).replace("'", "'\\''")
            concat_lines.append(f"file '{escaped}'")

        concat_file.write_text('\n'.join(concat_lines))

        meta_file = tmpdir / 'chapters.txt'
        meta_lines = [';FFMETADATA1']
        for i, (title, start_ms) in enumerate(chapters_meta):
            end_ms = (
                chapters_meta[i + 1][1] if i + 1 < len(chapters_meta)
                else cumulative_ms
            )
            meta_lines.extend([
                '', '[CHAPTER]', 'TIMEBASE=1/1000',
                f'START={start_ms}', f'END={end_ms}', f'title={title}',
            ])
        meta_file.write_text('\n'.join(meta_lines))

        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0', '-i', str(concat_file),
            '-i', str(meta_file),
        ]

        cover_path = None
        if cover_image:
            cover_path = tmpdir / 'cover.jpg'
            cover_path.write_bytes(cover_image)
            cmd.extend(['-i', str(cover_path)])
            cmd.extend([
                '-map', '0:a', '-map', '2:v',
                '-c:v', 'mjpeg',
                '-disposition:v:0', 'attached_pic',
            ])
        else:
            cmd.extend(['-map', '0:a'])

        cmd.extend([
            '-map_metadata', '1',
            '-c:a', 'aac', '-b:a', '64k',
            '-movflags', '+faststart',
            str(output_path),
        ])

        subprocess.run(cmd, check=True, capture_output=True)

    return output_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_beginner_japanese(
    epub_path: str,
    output_path: str,
    model: str | None = None,
    max_correction_rounds: int = 2,
) -> dict:
    """
    Full pipeline: Chinese EPUB chapter -> Beginner Japanese audio.

    Steps:
        1. Extract first content chapter
        2. LLM synthesis to N5-level Japanese
        3. Hard N5 validation + correction
        4. Audio generation (JP -> CN -> JP pattern)
        5. Assemble M4B audiobook

    Args:
        epub_path: Path to Chinese EPUB file.
        output_path: Path for output audio file.
        model: LLM model ID (OpenRouter or Anthropic).
        max_correction_rounds: How many times to attempt N5 correction.

    Returns:
        dict with 'status', 'output_path', 'sentence_count',
        'validation_report', 'pairs'.
    """
    if model is None:
        from .models import TIER_DEFAULTS
        model = TIER_DEFAULTS['free']

    # ------------------------------------------------------------------
    # Phase 1: Extract chapter
    # ------------------------------------------------------------------
    logger.info('Phase 1: Extracting first content chapter...')
    title, paragraphs = _extract_first_content_chapter(epub_path)
    logger.info(f'Chapter: "{title}" ({len(paragraphs)} paragraphs)')

    # ------------------------------------------------------------------
    # Phase 2: LLM synthesis
    # ------------------------------------------------------------------
    logger.info(f'Phase 2: Synthesizing beginner Japanese (model={model})...')
    pairs = synthesize_japanese_from_chinese(paragraphs, model=model)

    if not pairs:
        return {
            'status': 'error',
            'message': 'LLM synthesis returned no sentence pairs',
            'output_path': None,
            'sentence_count': 0,
            'validation_report': None,
            'pairs': [],
        }

    # ------------------------------------------------------------------
    # Phase 3: N5 validation + correction
    # ------------------------------------------------------------------
    logger.info('Phase 3: Validating N5 compliance...')
    pairs, validation_report = validate_and_correct(
        pairs, model=model, max_correction_rounds=max_correction_rounds,
    )

    # ------------------------------------------------------------------
    # Phase 4: Audio generation
    # ------------------------------------------------------------------
    logger.info(f'Phase 4: Generating audio for {len(pairs)} sentence pairs...')
    logger.info('Pattern: Japanese → pause → Chinese → pause → Japanese')

    semaphore = asyncio.Semaphore(5)  # Limit concurrent TTS
    silence_long = _get_silence_long()

    async def _generate_all_audio() -> list[tuple[int, bytes]]:
        tasks = []
        for i, (jp, cn) in enumerate(pairs):
            tasks.append(_generate_one(i, jp, cn, semaphore, silence_long))
        return await asyncio.gather(*tasks)

    async def _generate_one(idx, jp, cn, sem, long_silence) -> tuple[int, bytes]:
        audio = await _generate_sentence_audio(jp, cn, sem)
        # Add longer silence between sentences
        if audio and long_silence:
            audio = audio + long_silence
        return (idx, audio)

    results = asyncio.run(_generate_all_audio())
    results = sorted(results, key=lambda x: x[0])

    # Combine all audio
    all_audio = b''.join(audio for _, audio in results if audio)

    if not all_audio:
        return {
            'status': 'error',
            'message': 'Audio generation failed — no audio produced',
            'output_path': None,
            'sentence_count': len(pairs),
            'validation_report': validation_report,
            'pairs': pairs,
        }

    logger.info(f'Total audio: {len(all_audio) / 1024:.0f} KB')

    # ------------------------------------------------------------------
    # Phase 5: Assemble output
    # ------------------------------------------------------------------
    logger.info('Phase 5: Assembling audiobook...')

    output_path = str(Path(output_path))
    use_m4b = _has_ffmpeg()

    if use_m4b:
        output_path = str(Path(output_path).with_suffix('.m4b'))
    else:
        output_path = str(Path(output_path).with_suffix('.mp3'))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        if use_m4b:
            # Write as a single chapter M4B
            chapter_mp3 = tmpdir_path / 'chapter.mp3'
            chapter_mp3.write_bytes(all_audio)

            # Extract cover image
            cover_image = None
            try:
                from .audio_generator import _extract_cover_image
                cover_image = _extract_cover_image(epub_path)
            except Exception:
                pass

            _build_m4b(
                [(f'Beginner Japanese: {title}', chapter_mp3)],
                output_path,
                cover_image=cover_image,
            )
        else:
            # Just write MP3 directly
            Path(output_path).write_bytes(all_audio)

    logger.info(f'Output written to: {output_path}')

    # Save sentence pairs as JSON alongside audio
    pairs_json_path = str(Path(output_path).with_suffix('.json'))
    with open(pairs_json_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'chapter_title': title,
                'model': model,
                'sentence_pairs': [
                    {'japanese': jp, 'chinese': cn}
                    for jp, cn in pairs
                ],
                'validation': {
                    'total_tokens': validation_report['total_tokens'],
                    'violation_count': validation_report['violation_count'],
                    'violations': [
                        v['word'] for v in validation_report['violations']
                    ][:50],
                },
            },
            f, ensure_ascii=False, indent=2,
        )
    logger.info(f'Sentence pairs saved to: {pairs_json_path}')

    return {
        'status': 'success',
        'output_path': output_path,
        'pairs_json_path': pairs_json_path,
        'sentence_count': len(pairs),
        'validation_report': validation_report,
        'pairs': pairs,
    }
