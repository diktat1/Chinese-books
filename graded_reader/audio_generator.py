"""
Audiobook generator from Chinese EPUB books.

Generates chapter-by-chapter audio with optional multilingual narration.
Each paragraph is narrated three times: Chinese -> Target language -> Chinese.
Chapters rotate through a list of target languages.

TTS engines (tried in order):
  1. Qwen3-TTS — HuggingFace Space, multilingual neural voices
  2. edge-tts  — Microsoft Edge neural voices, free, best quality
  3. kokoro    — Local 82M-param model, Apache 2.0, CPU-friendly fallback

Output formats:
  - M4B audiobook with chapter markers and cover art (Apple Books, Audible, etc.)
  - ZIP of chapter MP3s (fallback when ffmpeg is unavailable)
"""

import asyncio
import io
import json
import logging
import re
import shutil
import subprocess
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import epub

from .chinese_processing import contains_chinese, split_sentences

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language names for chapter labels
# ---------------------------------------------------------------------------

_LANG_NAMES = {
    'fr': 'French',
    'de': 'German',
    'es': 'Spanish',
    'tr': 'Turkish',
    'pt': 'Portuguese',
    'it': 'Italian',
    'en': 'English',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ru': 'Russian',
    'zh-CN': 'Chinese',
}

# ---------------------------------------------------------------------------
# TTS voice configuration
# ---------------------------------------------------------------------------

# Best neural voice per language (edge-tts)
_EDGE_VOICES = {
    'zh-CN': 'zh-CN-XiaoxiaoNeural',
    'fr': 'fr-FR-DeniseNeural',
    'en': 'en-US-JennyNeural',
    'ja': 'ja-JP-NanamiNeural',
    'ko': 'ko-KR-SunHiNeural',
    'de': 'de-DE-KatjaNeural',
    'es': 'es-ES-ElviraNeural',
    'it': 'it-IT-ElsaNeural',
    'pt': 'pt-BR-FranciscaNeural',
    'tr': 'tr-TR-EmelNeural',
    'ru': 'ru-RU-SvetlanaNeural',
}

# Kokoro language codes and voice names
_KOKORO_VOICES = {
    'zh-CN': ('z', 'zf_xiaobei'),
    'fr': ('f', 'ff_siwis'),
    'en': ('a', 'af_heart'),
    'ja': ('j', 'jf_alpha'),
    'ko': ('k', 'kf_sarah'),
    'de': ('d', 'df_anna'),
    'es': ('e', 'ef_dora'),
    'it': ('i', 'if_sara'),
    'pt': ('p', 'pf_dora'),
}

# Qwen3-TTS supported languages and default speakers
_QWEN3_LANGS = {
    'zh-CN': 'Chinese',
    'en': 'English',
    'ja': 'Japanese',
    'ko': 'Korean',
    'fr': 'French',
    'de': 'German',
    'es': 'Spanish',
    'pt': 'Portuguese',
    'ru': 'Russian',
}

_QWEN3_SPEAKERS = {
    'zh-CN': 'Vivian',
    'en': 'Serena',
    'ja': 'Ono_anna',
    'ko': 'Sohee',
    'fr': 'Serena',
    'de': 'Serena',
    'es': 'Serena',
    'pt': 'Serena',
    'ru': 'Serena',
}

# Circuit breaker for Qwen3-TTS (disabled after repeated failures)
_qwen3_disabled = False
_qwen3_fail_count = 0
_QWEN3_MAX_FAILURES = 3  # Disable after this many consecutive failures
_qwen3_client = None

# Block elements to extract text from
_ALL_BLOCKS = [
    'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'li', 'td', 'th', 'blockquote', 'dt', 'dd',
    'figcaption', 'pre', 'div', 'section', 'article',
    'aside', 'header', 'footer', 'caption',
]

# Chinese conjunctions that should stay with the previous chunk
_ZH_CONJUNCTIONS = {
    '但是', '但', '可是', '不过', '然而', '而且', '并且', '而',
    '所以', '因此', '因而', '于是', '那么', '那', '则',
    '虽然', '尽管', '即使', '就算', '哪怕',
    '如果', '假如', '要是', '倘若',
    '另外', '此外', '同时', '接着', '然后', '最后',
}


# ---------------------------------------------------------------------------
# EPUB chapter extraction (spine-ordered)
# ---------------------------------------------------------------------------

def _get_spine_items(book: epub.EpubBook) -> list:
    """Return document items in spine (reading) order, not manifest order."""
    items_by_id = {}
    for item in book.get_items():
        items_by_id[item.get_id()] = item
        items_by_id[item.get_name()] = item

    spine_items = []
    for entry in book.spine:
        item_id = entry[0] if isinstance(entry, tuple) else entry
        item = items_by_id.get(item_id)
        if item and item.get_type() == 9:  # ITEM_DOCUMENT
            spine_items.append(item)

    if not spine_items:
        logger.warning('Could not resolve spine order, falling back to manifest order')
        spine_items = list(book.get_items_of_type(9))

    return spine_items


def _extract_chapters(epub_path: str) -> list[tuple[str, list[str]]]:
    """
    Extract chapters from EPUB as (title, [paragraphs]) in reading order.
    """
    book = epub.read_epub(epub_path)
    chapters = []

    for item in _get_spine_items(book):
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

        paragraphs = []
        for block in soup.find_all(_ALL_BLOCKS):
            if block.find(_ALL_BLOCKS):
                continue
            text = block.get_text().strip()
            if text and contains_chinese(text):
                paragraphs.append(text)

        if paragraphs:
            chapters.append((title, paragraphs))

    return chapters


# ---------------------------------------------------------------------------
# Cover image and metadata extraction
# ---------------------------------------------------------------------------

def _extract_cover_image(epub_path: str) -> bytes | None:
    """Extract cover image from EPUB. Returns image bytes or None."""
    try:
        book = epub.read_epub(epub_path)

        # Method 1: Check ITEM_COVER type (type=10)
        cover_items = list(book.get_items_of_type(10))
        if cover_items:
            return cover_items[0].get_content()

        # Method 2: Check OPF metadata for cover item ID
        cover_meta = book.get_metadata('OPF', 'cover')
        if cover_meta:
            cover_id = cover_meta[0][1].get('content', '') if cover_meta[0][1] else ''
            if cover_id:
                for item in book.get_items():
                    if item.get_id() == cover_id:
                        return item.get_content()

        # Method 3: Look for items with cover-image properties
        for item in book.get_items_of_type(6):  # ITEM_IMAGE
            name = item.get_name().lower()
            if 'cover' in name:
                return item.get_content()

    except Exception as e:
        logger.warning(f'Could not extract cover image: {e}')

    return None


def _get_book_metadata(epub_path: str, english: bool = False) -> dict:
    """
    Extract book metadata from EPUB.

    Args:
        epub_path: Path to the EPUB file.
        english: If True, transliterate Chinese title/author to pinyin.

    Returns:
        dict with 'title', 'author', 'year' keys.
    """
    try:
        book = epub.read_epub(epub_path)

        title = ''
        title_meta = book.get_metadata('DC', 'title')
        if title_meta:
            title = title_meta[0][0]

        author = ''
        creator_meta = book.get_metadata('DC', 'creator')
        if creator_meta:
            author = creator_meta[0][0]

        year = ''
        date_meta = book.get_metadata('DC', 'date')
        if date_meta:
            date_str = date_meta[0][0]
            year_match = re.search(r'(\d{4})', date_str)
            if year_match:
                year = year_match.group(1)

        if english and (contains_chinese(title) or contains_chinese(author)):
            try:
                from pypinyin import pinyin, Style
                if contains_chinese(title):
                    py = pinyin(title, style=Style.NORMAL)
                    title = ' '.join(p[0] for p in py).title()
                if contains_chinese(author):
                    py = pinyin(author, style=Style.NORMAL)
                    author = ' '.join(p[0] for p in py).title()
            except ImportError:
                pass

        return {'title': title, 'author': author, 'year': year}

    except Exception as e:
        logger.warning(f'Could not extract metadata: {e}')
        return {'title': '', 'author': '', 'year': ''}


# ---------------------------------------------------------------------------
# Paragraph splitting
# ---------------------------------------------------------------------------

def _split_paragraph(text: str, max_sentences: int = 4) -> list[str]:
    """
    Split a paragraph into chunks of max_sentences sentences.

    Keeps idea boundaries: if a sentence starts with a conjunction,
    it stays with the previous chunk.
    """
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return [text]

    chunks = []
    current_chunk = []

    for sent in sentences:
        # Check if this sentence starts with a conjunction
        starts_with_conj = any(sent.startswith(conj) for conj in _ZH_CONJUNCTIONS)

        if starts_with_conj and current_chunk:
            # Keep conjunction sentence with the previous chunk
            current_chunk.append(sent)
        elif len(current_chunk) >= max_sentences:
            # Current chunk is full, start a new one
            chunks.append(''.join(current_chunk))
            current_chunk = [sent]
        else:
            current_chunk.append(sent)

    if current_chunk:
        chunks.append(''.join(current_chunk))

    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# TTS engines
# ---------------------------------------------------------------------------

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


def _synthesize_kokoro(text: str, lang_code: str, voice_name: str) -> bytes:
    """Synthesize text to WAV bytes using Kokoro (local, CPU)."""
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=lang_code)
    buffer = io.BytesIO()

    import soundfile as sf
    all_audio = []
    for _, _, audio in pipeline(text, voice=voice_name):
        all_audio.append(audio)

    if not all_audio:
        raise RuntimeError('Kokoro returned no audio')

    import numpy as np
    combined = np.concatenate(all_audio)
    sf.write(buffer, combined, 24000, format='WAV')
    return buffer.getvalue()


def _synthesize_qwen3tts(text: str, lang: str) -> bytes:
    """Synthesize text using Qwen3-TTS via HuggingFace Gradio Client."""
    global _qwen3_client, _qwen3_disabled, _qwen3_fail_count

    if _qwen3_disabled:
        raise RuntimeError('Qwen3-TTS disabled (GPU quota exceeded)')

    qwen_lang = _QWEN3_LANGS.get(lang)
    if not qwen_lang:
        raise RuntimeError(f'Qwen3-TTS does not support lang={lang}')

    speaker = _QWEN3_SPEAKERS.get(lang, 'Chelsie')

    try:
        if _qwen3_client is None:
            from gradio_client import Client
            _qwen3_client = Client('Qwen/Qwen3-TTS')

        result = _qwen3_client.predict(
            text=text,
            language=qwen_lang,
            speaker=speaker,
            instruct='',
            model_size='1.7B',
            api_name='/generate_custom_voice',
        )

        # Result is (filepath, status_text)
        audio_path = result[0] if isinstance(result, (list, tuple)) else result

        with open(audio_path, 'rb') as f:
            return f.read()

    except Exception as e:
        _qwen3_fail_count += 1
        error_msg = str(e).lower()
        if ('gpu' in error_msg or 'quota' in error_msg or 'queue' in error_msg
                or _qwen3_fail_count >= _QWEN3_MAX_FAILURES):
            _qwen3_disabled = True
            logger.warning(f'Qwen3-TTS disabled after {_qwen3_fail_count} failures: {e}')
        raise


_kokoro_available = None


def _is_kokoro_available() -> bool:
    """Check if Kokoro TTS is installed."""
    global _kokoro_available
    if _kokoro_available is None:
        try:
            import kokoro  # noqa: F401
            import soundfile  # noqa: F401
            _kokoro_available = True
        except ImportError:
            _kokoro_available = False
    return _kokoro_available


async def _synthesize(text: str, lang: str) -> bytes:
    """
    Synthesize text using the best available TTS engine.

    Priority: Qwen3-TTS -> edge-tts -> Kokoro.
    """
    # Try Qwen3-TTS first (if available for this language)
    if lang in _QWEN3_LANGS and not _qwen3_disabled:
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, _synthesize_qwen3tts, text, lang,
            )
        except Exception as e:
            if not _qwen3_disabled:
                logger.warning(f'Qwen3-TTS failed: {e}')

    # Try edge-tts
    edge_voice = _EDGE_VOICES.get(lang)
    if edge_voice:
        try:
            return await _synthesize_edge(text, edge_voice)
        except Exception as e:
            logger.warning(f'edge-tts failed: {e}')

    # Fallback to Kokoro
    if _is_kokoro_available():
        kokoro_cfg = _KOKORO_VOICES.get(lang)
        if kokoro_cfg:
            lang_code, voice_name = kokoro_cfg
            try:
                return await asyncio.get_event_loop().run_in_executor(
                    None, _synthesize_kokoro, text, lang_code, voice_name,
                )
            except Exception as e:
                logger.warning(f'Kokoro TTS failed: {e}')

    raise RuntimeError(f'All TTS engines failed for lang={lang}')


# ---------------------------------------------------------------------------
# Silence generation
# ---------------------------------------------------------------------------

def _generate_silence_mp3(duration_ms: int = 300) -> bytes:
    """Generate a short silence as MP3 bytes using ffmpeg."""
    if not shutil.which('ffmpeg'):
        # Return empty bytes if ffmpeg not available
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


# Cached silence
_silence_bytes = None


def _get_silence() -> bytes:
    """Get cached 0.3s silence MP3."""
    global _silence_bytes
    if _silence_bytes is None:
        _silence_bytes = _generate_silence_mp3(300)
    return _silence_bytes


# ---------------------------------------------------------------------------
# Chunk-level audio generation (Chinese -> Target -> Chinese)
# ---------------------------------------------------------------------------

async def _generate_chunk_audio(
    chunk: str,
    source_lang: str,
    target_lang: str | None = None,
    translation: str | None = None,
    tts_semaphore: asyncio.Semaphore | None = None,
) -> bytes:
    """
    Generate audio for one text chunk with triple narration pattern.

    Pattern: Chinese -> Target language -> Chinese
    With 0.3s silence gaps between segments.
    """
    parts = []
    silence = _get_silence()

    async def _synth(text, lang):
        if tts_semaphore:
            async with tts_semaphore:
                return await _synthesize(text, lang)
        return await _synthesize(text, lang)

    # 1. Chinese audio
    try:
        zh_audio = await _synth(chunk, source_lang)
        parts.append(zh_audio)
    except Exception as e:
        logger.warning(f'Chinese TTS failed for "{chunk[:30]}...": {e}')
        return b''

    # 2. Target language audio (if bilingual mode)
    if target_lang:
        if not translation or not translation.strip():
            logger.warning(f'Empty translation for chunk "{chunk[:30]}..." — using Chinese-only for this chunk')
            # Still append silence + repeat Chinese so rhythm stays consistent
            if silence:
                parts.append(silence)
            # Skip target audio but still repeat Chinese
            if silence:
                parts.append(silence)
            try:
                zh_audio_2 = await _synth(chunk, source_lang)
                parts.append(zh_audio_2)
            except Exception as e:
                logger.warning(f'Chinese TTS (repeat) failed: {e}')
        else:
            # Normal triple pattern: ZH → silence → Target → silence → ZH
            if silence:
                parts.append(silence)
            try:
                target_audio = await _synth(translation, target_lang)
                parts.append(target_audio)
            except Exception as e:
                logger.warning(f'Target TTS failed: {e}')
            if silence:
                parts.append(silence)
            try:
                zh_audio_2 = await _synth(chunk, source_lang)
                parts.append(zh_audio_2)
            except Exception as e:
                logger.warning(f'Chinese TTS (repeat) failed: {e}')

    return b''.join(parts)


# ---------------------------------------------------------------------------
# M4B packaging (chapter-marked audiobook)
# ---------------------------------------------------------------------------

def _has_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    return shutil.which('ffmpeg') is not None


def _build_m4b(
    chapter_files: list[tuple[str, Path]],
    output_path: str,
    cover_image: bytes | None = None,
) -> str:
    """
    Combine chapter audio files into a single M4B with chapter markers.

    Args:
        chapter_files: List of (chapter_title, audio_file_path) tuples.
        output_path: Destination .m4b path.
        cover_image: Optional cover image bytes to embed.

    Returns:
        Path to the generated M4B file.
    """
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

        # Write chapter metadata
        meta_file = tmpdir / 'chapters.txt'
        meta_lines = [';FFMETADATA1']
        for i, (title, start_ms) in enumerate(chapters_meta):
            end_ms = (
                chapters_meta[i + 1][1] if i + 1 < len(chapters_meta)
                else cumulative_ms
            )
            meta_lines.extend([
                '',
                '[CHAPTER]',
                'TIMEBASE=1/1000',
                f'START={start_ms}',
                f'END={end_ms}',
                f'title={title}',
            ])
        meta_file.write_text('\n'.join(meta_lines))

        # Build ffmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0', '-i', str(concat_file),
            '-i', str(meta_file),
        ]

        # Add cover image if available
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
# Main entry point — 5-phase parallel pipeline
# ---------------------------------------------------------------------------

def generate_audiobook(
    epub_path: str,
    output_path: str,
    translation_target: str = 'fr',
    translation_source: str = 'zh-CN',
    bilingual: bool = True,
    llm_model: str | None = None,
    target_languages: list[str] | None = None,
    simplify_hsk4: bool = False,
    hsk_level: str = '4',
    max_sentence_words: int = 0,
    simplify_model: str | None = None,
    chapter_start: int = 0,
    chapter_count: int = 0,
    lang_start_index: int = 0,
) -> str:
    """
    Generate an audiobook from a Chinese EPUB with parallel processing.

    5-phase pipeline:
      1. Extract & split all chapters into chunks
      2. Parallel simplification (all chunks at once)
      3. Parallel translation (all chunks at once)
      4. Parallel TTS (all chapters concurrently)
      5. Assemble M4B with chapters, cover art

    Args:
        epub_path: Path to the input EPUB file.
        output_path: Path for the output file (.m4b or .zip).
        translation_target: Default target language code.
        translation_source: Source language code.
        bilingual: Enable bilingual narration (Chinese -> Target -> Chinese).
        llm_model: Model ID for LLM translation, or None for Google Translate.
        target_languages: List of language codes to rotate through chapters.
            Defaults to [translation_target].
        simplify_hsk4: Simplify Chinese text before TTS.
        hsk_level: HSK level for simplification (e.g., '4', '4-5').
        max_sentence_words: Max words per sentence for simplification (0=no limit).
        simplify_model: Model ID for simplification (defaults to llm_model).

    Returns:
        Path to the generated audiobook file.
    """
    if target_languages is None:
        target_languages = [translation_target]

    # -----------------------------------------------------------------------
    # Phase 1: Extract chapters and split into chunks
    # -----------------------------------------------------------------------
    logger.info(f'Extracting chapters from {epub_path}...')
    chapters = _extract_chapters(epub_path)
    logger.info(f'Found {len(chapters)} chapters')

    if not chapters:
        raise ValueError('No Chinese chapters found in the EPUB')

    # Slice chapters if chapter_start/chapter_count specified
    if chapter_start > 0 or chapter_count > 0:
        end = chapter_start + chapter_count if chapter_count > 0 else len(chapters)
        logger.info(f'Selecting chapters {chapter_start} to {end - 1} (0-based)')
        chapters = chapters[chapter_start:end]
        if not chapters:
            raise ValueError(f'No chapters in range [{chapter_start}:{end}]')
        logger.info(f'Processing {len(chapters)} chapters')

    # Extract cover image and metadata
    cover_image = _extract_cover_image(epub_path)
    if cover_image:
        logger.info(f'Cover image extracted ({len(cover_image) / 1024:.0f} KB)')
    else:
        logger.info('No cover image found')

    # Assign target language per chapter (rotating with offset)
    chapter_langs = []
    for i in range(len(chapters)):
        lang = target_languages[(lang_start_index + i) % len(target_languages)]
        lang_name = _LANG_NAMES.get(lang, lang)
        chapter_langs.append(lang)
        logger.info(f'Chapter {i + 1}: {chapters[i][0]} [{lang_name}]')

    # Split paragraphs into chunks
    # all_chunks[ch_idx] = list of chunk strings
    all_chunks: list[list[str]] = []
    total_chunks = 0
    for title, paragraphs in chapters:
        chapter_chunks = []
        for para in paragraphs:
            sub_chunks = split_sentences(para)
            if not sub_chunks:
                sub_chunks = [para]  # fallback for non-sentence text
            chapter_chunks.extend(sub_chunks)
        all_chunks.append(chapter_chunks)
        total_chunks += len(chapter_chunks)

    logger.info(f'Total chunks across all chapters: {total_chunks}')

    # -----------------------------------------------------------------------
    # Phase 2: Parallel simplification (if enabled)
    # -----------------------------------------------------------------------
    if simplify_hsk4:
        from .llm_simplifier import simplify_to_hsk4

        s_model = simplify_model or llm_model
        logger.info(f'Phase 2: Simplifying {total_chunks} chunks (model={s_model})...')

        flat_chunks = [(ch_idx, ck_idx, chunk)
                       for ch_idx, ch_chunks in enumerate(all_chunks)
                       for ck_idx, chunk in enumerate(ch_chunks)]

        def _safe_simplify(item):
            ch_idx, ck_idx, text = item
            try:
                result = simplify_to_hsk4(
                    text, model=s_model,
                    hsk_level=hsk_level,
                    max_sentence_words=max_sentence_words,
                )
                if result and not result.startswith('['):
                    return (ch_idx, ck_idx, result)
            except Exception as e:
                logger.warning(f'Simplification failed: {e}')
            return (ch_idx, ck_idx, text)

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(_safe_simplify, flat_chunks))

        # Put results back
        for ch_idx, ck_idx, text in results:
            all_chunks[ch_idx][ck_idx] = text

        logger.info('Phase 2: Simplification complete')
    else:
        logger.info('Phase 2: Simplification skipped')

    # -----------------------------------------------------------------------
    # Phase 3: Parallel translation (if bilingual)
    # -----------------------------------------------------------------------
    # all_translations[ch_idx] = list of translation strings (or None)
    all_translations: list[list[str] | None] = [None] * len(chapters)

    if bilingual:
        # Check which languages have TTS voices available
        for i, lang in enumerate(chapter_langs):
            has_voice = (
                lang in _EDGE_VOICES
                or lang in _QWEN3_LANGS
                or (_is_kokoro_available() and lang in _KOKORO_VOICES)
            )
            if not has_voice:
                logger.warning(
                    f'No TTS voice for {lang}, chapter {i+1} will be Chinese-only'
                )

        # Build flat list of (ch_idx, ck_idx, text, target_lang) to translate
        flat_translate = []
        for ch_idx, ch_chunks in enumerate(all_chunks):
            target = chapter_langs[ch_idx]
            for ck_idx, chunk in enumerate(ch_chunks):
                flat_translate.append((ch_idx, ck_idx, chunk, target))

        if llm_model:
            from .llm_translator import translate_text_llm

            def _safe_translate(item):
                ch_idx, ck_idx, text, target = item
                try:
                    result = translate_text_llm(
                        text, source=translation_source, target=target,
                        model=llm_model,
                    )
                    if result and not result.startswith('['):
                        return (ch_idx, ck_idx, result)
                except Exception as e:
                    logger.warning(f'Translation failed: {e}')
                return (ch_idx, ck_idx, '')
        else:
            from .translator import translate_text

            def _safe_translate(item):
                ch_idx, ck_idx, text, target = item
                try:
                    result = translate_text(
                        text, source=translation_source, target=target,
                    )
                    if result and not result.startswith('['):
                        return (ch_idx, ck_idx, result)
                except Exception as e:
                    logger.warning(f'Translation failed: {e}')
                return (ch_idx, ck_idx, '')

        logger.info(f'Phase 3: Translating {len(flat_translate)} chunks...')

        with ThreadPoolExecutor(max_workers=10) as executor:
            trans_results = list(executor.map(_safe_translate, flat_translate))

        # Organize translations by chapter
        for ch_idx in range(len(chapters)):
            all_translations[ch_idx] = [''] * len(all_chunks[ch_idx])

        for ch_idx, ck_idx, text in trans_results:
            all_translations[ch_idx][ck_idx] = text

        translated_count = sum(1 for _, _, t in trans_results if t)
        logger.info(f'Phase 3: Translation complete ({translated_count}/{len(flat_translate)} succeeded)')
    else:
        logger.info('Phase 3: Translation skipped (monolingual mode)')

    # -----------------------------------------------------------------------
    # Phase 4: Parallel TTS (all chapters concurrently)
    # -----------------------------------------------------------------------
    logger.info('Phase 4: Generating TTS audio...')

    # Log TTS engine status
    if not _qwen3_disabled and any(l in _QWEN3_LANGS for l in chapter_langs + [translation_source]):
        logger.info('TTS engines: Qwen3-TTS (primary) + edge-tts (fallback)')
    else:
        logger.info('TTS engine: edge-tts')
    if _is_kokoro_available():
        logger.info('Kokoro available as secondary fallback')

    tts_semaphore = asyncio.Semaphore(10)

    async def _generate_chapter(ch_idx: int) -> tuple[int, bytes]:
        """Generate all audio for one chapter."""
        title = chapters[ch_idx][0]
        ch_chunks = all_chunks[ch_idx]
        ch_trans = all_translations[ch_idx]
        target_lang = chapter_langs[ch_idx] if bilingual else None
        lang_name = _LANG_NAMES.get(target_lang, target_lang) if target_lang else None

        logger.info(
            f'TTS Chapter {ch_idx + 1}/{len(chapters)}: {title} '
            f'[{lang_name or "Chinese only"}] ({len(ch_chunks)} chunks)'
        )

        parts = []
        for ck_idx, chunk in enumerate(ch_chunks):
            trans = ch_trans[ck_idx] if ch_trans else None
            audio = await _generate_chunk_audio(
                chunk, translation_source,
                target_lang=target_lang,
                translation=trans,
                tts_semaphore=tts_semaphore,
            )
            if audio:
                parts.append(audio)

        return (ch_idx, b''.join(parts))

    async def _run_all_tts():
        tasks = [_generate_chapter(i) for i in range(len(chapters))]
        return await asyncio.gather(*tasks)

    chapter_audio_results = asyncio.run(_run_all_tts())

    # Sort by chapter index
    chapter_audio_results = sorted(chapter_audio_results, key=lambda x: x[0])

    logger.info('Phase 4: TTS complete')

    # -----------------------------------------------------------------------
    # Phase 4b: Translate chapter titles to English
    # -----------------------------------------------------------------------
    english_titles = {}
    if llm_model:
        from .llm_translator import translate_text_llm
        logger.info('Translating chapter titles to English...')
        for ch_idx, (title, _) in enumerate(chapters):
            if contains_chinese(title):
                try:
                    en_title = translate_text_llm(
                        title, source=translation_source, target='English',
                        model=llm_model,
                    )
                    if en_title and not en_title.startswith('['):
                        english_titles[ch_idx] = en_title
                except Exception as e:
                    logger.warning(f'Title translation failed for chapter {ch_idx}: {e}')
    # Fallback: use pypinyin transliteration
    if not english_titles:
        try:
            from pypinyin import pinyin as _pinyin, Style as _Style
            for ch_idx, (title, _) in enumerate(chapters):
                if contains_chinese(title):
                    py = _pinyin(title, style=_Style.NORMAL)
                    english_titles[ch_idx] = ' '.join(p[0] for p in py).title()
        except ImportError:
            pass

    # -----------------------------------------------------------------------
    # Phase 5: Assemble output
    # -----------------------------------------------------------------------
    logger.info('Phase 5: Assembling audiobook...')

    use_m4b = _has_ffmpeg()
    if use_m4b:
        output_path = str(Path(output_path).with_suffix('.m4b'))
        logger.info('Output format: M4B audiobook with chapter markers')
    else:
        output_path = str(Path(output_path).with_suffix('.zip'))
        logger.info('Output format: ZIP of MP3s (install ffmpeg for M4B with chapters)')

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        chapter_files = []

        zip_file = None if use_m4b else zipfile.ZipFile(output_path, 'w', zipfile.ZIP_STORED)

        try:
            for ch_idx, audio_data in chapter_audio_results:
                if not audio_data:
                    continue

                title = english_titles.get(ch_idx, chapters[ch_idx][0])
                lang = chapter_langs[ch_idx]
                lang_name = _LANG_NAMES.get(lang, lang)

                # Add language label to chapter title
                display_title = f'{title} [{lang_name}]' if bilingual else title

                safe_title = re.sub(r'[^\w\s-]', '', title)[:50].strip()
                safe_title = safe_title or f'chapter_{ch_idx + 1}'
                filename = f'{ch_idx + 1:02d}_{safe_title}.mp3'

                if use_m4b:
                    chapter_path = tmpdir_path / filename
                    chapter_path.write_bytes(audio_data)
                    chapter_files.append((display_title, chapter_path))
                    logger.info(f'  -> {display_title} ({len(audio_data) / 1024:.0f} KB)')
                else:
                    zip_file.writestr(filename, audio_data)
                    logger.info(f'  -> {filename} ({len(audio_data) / 1024:.0f} KB)')

        finally:
            if zip_file:
                zip_file.close()

        if use_m4b and chapter_files:
            logger.info('Assembling M4B with chapter markers...')
            _build_m4b(chapter_files, output_path, cover_image=cover_image)

    logger.info(f'Audiobook written to {output_path}')
    return output_path
