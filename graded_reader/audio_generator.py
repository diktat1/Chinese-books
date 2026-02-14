"""
Audiobook generator from Chinese EPUB books.

Generates chapter-by-chapter audio with optional bilingual narration
(target language paragraph followed by Chinese paragraph).

TTS engines (tried in order):
  1. edge-tts — Microsoft Edge neural voices, free, best quality
  2. kokoro   — Local 82M-param model, Apache 2.0, CPU-friendly fallback

Output formats:
  - M4B audiobook with chapter markers (plays in Apple Books, Audible, etc.)
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
from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import epub

from .chinese_processing import contains_chinese

logger = logging.getLogger(__name__)

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

_ALL_BLOCKS = [
    'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'li', 'td', 'th', 'blockquote', 'dt', 'dd',
    'figcaption', 'pre', 'div', 'section', 'article',
    'aside', 'header', 'footer', 'caption',
]


# ---------------------------------------------------------------------------
# EPUB chapter extraction (spine-ordered)
# ---------------------------------------------------------------------------

def _get_spine_items(book: epub.EpubBook) -> list:
    """Return document items in spine (reading) order, not manifest order."""
    items_by_id = {}
    for item in book.get_items():
        items_by_id[item.get_id()] = item
        # Also index by filename for fallback matching
        items_by_id[item.get_name()] = item

    spine_items = []
    for entry in book.spine:
        item_id = entry[0] if isinstance(entry, tuple) else entry
        item = items_by_id.get(item_id)
        if item and item.get_type() == 9:  # ITEM_DOCUMENT
            spine_items.append(item)

    # Fallback: if spine resolution fails, use manifest order
    if not spine_items:
        logger.warning('Could not resolve spine order, falling back to manifest order')
        spine_items = list(book.get_items_of_type(9))

    return spine_items


def _extract_chapters(epub_path: str) -> list[tuple[str, list[str]]]:
    """
    Extract chapters from EPUB as (title, [paragraphs]) in reading order.

    Uses the EPUB spine to determine chapter order, and the same
    block-detection logic as epub_processor to avoid duplicating text.
    """
    book = epub.read_epub(epub_path)
    chapters = []

    for item in _get_spine_items(book):
        html = item.get_content().decode('utf-8', errors='replace')
        soup = BeautifulSoup(html, 'lxml')

        # Extract chapter title from headings
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

    Tries edge-tts first (best quality), falls back to Kokoro (local CPU).
    """
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
# Chapter audio generation
# ---------------------------------------------------------------------------

async def _generate_chapter_audio(
    paragraphs: list[str],
    source_lang: str,
    target_lang: str | None = None,
    translations: list[str] | None = None,
) -> bytes:
    """
    Generate audio for one chapter by concatenating paragraph segments.

    If bilingual: narrates each paragraph in the target language first,
    then in Chinese, giving the reader comprehension before hearing
    the original.
    """
    parts = []

    for i, para in enumerate(paragraphs):
        # Bilingual: target language first
        if target_lang and translations and i < len(translations):
            trans = translations[i]
            if trans:
                try:
                    target_audio = await _synthesize(trans, target_lang)
                    parts.append(target_audio)
                except Exception as e:
                    logger.warning(f'Target TTS failed: {e}')

        # Chinese audio
        try:
            chinese_audio = await _synthesize(para, source_lang)
            parts.append(chinese_audio)
        except Exception as e:
            logger.warning(f'Chinese TTS failed for "{para[:30]}...": {e}')

    return b''.join(parts)


# ---------------------------------------------------------------------------
# M4B packaging (chapter-marked audiobook)
# ---------------------------------------------------------------------------

def _has_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    return shutil.which('ffmpeg') is not None


def _build_m4b(chapter_files: list[tuple[str, Path]], output_path: str) -> str:
    """
    Combine chapter audio files into a single M4B with chapter markers.

    Args:
        chapter_files: List of (chapter_title, audio_file_path) tuples.
        output_path: Destination .m4b path.

    Returns:
        Path to the generated M4B file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Build ffmpeg concat file and chapter metadata
        concat_file = tmpdir / 'concat.txt'
        chapters_meta = []
        cumulative_ms = 0

        # Probe durations and write concat list
        concat_lines = []
        for title, audio_path in chapter_files:
            # Get duration via ffprobe
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

            # Escape special chars for ffmpeg concat
            escaped = str(audio_path).replace("'", "'\\''")
            concat_lines.append(f"file '{escaped}'")

        concat_file.write_text('\n'.join(concat_lines))

        # Write ffmpeg chapter metadata file
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

        # Concatenate and convert to M4B (AAC in MP4 container)
        subprocess.run(
            [
                'ffmpeg', '-y',
                '-f', 'concat', '-safe', '0', '-i', str(concat_file),
                '-i', str(meta_file),
                '-map_metadata', '1',
                '-c:a', 'aac', '-b:a', '64k',
                '-movflags', '+faststart',
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )

    return output_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_audiobook(
    epub_path: str,
    output_path: str,
    translation_target: str = 'fr',
    translation_source: str = 'zh-CN',
    bilingual: bool = True,
    use_claude: bool = False,
    use_opus: bool = False,
) -> str:
    """
    Generate an audiobook from a Chinese EPUB.

    Produces an M4B file with chapter markers (plays in Apple Books, etc.)
    when ffmpeg is available, otherwise falls back to a ZIP of MP3s.

    Args:
        epub_path: Path to the input EPUB file.
        output_path: Path for the output file (.m4b or .zip).
        translation_target: Target language code for bilingual narration.
        translation_source: Source language code.
        bilingual: Narrate in target language then Chinese per paragraph.
        use_claude: Use Claude for translation instead of Google Translate.
        use_opus: Use Claude Opus model.

    Returns:
        Path to the generated audiobook file.
    """
    has_target_voice = (
        translation_target in _EDGE_VOICES
        or (_is_kokoro_available() and translation_target in _KOKORO_VOICES)
    )

    if bilingual and not has_target_voice:
        logger.warning(
            f'No voice available for {translation_target}, '
            f'falling back to Chinese-only audio'
        )
        bilingual = False

    target_lang = translation_target if bilingual else None

    # Set up translation for bilingual mode
    translate = None
    if bilingual:
        if use_claude:
            from .claude_translator import translate_text_claude

            def translate(text):
                return translate_text_claude(
                    text, source=translation_source, target=translation_target,
                    use_opus=use_opus,
                )
        else:
            from .translator import translate_text

            def translate(text):
                return translate_text(
                    text, source=translation_source, target=translation_target,
                )

    # Extract chapters in spine (reading) order
    logger.info(f'Extracting chapters from {epub_path}...')
    chapters = _extract_chapters(epub_path)
    logger.info(f'Found {len(chapters)} chapters')

    if not chapters:
        raise ValueError('No Chinese chapters found in the EPUB')

    # Log TTS engine status
    if _is_kokoro_available():
        logger.info('TTS engines: edge-tts (primary) + Kokoro (fallback)')
    else:
        logger.info('TTS engine: edge-tts (install kokoro + soundfile for local fallback)')

    # Decide output format
    use_m4b = _has_ffmpeg()
    if use_m4b:
        # Ensure output has .m4b extension
        output_path = str(Path(output_path).with_suffix('.m4b'))
        logger.info('Output format: M4B audiobook with chapter markers')
    else:
        output_path = str(Path(output_path).with_suffix('.zip'))
        logger.info('Output format: ZIP of MP3s (install ffmpeg for M4B with chapters)')

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        chapter_files = []  # (title, file_path) for M4B assembly

        # Also open a ZIP in case we need the fallback path
        zip_file = None if use_m4b else zipfile.ZipFile(output_path, 'w', zipfile.ZIP_STORED)

        try:
            for ch_idx, (title, paragraphs) in enumerate(chapters):
                logger.info(
                    f'Chapter {ch_idx + 1}/{len(chapters)}: {title} '
                    f'({len(paragraphs)} paragraphs)'
                )

                # Translate paragraphs if bilingual
                translations = None
                if translate:
                    translations = []
                    for para in paragraphs:
                        try:
                            t = translate(para)
                            if t and not t.startswith('['):
                                translations.append(t)
                            else:
                                translations.append('')
                        except Exception as e:
                            logger.warning(f'Translation failed: {e}')
                            translations.append('')

                # Generate audio
                audio_data = asyncio.run(_generate_chapter_audio(
                    paragraphs, translation_source,
                    target_lang=target_lang,
                    translations=translations,
                ))

                if not audio_data:
                    continue

                safe_title = re.sub(r'[^\w\s-]', '', title)[:50].strip()
                safe_title = safe_title or f'chapter_{ch_idx + 1}'
                filename = f'{ch_idx + 1:02d}_{safe_title}.mp3'

                if use_m4b:
                    # Write chapter to temp file for M4B assembly
                    chapter_path = tmpdir_path / filename
                    chapter_path.write_bytes(audio_data)
                    chapter_files.append((title, chapter_path))
                    logger.info(f'  -> {title} ({len(audio_data) / 1024:.0f} KB)')
                else:
                    zip_file.writestr(filename, audio_data)
                    logger.info(f'  -> {filename} ({len(audio_data) / 1024:.0f} KB)')

        finally:
            if zip_file:
                zip_file.close()

        # Assemble M4B
        if use_m4b and chapter_files:
            logger.info('Assembling M4B with chapter markers...')
            _build_m4b(chapter_files, output_path)

    logger.info(f'Audiobook written to {output_path}')
    return output_path
