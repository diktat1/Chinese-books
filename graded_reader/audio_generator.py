"""
Audiobook generator from Chinese EPUB books using edge-tts.

Generates chapter-by-chapter MP3 audio with optional bilingual narration
(target language paragraph followed by Chinese paragraph). Uses Microsoft
Edge's neural TTS voices — free, no API key required, works in CI.

Output: ZIP archive containing one MP3 per chapter.
"""

import asyncio
import io
import logging
import re
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import epub

from .chinese_processing import contains_chinese

logger = logging.getLogger(__name__)

# Best neural voice per language
_VOICES = {
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

_ALL_BLOCKS = [
    'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'li', 'td', 'th', 'blockquote', 'dt', 'dd',
    'figcaption', 'pre', 'div', 'section', 'article',
    'aside', 'header', 'footer', 'caption',
]


def _extract_chapters(epub_path: str) -> list[tuple[str, list[str]]]:
    """
    Extract chapters from EPUB as (title, [paragraphs]).

    Uses the same block-detection logic as epub_processor: skip any block
    that contains nested block children to avoid duplicating text.
    """
    book = epub.read_epub(epub_path)
    chapters = []

    for item in book.get_items_of_type(9):  # ITEM_DOCUMENT
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


async def _synthesize(text: str, voice: str) -> bytes:
    """Synthesize text to MP3 bytes using edge-tts."""
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk['type'] == 'audio':
            buffer.write(chunk['data'])
    return buffer.getvalue()


async def _generate_chapter_audio(
    paragraphs: list[str],
    voice_chinese: str,
    voice_target: str | None = None,
    translations: list[str] | None = None,
) -> bytes:
    """
    Generate audio for one chapter by concatenating paragraph MP3 segments.

    If bilingual: narrates each paragraph in the target language first,
    then in Chinese, giving the reader comprehension before hearing
    the original.
    """
    parts = []

    for i, para in enumerate(paragraphs):
        # Bilingual: target language first
        if voice_target and translations and i < len(translations):
            trans = translations[i]
            if trans:
                try:
                    target_audio = await _synthesize(trans, voice_target)
                    parts.append(target_audio)
                except Exception as e:
                    logger.warning(f'Target TTS failed: {e}')

        # Chinese audio
        try:
            chinese_audio = await _synthesize(para, voice_chinese)
            parts.append(chinese_audio)
        except Exception as e:
            logger.warning(f'Chinese TTS failed for "{para[:30]}...": {e}')

    return b''.join(parts)


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
    Generate an audiobook (ZIP of chapter MP3s) from a Chinese EPUB.

    Args:
        epub_path: Path to the input EPUB file.
        output_path: Path for the output .zip file.
        translation_target: Target language code for bilingual narration.
        translation_source: Source language code.
        bilingual: Narrate in target language then Chinese per paragraph.
        use_claude: Use Claude for translation instead of Google Translate.
        use_opus: Use Claude Opus model.

    Returns:
        Path to the generated .zip file.
    """
    voice_chinese = _VOICES.get(translation_source, _VOICES['zh-CN'])
    voice_target = _VOICES.get(translation_target) if bilingual else None

    if bilingual and not voice_target:
        logger.warning(
            f'No voice available for {translation_target}, '
            f'falling back to Chinese-only audio'
        )
        bilingual = False

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

    # Extract chapters
    logger.info(f'Extracting chapters from {epub_path}...')
    chapters = _extract_chapters(epub_path)
    logger.info(f'Found {len(chapters)} chapters')

    if not chapters:
        raise ValueError('No Chinese chapters found in the EPUB')

    # Generate audio per chapter, package into ZIP
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_STORED) as zf:
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
                paragraphs, voice_chinese,
                voice_target=voice_target,
                translations=translations,
            ))

            if audio_data:
                safe_title = re.sub(r'[^\w\s-]', '', title)[:50].strip()
                safe_title = safe_title or f'chapter_{ch_idx + 1}'
                filename = f'{ch_idx + 1:02d}_{safe_title}.mp3'
                zf.writestr(filename, audio_data)
                logger.info(f'  -> {filename} ({len(audio_data) / 1024:.0f} KB)')

    logger.info(f'Audiobook written to {output_path}')
    return output_path
