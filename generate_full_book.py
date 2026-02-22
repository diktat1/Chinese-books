#!/usr/bin/env python3
"""
Generate a full-book beginner Japanese audiobook + EPUB from a Chinese EPUB.

Processes ALL chapters of the source book through:
  1. EPUB chapter extraction
  2. LLM synthesis: Chinese → JLPT N5 Japanese (via OpenRouter)
  3. N5 vocabulary validation + automatic correction
  4. Edge-TTS audio generation (JP → 0.3s → CN → 0.3s → JP → 0.8s)
  5. Multi-chapter M4B audiobook with chapter markers + cover art
  6. Multi-chapter EPUB with furigana, romaji, and parallel Chinese

Supports resuming — saves per-chapter JSON progress so you can restart
without re-running LLM synthesis for already-completed chapters.

Usage:
    # Set your OpenRouter API key first:
    export OPENROUTER_API_KEY=sk-or-...

    # Process full book with DeepSeek V3 (standard tier, ~$0.10):
    python generate_full_book.py

    # Use a specific model:
    python generate_full_book.py --model deepseek/deepseek-chat

    # Use free tier (rate-limited):
    python generate_full_book.py --tier free

    # Process specific chapter range:
    python generate_full_book.py --chapter-start 3 --chapter-count 2

    # Skip LLM synthesis (audio + EPUB only from existing JSON):
    python generate_full_book.py --audio-only
"""
import argparse
import asyncio
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

EPUB_PATH = 'input-epubs/Only the Paranoid Survive_CN (Original Book as Published).epub'
OUTPUT_DIR = Path('.')
SCRIPT_DIR = Path('beginner_japanese_scripts')  # Per-chapter JSON cache
FULL_SCRIPT_PATH = 'beginner_japanese_full_script.json'
M4B_OUTPUT = 'beginner_japanese_full.m4b'
EPUB_OUTPUT = 'beginner_japanese_full.epub'

# Chapters to skip (front matter / back matter) by title substring
_SKIP_TITLES = {'titlepage', '版权信息', '致谢', 'copyright', 'thanks'}

# Edge-TTS voices
JP_VOICE = 'ja-JP-NanamiNeural'
CN_VOICE = 'zh-CN-XiaoxiaoNeural'


# ---------------------------------------------------------------------------
# Phase 1: Extract chapters from EPUB
# ---------------------------------------------------------------------------

def extract_content_chapters(epub_path: str) -> list[dict]:
    """
    Extract content chapters from the EPUB, skipping front/back matter.

    Returns list of dicts: [{'index': int, 'title': str, 'paragraphs': [str]}]
    """
    from graded_reader.audio_generator import _extract_chapters

    all_chapters = _extract_chapters(epub_path)
    content_chapters = []

    for i, (title, paragraphs) in enumerate(all_chapters):
        # Skip front/back matter
        title_lower = title.lower().strip()
        if any(skip in title_lower for skip in _SKIP_TITLES):
            logger.info(f'  Skipping: "{title}" (front/back matter)')
            continue
        # Skip very short chapters (< 5 paragraphs)
        if len(paragraphs) < 5:
            logger.info(f'  Skipping: "{title}" ({len(paragraphs)} paragraphs — too short)')
            continue

        content_chapters.append({
            'index': i,
            'title': title,
            'paragraphs': paragraphs,
        })

    return content_chapters


# ---------------------------------------------------------------------------
# Phase 2: LLM synthesis with per-chapter caching
# ---------------------------------------------------------------------------

def load_cached_chapter(chapter_num: int) -> dict | None:
    """Load a previously synthesized chapter from the cache directory."""
    cache_path = SCRIPT_DIR / f'chapter_{chapter_num:02d}.json'
    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if data.get('sentence_pairs'):
                return data
    return None


def save_chapter_cache(chapter_num: int, data: dict):
    """Save synthesized chapter data to cache."""
    SCRIPT_DIR.mkdir(exist_ok=True)
    cache_path = SCRIPT_DIR / f'chapter_{chapter_num:02d}.json'
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_existing_ch1_script() -> dict | None:
    """Load the existing Ch1 script (127 validated pairs from generate_script.py)."""
    path = Path('beginner_japanese_script.json')
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def _synthesize_segment(seg_idx, total_segments, segment, model):
    """Synthesize one segment via LLM (for parallel execution)."""
    from graded_reader.beginner_japanese import (
        _call_llm,
        _SYNTHESIS_SYSTEM_PROMPT,
        _parse_sentence_pairs,
    )

    user_prompt = (
        "Transform this Chinese text into JLPT N5 beginner Japanese vocabulary drill sentences.\n"
        "Follow the ideas in this text, but express them using ONLY N5 vocabulary and A1-A2 grammar.\n"
        "Repeat key words many times across sentences.\n\n"
        f"Chinese text:\n{segment}\n\n"
        "Remember: Output ONLY numbered JP/CN pairs. At least 15 pairs for this segment. Use only N5 vocabulary."
    )

    try:
        response = _call_llm(
            _SYNTHESIS_SYSTEM_PROMPT, user_prompt, model,
            max_tokens=4000, max_retries=3,
        )
        pairs = _parse_sentence_pairs(response)
        if pairs:
            logger.info(f'  Segment {seg_idx + 1}/{total_segments}: {len(pairs)} pairs')
            return pairs
        else:
            logger.warning(f'  Segment {seg_idx + 1}/{total_segments}: No pairs parsed')
            return []
    except Exception as e:
        logger.error(f'  Segment {seg_idx + 1}/{total_segments} failed: {e}')
        return []


def synthesize_chapter(
    chapter_num: int,
    title: str,
    paragraphs: list[str],
    model: str,
    max_workers: int = 5,
) -> dict:
    """
    Synthesize beginner Japanese for one chapter via LLM.
    Uses parallel segment processing for speed.

    Returns dict with 'chapter_title', 'sentence_pairs', 'validation'.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from graded_reader.beginner_japanese import validate_and_correct

    # Group paragraphs into segments of 3
    segments = []
    for i in range(0, len(paragraphs), 3):
        segment = '\n'.join(paragraphs[i:i + 3])
        segments.append(segment)

    logger.info(f'  Synthesizing N5 Japanese ({len(paragraphs)} paragraphs, {len(segments)} segments, {max_workers} workers)...')

    # Process segments in parallel
    all_pairs = [None] * len(segments)  # Maintain order
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_synthesize_segment, i, len(segments), seg, model): i
            for i, seg in enumerate(segments)
        }
        for future in as_completed(futures):
            idx = futures[future]
            all_pairs[idx] = future.result()

    # Flatten in order
    pairs = []
    for segment_pairs in all_pairs:
        if segment_pairs:
            pairs.extend(segment_pairs)

    if not pairs:
        logger.error(f'  Chapter {chapter_num}: LLM returned no pairs')
        return {
            'chapter_title': title,
            'sentence_pairs': [],
            'validation': {'is_valid': False, 'violation_count': -1},
        }

    logger.info(f'  Got {len(pairs)} pairs, validating N5 compliance...')
    pairs, report = validate_and_correct(pairs, model=model)

    sentence_pairs = [{'japanese': jp, 'chinese': cn} for jp, cn in pairs]

    data = {
        'chapter_title': title,
        'chapter_num': chapter_num,
        'model': model,
        'sentence_pairs': sentence_pairs,
        'validation': {
            'total_tokens': report.get('total_tokens', 0),
            'violation_count': report.get('violation_count', 0),
            'violations': [v['word'] for v in report.get('violations', [])][:30],
        },
    }

    save_chapter_cache(chapter_num, data)
    return data


# ---------------------------------------------------------------------------
# Phase 3: Audio generation (edge-tts)
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
        raise RuntimeError(f'edge-tts returned empty audio for: {text[:30]}')
    return result


def _generate_silence_mp3(duration_ms: int) -> bytes:
    """Generate silence as MP3 using ffmpeg."""
    result = subprocess.run(
        [
            'ffmpeg', '-y', '-f', 'lavfi',
            '-i', 'anullsrc=r=24000:cl=mono',
            '-t', str(duration_ms / 1000),
            '-c:a', 'libmp3lame', '-b:a', '32k',
            '-f', 'mp3', 'pipe:1',
        ],
        capture_output=True,
    )
    if result.returncode == 0:
        return result.stdout
    return b''


async def generate_chapter_audio(
    pairs: list[dict],
    chapter_num: int,
    total_chapters: int,
    semaphore: asyncio.Semaphore,
    silence_short: bytes,
    silence_long: bytes,
) -> bytes:
    """Generate audio for all sentence pairs in a chapter."""
    total = len(pairs)
    logger.info(f'  Chapter {chapter_num + 1}/{total_chapters}: {total} sentences')

    async def synth(text: str, voice: str) -> bytes:
        async with semaphore:
            return await _synthesize_edge(text, voice)

    chapter_parts = []
    succeeded = 0

    for i, pair in enumerate(pairs):
        jp = pair['japanese']
        cn = pair['chinese']
        parts = []

        try:
            # JP → 0.3s → CN → 0.3s → JP → 0.8s
            jp1 = await synth(jp, JP_VOICE)
            parts.append(jp1)

            if silence_short:
                parts.append(silence_short)

            cn_audio = await synth(cn, CN_VOICE)
            parts.append(cn_audio)

            if silence_short:
                parts.append(silence_short)

            jp2 = await synth(jp, JP_VOICE)
            parts.append(jp2)

            if silence_long:
                parts.append(silence_long)

            chapter_parts.append(b''.join(parts))
            succeeded += 1

            if (i + 1) % 20 == 0 or i == total - 1:
                logger.info(f'    [{i + 1}/{total}] {succeeded} OK')

        except Exception as e:
            logger.warning(f'    [{i + 1}/{total}] FAIL: {jp[:30]}... -> {e}')
            chapter_parts.append(b'')

    failed = total - succeeded
    if failed:
        logger.warning(f'  Chapter {chapter_num + 1}: {failed} sentences failed TTS')

    return b''.join(chapter_parts)


# ---------------------------------------------------------------------------
# Phase 4: M4B assembly (multi-chapter)
# ---------------------------------------------------------------------------

def build_multi_chapter_m4b(
    chapter_audio_files: list[tuple[str, Path]],
    output_path: str,
    cover_image: bytes | None = None,
):
    """Combine chapter MP3s into a single M4B with chapter markers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        concat_file = tmpdir / 'concat.txt'
        chapters_meta = []
        cumulative_ms = 0
        concat_lines = []

        for title, audio_path in chapter_audio_files:
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json',
                 '-show_format', str(audio_path)],
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
                '', '[CHAPTER]', 'TIMEBASE=1/1000',
                f'START={start_ms}', f'END={end_ms}', f'title={title}',
            ])
        meta_file.write_text('\n'.join(meta_lines))

        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0', '-i', str(concat_file),
            '-i', str(meta_file),
        ]

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

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f'ffmpeg error: {result.stderr[:500]}')
            raise RuntimeError('M4B assembly failed')

    logger.info(f'M4B written: {output_path}')


# ---------------------------------------------------------------------------
# Phase 5: EPUB generation (multi-chapter with furigana)
# ---------------------------------------------------------------------------

def build_full_epub(
    chapters_data: list[dict],
    source_book: str,
    output_path: str,
):
    """Build a multi-chapter EPUB with furigana, romaji, and Chinese."""
    import pykakasi
    from ebooklib import epub

    kks = pykakasi.kakasi()

    def add_furigana_html(text: str) -> tuple[str, str]:
        items = kks.convert(text)
        ruby_parts = []
        romaji_parts = []
        _PUNCT = set('。、！？「」（）．，.!?,')
        for item in items:
            orig = item['orig']
            hira = item['hira']
            hepburn = item['hepburn']
            has_kanji = any('\u4e00' <= c <= '\u9fff' for c in orig)
            is_punct = all(c in _PUNCT or c.isspace() for c in orig)
            if has_kanji and hira != orig:
                ruby_parts.append(
                    f'<ruby>{orig}<rp>(</rp><rt>{hira}</rt><rp>)</rp></ruby>'
                )
            else:
                ruby_parts.append(orig)
            # Add word boundary space (thin space for Japanese readability)
            if not is_punct:
                ruby_parts.append(' ')
            if hepburn and hepburn not in ('.', ',', '!', '?'):
                romaji_parts.append(hepburn)
        return ''.join(ruby_parts).strip(), ' '.join(romaji_parts)

    # CSS (same as generate_epub.py)
    EPUB_CSS = '''
body {
    font-family: "Hiragino Kaku Gothic ProN", "Noto Sans CJK JP", "Yu Gothic", sans-serif;
    line-height: 2.2;
    margin: 1em;
    color: #333;
}
h1 {
    font-size: 1.4em;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    color: #222;
    border-bottom: 2px solid #c0392b;
    padding-bottom: 0.3em;
}
h2 {
    font-size: 1.15em;
    margin-top: 1.8em;
    margin-bottom: 0.4em;
    color: #555;
    border-bottom: 1px solid #ddd;
    padding-bottom: 0.2em;
}
ruby { ruby-align: center; -epub-ruby-position: over; ruby-position: over; }
rt { font-size: 0.5em; color: #c0392b; ruby-align: center; }
rp { display: none; }
.sentence-block {
    margin-bottom: 1.8em;
    padding: 0.6em 0.8em;
    border-left: 3px solid #3498db;
    background-color: #f9f9f9;
    border-radius: 0 4px 4px 0;
}
.sentence-num { font-size: 0.75em; color: #999; font-weight: bold; margin-right: 0.3em; }
.jp-line { font-size: 1.15em; line-height: 2.4; margin-bottom: 0.3em; color: #222; }
.romaji-line { font-size: 0.8em; color: #888; font-style: italic; line-height: 1.4; margin-bottom: 0.3em; }
.cn-line { font-size: 0.95em; color: #555; line-height: 1.6; padding-left: 0.3em; border-left: 2px solid #e74c3c; margin-left: 0.2em; }
.title-page { text-align: center; margin-top: 3em; }
.title-page h1 { font-size: 1.6em; border: none; color: #c0392b; }
.title-page .subtitle { font-size: 1.0em; color: #666; margin-top: 0.5em; }
.title-page .meta { font-size: 0.85em; color: #999; margin-top: 2em; line-height: 1.8; }
.how-to-use { margin: 1em 0; padding: 0.8em; background-color: #f0f7ff; border: 1px solid #b3d4fc; border-radius: 4px; font-size: 0.9em; line-height: 1.8; }
.how-to-use h2 { border: none; color: #2c3e50; margin-top: 0; }
'''

    total_pairs = sum(len(ch['sentence_pairs']) for ch in chapters_data)

    book = epub.EpubBook()
    book.set_identifier('beginner-japanese-n5-full-book')
    book.set_title('初級日本語ドリル — 只有偏执狂才能生存')
    book.set_language('ja')
    book.add_author(f'Generated from: {source_book}')
    book.add_metadata('DC', 'subject', 'JLPT N5')
    book.add_metadata('DC', 'subject', 'Japanese Language Learning')
    book.add_metadata('DC', 'description',
                      'Full-book beginner Japanese vocabulary drill with furigana, '
                      'romaji, and parallel Chinese translation. JLPT N5 level.')

    css_item = epub.EpubItem(
        uid='style', file_name='style/default.css',
        media_type='text/css', content=EPUB_CSS.encode('utf-8'),
    )
    book.add_item(css_item)

    # Title page
    title_html = f'''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="ja">
<head><title>初級日本語ドリル</title></head>
<body>
<div class="title-page">
    <h1>初級日本語 ドリル</h1>
    <p class="subtitle">Beginner Japanese Vocabulary Drill</p>
    <p class="subtitle">初级日语词汇练习</p>
    <p class="subtitle" style="margin-top:1.5em; font-size:0.9em;">
        Full Book: {source_book}
    </p>
    <div class="meta">
        <p>Level: JLPT N5 (A1-A2)</p>
        <p>{total_pairs} sentence pairs across {len(chapters_data)} chapters</p>
        <p>Each sentence: furigana + romaji + Chinese translation</p>
    </div>
</div>
<div class="how-to-use">
    <h2>How to Use This Book / この本の使い方</h2>
    <p>Each sentence is shown in three forms:</p>
    <p><strong>1.</strong> Japanese with furigana (reading above kanji) — 日本語（ふりがな付き）</p>
    <p><strong>2.</strong> Romaji (Latin alphabet reading) — ローマ字</p>
    <p><strong>3.</strong> Chinese translation — 中文翻译</p>
    <p style="margin-top:0.5em;">All vocabulary is JLPT N5 level. Read the Japanese first,
    check the furigana if needed, then confirm meaning with the Chinese.</p>
    <p>所有词汇都是JLPT N5级别。先读日语，需要时查看假名注音，然后用中文确认意思。</p>
</div>
</body>
</html>'''

    title_page = epub.EpubHtml(
        title='Title Page', file_name='title.xhtml', lang='ja',
    )
    title_page.content = title_html.encode('utf-8')
    title_page.add_item(css_item)
    book.add_item(title_page)

    spine = ['nav', title_page]
    toc = []
    global_sentence_num = 0

    for ch_idx, ch_data in enumerate(chapters_data):
        ch_title = ch_data['chapter_title']
        pairs = ch_data['sentence_pairs']

        if not pairs:
            continue

        # Build sentence blocks
        blocks = []
        for i, pair in enumerate(pairs):
            jp = pair['japanese']
            cn = pair['chinese']
            global_sentence_num += 1

            ruby_html, romaji = add_furigana_html(jp)
            blocks.append(f'''<div class="sentence-block">
    <p class="jp-line"><span class="sentence-num">{global_sentence_num}.</span> {ruby_html}</p>
    <p class="romaji-line">{romaji}</p>
    <p class="cn-line">{cn}</p>
</div>''')

        body = '\n'.join(blocks)
        chapter_html = f'''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="ja">
<head><title>Ch.{ch_idx + 1}: {ch_title}</title></head>
<body>
<h1>Ch.{ch_idx + 1}: {ch_title}</h1>
<p style="color:#888; font-size:0.85em;">{len(pairs)} sentences</p>
{body}
</body>
</html>'''

        chapter = epub.EpubHtml(
            title=f'Ch.{ch_idx + 1}: {ch_title}',
            file_name=f'chapter_{ch_idx:02d}.xhtml',
            lang='ja',
        )
        chapter.content = chapter_html.encode('utf-8')
        chapter.add_item(css_item)
        book.add_item(chapter)

        spine.append(chapter)
        toc.append(chapter)

        logger.info(f'  EPUB Ch.{ch_idx + 1}: {ch_title} ({len(pairs)} pairs)')

    book.toc = toc
    book.spine = spine
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    epub.write_epub(output_path, book)
    size_kb = Path(output_path).stat().st_size / 1024
    logger.info(f'EPUB written: {output_path} ({size_kb:.0f} KB)')


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate full-book beginner Japanese audiobook + EPUB',
    )
    parser.add_argument(
        '--epub', default=EPUB_PATH,
        help='Path to source Chinese EPUB',
    )
    parser.add_argument(
        '--model', default=None,
        help='OpenRouter model ID (e.g. deepseek/deepseek-chat)',
    )
    parser.add_argument(
        '--tier', default='standard',
        choices=['free', 'standard', 'premium'],
        help='Model tier (default: standard = DeepSeek V3)',
    )
    parser.add_argument(
        '--chapter-start', type=int, default=0,
        help='0-based index of first chapter to process',
    )
    parser.add_argument(
        '--chapter-count', type=int, default=0,
        help='Number of chapters to process (0 = all)',
    )
    parser.add_argument(
        '--audio-only', action='store_true',
        help='Skip LLM synthesis, generate audio + EPUB from existing JSON cache',
    )
    parser.add_argument(
        '--no-audio', action='store_true',
        help='Skip audio generation (LLM synthesis + EPUB only)',
    )
    parser.add_argument(
        '--concurrency', type=int, default=10,
        help='Max concurrent TTS requests (default: 10)',
    )
    parser.add_argument(
        '--llm-workers', type=int, default=5,
        help='Max parallel LLM API calls per chapter (default: 5)',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve model
    if args.model:
        model = args.model
    else:
        from graded_reader.models import TIER_DEFAULTS
        model = TIER_DEFAULTS[args.tier]

    # Check API key (unless audio-only)
    if not args.audio_only:
        api_key = os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            logger.error(
                'OPENROUTER_API_KEY not set.\n'
                'Set it with: export OPENROUTER_API_KEY=sk-or-...\n'
                'Get a key at: https://openrouter.ai/keys'
            )
            sys.exit(1)
        logger.info(f'Using model: {model}')

    # ======================================================================
    # Phase 1: Extract chapters
    # ======================================================================
    logger.info('=' * 60)
    logger.info('PHASE 1: Extracting chapters from EPUB')
    logger.info('=' * 60)

    chapters = extract_content_chapters(args.epub)
    logger.info(f'Found {len(chapters)} content chapters:')
    for i, ch in enumerate(chapters):
        logger.info(f'  {i + 1}. {ch["title"]} ({len(ch["paragraphs"])} paragraphs)')

    # Apply chapter range filter
    if args.chapter_start > 0 or args.chapter_count > 0:
        start = args.chapter_start
        end = start + args.chapter_count if args.chapter_count > 0 else len(chapters)
        chapters = chapters[start:end]
        logger.info(f'Processing chapters {start + 1} to {start + len(chapters)}')

    # ======================================================================
    # Phase 2: Synthesize Japanese for each chapter
    # ======================================================================
    logger.info('')
    logger.info('=' * 60)
    logger.info('PHASE 2: LLM synthesis (Chinese → N5 Japanese)')
    logger.info('=' * 60)

    # Try to load existing Ch1 data
    ch1_data = load_existing_ch1_script()
    chapters_data = []

    for ch_idx, chapter in enumerate(chapters):
        title = chapter['title']
        paragraphs = chapter['paragraphs']

        logger.info(f'\n--- Chapter {ch_idx + 1}/{len(chapters)}: {title} ---')

        # Check cache first
        cached = load_cached_chapter(ch_idx)
        if cached and cached.get('sentence_pairs'):
            logger.info(f'  Loaded from cache: {len(cached["sentence_pairs"])} pairs')
            chapters_data.append(cached)
            continue

        # Try reusing existing validated script for the matching chapter
        if ch1_data and ch1_data.get('sentence_pairs') and '什么事' in title:
            logger.info(f'  Reusing existing Ch1 script: {len(ch1_data["sentence_pairs"])} validated pairs')
            data = {
                'chapter_title': title,
                'chapter_num': ch_idx,
                'sentence_pairs': ch1_data['sentence_pairs'],
                'validation': ch1_data.get('validation', {}),
            }
            save_chapter_cache(ch_idx, data)
            chapters_data.append(data)
            continue

        if args.audio_only:
            logger.warning(f'  No cached data for chapter {ch_idx + 1} — skipping (--audio-only)')
            continue

        # Synthesize via LLM
        data = synthesize_chapter(ch_idx, title, paragraphs, model, max_workers=args.llm_workers)
        chapters_data.append(data)

        logger.info(
            f'  Result: {len(data["sentence_pairs"])} pairs, '
            f'{data["validation"].get("violation_count", "?")} N5 violations'
        )

        # Brief pause between chapters to avoid rate limits
        if ch_idx < len(chapters) - 1:
            time.sleep(2)

    # Save full script JSON
    total_pairs = sum(len(ch['sentence_pairs']) for ch in chapters_data)
    logger.info(f'\nTotal: {total_pairs} sentence pairs across {len(chapters_data)} chapters')

    full_script = {
        'source_book': '只有偏执狂才能生存 (Only the Paranoid Survive)',
        'model': model,
        'total_pairs': total_pairs,
        'chapters': chapters_data,
    }
    with open(FULL_SCRIPT_PATH, 'w', encoding='utf-8') as f:
        json.dump(full_script, f, ensure_ascii=False, indent=2)
    logger.info(f'Full script saved: {FULL_SCRIPT_PATH}')

    if not chapters_data:
        logger.error('No chapters with data — nothing to generate')
        sys.exit(1)

    # ======================================================================
    # Phase 3: Generate EPUB
    # ======================================================================
    logger.info('')
    logger.info('=' * 60)
    logger.info('PHASE 3: Generating EPUB with furigana + romaji')
    logger.info('=' * 60)

    build_full_epub(
        chapters_data,
        source_book='只有偏执狂才能生存 (Only the Paranoid Survive)',
        output_path=EPUB_OUTPUT,
    )

    # ======================================================================
    # Phase 4: Generate audio + M4B
    # ======================================================================
    if args.no_audio:
        logger.info('\nSkipping audio generation (--no-audio)')
        logger.info(f'\nDone! EPUB: {EPUB_OUTPUT}')
        return

    logger.info('')
    logger.info('=' * 60)
    logger.info('PHASE 4: Generating TTS audio (edge-tts)')
    logger.info('=' * 60)
    logger.info(f'Pattern: JP → 0.3s → CN → 0.3s → JP → 0.8s')
    logger.info(f'Concurrency: {args.concurrency}')

    # Pre-generate silence
    silence_short = _generate_silence_mp3(300)
    silence_long = _generate_silence_mp3(800)

    semaphore = asyncio.Semaphore(args.concurrency)

    async def generate_all_audio():
        chapter_audio_files = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            for ch_idx, ch_data in enumerate(chapters_data):
                pairs = ch_data['sentence_pairs']
                if not pairs:
                    continue

                ch_title = ch_data['chapter_title']
                logger.info(f'\nChapter {ch_idx + 1}/{len(chapters_data)}: {ch_title}')

                audio_data = await generate_chapter_audio(
                    pairs, ch_idx, len(chapters_data),
                    semaphore, silence_short, silence_long,
                )

                if not audio_data:
                    logger.warning(f'  No audio for chapter {ch_idx + 1} — skipping')
                    continue

                # Save chapter MP3
                mp3_path = tmpdir / f'chapter_{ch_idx:02d}.mp3'
                mp3_path.write_bytes(audio_data)
                chapter_audio_files.append((
                    f'Ch.{ch_idx + 1}: {ch_title}',
                    mp3_path,
                ))

                size_mb = len(audio_data) / 1024 / 1024
                logger.info(f'  Chapter audio: {size_mb:.1f} MB')

            if not chapter_audio_files:
                logger.error('No audio generated for any chapter')
                return

            # Extract cover image
            logger.info('\nExtracting cover image...')
            cover_image = None
            try:
                from graded_reader.audio_generator import _extract_cover_image
                cover_image = _extract_cover_image(args.epub)
                if cover_image:
                    logger.info(f'  Cover: {len(cover_image) / 1024:.0f} KB')
            except Exception:
                pass

            # Assemble M4B
            logger.info(f'\nAssembling M4B: {M4B_OUTPUT}')
            logger.info(f'  {len(chapter_audio_files)} chapters')

            build_multi_chapter_m4b(
                chapter_audio_files,
                M4B_OUTPUT,
                cover_image=cover_image,
            )

    asyncio.run(generate_all_audio())

    # ======================================================================
    # Summary
    # ======================================================================
    logger.info('')
    logger.info('=' * 60)
    logger.info('DONE!')
    logger.info('=' * 60)
    logger.info(f'  Chapters:  {len(chapters_data)}')
    logger.info(f'  Sentences: {total_pairs}')
    logger.info(f'  EPUB:      {EPUB_OUTPUT}')

    if not args.no_audio and Path(M4B_OUTPUT).exists():
        m4b_size = Path(M4B_OUTPUT).stat().st_size / 1024 / 1024
        logger.info(f'  M4B:       {M4B_OUTPUT} ({m4b_size:.1f} MB)')

    logger.info(f'  Script:    {FULL_SCRIPT_PATH}')


if __name__ == '__main__':
    main()
