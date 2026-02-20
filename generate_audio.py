#!/usr/bin/env python3
"""
Generate beginner Japanese audiobook from the N5 script.

Pattern per sentence: JP -> 0.3s pause -> CN -> 0.3s pause -> JP -> 0.8s pause
Output: M4B audiobook with chapter marker.
"""
import asyncio
import io
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Edge-TTS voices
JP_VOICE = 'ja-JP-NanamiNeural'
CN_VOICE = 'zh-CN-XiaoxiaoNeural'


async def synthesize_edge(text: str, voice: str) -> bytes:
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


def generate_silence_mp3(duration_ms: int) -> bytes:
    """Generate silence as MP3 using ffmpeg."""
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


async def generate_sentence_audio(
    jp_text: str,
    cn_text: str,
    idx: int,
    total: int,
    semaphore: asyncio.Semaphore,
    silence_short: bytes,
    silence_long: bytes,
) -> tuple[int, bytes]:
    """Generate audio for one sentence: JP -> pause -> CN -> pause -> JP -> long pause."""
    parts = []

    async def synth(text, voice):
        async with semaphore:
            return await synthesize_edge(text, voice)

    try:
        # 1. Japanese (first time)
        jp1 = await synth(jp_text, JP_VOICE)
        parts.append(jp1)

        # 2. Short pause
        if silence_short:
            parts.append(silence_short)

        # 3. Chinese
        cn_audio = await synth(cn_text, CN_VOICE)
        parts.append(cn_audio)

        # 4. Short pause
        if silence_short:
            parts.append(silence_short)

        # 5. Japanese (repeat)
        jp2 = await synth(jp_text, JP_VOICE)
        parts.append(jp2)

        # 6. Long pause between sentences
        if silence_long:
            parts.append(silence_long)

        logger.info(f'  [{idx+1}/{total}] OK: {jp_text[:40]}...')
        return (idx, b''.join(parts))

    except Exception as e:
        logger.error(f'  [{idx+1}/{total}] FAIL: {jp_text[:40]}... -> {e}')
        return (idx, b'')


def build_m4b(audio_data: bytes, title: str, output_path: str, cover_image: bytes | None = None) -> str:
    """Wrap MP3 audio data into an M4B with chapter marker."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        mp3_path = tmpdir / 'chapter.mp3'
        mp3_path.write_bytes(audio_data)

        # Chapter metadata
        meta_file = tmpdir / 'chapters.txt'

        # Get duration
        probe = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(mp3_path)],
            capture_output=True, text=True,
        )
        duration_ms = 0
        try:
            info = json.loads(probe.stdout)
            duration_ms = int(float(info['format']['duration']) * 1000)
        except Exception:
            pass

        meta_file.write_text(
            ';FFMETADATA1\n\n'
            '[CHAPTER]\n'
            'TIMEBASE=1/1000\n'
            'START=0\n'
            f'END={duration_ms}\n'
            f'title={title}\n'
        )

        cmd = [
            'ffmpeg', '-y',
            '-i', str(mp3_path),
            '-i', str(meta_file),
        ]

        if cover_image:
            cover_path = tmpdir / 'cover.jpg'
            cover_path.write_bytes(cover_image)
            cmd.extend(['-i', str(cover_path)])
            cmd.extend(['-map', '0:a', '-map', '2:v', '-c:v', 'mjpeg',
                         '-disposition:v:0', 'attached_pic'])
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


async def main():
    # Load script
    with open('beginner_japanese_script.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    pairs = data['sentence_pairs']
    title = data['chapter_title']
    logger.info(f'Loaded {len(pairs)} sentence pairs')
    logger.info(f'Chapter: {title}')
    logger.info(f'Audio pattern: JP -> 0.3s -> CN -> 0.3s -> JP -> 0.8s')
    logger.info('')

    # Pre-generate silence
    logger.info('Generating silence segments...')
    silence_short = generate_silence_mp3(300)   # 0.3s between segments
    silence_long = generate_silence_mp3(800)    # 0.8s between sentences
    logger.info(f'  Short silence: {len(silence_short)} bytes')
    logger.info(f'  Long silence: {len(silence_long)} bytes')

    # Generate all sentence audio in parallel (limit concurrency)
    semaphore = asyncio.Semaphore(10)
    logger.info(f'\nGenerating TTS for {len(pairs)} sentences...')

    tasks = [
        generate_sentence_audio(
            pair['japanese'], pair['chinese'],
            i, len(pairs), semaphore, silence_short, silence_long,
        )
        for i, pair in enumerate(pairs)
    ]

    results = await asyncio.gather(*tasks)
    results = sorted(results, key=lambda x: x[0])

    # Combine
    all_audio = b''.join(audio for _, audio in results if audio)
    failed = sum(1 for _, audio in results if not audio)

    logger.info(f'\nTTS complete: {len(pairs) - failed}/{len(pairs)} succeeded')
    logger.info(f'Total audio: {len(all_audio) / 1024 / 1024:.1f} MB')

    if not all_audio:
        logger.error('No audio generated!')
        return

    # Extract cover from EPUB
    cover_image = None
    try:
        from graded_reader.audio_generator import _extract_cover_image
        cover_image = _extract_cover_image('input-epubs/Only the Paranoid Survive_CN (Original Book as Published).epub')
        if cover_image:
            logger.info(f'Cover image: {len(cover_image) / 1024:.0f} KB')
    except Exception:
        pass

    # Build M4B
    output_path = 'beginner_japanese_ch1.m4b'
    logger.info(f'\nAssembling M4B: {output_path}')
    build_m4b(all_audio, f'Beginner Japanese: {title}', output_path, cover_image)

    final_size = Path(output_path).stat().st_size / 1024 / 1024
    logger.info(f'Done! Output: {output_path} ({final_size:.1f} MB)')


if __name__ == '__main__':
    asyncio.run(main())
