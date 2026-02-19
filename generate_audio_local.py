#!/usr/bin/env python3
"""
Generate beginner Japanese audiobook using local espeak-ng TTS.

Pattern per sentence: JP -> 0.3s pause -> CN -> 0.3s pause -> JP -> 0.8s pause
Output: M4B audiobook with chapter marker.

Uses espeak-ng (local, no network required) instead of edge-tts.
"""
import json
import logging
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# espeak-ng voice settings
JP_VOICE = 'ja'
CN_VOICE = 'cmn'
SPEED = 120  # words-per-minute (slower for learners)
PITCH = 50   # default pitch


def synthesize_espeak(text: str, voice: str, speed: int = SPEED) -> bytes:
    """Synthesize text to WAV bytes using espeak-ng, then convert to MP3."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as wav_f:
        wav_path = wav_f.name

    # Generate WAV
    result = subprocess.run(
        ['espeak-ng', '-v', voice, '-s', str(speed), '-p', str(PITCH),
         text, '-w', wav_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f'espeak-ng failed: {result.stderr}')

    # Convert WAV to MP3
    mp3_result = subprocess.run(
        ['ffmpeg', '-y', '-i', wav_path,
         '-c:a', 'libmp3lame', '-b:a', '64k',
         '-f', 'mp3', 'pipe:1'],
        capture_output=True,
    )

    # Clean up WAV
    Path(wav_path).unlink(missing_ok=True)

    if mp3_result.returncode != 0:
        raise RuntimeError(f'ffmpeg conversion failed')

    return mp3_result.stdout


def generate_silence_mp3(duration_ms: int) -> bytes:
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


def generate_sentence_audio(
    jp_text: str,
    cn_text: str,
    idx: int,
    total: int,
    silence_short: bytes,
    silence_long: bytes,
) -> tuple[int, bytes]:
    """Generate audio for one sentence: JP -> pause -> CN -> pause -> JP -> long pause."""
    parts = []

    try:
        # 1. Japanese (first time)
        jp1 = synthesize_espeak(jp_text, JP_VOICE)
        parts.append(jp1)

        # 2. Short pause
        if silence_short:
            parts.append(silence_short)

        # 3. Chinese
        cn_audio = synthesize_espeak(cn_text, CN_VOICE)
        parts.append(cn_audio)

        # 4. Short pause
        if silence_short:
            parts.append(silence_short)

        # 5. Japanese (repeat)
        jp2 = synthesize_espeak(jp_text, JP_VOICE)
        parts.append(jp2)

        # 6. Long pause between sentences
        if silence_long:
            parts.append(silence_long)

        logger.info(f'  [{idx+1}/{total}] OK: {jp_text[:40]}')
        return (idx, b''.join(parts))

    except Exception as e:
        logger.error(f'  [{idx+1}/{total}] FAIL: {jp_text[:40]} -> {e}')
        return (idx, b'')


def build_m4b(audio_data: bytes, title: str, output_path: str) -> str:
    """Wrap MP3 audio data into an M4B with chapter marker."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        mp3_path = tmpdir / 'chapter.mp3'
        mp3_path.write_bytes(audio_data)

        # Get duration
        probe = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_format', str(mp3_path)],
            capture_output=True, text=True,
        )
        duration_ms = 0
        try:
            info = json.loads(probe.stdout)
            duration_ms = int(float(info['format']['duration']) * 1000)
        except Exception:
            pass

        # Chapter metadata
        meta_file = tmpdir / 'chapters.txt'
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
            '-map', '0:a',
            '-map_metadata', '1',
            '-c:a', 'aac', '-b:a', '64k',
            '-movflags', '+faststart',
            str(output_path),
        ]

        subprocess.run(cmd, check=True, capture_output=True)

    return output_path


def main():
    # Load script
    with open('beginner_japanese_script.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    pairs = data['sentence_pairs']
    title = data['chapter_title']
    logger.info(f'Loaded {len(pairs)} sentence pairs')
    logger.info(f'Chapter: {title}')
    logger.info(f'Audio pattern: JP -> 0.3s -> CN -> 0.3s -> JP -> 0.8s')
    logger.info(f'TTS engine: espeak-ng (local, no network)')
    logger.info('')

    # Pre-generate silence
    logger.info('Generating silence segments...')
    silence_short = generate_silence_mp3(300)   # 0.3s between segments
    silence_long = generate_silence_mp3(800)    # 0.8s between sentences
    logger.info(f'  Short silence: {len(silence_short)} bytes')
    logger.info(f'  Long silence: {len(silence_long)} bytes')

    # Generate all sentence audio sequentially (espeak-ng is synchronous)
    logger.info(f'\nGenerating TTS for {len(pairs)} sentences...')

    results = []
    for i, pair in enumerate(pairs):
        result = generate_sentence_audio(
            pair['japanese'], pair['chinese'],
            i, len(pairs), silence_short, silence_long,
        )
        results.append(result)

    results = sorted(results, key=lambda x: x[0])

    # Combine
    all_audio = b''.join(audio for _, audio in results if audio)
    failed = sum(1 for _, audio in results if not audio)

    logger.info(f'\nTTS complete: {len(pairs) - failed}/{len(pairs)} succeeded')
    logger.info(f'Total audio: {len(all_audio) / 1024 / 1024:.1f} MB')

    if not all_audio:
        logger.error('No audio generated!')
        return

    # Build M4B
    output_path = 'beginner_japanese_ch1.m4b'
    logger.info(f'\nAssembling M4B: {output_path}')
    build_m4b(all_audio, f'Beginner Japanese: {title}', output_path)

    final_size = Path(output_path).stat().st_size / 1024 / 1024
    logger.info(f'Done! Output: {output_path} ({final_size:.1f} MB)')


if __name__ == '__main__':
    main()
