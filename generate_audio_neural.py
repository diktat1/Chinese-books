#!/usr/bin/env python3
"""
Generate beginner Japanese audiobook using Kokoro neural TTS.

Pattern per sentence: JP -> 0.3s pause -> CN -> 0.3s pause -> JP -> 0.8s pause
Output: M4B audiobook with chapter marker.

Uses kokoro-onnx for high-quality neural voices (local, no network required).
"""
import io
import json
import logging
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Model paths
MODEL_PATH = 'models/kokoro-v1.0.int8.onnx'
VOICES_PATH = 'models/voices-v1.0.bin'

# Voice settings
JP_VOICE = 'jf_alpha'      # Japanese female neural voice
CN_VOICE = 'zf_xiaoxiao'   # Chinese female neural voice
JP_LANG = 'ja'
CN_LANG = 'cmn'

SAMPLE_RATE = 24000


def generate_silence(duration_ms: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate silence as numpy array."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return np.zeros(num_samples, dtype=np.float32)


def samples_to_mp3(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert numpy audio samples to MP3 bytes via ffmpeg."""
    # Write to WAV in memory
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, samples, sample_rate, format='WAV')
    wav_bytes = wav_buffer.getvalue()

    # Convert WAV to MP3
    result = subprocess.run(
        ['ffmpeg', '-y', '-i', 'pipe:0',
         '-c:a', 'libmp3lame', '-b:a', '64k',
         '-f', 'mp3', 'pipe:1'],
        input=wav_bytes,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg conversion failed: {result.stderr[:200]}')
    return result.stdout


def main():
    from kokoro_onnx import Kokoro

    # Load script
    with open('beginner_japanese_script.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    pairs = data['sentence_pairs']
    title = data['chapter_title']
    logger.info(f'Loaded {len(pairs)} sentence pairs')
    logger.info(f'Chapter: {title}')
    logger.info(f'Audio pattern: JP -> 0.3s -> CN -> 0.3s -> JP -> 0.8s')
    logger.info(f'TTS engine: Kokoro neural (ONNX, local)')
    logger.info(f'JP voice: {JP_VOICE} | CN voice: {CN_VOICE}')
    logger.info('')

    # Load model
    logger.info('Loading Kokoro model...')
    t0 = time.time()
    kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
    logger.info(f'Model loaded in {time.time() - t0:.1f}s')

    # Pre-generate silence
    silence_short = generate_silence(300)   # 0.3s
    silence_long = generate_silence(800)    # 0.8s

    # Generate all sentence audio
    logger.info(f'\nGenerating TTS for {len(pairs)} sentences...')
    all_samples = []
    failed = 0
    total_start = time.time()

    for i, pair in enumerate(pairs):
        jp_text = pair['japanese']
        cn_text = pair['chinese']

        try:
            # 1. Japanese (first time)
            jp1, _ = kokoro.create(jp_text, voice=JP_VOICE, lang=JP_LANG)
            all_samples.append(jp1)

            # 2. Short pause
            all_samples.append(silence_short)

            # 3. Chinese
            cn_audio, _ = kokoro.create(cn_text, voice=CN_VOICE, lang=CN_LANG)
            all_samples.append(cn_audio)

            # 4. Short pause
            all_samples.append(silence_short)

            # 5. Japanese (repeat)
            jp2, _ = kokoro.create(jp_text, voice=JP_VOICE, lang=JP_LANG)
            all_samples.append(jp2)

            # 6. Long pause between sentences
            all_samples.append(silence_long)

            elapsed = time.time() - total_start
            avg = elapsed / (i + 1)
            remaining = avg * (len(pairs) - i - 1)
            logger.info(
                f'  [{i+1}/{len(pairs)}] OK ({elapsed:.0f}s elapsed, '
                f'~{remaining/60:.0f}m remaining): {jp_text[:40]}'
            )

        except Exception as e:
            logger.error(f'  [{i+1}/{len(pairs)}] FAIL: {jp_text[:40]} -> {e}')
            failed += 1

    total_elapsed = time.time() - total_start
    logger.info(f'\nTTS complete: {len(pairs) - failed}/{len(pairs)} succeeded')
    logger.info(f'Total time: {total_elapsed/60:.1f} minutes')

    if not all_samples:
        logger.error('No audio generated!')
        return

    # Combine all samples
    logger.info('Combining audio...')
    combined = np.concatenate(all_samples)
    duration_s = len(combined) / SAMPLE_RATE
    logger.info(f'Total duration: {duration_s/60:.1f} minutes ({duration_s:.0f}s)')

    # Convert to MP3
    logger.info('Converting to MP3...')
    mp3_data = samples_to_mp3(combined)
    logger.info(f'MP3 size: {len(mp3_data) / 1024 / 1024:.1f} MB')

    # Build M4B
    output_path = 'beginner_japanese_ch1_neural.m4b'
    logger.info(f'Assembling M4B: {output_path}')

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        mp3_path = tmpdir / 'chapter.mp3'
        mp3_path.write_bytes(mp3_data)

        # Get duration
        probe = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_format', str(mp3_path)],
            capture_output=True, text=True,
        )
        duration_ms = int(duration_s * 1000)
        try:
            info = json.loads(probe.stdout)
            duration_ms = int(float(info['format']['duration']) * 1000)
        except Exception:
            pass

        meta_file = tmpdir / 'chapters.txt'
        meta_file.write_text(
            ';FFMETADATA1\n\n'
            '[CHAPTER]\n'
            'TIMEBASE=1/1000\n'
            'START=0\n'
            f'END={duration_ms}\n'
            f'title=Beginner Japanese: {title}\n'
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

    final_size = Path(output_path).stat().st_size / 1024 / 1024
    logger.info(f'Done! Output: {output_path} ({final_size:.1f} MB)')


if __name__ == '__main__':
    main()
