#!/usr/bin/env python3
"""
Chinese Graded Reader Converter

Converts a Chinese EPUB ebook into a graded reader with:
  - Pinyin annotations above each Chinese character (using <ruby> tags)
  - English translations after each paragraph

Usage:
    python convert.py input.epub
    python convert.py input.epub -o output.epub
    python convert.py input.epub --pinyin-only
    python convert.py input.epub --translation-only
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from graded_reader.epub_processor import process_epub
from graded_reader.calibre import (
    is_calibre_installed,
    convert_epub_to_azw3,
    CalibreNotFoundError,
)
from graded_reader.claude_simplifier import is_anthropic_available, get_api_key


def main():
    parser = argparse.ArgumentParser(
        description='Convert a Chinese EPUB into a graded reader with pinyin and English translations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python convert.py book.epub                          # Full conversion (pinyin + translation)
  python convert.py book.epub -o my_reader.epub        # Specify output filename
  python convert.py book.epub --pinyin-only            # Only add pinyin, no translation
  python convert.py book.epub --translation-only       # Only add translation, no pinyin
  python convert.py book.epub --target ja              # Translate to Japanese instead of English
  python convert.py book.epub --word-spacing           # Add spaces between words for Kindle lookup
  python convert.py book.epub --kindle                 # Output AZW3 for Kindle (requires Calibre)
  python convert.py book.epub --kindle --no-keep-epub  # AZW3 only, delete intermediate EPUB
  python convert.py book.epub --kindle-format          # Paragraph format (Chinese, pinyin, English)
  python convert.py book.epub --simplify-hsk4          # Simplify to HSK 4 vocabulary (requires Claude)
  python convert.py book.epub --use-claude             # Use Claude for translation (higher quality)
  python convert.py book.epub --simplify-hsk4 --use-opus  # Use Opus model for best quality
  python convert.py book.epub --anki                      # Generate Anki deck (sentence cards)
  python convert.py book.epub --anki --target fr           # Anki deck with French translations
  python convert.py book.epub --anki --no-audio            # Anki deck without TTS audio
  python convert.py book.epub --anki --max-sentences 100   # Limit to first 100 sentences
        ''',
    )

    parser.add_argument('input', help='Path to the input Chinese EPUB file')
    parser.add_argument(
        '-o', '--output',
        help='Path for the output EPUB file (default: <input>_graded.epub)',
    )
    parser.add_argument(
        '--pinyin-only',
        action='store_true',
        help='Only add pinyin annotations, skip translation',
    )
    parser.add_argument(
        '--translation-only',
        action='store_true',
        help='Only add translations, skip pinyin annotations',
    )
    parser.add_argument(
        '--source',
        default='zh-CN',
        help='Source language code for translation (default: zh-CN)',
    )
    parser.add_argument(
        '--target',
        default='en',
        help='Target language code for translation (default: en)',
    )
    parser.add_argument(
        '--word-spacing',
        action='store_true',
        help='Add spaces between Chinese words for easier dictionary lookup on e-readers',
    )
    parser.add_argument(
        '--kindle-format',
        action='store_true',
        help='Use paragraph-by-paragraph format instead of ruby annotations. '
             'Outputs: Chinese paragraph (with word spacing), pinyin paragraph, '
             'English translation. Works better on Kindle which has poor ruby support.',
    )
    parser.add_argument(
        '--kindle', '--azw3',
        action='store_true',
        dest='kindle',
        help='Convert output to AZW3 format for Kindle (requires Calibre)',
    )
    parser.add_argument(
        '--kindle-profile',
        default='kindle_pw3',
        choices=['kindle', 'kindle_dx', 'kindle_fire', 'kindle_oasis',
                 'kindle_pw', 'kindle_pw3', 'kindle_scribe', 'kindle_voyage'],
        help='Calibre output profile for Kindle conversion (default: kindle_pw3)',
    )
    parser.add_argument(
        '--no-keep-epub',
        action='store_true',
        help='Delete the intermediate EPUB after AZW3 conversion',
    )
    parser.add_argument(
        '--simplify-hsk4',
        action='store_true',
        help='Simplify Chinese vocabulary to HSK 4 level using Claude AI. '
             'Requires ANTHROPIC_API_KEY environment variable.',
    )
    parser.add_argument(
        '--use-claude',
        action='store_true',
        help='Use Claude AI for translation instead of Google Translate. '
             'Provides higher quality translations. Requires ANTHROPIC_API_KEY.',
    )
    parser.add_argument(
        '--use-opus',
        action='store_true',
        help='Use Claude Opus model for highest quality (slower, more expensive). '
             'Applies to both HSK simplification and Claude translation.',
    )
    # Anki deck generation
    parser.add_argument(
        '--anki',
        action='store_true',
        help='Generate an Anki deck (.apkg) with sentence cards. '
             'Front: Chinese + audio. Back: pinyin + translation.',
    )
    parser.add_argument(
        '--no-audio',
        action='store_true',
        help='Skip TTS audio generation for Anki cards (faster)',
    )
    parser.add_argument(
        '--max-sentences',
        type=int,
        default=0,
        help='Maximum number of sentences for Anki deck (0 = all)',
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging',
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f'Error: Input file not found: {input_path}', file=sys.stderr)
        sys.exit(1)

    if input_path.suffix.lower() != '.epub':
        print(f'Warning: Input file does not have .epub extension: {input_path}', file=sys.stderr)

    # Check Calibre availability if --kindle is requested
    if args.kindle and not is_calibre_installed():
        error = CalibreNotFoundError()
        print(f'Error: {error.message}', file=sys.stderr)
        sys.exit(1)

    # Check Anthropic availability if Claude features are requested
    if args.simplify_hsk4 or args.use_claude:
        if not is_anthropic_available():
            print('Error: Anthropic SDK not installed. Install with: pip install anthropic', file=sys.stderr)
            sys.exit(1)
        if not get_api_key():
            print('Error: ANTHROPIC_API_KEY environment variable not set.', file=sys.stderr)
            print('Get your API key from https://console.anthropic.com/', file=sys.stderr)
            sys.exit(1)

    # Anki deck mode - separate flow
    if args.anki:
        from graded_reader.anki_generator import generate_anki_deck

        if args.output:
            anki_output = Path(args.output)
            if anki_output.suffix.lower() != '.apkg':
                anki_output = anki_output.with_suffix('.apkg')
        else:
            anki_output = input_path.with_stem(input_path.stem + '_anki').with_suffix('.apkg')

        print(f'Input:  {input_path}')
        print(f'Output: {anki_output}')
        print(f'Mode:   Anki deck ({args.source} -> {args.target})')
        if not args.no_audio:
            print(f'Audio:  TTS enabled (Chinese)')
        if args.max_sentences:
            print(f'Limit:  {args.max_sentences} sentences')
        print()

        generate_anki_deck(
            epub_path=str(input_path),
            output_path=str(anki_output),
            translation_target=args.target,
            translation_source=args.source,
            use_claude=args.use_claude,
            use_opus=args.use_opus,
            include_audio=not args.no_audio,
            max_sentences=args.max_sentences,
        )
        print(f'\nAnki deck written to: {anki_output}')
        return

    # Determine output paths
    if args.kindle:
        # For AZW3 output, we need both an intermediate EPUB and final AZW3 path
        if args.output:
            azw3_path = Path(args.output)
            if azw3_path.suffix.lower() != '.azw3':
                azw3_path = azw3_path.with_suffix('.azw3')
        else:
            azw3_path = input_path.with_stem(input_path.stem + '_graded').with_suffix('.azw3')
        epub_output_path = azw3_path.with_suffix('.epub')
        output_path = epub_output_path  # process_epub writes to this
    else:
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.with_stem(input_path.stem + '_graded')

    # Determine what to add
    add_pinyin = not args.translation_only
    add_translation = not args.pinyin_only

    if not add_pinyin and not add_translation:
        print('Error: Cannot use --pinyin-only and --translation-only together', file=sys.stderr)
        sys.exit(1)

    mode_parts = []
    if args.simplify_hsk4:
        model_note = ' (Opus)' if args.use_opus else ''
        mode_parts.append(f'HSK4-simplify{model_note}')
    if add_pinyin:
        mode_parts.append('pinyin')
    if add_translation:
        translator = 'Claude' if args.use_claude else 'Google'
        model_note = ' Opus' if args.use_claude and args.use_opus else ''
        mode_parts.append(f'translation ({args.source} -> {args.target}, {translator}{model_note})')
    if args.word_spacing:
        mode_parts.append('word-spacing')
    if args.kindle_format:
        mode_parts.append('kindle-format (paragraph-by-paragraph)')
    if args.kindle:
        mode_parts.append(f'kindle ({args.kindle_profile})')

    print(f'Input:  {input_path}')
    if args.kindle:
        print(f'Output: {azw3_path} (AZW3 for Kindle)')
    else:
        print(f'Output: {output_path}')
    print(f'Mode:   {" + ".join(mode_parts)}')
    print()

    process_epub(
        input_path=str(input_path),
        output_path=str(output_path),
        add_pinyin=add_pinyin,
        add_translation=add_translation,
        translation_source=args.source,
        translation_target=args.target,
        word_spacing=args.word_spacing,
        kindle_format=args.kindle_format,
        simplify_hsk4=args.simplify_hsk4,
        use_claude_translator=args.use_claude,
        use_opus=args.use_opus,
    )

    if args.kindle:
        # Convert EPUB to AZW3
        keep_epub = not args.no_keep_epub
        try:
            convert_epub_to_azw3(
                epub_path=str(epub_output_path),
                azw3_path=str(azw3_path),
                output_profile=args.kindle_profile,
                keep_epub=keep_epub,
            )
            print(f'\nKindle AZW3 written to: {azw3_path}')
            if keep_epub:
                print(f'Intermediate EPUB kept at: {epub_output_path}')
        except subprocess.CalledProcessError as e:
            print(f'Error: Calibre conversion failed: {e.stderr}', file=sys.stderr)
            sys.exit(1)
    else:
        print(f'\nOutput written to: {output_path}')
        print('Tip: Use --kindle flag to automatically convert to AZW3 for Kindle.')


if __name__ == '__main__':
    main()
