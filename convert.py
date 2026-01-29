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
import sys
from pathlib import Path

from graded_reader.epub_processor import process_epub


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

    # Determine output path
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
    if add_pinyin:
        mode_parts.append('pinyin')
    if add_translation:
        mode_parts.append(f'translation ({args.source} -> {args.target})')
    if args.word_spacing:
        mode_parts.append('word-spacing')

    print(f'Input:  {input_path}')
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
    )

    print(f'\nOutput written to: {output_path}')
    print('Tip: For Kindle, convert the output EPUB to AZW3 using Calibre for best ruby support.')


if __name__ == '__main__':
    main()
