#!/usr/bin/env python3
"""
Build an EPUB graded reader from a JSON cache + source EPUB.

No LLM API calls are made. Pinyin is computed deterministically.
This is instant and repeatable — tweak CSS, switch languages, toggle
simplified/original, and rebuild in seconds.

Usage:
    # Simplified interlinear with word spacing
    python build.py book_data/paranoid.json \\
      --interlinear --word-spacing --use-simplified \\
      --target-languages pt,it,fr,de,es,tr \\
      -o output_simplified.epub

    # Original interlinear
    python build.py book_data/paranoid.json \\
      --interlinear --word-spacing \\
      --target-languages pt,it,fr,de,es,tr \\
      -o output_original.epub

    # Parallel text layout
    python build.py book_data/paranoid.json \\
      --parallel-text \\
      --target-languages pt,it \\
      -o output_parallel.epub
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from graded_reader.json_cache import BuildConfig, build_epub_from_json


def main():
    parser = argparse.ArgumentParser(
        description='Build an EPUB graded reader from a JSON cache (no API calls).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python build.py book.json -o output.epub
  python build.py book.json --interlinear --word-spacing -o output.epub
  python build.py book.json --use-simplified --interlinear -o simplified.epub
  python build.py book.json --parallel-text -o parallel.epub
  python build.py book.json --target-languages pt,fr -o two_lang.epub
  python build.py book.json --source-epub other.epub -o output.epub
        ''',
    )

    parser.add_argument('json_input', help='Path to the JSON cache file')
    parser.add_argument(
        '-o', '--output',
        help='Path for the output EPUB (default: <json_stem>.epub)',
    )

    # Layout mode (mutually exclusive)
    mode_group = parser.add_argument_group('layout mode')
    mode_exc = mode_group.add_mutually_exclusive_group()
    mode_exc.add_argument(
        '--interlinear',
        action='store_true',
        default=True,
        help='Sentence-by-sentence Chinese + translation (default)',
    )
    mode_exc.add_argument(
        '--parallel-text',
        action='store_true',
        help='Two-column table: Chinese left, translation right',
    )

    # Formatting options
    parser.add_argument(
        '--word-spacing',
        action='store_true',
        default=True,
        help='Add word boundary spacing for Kindle dictionary lookup (default: on)',
    )
    parser.add_argument(
        '--no-word-spacing',
        action='store_true',
        help='Disable word boundary spacing',
    )
    parser.add_argument(
        '--use-simplified',
        action='store_true',
        help='Use HSK-simplified Chinese text (if available in JSON)',
    )
    parser.add_argument(
        '--target-languages',
        help='Comma-separated target language codes for chapter rotation '
             '(default: all languages in JSON)',
    )
    parser.add_argument(
        '--lang-start',
        type=int,
        default=0,
        help='Language rotation offset index (default: 0)',
    )

    # CSS overrides
    parser.add_argument(
        '--zh-font-size',
        help='Override Chinese text font size (e.g., "1.2em")',
    )
    parser.add_argument(
        '--tr-font-size',
        help='Override translation text font size (e.g., "0.8em")',
    )
    parser.add_argument(
        '--zh-line-height',
        help='Override Chinese text line height (e.g., "2.0")',
    )
    parser.add_argument(
        '--tr-line-height',
        help='Override translation text line height (e.g., "1.4")',
    )
    parser.add_argument(
        '--tr-color',
        help='Override translation text color (e.g., "#888")',
    )

    # Source EPUB override
    parser.add_argument(
        '--source-epub',
        help='Path to source EPUB (overrides metadata.source_epub in JSON)',
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
    json_path = Path(args.json_input)
    if not json_path.exists():
        print(f'Error: JSON file not found: {json_path}', file=sys.stderr)
        sys.exit(1)

    # Load JSON to get metadata for defaults
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Resolve output path
    if args.output:
        output_path = args.output
    else:
        output_path = str(json_path.with_suffix('.epub'))

    # Determine layout mode
    if args.parallel_text:
        mode = 'parallel_text'
    else:
        mode = 'interlinear'

    # Determine target languages
    if args.target_languages:
        target_languages = [l.strip() for l in args.target_languages.split(',')]
    else:
        target_languages = data['metadata'].get('target_languages', ['en'])

    # Word spacing
    word_spacing = True
    if args.no_word_spacing:
        word_spacing = False

    # CSS overrides
    css_overrides = {}
    if args.zh_font_size:
        css_overrides['zh_font_size'] = args.zh_font_size
    if args.tr_font_size:
        css_overrides['tr_font_size'] = args.tr_font_size
    if args.zh_line_height:
        css_overrides['zh_line_height'] = args.zh_line_height
    if args.tr_line_height:
        css_overrides['tr_line_height'] = args.tr_line_height
    if args.tr_color:
        css_overrides['tr_color'] = args.tr_color

    config = BuildConfig(
        mode=mode,
        target_languages=target_languages,
        lang_start_index=args.lang_start,
        use_simplified=args.use_simplified,
        word_spacing=word_spacing,
        css_overrides=css_overrides or None,
    )

    # Print summary
    print(f'Input:      {json_path}')
    print(f'Source:     {args.source_epub or data["metadata"]["source_epub"]}')
    print(f'Output:     {output_path}')
    print(f'Mode:       {mode}')
    print(f'Languages:  {", ".join(target_languages)}')
    print(f'Simplified: {"yes" if args.use_simplified else "no"}')
    print(f'Word space: {"yes" if word_spacing else "no"}')
    if css_overrides:
        print(f'CSS tweaks: {css_overrides}')
    print()

    build_epub_from_json(
        json_path=str(json_path),
        output_path=output_path,
        config=config,
        source_epub=args.source_epub,
    )

    print(f'\nEPUB written to: {output_path}')


if __name__ == '__main__':
    main()
