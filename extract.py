#!/usr/bin/env python3
"""
Extract all LLM-dependent data from a Chinese EPUB into a JSON cache.

This runs all expensive LLM calls (simplification, word spacing, translations)
once, storing results in a JSON file. The JSON can then be used with build.py
to instantly generate EPUBs with different formatting—no API calls needed.

Usage:
    python extract.py input-epubs/paranoid.epub \\
      --simplify-hsk4 \\
      --target-languages pt,it,fr,de,es,tr \\
      --tier standard \\
      -o book_data/paranoid.json

    # Incremental: add new languages to existing JSON
    python extract.py input-epubs/paranoid.epub \\
      --target-languages ja,ko \\
      --tier standard \\
      -o book_data/paranoid.json
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from graded_reader.json_cache import extract_to_json
from graded_reader.llm_simplifier import is_openrouter_available, get_api_key
from graded_reader.models import MODELS, TIER_DEFAULTS


def main():
    parser = argparse.ArgumentParser(
        description='Extract LLM data from a Chinese EPUB into a JSON cache.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python extract.py input.epub -o book_data/book.json
  python extract.py input.epub --simplify-hsk4 --tier standard -o book.json
  python extract.py input.epub --target-languages pt,it,fr,de,es,tr -o book.json
  python extract.py input.epub --target-languages ja,ko -o book.json  # add languages incrementally
        ''',
    )

    parser.add_argument('input', help='Path to the input Chinese EPUB file')
    parser.add_argument(
        '-o', '--output',
        help='Path for the output JSON file (default: <input_stem>.json)',
    )
    parser.add_argument(
        '--simplify-hsk4',
        action='store_true',
        help='Simplify Chinese vocabulary to HSK 4 level using an LLM',
    )
    parser.add_argument(
        '--target-languages',
        default='pt,it,fr,de,es,tr',
        help='Comma-separated target language codes (default: pt,it,fr,de,es,tr)',
    )

    # Model selection
    model_group = parser.add_argument_group('model selection')
    model_group.add_argument(
        '--tier',
        choices=['free', 'standard', 'premium'],
        default='standard',
        help='Translation quality tier (default: standard)',
    )
    model_group.add_argument(
        '--model',
        help='Specific model ID (overrides --tier)',
    )

    # Chapter range
    parser.add_argument(
        '--chapter-start',
        type=int,
        default=0,
        help='0-based index of first chapter to process (default: 0)',
    )
    parser.add_argument(
        '--chapter-count',
        type=int,
        default=0,
        help='Number of chapters to process (default: 0 = all)',
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=8,
        help='Max parallel translation workers (default: 8)',
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

    # Resolve output path
    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path.with_suffix('.json'))

    # Resolve model
    llm_model = None
    if args.model:
        llm_model = args.model
    elif args.tier and args.tier != 'free':
        llm_model = TIER_DEFAULTS.get(args.tier)

    if args.simplify_hsk4 and not llm_model:
        llm_model = TIER_DEFAULTS["premium"]

    if not llm_model:
        llm_model = TIER_DEFAULTS["standard"]

    # Validate API keys
    def _is_anthropic_model(m):
        return m and m.startswith('claude-') and '/' not in m

    if _is_anthropic_model(llm_model):
        if not os.environ.get('ANTHROPIC_API_KEY'):
            print('Error: ANTHROPIC_API_KEY environment variable not set.', file=sys.stderr)
            sys.exit(1)
    else:
        if not is_openrouter_available():
            print('Error: OpenAI SDK not installed. Install with: pip install openai', file=sys.stderr)
            sys.exit(1)
        if not get_api_key():
            print('Error: OPENROUTER_API_KEY environment variable not set.', file=sys.stderr)
            sys.exit(1)

    # Parse target languages
    target_languages = [l.strip() for l in args.target_languages.split(',')]

    # Print summary
    model_name = MODELS[llm_model]["name"] if llm_model in MODELS else llm_model
    print(f'Input:     {input_path}')
    print(f'Output:    {output_path}')
    print(f'Model:     {model_name}')
    print(f'Languages: {", ".join(target_languages)}')
    if args.simplify_hsk4:
        print(f'Simplify:  HSK 4')
    if args.chapter_start > 0 or args.chapter_count > 0:
        end = args.chapter_start + args.chapter_count if args.chapter_count > 0 else '...'
        print(f'Chapters:  {args.chapter_start}-{end}')
    print()

    extract_to_json(
        epub_path=str(input_path),
        output_json=output_path,
        target_languages=target_languages,
        simplify_hsk4=args.simplify_hsk4,
        model=llm_model,
        chapter_start=args.chapter_start,
        chapter_count=args.chapter_count,
        max_workers=args.max_workers,
    )

    print(f'\nJSON cache written to: {output_path}')


if __name__ == '__main__':
    main()
