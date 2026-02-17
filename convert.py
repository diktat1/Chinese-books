#!/usr/bin/env python3
"""
Chinese Graded Reader Converter

Converts a Chinese EPUB ebook into a graded reader with:
  - Pinyin annotations above each Chinese character (using <ruby> tags)
  - Translations after each paragraph (Google Translate or OpenRouter LLM)

Usage:
    python convert.py input.epub
    python convert.py input.epub -o output.epub
    python convert.py input.epub --pinyin-only
    python convert.py input.epub --tier standard
    python convert.py input.epub --model deepseek/deepseek-chat
    python convert.py input.epub --list-models
"""

import argparse
import logging
import sys
from pathlib import Path

from graded_reader.epub_processor import process_epub
from graded_reader.llm_simplifier import is_openrouter_available, get_api_key
from graded_reader.models import MODELS, TIER_DEFAULTS, format_model_table


def main():
    parser = argparse.ArgumentParser(
        description='Convert a Chinese EPUB into a graded reader with pinyin and translations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python convert.py book.epub                              # Pinyin + Google Translate (free)
  python convert.py book.epub -o my_reader.epub            # Specify output filename
  python convert.py book.epub --pinyin-only                # Only add pinyin, no translation
  python convert.py book.epub --translation-only           # Only add translation, no pinyin
  python convert.py book.epub --target ja                  # Translate to Japanese
  python convert.py book.epub --word-spacing               # Kindle dictionary lookup support
  python convert.py book.epub --parallel-text              # Side-by-side Chinese + translation
  python convert.py book.epub --tier standard              # Use DeepSeek V3 (~$0.10/book)
  python convert.py book.epub --tier premium               # Use Claude Sonnet 4.5 (~$1.65/book)
  python convert.py book.epub --model deepseek/deepseek-chat  # Specific model
  python convert.py book.epub --simplify-hsk4              # Simplify to HSK 4 vocabulary
  python convert.py book.epub --anki                       # Anki flashcard deck
  python convert.py book.epub --anki --target fr           # Anki deck with French translations
  python convert.py book.epub --anki --no-audio            # Anki deck without TTS audio
  python convert.py book.epub --anki --max-sentences 100   # Limit to first 100 sentences
  python convert.py book.epub --audio --target fr          # Bilingual audiobook (French + Chinese)
  python convert.py book.epub --audio --no-bilingual       # Chinese-only audiobook
  python convert.py --list-models                          # Show available models and pricing
        ''',
    )

    parser.add_argument('input', nargs='?', help='Path to the input Chinese EPUB file')
    parser.add_argument(
        '-o', '--output',
        help='Path for the output file (default: <input>_graded.epub)',
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
        '--parallel-text',
        action='store_true',
        help='Two-column layout: Chinese sentences on the left, translations on '
             'the right, aligned row-by-row.',
    )

    # Model selection
    model_group = parser.add_argument_group('translation engine')
    model_group.add_argument(
        '--tier',
        choices=['free', 'standard', 'premium'],
        help='Translation quality tier. '
             'free=Google Translate (no API key), '
             'standard=DeepSeek V3 (~$0.10/book), '
             'premium=Claude Sonnet 4.5 (~$1.65/book)',
    )
    model_group.add_argument(
        '--model',
        help='OpenRouter model ID (e.g., deepseek/deepseek-chat). '
             'Overrides --tier. Requires OPENROUTER_API_KEY.',
    )
    model_group.add_argument(
        '--list-models',
        action='store_true',
        help='Show available models with pricing and exit',
    )

    # HSK simplification
    parser.add_argument(
        '--simplify-hsk4',
        action='store_true',
        help='Simplify Chinese vocabulary to HSK 4 level using an LLM. '
             'Requires OPENROUTER_API_KEY environment variable.',
    )

    # Anki deck generation
    parser.add_argument(
        '--anki',
        action='store_true',
        help='Generate an Anki deck (.apkg) with sentence cards.',
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

    # Audiobook generation
    parser.add_argument(
        '--audio',
        action='store_true',
        help='Generate audiobook using edge-tts. No API key required for TTS.',
    )
    parser.add_argument(
        '--no-bilingual',
        action='store_true',
        help='Chinese-only audio (skip target language narration)',
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging',
    )

    args = parser.parse_args()

    # Handle --list-models
    if args.list_models:
        print(format_model_table())
        return

    # Input is required for all other operations
    if not args.input:
        parser.error('the following arguments are required: input')

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

    # Resolve LLM model from --model or --tier
    llm_model = None
    if args.model:
        llm_model = args.model
    elif args.tier and args.tier != 'free':
        llm_model = TIER_DEFAULTS.get(args.tier)
    # No --model and no --tier (or --tier free) => Google Translate (llm_model=None)

    # HSK simplification always needs an LLM
    if args.simplify_hsk4 and not llm_model:
        llm_model = TIER_DEFAULTS["premium"]

    # Validate OpenRouter availability if LLM is needed
    if llm_model:
        if not is_openrouter_available():
            print('Error: OpenAI SDK not installed. Install with: pip install openai', file=sys.stderr)
            sys.exit(1)
        if not get_api_key():
            print('Error: OPENROUTER_API_KEY environment variable not set.', file=sys.stderr)
            print('Get your API key from https://openrouter.ai/', file=sys.stderr)
            sys.exit(1)

    # Anki deck mode
    if args.anki:
        from graded_reader.anki_generator import generate_anki_deck

        if args.output:
            anki_output = Path(args.output)
            if anki_output.suffix.lower() != '.apkg':
                anki_output = anki_output.with_suffix('.apkg')
        else:
            anki_output = input_path.with_stem(input_path.stem + '_anki').with_suffix('.apkg')

        model_name = MODELS[llm_model]["name"] if llm_model and llm_model in MODELS else (llm_model or "Google Translate")
        print(f'Input:  {input_path}')
        print(f'Output: {anki_output}')
        print(f'Mode:   Anki deck ({args.source} -> {args.target})')
        print(f'Engine: {model_name}')
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
            llm_model=llm_model,
            include_audio=not args.no_audio,
            max_sentences=args.max_sentences,
        )
        print(f'\nAnki deck written to: {anki_output}')
        return

    # Audiobook mode
    if args.audio:
        from graded_reader.audio_generator import generate_audiobook

        if args.output:
            audio_output = Path(args.output)
        else:
            audio_output = input_path.with_stem(input_path.stem + '_audiobook').with_suffix('.m4b')

        bilingual = not args.no_bilingual

        model_name = MODELS[llm_model]["name"] if llm_model and llm_model in MODELS else (llm_model or "Google Translate")
        print(f'Input:  {input_path}')
        print(f'Output: {audio_output} (format auto-detected)')
        print(f'Mode:   Audiobook')
        print(f'Engine: {model_name}')
        if bilingual:
            print(f'Audio:  Bilingual ({args.target} + {args.source})')
        else:
            print(f'Audio:  Chinese only ({args.source})')
        print()

        result = generate_audiobook(
            epub_path=str(input_path),
            output_path=str(audio_output),
            translation_target=args.target,
            translation_source=args.source,
            bilingual=bilingual,
            llm_model=llm_model,
        )
        print(f'\nAudiobook written to: {result}')
        return

    # EPUB graded reader mode
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

    # Validate flag combinations
    if args.parallel_text and args.pinyin_only:
        print('Error: --parallel-text requires translation. Cannot use with --pinyin-only.', file=sys.stderr)
        sys.exit(1)

    model_name = MODELS[llm_model]["name"] if llm_model and llm_model in MODELS else (llm_model or "Google Translate")
    mode_parts = []
    if args.simplify_hsk4:
        mode_parts.append('HSK4-simplify')
    if add_pinyin:
        mode_parts.append('pinyin')
    if add_translation:
        mode_parts.append(f'translation ({args.source} -> {args.target})')
    if args.word_spacing:
        mode_parts.append('word-spacing')
    if args.parallel_text:
        mode_parts.append('parallel-text')

    print(f'Input:  {input_path}')
    print(f'Output: {output_path}')
    print(f'Engine: {model_name}')
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
        parallel_text=args.parallel_text,
        simplify_hsk4=args.simplify_hsk4,
        llm_model=llm_model,
    )

    print(f'\nOutput written to: {output_path}')


if __name__ == '__main__':
    main()
