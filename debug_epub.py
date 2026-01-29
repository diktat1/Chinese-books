#!/usr/bin/env python3
"""Debug script to analyze EPUB files and output readable text for terminal inspection."""

import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


def analyze_epub(epub_path: str, output_md: str = None):
    """Analyze an EPUB and output structure + sample content."""

    print(f"\n{'='*60}")
    print(f"ANALYZING: {epub_path}")
    print('='*60)

    output_lines = []

    with zipfile.ZipFile(epub_path, 'r') as zf:
        # List all files
        files = zf.namelist()
        print(f"\n## Files in EPUB ({len(files)} total):")

        html_files = []
        image_files = []
        other_files = []

        for f in files:
            if f.endswith(('.xhtml', '.html', '.htm')):
                html_files.append(f)
            elif f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.svg')):
                image_files.append(f)
            else:
                other_files.append(f)

        print(f"\n  HTML/XHTML: {len(html_files)}")
        for f in html_files[:10]:
            size = zf.getinfo(f).file_size
            print(f"    - {f} ({size} bytes)")
        if len(html_files) > 10:
            print(f"    ... and {len(html_files)-10} more")

        print(f"\n  Images: {len(image_files)}")
        for f in image_files[:5]:
            size = zf.getinfo(f).file_size
            print(f"    - {f} ({size} bytes)")
        if len(image_files) > 5:
            print(f"    ... and {len(image_files)-5} more")

        print(f"\n  Other: {len(other_files)}")
        for f in other_files:
            size = zf.getinfo(f).file_size
            print(f"    - {f} ({size} bytes)")

        # Check OPF manifest
        opf_file = None
        for f in files:
            if f.endswith('.opf'):
                opf_file = f
                break

        if opf_file:
            print(f"\n## OPF Manifest: {opf_file}")
            opf_content = zf.read(opf_file).decode('utf-8')

            # Check for nav property
            if 'properties="nav"' in opf_content:
                print("  [OK] Has nav property")
            else:
                print("  [MISSING] No nav property found!")

            # Check spine
            if '<spine' in opf_content:
                # Count itemrefs
                import re
                itemrefs = re.findall(r'<itemref[^>]*idref="([^"]*)"', opf_content)
                print(f"  Spine has {len(itemrefs)} items: {itemrefs[:5]}...")

        # Sample content from first HTML file
        if html_files:
            print(f"\n## Sample content from first HTML file:")
            first_html = html_files[0]
            content = zf.read(first_html).decode('utf-8')

            # Strip tags for plain text preview
            import re
            # Remove script/style
            content = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', content, flags=re.DOTALL)
            # Extract text
            text = re.sub(r'<[^>]+>', ' ', content)
            text = re.sub(r'\s+', ' ', text).strip()

            print(f"  File: {first_html}")
            print(f"  Plain text preview (first 500 chars):")
            print(f"  {text[:500]}...")

            # Check for ruby tags
            ruby_count = content.count('<ruby')
            rt_count = content.count('<rt')
            print(f"\n  Ruby tags: {ruby_count}, RT tags: {rt_count}")

            output_lines.append(f"# {first_html}\n")
            output_lines.append(f"```\n{text[:2000]}\n```\n")

    # Write markdown output if requested
    if output_md:
        with open(output_md, 'w', encoding='utf-8') as f:
            f.write(f"# EPUB Analysis: {epub_path}\n\n")
            f.write('\n'.join(output_lines))
        print(f"\n[Markdown written to {output_md}]")

    print("\n" + "="*60 + "\n")


def extract_text_content(epub_path: str, output_file: str):
    """Extract all text content from EPUB to a readable file."""

    import re

    with zipfile.ZipFile(epub_path, 'r') as zf:
        files = sorted([f for f in zf.namelist() if f.endswith(('.xhtml', '.html'))])

        with open(output_file, 'w', encoding='utf-8') as out:
            out.write(f"# Text content from: {epub_path}\n\n")

            for f in files:
                content = zf.read(f).decode('utf-8')

                # Remove script/style
                content = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', content, flags=re.DOTALL)

                # For ruby tags, show format: 字(pinyin)
                content = re.sub(r'<ruby>([^<]*)<rp>\(</rp><rt>([^<]*)</rt><rp>\)</rp></ruby>',
                               r'\1(\2)', content)

                # Remove remaining tags
                text = re.sub(r'<[^>]+>', '', content)
                text = re.sub(r'\s+', ' ', text).strip()

                if text:
                    out.write(f"\n## {f}\n\n")
                    out.write(text[:3000])  # First 3000 chars per file
                    if len(text) > 3000:
                        out.write(f"\n... [{len(text)-3000} more characters]\n")
                    out.write("\n")

    print(f"Text content extracted to: {output_file}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python debug_epub.py <epub_file>           # Analyze structure")
        print("  python debug_epub.py <epub_file> --text    # Extract text to .txt")
        sys.exit(1)

    epub_path = sys.argv[1]

    if not Path(epub_path).exists():
        print(f"Error: File not found: {epub_path}")
        sys.exit(1)

    if '--text' in sys.argv:
        output_file = Path(epub_path).stem + '_content.txt'
        extract_text_content(epub_path, output_file)
    else:
        analyze_epub(epub_path)
