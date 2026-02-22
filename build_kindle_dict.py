#!/usr/bin/env python3
"""Build a Chinese-English Kindle dictionary from CC-CEDICT.

Downloads the CC-CEDICT dataset (~124K entries), converts pinyin numbers to
tone marks, and generates Kindle-compatible dictionary source files (HTML + OPF).
Optionally compiles to .mobi with kindlegen.

Usage:
    python build_kindle_dict.py                    # generate source files
    python build_kindle_dict.py --compile          # also compile to .mobi
    python build_kindle_dict.py --test 100         # quick test with 100 entries
    python build_kindle_dict.py --force-download   # re-download CC-CEDICT
"""

import argparse
import gzip
import html
import logging
import re
import shutil
import subprocess
import unicodedata
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CEDICT_URL = "https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz"
CEDICT_GZ = "cedict_1_0_ts_utf-8_mdbg.txt.gz"
CEDICT_TXT = "cedict_1_0_ts_utf-8_mdbg.txt"
ENTRIES_PER_FILE = 1000
DICT_TITLE = "CC-CEDICT Chinese-English Dictionary"
DICT_ID = "cc-cedict-zh-en"

# --- Pinyin tone-mark conversion ---

TONE_MARKS = {
    "a": "āáǎàa", "e": "ēéěèe", "i": "īíǐìi",
    "o": "ōóǒòo", "u": "ūúǔùu", "ü": "ǖǘǚǜü",
}

PINYIN_RE = re.compile(
    r"([BCDFGHJKLMNPQRSTWXYZbcdfghjklmnpqrstwxyz]*)"  # initial
    r"([aeiouüvAEIOUÜV]*)"                              # vowel cluster (v = ü)
    r"([nNgGrR]*)"                                       # final
    r"([1-5])",                                          # tone number
)


def _apply_tone(vowels: str, tone: int) -> str:
    """Apply a tone mark to the correct vowel in a pinyin syllable."""
    if not vowels or tone == 5:
        return vowels
    lower = vowels.lower()
    # Rule: 'a' or 'e' always gets the mark
    for i, ch in enumerate(lower):
        if ch in ("a", "e"):
            return _mark_char(vowels, i, tone)
    # Rule: 'ou' → mark on 'o'
    if "ou" in lower:
        idx = lower.index("o")
        return _mark_char(vowels, idx, tone)
    # Otherwise mark the last vowel (v counts as ü)
    for i in range(len(lower) - 1, -1, -1):
        if lower[i] in TONE_MARKS or lower[i] == "v":
            return _mark_char(vowels, i, tone)
    return vowels


def _mark_char(vowels: str, idx: int, tone: int) -> str:
    ch = vowels[idx]
    is_upper = ch.isupper()
    base = ch.lower()
    if base == "v":
        base = "ü"
    if base not in TONE_MARKS:
        return vowels
    marked = TONE_MARKS[base][tone - 1]
    if is_upper:
        marked = marked.upper()
    return vowels[:idx] + marked + vowels[idx + 1:]


def numbered_to_tonemarks(pinyin: str) -> str:
    """Convert numbered pinyin like 'xue2 xi2' to 'xué xí'.

    Handles u: → ü, preserves capitalization, and tone 5 (neutral).
    """
    # Replace u: with ü before processing
    text = pinyin.replace("u:", "ü").replace("U:", "Ü")

    def replace_syllable(m: re.Match) -> str:
        initial, vowels, final, tone_str = m.groups()
        tone = int(tone_str)
        vowels_marked = _apply_tone(vowels, tone)
        return initial + vowels_marked + final

    return PINYIN_RE.sub(replace_syllable, text)


# --- CC-CEDICT parsing ---

CEDICT_LINE_RE = re.compile(r"^(\S+)\s+(\S+)\s+\[([^\]]+)\]\s+/(.+)/$")


@dataclass
class CedictEntry:
    traditional: str
    simplified: str
    pinyin_numbered: str
    pinyin_tonemarks: str
    definitions: list[str]
    raw_defs: str

    @property
    def same_form(self) -> bool:
        return self.traditional == self.simplified


def norm(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def download_cedict(output_dir: Path, force: bool = False) -> Path:
    """Download and extract CC-CEDICT, returning path to the text file."""
    gz_path = output_dir / CEDICT_GZ
    txt_path = output_dir / CEDICT_TXT

    if txt_path.exists() and not force:
        log.info("Using cached %s", txt_path)
        return txt_path

    log.info("Downloading CC-CEDICT from %s ...", CEDICT_URL)
    urllib.request.urlretrieve(CEDICT_URL, gz_path)
    log.info("Extracting ...")
    with gzip.open(gz_path, "rb") as f_in:
        txt_path.write_bytes(f_in.read())
    gz_path.unlink()
    log.info("Saved %s", txt_path)
    return txt_path


def parse_cedict(txt_path: Path) -> list[CedictEntry]:
    """Parse CC-CEDICT text file into a list of entries."""
    entries = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = CEDICT_LINE_RE.match(line)
        if not m:
            continue
        trad, simp, pinyin_num, raw_defs = m.groups()
        trad = norm(trad)
        simp = norm(simp)
        pinyin_num = norm(pinyin_num)
        defs = [norm(d.strip()) for d in raw_defs.split("/") if d.strip()]
        pinyin_tm = numbered_to_tonemarks(pinyin_num)
        entries.append(CedictEntry(
            traditional=trad,
            simplified=simp,
            pinyin_numbered=pinyin_num,
            pinyin_tonemarks=pinyin_tm,
            definitions=defs,
            raw_defs=raw_defs,
        ))
    log.info("Parsed %d entries", len(entries))
    return entries


# --- Kindle dictionary HTML generation ---

INLINE_CSS = """\
<style>
  .entry { margin-bottom: 0.8em; }
  .hw { font-size: 1.4em; font-weight: bold; }
  .py { color: #555; font-style: italic; }
  .var { color: #888; font-size: 0.9em; }
  .cl { color: #666; font-size: 0.9em; }
  ol { margin: 0.2em 0 0 1.2em; padding: 0; }
  li { margin-bottom: 0.1em; }
</style>"""


def _format_defs(defs: list[str]) -> str:
    """Format definitions, separating measure words from regular defs."""
    regular = []
    classifiers = []
    for d in defs:
        if d.startswith("CL:"):
            classifiers.append(d)
        else:
            regular.append(html.escape(d))

    parts = []
    if regular:
        if len(regular) == 1:
            parts.append(f"<p>{regular[0]}</p>")
        else:
            items = "".join(f"<li>{d}</li>" for d in regular)
            parts.append(f"<ol>{items}</ol>")
    if classifiers:
        cl_text = html.escape("; ".join(classifiers))
        parts.append(f'<p class="cl">{cl_text}</p>')
    return "\n    ".join(parts)


def _make_entry_block(
    entry: CedictEntry,
    headword: str,
    entry_id: str,
    variant_label: str | None = None,
    cross_ref_id: str | None = None,
    cross_ref_hw: str | None = None,
) -> str:
    """Generate one <idx:entry> block."""
    hw_esc = html.escape(headword)
    py_esc = html.escape(entry.pinyin_tonemarks)
    defs_html = _format_defs(entry.definitions)

    variant_html = ""
    if variant_label and cross_ref_id and cross_ref_hw:
        cr_esc = html.escape(cross_ref_hw)
        variant_html = (
            f'\n    <span class="var">{variant_label} '
            f'<a href="#{cross_ref_id}">{cr_esc}</a></span>'
        )

    return f"""\
<idx:entry name="default" scriptable="yes" spell="yes">
  <idx:short><a id="{entry_id}"></a>
    <idx:orth value="{hw_esc}">
      <idx:infl><idx:iform value="{hw_esc}" /></idx:infl>
    </idx:orth>
  </idx:short>
  <div class="entry">
    <span class="hw">{hw_esc}</span> <span class="py">{py_esc}</span>{variant_html}
    {defs_html}
  </div>
</idx:entry>"""


@dataclass
class EntryBlock:
    """One or two HTML blocks for a single CC-CEDICT entry."""
    simplified: str
    blocks: list[str] = field(default_factory=list)
    anchors: set[str] = field(default_factory=set)


def generate_entry_blocks(entries: list[CedictEntry]) -> list[EntryBlock]:
    """Generate Kindle entry blocks from parsed CC-CEDICT entries."""
    results = []
    for i, e in enumerate(entries):
        eb = EntryBlock(simplified=e.simplified)
        simp_id = f"s{i}"
        trad_id = f"t{i}"

        if e.same_form:
            block = _make_entry_block(e, e.simplified, simp_id)
            eb.blocks.append(block)
            eb.anchors.add(simp_id)
        else:
            # Simplified entry
            simp_block = _make_entry_block(
                e, e.simplified, simp_id,
                variant_label="Trad.",
                cross_ref_id=trad_id,
                cross_ref_hw=e.traditional,
            )
            eb.blocks.append(simp_block)
            eb.anchors.add(simp_id)

            # Traditional entry
            trad_block = _make_entry_block(
                e, e.traditional, trad_id,
                variant_label="Simp.",
                cross_ref_id=simp_id,
                cross_ref_hw=e.simplified,
            )
            eb.blocks.append(trad_block)
            eb.anchors.add(trad_id)

        results.append(eb)
    return results


def wrap_html(body: str, title: str) -> str:
    """Wrap entry blocks in a complete Kindle dictionary HTML page."""
    return f"""\
<html xmlns:mbp="https://kindlegen.s3.amazonaws.com/AmazonKindlePublishingGuidelines.pdf"
      xmlns:idx="https://kindlegen.s3.amazonaws.com/AmazonKindlePublishingGuidelines.pdf">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>{html.escape(title)}</title>
{INLINE_CSS}
</head>
<body>
<mbp:frameset>
{body}
</mbp:frameset>
</body>
</html>"""


def make_cover_page() -> str:
    return wrap_html(
        f"<h1>{html.escape(DICT_TITLE)}</h1>\n"
        "<p>Built from CC-CEDICT — a community-maintained free Chinese-English dictionary.</p>\n"
        "<p>Tap any Chinese word to look up its pinyin and English definition.</p>",
        DICT_TITLE,
    )


def make_about_page() -> str:
    return wrap_html(
        f"<h2>About {html.escape(DICT_TITLE)}</h2>\n"
        "<p>This dictionary was generated from the CC-CEDICT project "
        "(https://cc-cedict.org/wiki/), a continuation of the CEDICT project.</p>\n"
        "<p>CC-CEDICT is licensed under the Creative Commons Attribution-ShareAlike 4.0 "
        "International License.</p>\n"
        "<p>It covers 124,000+ entries with simplified &amp; traditional Chinese, "
        "pinyin with tone marks, and English definitions.</p>",
        "About",
    )


# --- OPF manifest ---

def make_opf(entry_filenames: list[str]) -> str:
    manifest_items = [
        '<item id="cover" href="cover.html" media-type="application/xhtml+xml" />',
        '<item id="about" href="about.html" media-type="application/xhtml+xml" />',
    ]
    spine_refs = [
        '<itemref idref="cover" />',
        '<itemref idref="about" />',
    ]

    for fname in entry_filenames:
        item_id = fname.replace(".html", "")
        manifest_items.append(
            f'<item id="{item_id}" href="{fname}" media-type="application/xhtml+xml" />'
        )
        spine_refs.append(f'<itemref idref="{item_id}" />')

    manifest_str = "\n    ".join(manifest_items)
    spine_str = "\n    ".join(spine_refs)

    return f"""\
<?xml version="1.0" encoding="utf-8"?>
<package version="2.0" xmlns="http://www.idpf.org/2007/opf" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:opf="http://www.idpf.org/2007/opf">
    <dc:title>{html.escape(DICT_TITLE)}</dc:title>
    <dc:language>zh</dc:language>
    <dc:identifier id="uid">{DICT_ID}</dc:identifier>
    <dc:creator>CC-CEDICT Community</dc:creator>
    <x-metadata>
      <DictionaryInLanguage>zh</DictionaryInLanguage>
      <DictionaryOutLanguage>en</DictionaryOutLanguage>
      <DefaultLookupIndex>default</DefaultLookupIndex>
    </x-metadata>
  </metadata>
  <manifest>
    {manifest_str}
  </manifest>
  <spine>
    {spine_str}
  </spine>
</package>"""


# --- Main build pipeline ---

def build(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download & parse
    txt_path = download_cedict(output_dir, force=args.force_download)
    entries = parse_cedict(txt_path)

    if args.test:
        entries = entries[:args.test]
        log.info("Test mode: using first %d entries", args.test)

    # Step 2-3: Sort by simplified headword, generate blocks
    entries.sort(key=lambda e: e.simplified)
    log.info("Generating entry blocks ...")
    all_blocks = generate_entry_blocks(entries)

    # Step 4: Split into files
    entry_filenames = []
    file_idx = 0
    for chunk_start in range(0, len(all_blocks), ENTRIES_PER_FILE):
        chunk = all_blocks[chunk_start:chunk_start + ENTRIES_PER_FILE]
        body = "\n\n".join(block for eb in chunk for block in eb.blocks)
        fname = f"entries_{file_idx:04d}.html"
        (output_dir / fname).write_text(wrap_html(body, f"Entries {file_idx}"), encoding="utf-8")
        entry_filenames.append(fname)
        file_idx += 1

    log.info("Generated %d entry files", len(entry_filenames))

    # Cover & about pages
    (output_dir / "cover.html").write_text(make_cover_page(), encoding="utf-8")
    (output_dir / "about.html").write_text(make_about_page(), encoding="utf-8")

    # Step 5: OPF manifest
    opf_content = make_opf(entry_filenames)
    opf_path = output_dir / "dictionary.opf"
    opf_path.write_text(opf_content, encoding="utf-8")
    log.info("Wrote %s", opf_path)

    # Clean up raw CEDICT text (keep it cached for re-runs)
    total_entries = sum(len(eb.blocks) for eb in all_blocks)
    log.info("Done! %d lookup entries across %d files.", total_entries, len(entry_filenames))

    # Step 6: Optional compilation
    if args.compile:
        kindlegen = shutil.which("kindlegen")
        if kindlegen:
            log.info("Compiling with kindlegen ...")
            result = subprocess.run(
                [kindlegen, str(opf_path)],
                capture_output=True, text=True,
            )
            # kindlegen returns 1 for warnings (still produces output)
            if result.returncode <= 1:
                mobi_path = opf_path.with_suffix(".mobi")
                log.info("Success! Dictionary at %s", mobi_path)
            else:
                log.error("kindlegen failed:\n%s", result.stderr or result.stdout)
        else:
            log.warning(
                "kindlegen not found on PATH. To compile:\n"
                "  1. Download kindlegen from Amazon\n"
                "  2. Run: kindlegen %s", opf_path
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Chinese-English Kindle dictionary from CC-CEDICT",
    )
    parser.add_argument(
        "--output-dir", default="kindle-dictionary",
        help="Output directory (default: kindle-dictionary)",
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Compile to .mobi with kindlegen (must be on PATH)",
    )
    parser.add_argument(
        "--test", type=int, metavar="N",
        help="Only use the first N entries (for quick testing)",
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Re-download CC-CEDICT even if cached",
    )
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
