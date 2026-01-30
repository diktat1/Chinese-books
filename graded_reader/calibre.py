"""
Calibre integration: detect installation and convert EPUB to AZW3.

Uses Calibre's ebook-convert CLI tool for format conversion.
AZW3 (KF8) format properly supports ruby tags for pinyin display,
unlike MOBI which does not support ruby at all.
"""

import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Platform-specific default paths for Calibre
CALIBRE_PATHS = {
    'Darwin': [
        '/Applications/calibre.app/Contents/MacOS/ebook-convert',
    ],
    'Windows': [
        r'C:\Program Files\Calibre2\ebook-convert.exe',
        r'C:\Program Files (x86)\Calibre2\ebook-convert.exe',
    ],
    'Linux': [
        '/usr/bin/ebook-convert',
        '/opt/calibre/ebook-convert',
    ],
}


def find_ebook_convert() -> Optional[str]:
    """
    Find the ebook-convert executable.

    Returns:
        Path to ebook-convert executable, or None if not found.
    """
    # First try PATH
    ebook_convert = shutil.which('ebook-convert')
    if ebook_convert:
        logger.debug(f'Found ebook-convert in PATH: {ebook_convert}')
        return ebook_convert

    # Try platform-specific default paths
    system = platform.system()
    for path in CALIBRE_PATHS.get(system, []):
        if os.path.isfile(path) and os.access(path, os.X_OK):
            logger.debug(f'Found ebook-convert at: {path}')
            return path

    return None


def is_calibre_installed() -> bool:
    """Check if Calibre is installed and ebook-convert is available."""
    return find_ebook_convert() is not None


def get_calibre_version() -> Optional[str]:
    """
    Get Calibre version string.

    Returns:
        Version string or None if Calibre not found.
    """
    ebook_convert = find_ebook_convert()
    if not ebook_convert:
        return None

    try:
        result = subprocess.run(
            [ebook_convert, '--version'],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Output format: "ebook-convert (calibre X.Y.Z)"
        return result.stdout.strip()
    except Exception as e:
        logger.warning(f'Failed to get Calibre version: {e}')
        return None


class CalibreNotFoundError(Exception):
    """Raised when Calibre is not installed."""

    def __init__(self):
        self.message = self._build_message()
        super().__init__(self.message)

    def _build_message(self) -> str:
        system = platform.system()

        if system == 'Darwin':
            install_cmd = 'Download from https://calibre-ebook.com/download_osx'
        elif system == 'Windows':
            install_cmd = 'Download from https://calibre-ebook.com/download_windows'
        else:
            install_cmd = (
                'Install via package manager:\n'
                '  Ubuntu/Debian: sudo apt install calibre\n'
                '  Fedora: sudo dnf install calibre\n'
                '  Or download from https://calibre-ebook.com/download_linux'
            )

        return (
            'Calibre is not installed or ebook-convert is not in PATH.\n\n'
            f'To install Calibre:\n{install_cmd}\n\n'
            'After installation, ensure ebook-convert is available in your PATH.'
        )


def convert_epub_to_azw3(
    epub_path: str,
    azw3_path: str,
    output_profile: str = 'kindle_pw3',
    keep_epub: bool = True,
) -> str:
    """
    Convert an EPUB file to AZW3 format using Calibre.

    Args:
        epub_path: Path to the input EPUB file.
        azw3_path: Path for the output AZW3 file.
        output_profile: Calibre output profile (default: kindle_pw3 for Paperwhite 3+).
        keep_epub: Whether to keep the input EPUB (default: True).

    Returns:
        Path to the created AZW3 file.

    Raises:
        CalibreNotFoundError: If Calibre is not installed.
        FileNotFoundError: If the input EPUB doesn't exist.
        subprocess.CalledProcessError: If conversion fails.
    """
    ebook_convert = find_ebook_convert()
    if not ebook_convert:
        raise CalibreNotFoundError()

    epub_path = Path(epub_path)
    azw3_path = Path(azw3_path)

    if not epub_path.exists():
        raise FileNotFoundError(f'EPUB file not found: {epub_path}')

    # Build the conversion command
    # --output-profile: Use Kindle-specific profile for best compatibility
    # AZW3 (KF8) format inherently supports ruby tags
    cmd = [
        ebook_convert,
        str(epub_path),
        str(azw3_path),
        '--output-profile', output_profile,
    ]

    logger.info(f'Converting {epub_path.name} to AZW3...')
    logger.debug(f'Command: {" ".join(cmd)}')

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for large books
        )

        if result.returncode != 0:
            logger.error(f'ebook-convert stderr: {result.stderr}')
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )

        logger.info(f'Created AZW3: {azw3_path}')

        if not keep_epub:
            epub_path.unlink()
            logger.debug(f'Removed intermediate EPUB: {epub_path}')

        return str(azw3_path)

    except subprocess.TimeoutExpired:
        logger.error('Conversion timed out after 5 minutes')
        raise
