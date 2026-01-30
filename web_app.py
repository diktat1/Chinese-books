#!/usr/bin/env python3
"""
Simple Flask web app for drag-and-drop EPUB conversion.

Run with:
    python web_app.py

Then open http://localhost:5000 in your browser.
"""

import os
import tempfile
import logging
from pathlib import Path

from flask import Flask, request, send_file, render_template_string, jsonify

from graded_reader.epub_processor import process_epub
from graded_reader.calibre import (
    is_calibre_installed,
    get_calibre_version,
    convert_epub_to_azw3,
    CalibreNotFoundError,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload

HTML_PAGE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Chinese Graded Reader Converter</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
            padding: 40px;
            max-width: 600px;
            width: 90%;
        }
        h1 {
            font-size: 1.5em;
            margin-bottom: 8px;
        }
        .subtitle {
            color: #888;
            margin-bottom: 30px;
            font-size: 0.95em;
        }
        .drop-zone {
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 50px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 20px;
        }
        .drop-zone:hover, .drop-zone.drag-over {
            border-color: #e74c3c;
            background: #fef5f5;
        }
        .drop-zone p {
            font-size: 1.1em;
            color: #666;
        }
        .drop-zone .icon {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .drop-zone .filename {
            margin-top: 10px;
            font-weight: bold;
            color: #333;
        }
        .options {
            margin-bottom: 20px;
        }
        .options label {
            display: block;
            padding: 8px 0;
            cursor: pointer;
        }
        .options input[type="checkbox"] {
            margin-right: 8px;
        }
        button {
            background: #e74c3c;
            color: white;
            border: none;
            padding: 14px 28px;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
            width: 100%;
            transition: background 0.2s;
        }
        button:hover { background: #c0392b; }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .status {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            display: none;
        }
        .status.processing {
            display: block;
            background: #fff3cd;
            color: #856404;
        }
        .status.done {
            display: block;
            background: #d4edda;
            color: #155724;
        }
        .status.error {
            display: block;
            background: #f8d7da;
            color: #721c24;
        }
        .note {
            margin-top: 20px;
            font-size: 0.85em;
            color: #999;
            line-height: 1.5;
        }
        .kindle-status {
            font-size: 0.85em;
            padding: 8px 12px;
            border-radius: 4px;
            margin-top: 8px;
            margin-bottom: 8px;
        }
        .kindle-status.available {
            background: #d4edda;
            color: #155724;
        }
        .kindle-status.missing {
            background: #f8d7da;
            color: #721c24;
        }
        .kindle-status a {
            color: inherit;
            font-weight: bold;
        }
        .options input:disabled + span {
            color: #999;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>Chinese Graded Reader Converter</h1>
    <p class="subtitle">Upload a Chinese EPUB to add pinyin and English translations</p>

    <form id="uploadForm" enctype="multipart/form-data">
        <div class="drop-zone" id="dropZone">
            <div class="icon">&#128214;</div>
            <p>Drag &amp; drop an EPUB file here<br>or click to browse</p>
            <input type="file" name="file" id="fileInput" accept=".epub" hidden>
            <div class="filename" id="fileName"></div>
        </div>

        <div class="options">
            <label><input type="checkbox" name="add_pinyin" checked> Add pinyin above characters</label>
            <label><input type="checkbox" name="add_translation" checked> Add English translation after paragraphs</label>
            <label><input type="checkbox" name="kindle_output" id="kindleOutput"> <span>Output as AZW3 for Kindle</span></label>
            <div class="kindle-status" id="kindleStatus"></div>
        </div>

        <button type="submit" id="convertBtn" disabled>Convert to Graded Reader</button>
    </form>

    <div class="status" id="status"></div>

    <p class="note">
        Pinyin-only conversion is fast (seconds). Adding translations takes longer
        because each paragraph is sent to Google Translate.<br>
        For Kindle: check "Output as AZW3" for direct Kindle support (requires Calibre).
    </p>
</div>

<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const form = document.getElementById('uploadForm');
const btn = document.getElementById('convertBtn');
const status = document.getElementById('status');
const kindleCheckbox = document.getElementById('kindleOutput');
const kindleStatus = document.getElementById('kindleStatus');

// Check Calibre availability on page load
async function checkCalibre() {
    try {
        const resp = await fetch('/check-calibre');
        const data = await resp.json();
        if (data.available) {
            kindleStatus.className = 'kindle-status available';
            kindleStatus.textContent = 'Calibre detected: ' + data.version;
        } else {
            kindleStatus.className = 'kindle-status missing';
            kindleStatus.innerHTML = 'Calibre not installed. <a href="https://calibre-ebook.com/download" target="_blank">Install Calibre</a> for AZW3 support.';
            kindleCheckbox.disabled = true;
        }
    } catch (e) {
        kindleStatus.className = 'kindle-status missing';
        kindleStatus.textContent = 'Could not check Calibre status';
        kindleCheckbox.disabled = true;
    }
}
checkCalibre();

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        updateFileName();
    }
});

fileInput.addEventListener('change', updateFileName);

function updateFileName() {
    if (fileInput.files.length) {
        fileName.textContent = fileInput.files[0].name;
        btn.disabled = false;
    }
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!fileInput.files.length) return;

    const kindleOutput = kindleCheckbox.checked;
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('add_pinyin', form.querySelector('[name=add_pinyin]').checked);
    formData.append('add_translation', form.querySelector('[name=add_translation]').checked);
    formData.append('kindle_output', kindleOutput);

    btn.disabled = true;
    status.className = 'status processing';
    const formatMsg = kindleOutput ? ' Converting to AZW3 for Kindle.' : '';
    status.textContent = 'Converting... This may take a while if translation is enabled.' + formatMsg;

    try {
        const resp = await fetch('/convert', { method: 'POST', body: formData });
        if (!resp.ok) {
            const err = await resp.text();
            throw new Error(err);
        }
        // Trigger download
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        const origName = fileInput.files[0].name.replace('.epub', '');
        const ext = kindleOutput ? '.azw3' : '.epub';
        a.href = url;
        a.download = origName + '_graded' + ext;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);

        status.className = 'status done';
        const formatDone = kindleOutput ? ' (AZW3 for Kindle)' : '';
        status.textContent = 'Conversion complete!' + formatDone + ' Your download should start automatically.';
    } catch (err) {
        status.className = 'status error';
        status.textContent = 'Error: ' + err.message;
    } finally {
        btn.disabled = false;
    }
});
</script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


@app.route('/check-calibre')
def check_calibre():
    """Check if Calibre is installed and return status."""
    available = is_calibre_installed()
    version = get_calibre_version() if available else None
    return jsonify({
        'available': available,
        'version': version,
    })


@app.route('/convert', methods=['POST'])
def convert():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if not file.filename or not file.filename.lower().endswith('.epub'):
        return 'Please upload an .epub file', 400

    add_pinyin = request.form.get('add_pinyin', 'true') == 'true'
    add_translation = request.form.get('add_translation', 'true') == 'true'
    kindle_output = request.form.get('kindle_output', 'false') == 'true'

    if not add_pinyin and not add_translation:
        return 'Select at least one option (pinyin or translation)', 400

    if kindle_output and not is_calibre_installed():
        return 'Calibre is not installed. Cannot convert to AZW3.', 400

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, 'input.epub')
        epub_output_path = os.path.join(tmpdir, 'output.epub')

        file.save(input_path)

        try:
            process_epub(
                input_path=input_path,
                output_path=epub_output_path,
                add_pinyin=add_pinyin,
                add_translation=add_translation,
            )

            if kindle_output:
                azw3_output_path = os.path.join(tmpdir, 'output.azw3')
                convert_epub_to_azw3(
                    epub_path=epub_output_path,
                    azw3_path=azw3_output_path,
                    keep_epub=False,
                )
                return send_file(
                    azw3_output_path,
                    as_attachment=True,
                    download_name=file.filename.replace('.epub', '_graded.azw3'),
                    mimetype='application/x-mobi8-ebook',
                )
            else:
                return send_file(
                    epub_output_path,
                    as_attachment=True,
                    download_name=file.filename.replace('.epub', '_graded.epub'),
                    mimetype='application/epub+zip',
                )

        except CalibreNotFoundError as e:
            return f'Calibre error: {e.message}', 500
        except Exception as e:
            logging.exception('Conversion failed')
            return f'Conversion failed: {e}', 500


if __name__ == '__main__':
    print('Starting server at http://localhost:5000')
    print('Open this URL in your browser to upload an EPUB file.')
    app.run(host='0.0.0.0', port=5000, debug=False)
