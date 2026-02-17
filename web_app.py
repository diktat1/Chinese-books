#!/usr/bin/env python3
"""
Web app for converting Chinese EPUBs into learning materials.

Upload an EPUB and get back any combination of:
  - Graded reader EPUB/AZW3 (pinyin + translation)
  - Anki flashcard deck
  - M4B audiobook with chapter markers

Run with:
    python web_app.py

Then open http://localhost:5000 in your browser.
"""

import os
import tempfile
import logging
import zipfile
from pathlib import Path

from flask import Flask, request, send_file, render_template_string, jsonify

from graded_reader.epub_processor import process_epub
from graded_reader.calibre import (
    is_calibre_installed,
    get_calibre_version,
    convert_epub_to_azw3,
    CalibreNotFoundError,
)
from graded_reader.claude_simplifier import is_openrouter_available, get_api_key

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload

HTML_PAGE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Chinese Graded Reader</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            min-height: 100vh;
            padding: 20px;
        }
        .page {
            max-width: 520px;
            margin: 0 auto;
        }
        h1 { font-size: 1.4em; margin-bottom: 4px; }
        .sub { color: #888; font-size: .9em; margin-bottom: 20px; }
        .card {
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,.08);
            padding: 24px;
            margin-bottom: 16px;
        }
        .section-title {
            font-weight: 600; font-size: .92em; color: #555;
            margin-bottom: 8px; text-transform: uppercase; letter-spacing: .5px;
        }
        /* Drop zone */
        .drop-zone {
            border: 2px dashed #ccc; border-radius: 8px;
            padding: 40px 16px; text-align: center; cursor: pointer;
            transition: all 0.2s; margin-bottom: 16px;
        }
        .drop-zone:hover, .drop-zone.active { border-color: #e74c3c; background: #fef5f5; }
        .drop-zone .icon { font-size: 2.2em; margin-bottom: 6px; }
        .drop-zone .name { font-weight: 600; color: #333; margin-top: 8px; word-break: break-all; }
        .drop-zone input { display: none; }
        /* Outputs */
        .outputs { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
        .output-btn {
            flex: 1; min-width: 100px; padding: 14px 8px; border: 2px solid #e0e0e0;
            border-radius: 10px; background: #fff; cursor: pointer; text-align: center;
            transition: all .15s; position: relative;
        }
        .output-btn:hover { border-color: #ccc; }
        .output-btn.selected { border-color: #e74c3c; background: #fef5f5; }
        .output-btn .out-icon { font-size: 1.6em; display: block; margin-bottom: 4px; }
        .output-btn .out-label { font-size: .82em; font-weight: 600; }
        .output-btn .out-desc { font-size: .72em; color: #999; margin-top: 2px; }
        .output-btn .check {
            position: absolute; top: 6px; right: 8px; font-size: .8em; color: #e74c3c;
            display: none;
        }
        .output-btn.selected .check { display: block; }
        /* Options grid */
        .opt-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 16px; }
        .opt-grid label { font-size: .9em; padding: 4px 0; cursor: pointer; }
        .opt-grid input { margin-right: 6px; }
        select {
            width: 100%; padding: 8px 10px; border: 1px solid #ddd;
            border-radius: 6px; font-size: .95em; background: #fff;
        }
        hr { border: none; border-top: 1px solid #eee; margin: 14px 0; }
        /* Button */
        .convert-btn {
            background: #e74c3c; color: #fff; border: none; padding: 16px;
            border-radius: 10px; font-size: 1.05em; font-weight: 600;
            cursor: pointer; width: 100%; transition: background .2s;
        }
        .convert-btn:hover { background: #c0392b; }
        .convert-btn:disabled { background: #ccc; cursor: not-allowed; }
        /* Status */
        .status {
            margin-top: 14px; padding: 14px; border-radius: 8px;
            font-size: .9em; display: none;
        }
        .status.info { display: block; background: #fff3cd; color: #856404; }
        .status.ok { display: block; background: #d4edda; color: #155724; }
        .status.err { display: block; background: #f8d7da; color: #721c24; }
        /* Dep badges */
        .dep-badge {
            display: inline-block; font-size: .78em; padding: 3px 8px;
            border-radius: 4px; margin-right: 6px; margin-top: 6px;
        }
        .dep-badge.ok { background: #d4edda; color: #155724; }
        .dep-badge.miss { background: #f8d7da; color: #721c24; }
        .dep-badge a { color: inherit; font-weight: 600; }
        .small { font-size: .78em; color: #aaa; line-height: 1.5; margin-top: 10px; }
        /* Progress bar */
        .progress-bar {
            height: 4px; background: #eee; border-radius: 2px;
            margin-top: 8px; overflow: hidden; display: none;
        }
        .progress-bar.active { display: block; }
        .progress-bar .fill {
            height: 100%; background: #e74c3c; border-radius: 2px;
            animation: progress 2s ease-in-out infinite;
        }
        @keyframes progress {
            0% { width: 0; } 50% { width: 80%; } 100% { width: 100%; }
        }
    </style>
</head>
<body>
<div class="page">

<div class="card">
    <h1>Chinese Graded Reader</h1>
    <p class="sub">Turn any Chinese EPUB into a learning kit</p>

    <div class="drop-zone" id="dropZone">
        <div class="icon">&#128214;</div>
        <p>Drop an EPUB here or tap to browse</p>
        <div class="name" id="fileName"></div>
        <input type="file" id="fileInput" accept=".epub">
    </div>

    <div class="section-title">What do you want?</div>
    <div class="outputs" id="outputs">
        <div class="output-btn selected" data-output="epub">
            <span class="check">&#10003;</span>
            <span class="out-icon">&#128216;</span>
            <span class="out-label">Graded Reader</span>
            <span class="out-desc">EPUB with pinyin</span>
        </div>
        <div class="output-btn" data-output="anki">
            <span class="check">&#10003;</span>
            <span class="out-icon">&#127183;</span>
            <span class="out-label">Flashcards</span>
            <span class="out-desc">Anki deck</span>
        </div>
        <div class="output-btn" data-output="audio">
            <span class="check">&#10003;</span>
            <span class="out-icon">&#127911;</span>
            <span class="out-label">Audiobook</span>
            <span class="out-desc">M4B for iPhone</span>
        </div>
    </div>
</div>

<div class="card" id="optionsCard">
    <div class="section-title">Options</div>
    <div class="opt-grid">
        <div>
            <label>Target language</label>
            <select id="optTarget">
                <option value="fr">French</option>
                <option value="en" selected>English</option>
                <option value="ja">Japanese</option>
                <option value="ko">Korean</option>
                <option value="de">German</option>
                <option value="es">Spanish</option>
                <option value="it">Italian</option>
                <option value="pt">Portuguese</option>
            </select>
        </div>
        <div>
            <label>Quality</label>
            <select id="optQuality">
                <option value="standard">Standard (Google)</option>
                <option value="claude">Premium (Claude)</option>
                <option value="simplified">Simplified (HSK 4)</option>
            </select>
        </div>
    </div>
    <hr>
    <div class="opt-grid" id="epubOptions">
        <label><input type="checkbox" id="optPinyin" checked> Pinyin annotations</label>
        <label><input type="checkbox" id="optTranslation" checked> Translation</label>
        <label><input type="checkbox" id="optWordSpacing"> Word spacing</label>
        <label><input type="checkbox" id="optParallelText"> Parallel text</label>
        <label><input type="checkbox" id="optKindleOutput"> Output as AZW3</label>
    </div>
    <div class="opt-grid" id="audioOptions" style="display:none">
        <label><input type="checkbox" id="optBilingual" checked> Bilingual narration</label>
    </div>
    <div id="deps"></div>
</div>

<div class="card">
    <button class="convert-btn" id="convertBtn" disabled>Convert</button>
    <div class="progress-bar" id="progressBar"><div class="fill"></div></div>
    <div class="status" id="status"></div>
    <p class="small">
        Graded reader: seconds (pinyin only) to minutes (with translation).<br>
        Anki: ~1 min per 100 sentences. Audiobook: ~2 min per chapter.<br>
        Multiple outputs are bundled in a single ZIP download.
    </p>
</div>

</div>

<script>
const $ = id => document.getElementById(id);

// --- Output selection ---
const selected = new Set(['epub']);
document.querySelectorAll('.output-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const out = btn.dataset.output;
        if (selected.has(out)) selected.delete(out);
        else selected.add(out);
        btn.classList.toggle('selected');
        updateUI();
    });
});

function updateUI() {
    $('epubOptions').style.display = selected.has('epub') ? '' : 'none';
    $('audioOptions').style.display = selected.has('audio') ? '' : 'none';
    const label = selected.size > 1 ? 'Convert & Download ZIP' :
                  selected.has('epub') ? 'Convert to Graded Reader' :
                  selected.has('anki') ? 'Generate Flashcards' :
                  selected.has('audio') ? 'Generate Audiobook' : 'Select an output';
    $('convertBtn').textContent = label;
}

// --- File picker ---
const dropZone = $('dropZone');
const fileInput = $('fileInput');
let selectedFile = null;

dropZone.onclick = () => fileInput.click();
dropZone.ondragover = e => { e.preventDefault(); dropZone.classList.add('active'); };
dropZone.ondragleave = () => dropZone.classList.remove('active');
dropZone.ondrop = e => {
    e.preventDefault(); dropZone.classList.remove('active');
    if (e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; pickFile(); }
};
fileInput.onchange = pickFile;

function pickFile() {
    if (fileInput.files.length) {
        selectedFile = fileInput.files[0];
        $('fileName').textContent = selectedFile.name;
        dropZone.classList.add('active');
        $('convertBtn').disabled = false;
    }
}

// --- Dependency checks ---
async function checkDeps() {
    const deps = $('deps');
    let html = '';
    try {
        const r1 = await fetch('/check-deps');
        const d = await r1.json();
        if (d.calibre) html += '<span class="dep-badge ok">Calibre ' + d.calibre_version + '</span>';
        else html += '<span class="dep-badge miss">No Calibre (AZW3 disabled)</span>';
        if (d.claude) html += '<span class="dep-badge ok">Claude API ready</span>';
        else html += '<span class="dep-badge miss">No Claude API</span>';
        if (d.ffmpeg) html += '<span class="dep-badge ok">ffmpeg (M4B)</span>';
        else html += '<span class="dep-badge miss">No ffmpeg (ZIP fallback)</span>';

        if (!d.calibre) $('optKindleOutput').disabled = true;
        if (!d.claude) {
            const q = $('optQuality');
            q.querySelectorAll('option').forEach(o => {
                if (o.value !== 'standard') o.disabled = true;
            });
        }
    } catch(e) {
        html = '<span class="dep-badge miss">Could not check dependencies</span>';
    }
    deps.innerHTML = html;
}
checkDeps();

// --- Convert ---
$('convertBtn').onclick = async () => {
    if (!selectedFile || selected.size === 0) return;

    const btn = $('convertBtn');
    const status = $('status');
    const bar = $('progressBar');

    btn.disabled = true;
    bar.classList.add('active');
    status.className = 'status info';
    status.style.display = 'block';
    status.textContent = 'Processing...';

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('outputs', JSON.stringify([...selected]));
    formData.append('target', $('optTarget').value);
    formData.append('quality', $('optQuality').value);
    formData.append('add_pinyin', $('optPinyin').checked);
    formData.append('add_translation', $('optTranslation').checked);
    formData.append('word_spacing', $('optWordSpacing').checked);
    formData.append('parallel_text', $('optParallelText').checked);
    formData.append('kindle_output', $('optKindleOutput').checked);
    formData.append('bilingual', $('optBilingual').checked);

    try {
        const resp = await fetch('/convert', { method: 'POST', body: formData });
        if (!resp.ok) throw new Error(await resp.text());

        const blob = await resp.blob();
        const cd = resp.headers.get('Content-Disposition') || '';
        const match = cd.match(/filename="?([^"]+)"?/);
        const filename = match ? match[1] : 'output.zip';

        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = filename;
        document.body.appendChild(a); a.click(); a.remove();
        URL.revokeObjectURL(url);

        status.className = 'status ok';
        status.textContent = 'Done! Download started.';
    } catch(err) {
        status.className = 'status err';
        status.textContent = 'Error: ' + err.message;
    } finally {
        btn.disabled = false;
        bar.classList.remove('active');
    }
};

// Parallel text requires translation
$('optParallelText').addEventListener('change', function() {
    if (this.checked && !$('optTranslation').checked) {
        $('optTranslation').checked = true;
    }
});
$('optTranslation').addEventListener('change', function() {
    if (!this.checked && $('optParallelText').checked) {
        $('optParallelText').checked = false;
    }
});

updateUI();
</script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


@app.route('/check-deps')
def check_deps():
    """Check all optional dependency availability at once."""
    import shutil

    calibre = is_calibre_installed()
    return jsonify({
        'calibre': calibre,
        'calibre_version': get_calibre_version() if calibre else None,
        'claude': is_openrouter_available() and bool(get_api_key()),
        'ffmpeg': shutil.which('ffmpeg') is not None,
    })


# Keep old endpoints for backwards compat
@app.route('/check-calibre')
def check_calibre():
    available = is_calibre_installed()
    return jsonify({'available': available, 'version': get_calibre_version() if available else None})


@app.route('/check-claude')
def check_claude():
    if not is_openrouter_available():
        return jsonify({'available': False, 'reason': 'no_sdk'})
    if not get_api_key():
        return jsonify({'available': False, 'reason': 'no_api_key'})
    return jsonify({'available': True})


@app.route('/convert', methods=['POST'])
def convert():
    import json

    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if not file.filename or not file.filename.lower().endswith('.epub'):
        return 'Please upload an .epub file', 400

    outputs = json.loads(request.form.get('outputs', '["epub"]'))
    target = request.form.get('target', 'en')
    quality = request.form.get('quality', 'standard')

    add_pinyin = request.form.get('add_pinyin', 'true') == 'true'
    add_translation = request.form.get('add_translation', 'true') == 'true'
    word_spacing = request.form.get('word_spacing', 'false') == 'true'
    parallel_text = request.form.get('parallel_text', 'false') == 'true'
    kindle_output = request.form.get('kindle_output', 'false') == 'true'
    bilingual = request.form.get('bilingual', 'true') == 'true'

    use_claude = quality in ('claude', 'simplified')
    use_opus = False
    simplify_hsk4 = quality == 'simplified'

    if not outputs:
        return 'Select at least one output type', 400

    if kindle_output and not is_calibre_installed():
        return 'Calibre is not installed. Cannot convert to AZW3.', 400

    if (simplify_hsk4 or use_claude) and (not is_openrouter_available() or not get_api_key()):
        return 'Claude API not available. Select Standard quality.', 400

    orig_name = Path(file.filename).stem

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, 'input.epub')
        file.save(input_path)

        generated_files = []  # (filename, path)

        try:
            # --- Graded reader EPUB/AZW3 ---
            if 'epub' in outputs:
                epub_path = os.path.join(tmpdir, 'graded.epub')
                process_epub(
                    input_path=input_path,
                    output_path=epub_path,
                    add_pinyin=add_pinyin,
                    add_translation=add_translation,
                    translation_target=target,
                    parallel_text=parallel_text,
                    word_spacing=word_spacing,
                    simplify_hsk4=simplify_hsk4,
                    use_claude_translator=use_claude,
                    use_opus=use_opus,
                )
                if kindle_output:
                    azw3_path = os.path.join(tmpdir, 'graded.azw3')
                    convert_epub_to_azw3(epub_path=epub_path, azw3_path=azw3_path)
                    generated_files.append((f'{orig_name}_graded.azw3', azw3_path))
                else:
                    generated_files.append((f'{orig_name}_graded.epub', epub_path))

            # --- Anki deck ---
            if 'anki' in outputs:
                from graded_reader.anki_generator import generate_anki_deck
                anki_path = os.path.join(tmpdir, 'flashcards.apkg')
                generate_anki_deck(
                    epub_path=input_path,
                    output_path=anki_path,
                    translation_target=target,
                    use_claude=use_claude,
                    use_opus=use_opus,
                )
                generated_files.append((f'{orig_name}_flashcards.apkg', anki_path))

            # --- Audiobook ---
            if 'audio' in outputs:
                from graded_reader.audio_generator import generate_audiobook
                audio_path = os.path.join(tmpdir, 'audiobook.m4b')
                result_path = generate_audiobook(
                    epub_path=input_path,
                    output_path=audio_path,
                    translation_target=target,
                    bilingual=bilingual,
                    use_claude=use_claude,
                    use_opus=use_opus,
                )
                result_ext = Path(result_path).suffix
                generated_files.append((
                    f'{orig_name}_audiobook{result_ext}', result_path,
                ))

        except CalibreNotFoundError as e:
            return f'Calibre error: {e.message}', 500
        except Exception as e:
            logging.exception('Conversion failed')
            return f'Conversion failed: {e}', 500

        if not generated_files:
            return 'No outputs were generated', 500

        # Single file: send directly
        if len(generated_files) == 1:
            fname, fpath = generated_files[0]
            return send_file(fpath, as_attachment=True, download_name=fname)

        # Multiple files: bundle into ZIP
        zip_path = os.path.join(tmpdir, 'bundle.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
            for fname, fpath in generated_files:
                zf.write(fpath, fname)

        return send_file(
            zip_path, as_attachment=True,
            download_name=f'{orig_name}_learning_kit.zip',
            mimetype='application/zip',
        )


if __name__ == '__main__':
    print('Starting server at http://localhost:5000')
    print('Open this URL in your browser to upload an EPUB file.')
    app.run(host='0.0.0.0', port=5000, debug=False)
