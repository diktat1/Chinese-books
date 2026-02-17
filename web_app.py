#!/usr/bin/env python3
"""
Web app for converting Chinese EPUBs into learning materials.

Upload an EPUB and get back any combination of:
  - Graded reader EPUB (pinyin + translation)
  - Anki flashcard deck
  - M4B audiobook with chapter markers

Run with:
    python web_app.py

Then open http://localhost:5000 in your browser.
"""

import json
import os
import tempfile
import logging
import zipfile
import uuid
import threading
from pathlib import Path

from flask import Flask, request, send_file, render_template_string, jsonify, Response

from graded_reader.epub_processor import process_epub
from graded_reader.llm_simplifier import is_openrouter_available, get_api_key
from graded_reader.models import MODELS, TIER_DEFAULTS, estimate_book_cost

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload

# In-memory job store for SSE progress tracking
_jobs = {}  # job_id -> {status, progress, message, result_files, tmpdir, error}

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
            max-width: 560px;
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
        /* Tier buttons */
        .tier-group { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
        .tier-btn {
            flex: 1; min-width: 130px; padding: 12px 8px; border: 2px solid #e0e0e0;
            border-radius: 10px; background: #fff; cursor: pointer; text-align: center;
            transition: all .15s;
        }
        .tier-btn:hover { border-color: #ccc; }
        .tier-btn.selected { border-color: #e74c3c; background: #fef5f5; }
        .tier-btn.disabled { opacity: 0.45; cursor: not-allowed; }
        .tier-btn .tier-name { font-weight: 700; font-size: .95em; display: block; }
        .tier-btn .tier-model { font-size: .75em; color: #888; display: block; margin-top: 2px; }
        .tier-btn .tier-cost { font-size: .72em; color: #aaa; display: block; margin-top: 1px; }
        /* Model picker */
        .model-toggle {
            font-size: .85em; color: #888; cursor: pointer; padding: 6px 0;
            user-select: none;
        }
        .model-toggle:hover { color: #e74c3c; }
        .model-picker {
            display: none; margin-top: 8px; border: 1px solid #eee;
            border-radius: 8px; padding: 12px; background: #fafafa;
        }
        .model-picker.open { display: block; }
        .model-tier-label {
            font-size: .75em; font-weight: 700; color: #aaa; text-transform: uppercase;
            letter-spacing: .5px; margin-top: 10px; margin-bottom: 4px;
            padding-bottom: 3px; border-bottom: 1px solid #eee;
        }
        .model-tier-label:first-child { margin-top: 0; }
        .model-option {
            display: flex; align-items: flex-start; gap: 8px; padding: 6px 4px;
            cursor: pointer; border-radius: 6px; transition: background .1s;
        }
        .model-option:hover { background: #f0f0f0; }
        .model-option.selected { background: #fef5f5; }
        .model-option input[type="radio"] { margin-top: 3px; }
        .model-info { flex: 1; }
        .model-name { font-size: .88em; font-weight: 600; }
        .model-desc { font-size: .75em; color: #888; }
        .model-meta { font-size: .72em; color: #aaa; }
        .model-meta .quality-badge {
            display: inline-block; padding: 1px 6px; border-radius: 3px;
            font-size: .9em; font-weight: 600;
        }
        .quality-badge.excellent { background: #d4edda; color: #155724; }
        .quality-badge.very_good { background: #fff3cd; color: #856404; }
        .quality-badge.good { background: #e2e3e5; color: #383d41; }
        .custom-model-input {
            width: 100%; padding: 6px 10px; border: 1px solid #ddd;
            border-radius: 6px; font-size: .85em; margin-top: 8px;
        }
        .custom-model-label { font-size: .78em; color: #999; margin-top: 10px; }
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
        .small { font-size: .78em; color: #aaa; line-height: 1.5; margin-top: 10px; }
        /* Progress bar */
        .progress-bar {
            height: 4px; background: #eee; border-radius: 2px;
            margin-top: 8px; overflow: hidden; display: none;
        }
        .progress-bar.active { display: block; }
        .progress-bar .fill {
            height: 100%; background: #e74c3c; border-radius: 2px;
            transition: width 0.3s ease;
            width: 0%;
        }
        .progress-bar.indeterminate .fill {
            animation: indeterminate 2s ease-in-out infinite;
        }
        @keyframes indeterminate {
            0% { width: 0; margin-left: 0; } 50% { width: 60%; margin-left: 20%; } 100% { width: 0; margin-left: 100%; }
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

<div class="card" id="engineCard">
    <div class="section-title">Translation Engine</div>
    <div class="tier-group" id="tierGroup">
        <div class="tier-btn selected" data-tier="google">
            <span class="tier-name">Free</span>
            <span class="tier-model">Google Translate</span>
            <span class="tier-cost">No API key needed</span>
        </div>
        <div class="tier-btn" data-tier="standard">
            <span class="tier-name">Standard</span>
            <span class="tier-model">DeepSeek V3</span>
            <span class="tier-cost">~$0.10/book</span>
        </div>
        <div class="tier-btn" data-tier="premium">
            <span class="tier-name">Premium</span>
            <span class="tier-model">Claude Sonnet 4.5</span>
            <span class="tier-cost">~$1.65/book</span>
        </div>
    </div>

    <div class="model-toggle" id="modelToggle">&#9654; Choose a specific model</div>
    <div class="model-picker" id="modelPicker">
        <!-- Populated by JS from /models endpoint -->
    </div>
</div>

<div class="card" id="optionsCard">
    <div class="section-title">Options</div>
    <div style="margin-bottom:10px">
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
    <hr>
    <div class="opt-grid" id="epubOptions">
        <label><input type="checkbox" id="optPinyin" checked> Pinyin annotations</label>
        <label><input type="checkbox" id="optTranslation" checked> Translation</label>
        <label><input type="checkbox" id="optWordSpacing"> Word spacing</label>
        <label><input type="checkbox" id="optParallelText"> Parallel text</label>
    </div>
    <div class="opt-grid" id="hskOptions">
        <label><input type="checkbox" id="optSimplifyHsk4"> Simplify to HSK 4</label>
    </div>
    <div class="opt-grid" id="audioOptions" style="display:none">
        <label><input type="checkbox" id="optBilingual" checked> Bilingual narration</label>
    </div>
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

// --- Model data (loaded from server) ---
let modelCatalog = {};
let hasApiKey = false;
let selectedTier = 'google';
let selectedModel = '';  // empty = Google Translate, else OpenRouter model ID

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

// --- Tier selection ---
document.querySelectorAll('.tier-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (btn.classList.contains('disabled')) return;
        document.querySelectorAll('.tier-btn').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
        selectedTier = btn.dataset.tier;

        // Update selected model
        if (selectedTier === 'google') {
            selectedModel = '';
        } else {
            const tierDefaults = { standard: 'deepseek/deepseek-chat', premium: 'anthropic/claude-sonnet-4-5' };
            selectedModel = tierDefaults[selectedTier] || '';
        }

        // Update radio in model picker
        updateModelPickerSelection();
    });
});

// --- Model picker toggle ---
$('modelToggle').addEventListener('click', () => {
    const picker = $('modelPicker');
    const isOpen = picker.classList.toggle('open');
    $('modelToggle').innerHTML = (isOpen ? '&#9660;' : '&#9654;') + ' Choose a specific model';
});

function updateModelPickerSelection() {
    document.querySelectorAll('.model-option input[type="radio"]').forEach(r => {
        r.checked = (r.value === selectedModel);
        r.closest('.model-option').classList.toggle('selected', r.checked);
    });
}

function buildModelPicker(models) {
    const picker = $('modelPicker');
    picker.innerHTML = '';
    const tiers = ['free', 'standard', 'premium'];
    const tierLabels = { free: 'Free (OpenRouter)', standard: 'Standard', premium: 'Premium' };

    tiers.forEach(tier => {
        const tierModels = Object.entries(models).filter(([_, m]) => m.tier === tier);
        if (!tierModels.length) return;

        const label = document.createElement('div');
        label.className = 'model-tier-label';
        label.textContent = tierLabels[tier];
        picker.appendChild(label);

        tierModels.forEach(([id, m]) => {
            const opt = document.createElement('div');
            opt.className = 'model-option';
            const qualityClass = m.chinese_quality.replace(' ', '_');
            const cost = m.input_price === 0 ? '$0.00' : '~$' + estimateCost(m);
            const note = m.note ? ' ' + m.note : '';
            opt.innerHTML = `
                <input type="radio" name="llm_model" value="${id}">
                <div class="model-info">
                    <div class="model-name">${m.name} <span style="color:#aaa;font-weight:400;font-size:.8em">${m.provider}</span></div>
                    <div class="model-desc">${m.description}${note}</div>
                    <div class="model-meta">
                        <span class="quality-badge ${qualityClass}">${m.chinese_quality.replace('_', ' ')}</span>
                        ${cost}/book &middot; ${(m.context_window/1000).toFixed(0)}K context
                    </div>
                </div>
            `;
            opt.addEventListener('click', () => {
                selectedModel = id;
                selectedTier = m.tier === 'free' ? 'free' : m.tier;
                // Update tier buttons
                document.querySelectorAll('.tier-btn').forEach(b => {
                    const t = b.dataset.tier;
                    b.classList.toggle('selected',
                        (t === 'google' && !selectedModel) ||
                        (t === selectedTier && selectedModel));
                });
                // Deselect Google if we picked a model
                if (selectedModel) {
                    document.querySelector('.tier-btn[data-tier="google"]').classList.remove('selected');
                    // Select the matching tier
                    const tierBtn = document.querySelector(`.tier-btn[data-tier="${selectedTier}"]`);
                    if (tierBtn) tierBtn.classList.add('selected');
                }
                updateModelPickerSelection();
            });
            picker.appendChild(opt);
        });
    });

    // Custom model input
    const customLabel = document.createElement('div');
    customLabel.className = 'custom-model-label';
    customLabel.textContent = 'Or enter any OpenRouter model ID:';
    picker.appendChild(customLabel);

    const customInput = document.createElement('input');
    customInput.className = 'custom-model-input';
    customInput.type = 'text';
    customInput.placeholder = 'e.g. mistralai/mistral-nemo';
    customInput.addEventListener('input', () => {
        const val = customInput.value.trim();
        if (val) {
            selectedModel = val;
            selectedTier = 'custom';
            document.querySelectorAll('.tier-btn').forEach(b => b.classList.remove('selected'));
            updateModelPickerSelection();
        }
    });
    picker.appendChild(customInput);

    updateModelPickerSelection();
}

function estimateCost(m) {
    const inCost = m.input_price * 50000 / 1000000;
    const outCost = m.output_price * 100000 / 1000000;
    return (inCost + outCost).toFixed(2);
}

// --- Check deps and load models ---
fetch('/check-deps').then(r => r.json()).then(d => {
    hasApiKey = d.openrouter;
    if (!hasApiKey) {
        document.querySelectorAll('.tier-btn').forEach(btn => {
            if (btn.dataset.tier !== 'google') {
                btn.classList.add('disabled');
                btn.querySelector('.tier-cost').textContent = 'Needs OPENROUTER_API_KEY';
            }
        });
    }
}).catch(() => {});

fetch('/models').then(r => r.json()).then(models => {
    modelCatalog = models;
    buildModelPicker(models);
}).catch(() => {});

// --- HSK simplification requires an LLM ---
$('optSimplifyHsk4').addEventListener('change', function() {
    if (this.checked && selectedTier === 'google' && !selectedModel) {
        // Auto-select premium tier for HSK simplification
        document.querySelectorAll('.tier-btn').forEach(b => b.classList.remove('selected'));
        const premBtn = document.querySelector('.tier-btn[data-tier="premium"]');
        if (premBtn && !premBtn.classList.contains('disabled')) {
            premBtn.classList.add('selected');
            selectedTier = 'premium';
            selectedModel = 'anthropic/claude-sonnet-4-5';
            updateModelPickerSelection();
        }
    }
});

// --- Convert with SSE progress ---
$('convertBtn').onclick = async () => {
    if (!selectedFile || selected.size === 0) return;

    const btn = $('convertBtn');
    const status = $('status');
    const bar = $('progressBar');
    const fill = bar.querySelector('.fill');

    btn.disabled = true;
    bar.classList.add('active', 'indeterminate');
    fill.style.width = '0%';
    status.className = 'status info';
    status.style.display = 'block';
    status.textContent = 'Uploading...';

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('outputs', JSON.stringify([...selected]));
    formData.append('target', $('optTarget').value);
    formData.append('llm_model', selectedModel);
    formData.append('simplify_hsk4', $('optSimplifyHsk4').checked);
    formData.append('add_pinyin', $('optPinyin').checked);
    formData.append('add_translation', $('optTranslation').checked);
    formData.append('word_spacing', $('optWordSpacing').checked);
    formData.append('parallel_text', $('optParallelText').checked);
    formData.append('bilingual', $('optBilingual').checked);

    try {
        // Step 1: Start the job
        const startResp = await fetch('/convert', { method: 'POST', body: formData });
        if (!startResp.ok) throw new Error(await startResp.text());
        const { job_id } = await startResp.json();

        // Step 2: Listen for SSE progress
        bar.classList.remove('indeterminate');
        status.textContent = 'Processing...';

        await new Promise((resolve, reject) => {
            const es = new EventSource('/progress/' + job_id);
            es.onmessage = (e) => {
                const d = JSON.parse(e.data);
                if (d.progress !== undefined) {
                    fill.style.width = d.progress + '%';
                }
                if (d.message) {
                    status.textContent = d.message;
                }
                if (d.status === 'done') {
                    es.close();
                    resolve();
                } else if (d.status === 'error') {
                    es.close();
                    reject(new Error(d.message || 'Conversion failed'));
                }
            };
            es.onerror = () => { es.close(); reject(new Error('Connection lost')); };
        });

        // Step 3: Download the result
        fill.style.width = '100%';
        status.textContent = 'Downloading...';
        const dlResp = await fetch('/download/' + job_id);
        if (!dlResp.ok) throw new Error(await dlResp.text());

        const blob = await dlResp.blob();
        const cd = dlResp.headers.get('Content-Disposition') || '';
        const match = cd.match(/filename="?([^"]+)"?/);
        const filename = match ? match[1] : 'output.epub';

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
        bar.classList.remove('active', 'indeterminate');
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
    """Check optional dependency availability."""
    return jsonify({
        'openrouter': is_openrouter_available() and bool(get_api_key()),
    })


@app.route('/models')
def models_endpoint():
    """Return the model catalog as JSON."""
    result = {}
    for model_id, info in MODELS.items():
        result[model_id] = {
            **info,
            'estimated_book_cost': estimate_book_cost(model_id),
        }
    return jsonify(result)


def _run_conversion(job_id, input_path, orig_name, tmpdir, outputs,
                     target, add_pinyin, add_translation, word_spacing,
                     parallel_text, bilingual,
                     llm_model, simplify_hsk4):
    """Background conversion worker."""
    job = _jobs[job_id]

    def progress(step, total, message):
        pct = int(step / max(total, 1) * 100) if total else 0
        job['progress'] = pct
        job['message'] = message

    generated_files = []

    try:
        # --- Graded reader EPUB ---
        if 'epub' in outputs:
            progress(0, 1, 'Building graded reader...')
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
                llm_model=llm_model or None,
                progress_callback=progress,
            )
            generated_files.append((f'{orig_name}_graded.epub', epub_path))

        # --- Anki deck ---
        if 'anki' in outputs:
            progress(0, 1, 'Generating flashcards...')
            from graded_reader.anki_generator import generate_anki_deck
            anki_path = os.path.join(tmpdir, 'flashcards.apkg')
            generate_anki_deck(
                epub_path=input_path,
                output_path=anki_path,
                translation_target=target,
                llm_model=llm_model or None,
            )
            generated_files.append((f'{orig_name}_flashcards.apkg', anki_path))

        # --- Audiobook ---
        if 'audio' in outputs:
            progress(0, 1, 'Generating audiobook...')
            from graded_reader.audio_generator import generate_audiobook
            audio_path = os.path.join(tmpdir, 'audiobook.m4b')
            result_path = generate_audiobook(
                epub_path=input_path,
                output_path=audio_path,
                translation_target=target,
                bilingual=bilingual,
                llm_model=llm_model or None,
            )
            result_ext = Path(result_path).suffix
            generated_files.append((
                f'{orig_name}_audiobook{result_ext}', result_path,
            ))

        if not generated_files:
            job['status'] = 'error'
            job['message'] = 'No outputs were generated'
            return

        # Bundle if multiple files
        if len(generated_files) == 1:
            job['result_files'] = generated_files
        else:
            zip_path = os.path.join(tmpdir, 'bundle.zip')
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
                for fname, fpath in generated_files:
                    zf.write(fpath, fname)
            job['result_files'] = [(f'{orig_name}_learning_kit.zip', zip_path)]

        job['status'] = 'done'
        job['progress'] = 100
        job['message'] = 'Done!'

    except Exception as e:
        logging.exception('Conversion failed')
        job['status'] = 'error'
        job['message'] = f'Conversion failed: {e}'


@app.route('/convert', methods=['POST'])
def convert():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if not file.filename or not file.filename.lower().endswith('.epub'):
        return 'Please upload an .epub file', 400

    outputs = json.loads(request.form.get('outputs', '["epub"]'))
    target = request.form.get('target', 'en')
    llm_model = request.form.get('llm_model', '').strip()
    simplify_hsk4 = request.form.get('simplify_hsk4', 'false') == 'true'

    add_pinyin = request.form.get('add_pinyin', 'true') == 'true'
    add_translation = request.form.get('add_translation', 'true') == 'true'
    word_spacing = request.form.get('word_spacing', 'false') == 'true'
    parallel_text = request.form.get('parallel_text', 'false') == 'true'
    bilingual = request.form.get('bilingual', 'true') == 'true'

    # HSK simplification always needs an LLM
    if simplify_hsk4 and not llm_model:
        llm_model = TIER_DEFAULTS["premium"]

    if llm_model and (not is_openrouter_available() or not get_api_key()):
        return 'OPENROUTER_API_KEY required for LLM models. Select the Free tier.', 400

    if not outputs:
        return 'Select at least one output type', 400

    orig_name = Path(file.filename).stem

    # Validate EPUB before starting
    tmpdir = tempfile.mkdtemp()
    input_path = os.path.join(tmpdir, 'input.epub')
    file.save(input_path)

    try:
        from ebooklib import epub as epub_lib
        test_book = epub_lib.read_epub(input_path, options={'ignore_ncx': True})
        items = list(test_book.get_items_of_type(9))
        if not items:
            return 'This EPUB has no readable content (no HTML documents found).', 400
    except Exception as e:
        return f'Invalid or corrupted EPUB file: {e}', 400

    # Start background job
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        'status': 'running',
        'progress': 0,
        'message': 'Starting...',
        'result_files': None,
        'tmpdir': tmpdir,
    }

    thread = threading.Thread(
        target=_run_conversion,
        args=(job_id, input_path, orig_name, tmpdir, outputs,
              target, add_pinyin, add_translation, word_spacing,
              parallel_text, bilingual,
              llm_model, simplify_hsk4),
        daemon=True,
    )
    thread.start()

    return jsonify({'job_id': job_id})


@app.route('/progress/<job_id>')
def progress(job_id):
    """SSE endpoint for real-time progress updates."""
    import time

    def stream():
        while True:
            job = _jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'status': 'error', 'message': 'Job not found'})}\n\n"
                break

            yield f"data: {json.dumps({'status': job['status'], 'progress': job['progress'], 'message': job['message']})}\n\n"

            if job['status'] in ('done', 'error'):
                break
            time.sleep(0.5)

    return Response(stream(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/download/<job_id>')
def download(job_id):
    """Download completed conversion result."""
    job = _jobs.get(job_id)
    if not job:
        return 'Job not found', 404
    if job['status'] != 'done':
        return 'Job not ready', 400
    if not job['result_files']:
        return 'No files generated', 500

    fname, fpath = job['result_files'][0]
    return send_file(fpath, as_attachment=True, download_name=fname)


if __name__ == '__main__':
    print('Starting server at http://localhost:5000')
    print('Open this URL in your browser to upload an EPUB file.')
    app.run(host='0.0.0.0', port=5000, debug=False)
