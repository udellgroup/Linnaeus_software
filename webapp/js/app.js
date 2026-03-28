/* Linnaeus — Main application: Pyodide integration, tab switching, check handler */

(function () {
  'use strict';

  let pyodide = null;
  let pyodideReady = false;

  // ---- Tab switching ----
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
    });
  });

  // ---- Helpers ----
  const overlay = document.getElementById('loading-overlay');
  const progressBar = document.getElementById('progress-bar');
  const loadingText = document.querySelector('.loading-text');
  const btnCheck = document.getElementById('btn-check');
  const algoInput = document.getElementById('algo-input');
  const resultsPanel = document.getElementById('results-panel');

  function setProgress(pct, text) {
    progressBar.style.width = pct + '%';
    if (text) loadingText.textContent = text;
  }

  function showOverlay(text) {
    loadingText.textContent = text || 'Loading...';
    overlay.classList.add('visible');
  }

  function hideOverlay() {
    overlay.classList.remove('visible');
  }

  function showError(msg) {
    resultsPanel.innerHTML =
      '<h2 class="panel-title">Results</h2>' +
      '<div class="error-display">' + escapeHtml(msg) + '</div>';
  }

  function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  // ---- Pyodide initialization ----
  async function initPyodide() {
    showOverlay('Loading Python runtime...');
    setProgress(10, 'Loading Pyodide...');

    pyodide = await loadPyodide();
    setProgress(40, 'Installing SymPy...');

    await pyodide.loadPackage('sympy');
    setProgress(70, 'Loading algorithm library...');

    // Fetch Python source files and write to virtual filesystem
    // Cache-bust to ensure latest versions are loaded after updates
    const cacheBust = '?v=' + Date.now();
    const pyFiles = ['parser.py', 'compute.py', 'equivalence.py', 'library.py'];
    for (const fname of pyFiles) {
      const resp = await fetch('python/' + fname + cacheBust);
      const text = await resp.text();
      pyodide.FS.writeFile('/home/pyodide/' + fname, text);
    }

    // Fetch algorithms.json
    const jsonResp = await fetch('data/algorithms.json' + cacheBust);
    const jsonText = await jsonResp.text();
    pyodide.FS.writeFile('/home/pyodide/algorithms.json', jsonText);

    setProgress(85, 'Initializing library...');

    await pyodide.runPythonAsync(`
from library import load_library
_library = load_library('algorithms.json')
`);

    setProgress(100, 'Ready!');
    await new Promise(r => setTimeout(r, 300));
    hideOverlay();

    pyodideReady = true;
    btnCheck.disabled = false;
  }

  // ---- Check Equivalence handler ----
  async function runCheck() {
    if (!pyodideReady) return;

    const input = algoInput.value.trim();
    if (!input) {
      showError('Please enter algorithm equations.');
      return;
    }

    // Show computing state
    resultsPanel.innerHTML =
      '<h2 class="panel-title">Results</h2>' +
      '<p class="computing">Computing transfer function...</p>';
    btnCheck.disabled = true;

    try {
      // Pass user input safely via Pyodide globals (no string interpolation)
      pyodide.globals.set('_raw_input', input);

      const resultJson = await pyodide.runPythonAsync(`
import json
from sympy import symbols, latex, Symbol, Matrix as _Matrix
from parser import parse_equations
from compute import compute_transfer_function
from library import check_all_equivalences

_input_text = _raw_input
_lines = [l.strip() for l in _input_text.split('\\n')
          if l.strip() and not l.strip().startswith('#')]

parsed = parse_equations(_lines)
_z = parsed['z_var']
H_user = compute_transfer_function(
    parsed['state_vars'],
    parsed['oracle_inputs'],
    parsed['oracle_outputs'],
    parsed['z_equations'],
    _z
)

_user_distributed = parsed.get('has_mixing_matrix', False)
_user_universal = [symbols('lambda')] if _user_distributed else []
matches = check_all_equivalences(
    H_user, parsed['oracle_types'], _library, _z,
    user_distributed=_user_distributed,
    user_universal_params=_user_universal,
)

# For display, substitute the internal z symbol with a clean 'z'
_display_z = Symbol('z')

def _to_display_latex(expr):
    clean = expr.subs(_z, _display_z)
    if isinstance(clean, _Matrix) and clean.shape == (1, 1):
        return latex(clean[0, 0])
    return latex(clean)

# Build result JSON
_match_list = []
for m in matches:
    algo = m['algorithm']
    _lib_tf = algo['tf']
    if isinstance(_lib_tf, _Matrix) and _lib_tf.shape == (1, 1):
        _lib_latex = latex(_lib_tf[0, 0])
    else:
        _lib_latex = latex(_lib_tf)
    entry = {
        'name': algo['name'],
        'citation': algo.get('citation', ''),
        'doi': algo.get('doi', ''),
        'bibtex': algo.get('bibtex', ''),
        'type': m['type'],
        'lib_tf_latex': _lib_latex,
        'lib_oracles': algo.get('oracles', []),
    }
    details = m.get('details', {})
    if details.get('params'):
        entry['params'] = {latex(k): latex(v) for k, v in details['params'].items()}
    if details.get('user_params'):
        entry['user_params'] = {latex(k): latex(v) for k, v in details['user_params'].items()}
    if details.get('free_params'):
        entry['free_params'] = [latex(s) for s in details['free_params']]
    if details.get('shift_vector'):
        entry['shift_vector'] = details['shift_vector']
    if m.get('permuted'):
        entry['permuted'] = True
    if algo.get('distributed', False):
        entry['distributed'] = True
    if details.get('condition_note'):
        entry['condition_note'] = details['condition_note']
    if m.get('conditional'):
        entry['conditional'] = True
    _match_list.append(entry)

_result = {
    'tf_latex': _to_display_latex(H_user),
    'oracle_types': parsed['oracle_types'],
    'user_params': [latex(symbols(p)) for p in parsed['parameters']],
    'matches': _match_list,
    'user_distributed': _user_distributed,
}
json.dumps(_result)
`);

      const result = JSON.parse(resultJson);
      displayResults(result);

    } catch (err) {
      const msg = err.message || String(err);
      // Extract Python ValueError messages for user-facing errors
      const valMatch = msg.match(/ValueError:\s*(.+)/);
      if (valMatch) {
        showError(valMatch[1]);
      } else {
        showError('Error: ' + msg);
      }
    } finally {
      btnCheck.disabled = false;
    }
  }

  // Button click
  btnCheck.addEventListener('click', runCheck);

  // Ctrl+Enter shortcut
  algoInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      runCheck();
    }
  });

  // ---- Load library UI immediately (no Pyodide needed) ----
  if (typeof loadLibraryUI === 'function') {
    loadLibraryUI();
  }

  // ---- Start Pyodide loading ----
  initPyodide().catch(err => {
    hideOverlay();
    console.error('Pyodide init failed:', err);
    showError('Failed to load Python runtime. Please refresh the page.');
  });

})();
