/* Linnaeus — Main application: Pyodide integration, tab switching, check handler */

(function () {
  'use strict';

  let pyodide = null;
  let pyodideReady = false;

  // ---- Render inline KaTeX formulas in the About section ----
  document.querySelectorAll('.katex-inline[data-formula]').forEach(el => {
    try {
      katex.render(el.getAttribute('data-formula'), el, { throwOnError: false });
    } catch (e) {
      el.textContent = el.getAttribute('data-formula');
    }
  });

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
from compute import compute_transfer_function, compute_char_poly
from library import check_all_equivalences

_input_text = _raw_input
_lines = [l.strip() for l in _input_text.split('\\n')
          if l.strip() and not l.strip().startswith('#')]

parsed = parse_equations(_lines)
_z = parsed['z_var']
_user_is_consensus = (len(parsed['oracle_types']) == 0)

if _user_is_consensus:
    H_user = None
    _user_char_poly = compute_char_poly(
        parsed['state_vars'], parsed['z_equations'], _z)
else:
    H_user = compute_transfer_function(
        parsed['state_vars'],
        parsed['oracle_inputs'],
        parsed['oracle_outputs'],
        parsed['z_equations'],
        _z
    )
    _user_char_poly = None

_user_distributed = parsed.get('has_mixing_matrix', False)
_user_universal = [symbols('lambda')] if _user_distributed else []
_user_has_projection = parsed.get('has_projection', False)
matches = check_all_equivalences(
    H_user, parsed['oracle_types'], _library, _z,
    user_distributed=_user_distributed,
    user_universal_params=_user_universal,
    user_has_projection=_user_has_projection,
    user_char_poly=_user_char_poly,
    user_equations=_lines,
)

# For display, substitute the internal z symbol with a clean 'z'
_display_z = Symbol('z')

def _poly_latex(expr, z):
    """Format a polynomial in z with factored coefficients."""
    from sympy import Poly, factor, latex as _latex, Add, Mul, signsimp
    try:
        p = Poly(expr, z)
    except Exception:
        return _latex(expr)
    deg = p.degree()
    if deg < 0:  # Poly(0, z) has degree -oo
        return '0'
    terms = []
    for i in range(deg, -1, -1):
        c = p.nth(i)
        if c == 0:
            continue
        c_factored = factor(c)
        # For Add expressions like -beta-1, extract minus sign for -(beta+1)
        if isinstance(c_factored, Add) and c_factored.could_extract_minus_sign():
            pos_part = factor(-c_factored)
            if isinstance(pos_part, Add):
                c_str = '-\\\\left(' + _latex(pos_part) + '\\\\right)'
            else:
                c_str = '-' + _latex(pos_part) if str(pos_part) != '1' else '-1'
            is_negative_wrapped = True
        elif isinstance(c_factored, Add):
            c_str = _latex(c_factored)
            is_negative_wrapped = False
        else:
            c_str = _latex(c_factored)
            is_negative_wrapped = False
        if i == 0:
            terms.append(c_str)
        elif i == 1:
            zstr = 'z'
            if c_factored == 1:
                terms.append(zstr)
            elif c_factored == -1:
                terms.append('-' + zstr)
            elif is_negative_wrapped:
                pos = factor(-c_factored)
                pos_str = _latex(pos)
                if isinstance(pos, Add):
                    terms.append('-\\\\left(' + pos_str + '\\\\right) ' + zstr)
                else:
                    terms.append('-' + pos_str + ' ' + zstr)
            else:
                needs_parens = isinstance(c_factored, Add)
                if needs_parens:
                    terms.append('\\\\left(' + c_str + '\\\\right) ' + zstr)
                else:
                    terms.append(c_str + ' ' + zstr)
        else:
            zstr = 'z^{' + str(i) + '}'
            if c_factored == 1:
                terms.append(zstr)
            elif c_factored == -1:
                terms.append('-' + zstr)
            elif is_negative_wrapped:
                pos = factor(-c_factored)
                pos_str = _latex(pos)
                if isinstance(pos, Add):
                    terms.append('-\\\\left(' + pos_str + '\\\\right) ' + zstr)
                else:
                    terms.append('-' + pos_str + ' ' + zstr)
            else:
                needs_parens = isinstance(c_factored, Add)
                if needs_parens:
                    terms.append('\\\\left(' + c_str + '\\\\right) ' + zstr)
                else:
                    terms.append(c_str + ' ' + zstr)
    if not terms:
        return '0'
    # Join with + but handle leading minus
    result = terms[0]
    for t in terms[1:]:
        if t.startswith('-'):
            result += ' ' + t
        else:
            result += ' + ' + t
    return result

def _to_display_latex(expr):
    from sympy import fraction, Poly, latex as _latex
    clean = expr.subs(_z, _display_z)
    if isinstance(clean, _Matrix) and clean.shape == (1, 1):
        clean = clean[0, 0]
    if isinstance(clean, _Matrix):
        # For matrix TFs, format each entry
        rows = clean.rows
        cols = clean.cols
        entries = []
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(_to_display_latex_scalar(clean[i, j]))
            entries.append(row)
        # Build matrix latex (use chr(92) to avoid JS/Python escaping issues)
        _bs = chr(92)
        _rowsep = ' ' + _bs + _bs + ' '
        lines = _rowsep.join(' & '.join(r) for r in entries)
        return _bs + 'begin{bmatrix} ' + lines + ' ' + _bs + 'end{bmatrix}'
    return _to_display_latex_scalar(clean)

def _factored_product_latex(expr, z):
    from sympy import Mul, Pow, Rational, Integer, Number, Add, Poly, factor, cancel, latex as _latex
    expr_f = factor(expr)
    if isinstance(expr_f, Mul):
        raw_factors = list(expr_f.args)
    else:
        raw_factors = [expr_f]
    sign = 1
    numerics = []
    params = []
    z_polys = []
    for f in raw_factors:
        if f == -1:
            sign *= -1
            continue
        if f.has(z):
            z_polys.append(f)
        elif isinstance(f, (Integer, Rational, Number)):
            if f < 0:
                sign *= -1
                numerics.append(abs(f))
            else:
                numerics.append(f)
        else:
            params.append(f)
    parts = []
    prefix = '-' if sign == -1 else ''
    for n in numerics:
        if n != 1:
            parts.append(_latex(n))
    for p in params:
        p_str = _latex(p)
        if isinstance(p, Add):
            p_str = '\\\\left(' + p_str + '\\\\right)'
        parts.append(p_str)
    total_factors = len(numerics) + len(params) + len(z_polys)
    for zp in z_polys:
        base = zp.base if isinstance(zp, Pow) else zp
        exp = zp.exp if isinstance(zp, Pow) else 1
        base_str = _poly_latex(base, z)
        has_multiple_terms = ('+' in base_str or base_str.count('-') > (1 if base_str.startswith('-') else 0))
        needs_parens = has_multiple_terms and (
            total_factors > 1 or sign == -1 or exp != 1
        )
        if needs_parens:
            base_str = '\\\\left(' + base_str + '\\\\right)'
        if exp != 1:
            parts.append(base_str + '^{' + _latex(exp) + '}')
        else:
            parts.append(base_str)
    if not parts:
        return prefix + '1' if prefix else '1'
    return prefix + ' '.join(parts)

def _to_display_latex_scalar(expr):
    from sympy import fraction, cancel
    expr = cancel(expr)
    num, den = fraction(expr)
    if den == 1:
        return _factored_product_latex(num, _display_z)
    return '\\\\frac{' + _factored_product_latex(num, _display_z) + '}{' + _factored_product_latex(den, _display_z) + '}'

# Build result JSON
_match_list = []
for m in matches:
    algo = m['algorithm']
    _lib_tf = algo['tf']
    if _lib_tf is None:
        # Consensus algorithm — show reciprocal of char poly (like a TF)
        _lib_cp = algo.get('char_poly')
        if _lib_cp is not None:
            _lib_latex = _to_display_latex_scalar((1 / _lib_cp).subs(_z, _display_z))
        else:
            _lib_latex = ''
    elif isinstance(_lib_tf, _Matrix) and _lib_tf.shape == (1, 1):
        _lib_latex = _to_display_latex_scalar(_lib_tf[0, 0].subs(_z, _display_z))
    elif isinstance(_lib_tf, _Matrix):
        _lib_latex = _to_display_latex(_lib_tf)
    else:
        _lib_latex = _to_display_latex_scalar(_lib_tf.subs(_z, _display_z) if hasattr(_lib_tf, 'subs') else _lib_tf)
    entry = {
        'name': algo['name'],
        'shortName': algo.get('shortName', ''),
        'citations': algo.get('citations', []),
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
    if details.get('func_mapping'):
        entry['func_mapping'] = details['func_mapping']
    _match_list.append(entry)

if _user_is_consensus:
    _cp_display = _to_display_latex_scalar((1 / _user_char_poly).subs(_z, _display_z))
    _result = {
        'tf_latex': _cp_display,
        'oracle_types': [],
        'user_params': [latex(symbols(p)) for p in parsed['parameters']],
        'matches': _match_list,
        'user_distributed': _user_distributed,
        'user_is_consensus': True,
    }
else:
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
