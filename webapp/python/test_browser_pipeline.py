"""Tests that exercise the EXACT Python code path used in the browser.

The browser runs Python code embedded in app.js via Pyodide. That code
has its own imports, variable names, and logic that differ from the
unit-test pipeline. These tests replicate that exact code path to catch
errors like missing imports, scoping bugs, and NameErrors that only
manifest in the browser.

If a test here fails, the webapp is broken for users.
"""
import json
import pytest
import os
from sympy import symbols, latex, Symbol, Matrix as _Matrix
from parser import parse_equations
from compute import compute_transfer_function
from library import load_library, check_all_equivalences


# Load library once (same as Pyodide init does)
_json_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'algorithms.json')
_library = load_library(_json_path)


def _browser_check(raw_input):
    """Replicate the EXACT Python code that app.js runs via Pyodide.

    This function mirrors the code in app.js's runPythonAsync block.
    If you change app.js, update this function to match.
    """
    # --- BEGIN: mirrors app.js Python code ---
    _input_text = raw_input
    _lines = [l.strip() for l in _input_text.split('\n')
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

    matches = check_all_equivalences(H_user, parsed['oracle_types'], _library, _z)

    _display_z = Symbol('z')

    def _to_display_latex(expr):
        clean = expr.subs(_z, _display_z)
        if isinstance(clean, _Matrix) and clean.shape == (1, 1):
            return latex(clean[0, 0])
        return latex(clean)

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
            'type': m['type'],
            'lib_tf_latex': _lib_latex,
        }
        details = m.get('details', {})
        if details.get('params'):
            entry['params'] = {latex(k): latex(v) for k, v in details['params'].items()}
        if details.get('shift_vector'):
            entry['shift_vector'] = details['shift_vector']
        _match_list.append(entry)

    _result = {
        'tf_latex': _to_display_latex(H_user),
        'oracle_types': parsed['oracle_types'],
        'matches': _match_list,
    }
    # --- END: mirrors app.js Python code ---

    # Verify it produces valid JSON (as app.js expects)
    result_json = json.dumps(_result)
    return json.loads(result_json)


# ---- Test every example chip (these are what users click first) ----

def _get_algorithm_equations():
    """Load all non-catalog algorithms with their equations from the library."""
    with open(_json_path) as f:
        data = json.load(f)
    return [(a['name'], a['equations']) for a in data
            if not a.get('catalogOnly', False) and a.get('equations')]


@pytest.mark.parametrize("name,equations", _get_algorithm_equations(),
                         ids=[a[0] for a in _get_algorithm_equations()])
def test_example_chip(name, equations):
    """Clicking any example chip in the UI must not crash.

    This test runs the full browser pipeline for every non-catalog
    algorithm in the library — exactly what happens when a user clicks
    an example chip.
    """
    input_text = '\n'.join(equations)
    result = _browser_check(input_text)

    # Must produce valid result structure
    assert 'tf_latex' in result, f"{name}: missing tf_latex"
    assert 'matches' in result, f"{name}: missing matches"
    assert isinstance(result['tf_latex'], str), f"{name}: tf_latex not a string"
    assert len(result['tf_latex']) > 0, f"{name}: empty tf_latex"

    # LaTeX should not contain internal symbol names
    assert '__ztf' not in result['tf_latex'], \
        f"{name}: internal z symbol leaked into display: {result['tf_latex']}"

    # Each match must have required fields
    for match in result['matches']:
        assert 'name' in match, f"{name}: match missing name"
        assert 'type' in match, f"{name}: match missing type"
        assert match['type'] in ('oracle', 'shift', 'lft'), \
            f"{name}: invalid match type: {match['type']}"
        assert 'lib_tf_latex' in match, f"{name}: match missing lib_tf_latex"


# ---- Test specific user inputs that are common first things to try ----

def test_browser_gradient_descent():
    """The simplest possible input — must work and find a match."""
    result = _browser_check("x[k+1] = x[k] - alpha * grad_f(x[k])")
    assert len(result['matches']) > 0
    names = [m['name'] for m in result['matches']]
    assert 'Gradient Descent' in names


def test_browser_nesterov():
    """Two-line Nesterov — a very common thing for experts to try."""
    result = _browser_check(
        "y[k] = x[k] + beta * (x[k] - x[k-1])\n"
        "x[k+1] = y[k] - alpha * grad_f(y[k])"
    )
    assert len(result['matches']) > 0
    names = [m['name'] for m in result['matches']]
    assert "Nesterov's Accelerated Method" in names


def test_browser_douglas_rachford():
    """Three-line DR with two oracles — tests matrix TF display."""
    result = _browser_check(
        "x1[k+1] = prox_f(x3[k])\n"
        "x2[k+1] = prox_g(2 * x1[k+1] - x3[k])\n"
        "x3[k+1] = x3[k] + x2[k+1] - x1[k+1]"
    )
    assert len(result['matches']) > 0
    names = [m['name'] for m in result['matches']]
    assert 'Douglas-Rachford Splitting' in names
    # The library TF is a matrix — check it renders
    dr_match = next(m for m in result['matches']
                    if m['name'] == 'Douglas-Rachford Splitting')
    assert len(dr_match['lib_tf_latex']) > 0


def test_browser_numeric_params():
    """Numeric step size — tests parametric matching display."""
    result = _browser_check("x[k+1] = x[k] - 0.01 * grad_f(x[k])")
    assert len(result['matches']) > 0
    gd = next(m for m in result['matches'] if m['name'] == 'Gradient Descent')
    # Should have parameter mapping
    assert 'params' in gd


def test_browser_no_match():
    """Algorithm that doesn't match anything — tests no-match display."""
    result = _browser_check("x[k+1] = 3 * x[k] - 2 * grad_f(x[k])")
    assert len(result['matches']) == 0


def test_browser_with_comments():
    """Input with comments and blank lines — common user behavior."""
    result = _browser_check(
        "# Gradient descent with step alpha\n"
        "\n"
        "x[k+1] = x[k] - alpha * grad_f(x[k])\n"
        "\n"
        "# end"
    )
    assert len(result['matches']) > 0


def test_browser_heavy_ball():
    """Heavy ball — tests algorithm with k-1 reference."""
    result = _browser_check(
        "x[k+1] = x[k] - alpha * grad_f(x[k]) + beta * (x[k] - x[k-1])"
    )
    assert len(result['matches']) > 0
    names = [m['name'] for m in result['matches']]
    assert 'Heavy Ball' in names


def test_browser_proximal_gradient():
    """Proximal gradient — tests mixed oracle algorithm."""
    result = _browser_check(
        "y[k] = x[k] - t * grad_f(x[k])\n"
        "x[k+1] = prox_g(y[k])"
    )
    assert len(result['matches']) > 0
    names = [m['name'] for m in result['matches']]
    assert 'Proximal Gradient' in names


def test_browser_error_nonlinear():
    """Nonlinear input — must raise ValueError, not crash."""
    with pytest.raises(ValueError):
        _browser_check("x[k+1] = x[k] * x[k]")


def test_browser_error_empty():
    """Empty input — must raise ValueError."""
    with pytest.raises(ValueError):
        _browser_check("")


def test_browser_error_no_equals():
    """Missing equals sign — must raise ValueError."""
    with pytest.raises(ValueError):
        _browser_check("x[k+1] + grad_f(x[k])")


def test_browser_rescaled_relaxed_proximal_point():
    """RPP with doubled relaxation parameter — must match RPP, not fail.
    Tests that parametric matching handles shared parameter names correctly
    through the full browser pipeline.
    """
    result = _browser_check(
        "x[k+1] = (1 - 2 * alpha_rpp) * x[k] + 2 * alpha_rpp * prox_f(x[k])"
    )
    assert len(result['matches']) > 0, \
        "Rescaled RPP should match Relaxed Proximal Point"
    names = [m['name'] for m in result['matches']]
    assert 'Relaxed Proximal Point' in names, \
        f"Expected 'Relaxed Proximal Point' in matches, got: {names}"
    rpp = next(m for m in result['matches'] if m['name'] == 'Relaxed Proximal Point')
    # Must show parameter mapping
    assert 'params' in rpp, "Match should include parameter mapping"
    assert len(rpp['params']) > 0, "Parameter mapping should not be empty"


def test_browser_latex_quality():
    """LaTeX output must be valid KaTeX — no raw SymPy artifacts."""
    result = _browser_check("x[k+1] = x[k] - alpha * grad_f(x[k])")

    tf = result['tf_latex']
    # Must contain proper LaTeX fraction, not Python division
    assert '/' not in tf or '\\frac' in tf or '}{' in tf, \
        f"LaTeX looks like raw Python: {tf}"
    # Must not contain Python-style multiplication
    assert '**' not in tf, f"LaTeX contains **: {tf}"
    # Must contain alpha as LaTeX
    assert '\\alpha' in tf, f"LaTeX missing \\alpha: {tf}"
