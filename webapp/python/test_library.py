# Save as webapp/python/test_library.py
import os
import pytest

def test_load_library():
    from library import load_library
    json_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'algorithms.json')
    algos = load_library(json_path)
    assert len(algos) >= 17

    # Check that non-catalog entries have valid TFs
    for a in algos:
        if not a.get('catalogOnly', False) and a.get('transferFunction', ''):
            assert a['tf'] is not None, f"{a['name']} failed to parse TF: {a.get('tf_error', 'unknown')}"

def test_gradient_descent_tf():
    from library import load_library, z
    from sympy import cancel, symbols
    json_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'algorithms.json')
    algos = load_library(json_path)

    gd = next(a for a in algos if a['id'] == 'gradient_descent')
    alpha = symbols('alpha')
    expected_tf = -alpha / (z - 1)
    assert cancel(gd['tf'][0, 0] - expected_tf) == 0
