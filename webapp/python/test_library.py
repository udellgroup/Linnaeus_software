# Save as webapp/python/test_library.py
import os
import pytest


JSON_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'algorithms.json')


def test_load_library():
    from library import load_library
    algos = load_library(JSON_PATH)
    assert len(algos) >= 22

    # Check that non-catalog entries have valid TFs
    for a in algos:
        if not a.get('catalogOnly', False) and a.get('transferFunction', ''):
            assert a['tf'] is not None, f"{a['name']} failed to parse TF: {a.get('tf_error', 'unknown')}"


def test_gradient_descent_tf():
    from library import load_library, z
    from sympy import cancel, symbols
    algos = load_library(JSON_PATH)

    gd = next(a for a in algos if a['id'] == 'gradient_descent')
    alpha = symbols('alpha')
    expected_tf = -alpha / (z - 1)
    assert cancel(gd['tf'][0, 0] - expected_tf) == 0


def _get_library():
    from library import load_library
    return load_library(JSON_PATH)


def _run_pipeline(equations):
    from parser import parse_equations
    from compute import compute_transfer_function
    parsed = parse_equations(equations)
    z = parsed['z_var']
    H = compute_transfer_function(
        parsed['state_vars'], parsed['oracle_inputs'],
        parsed['oracle_outputs'], parsed['z_equations'], z
    )
    return H, parsed['oracle_types'], z


@pytest.fixture(scope='module')
def library():
    return _get_library()


class TestSelfMatch:
    """Every algorithm with equations must match itself via the full pipeline."""

    @pytest.fixture(scope='class')
    def lib(self):
        return _get_library()

    def _algo_ids(self):
        lib = _get_library()
        return [
            a['id'] for a in lib
            if a['tf'] is not None and a.get('equations')
        ]

    @pytest.mark.parametrize('algo_id', [
        'gradient_descent', 'heavy_ball', 'nesterov', 'triple_momentum',
        'quasi_hyperbolic_momentum', 'stochastic_unified_momentum',
        'projected_reflected_gradient', 'modified_arrow_hurwicz',
        'extrapolation_from_past', 'optimistic_mirror_descent',
        'proximal_point', 'relaxed_proximal_point',
        'douglas_rachford', 'admm', 'peaceman_rachford',
        'chambolle_pock', 'proximal_gradient', 'conjugate_proximal_gradient',
        'davis_yin', 'extragradient', 'projected_gradient',
        'extragradient_korpelevich', 'extragradient_tseng',
        'nids', 'exact_diffusion',
    ])
    def test_self_match(self, lib, algo_id):
        from library import check_all_equivalences
        algo = next(a for a in lib if a['id'] == algo_id)
        H, oracle_types, z = _run_pipeline(algo['equations'])
        matches = check_all_equivalences(H, oracle_types, lib, z)
        match_names = [m['algorithm']['name'] for m in matches]
        assert algo['name'] in match_names, (
            f"{algo['name']} did not match itself. Matches: {match_names}"
        )
