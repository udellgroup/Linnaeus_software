"""Load algorithm library and parse transfer functions into SymPy objects."""
import json
from sympy import symbols, Matrix
from sympy.parsing.sympy_parser import parse_expr, standard_transformations


z = symbols('z')

# All possible parameters used across library entries
PARAM_NAMES = [
    'alpha', 'beta', 'gamma', 'eta', 't', 'nu', 's',
    'alpha_rpp', 'beta_qhm', 'beta_sum', 'delta'
]
PARAMS = {name: symbols(name) for name in PARAM_NAMES}


def load_library(json_path):
    """Load algorithms.json and parse TF strings into SymPy objects.

    Returns list of dicts, each with original fields plus 'tf' (SymPy Matrix).
    """
    with open(json_path) as f:
        data = json.load(f)

    local_dict = {'z': z, 'Matrix': Matrix, **PARAMS}
    algorithms = []

    for entry in data:
        algo = dict(entry)
        tf_str = entry.get('transferFunction', '')
        if tf_str and not entry.get('catalogOnly', False):
            try:
                tf = parse_expr(tf_str, local_dict=local_dict,
                                transformations=standard_transformations)
                if isinstance(tf, Matrix):
                    algo['tf'] = tf
                else:
                    algo['tf'] = Matrix([[tf]])
            except Exception as e:
                algo['tf'] = None
                algo['tf_error'] = str(e)
        else:
            algo['tf'] = None

        # Parse parameter symbols for this entry
        algo['param_symbols'] = [PARAMS[p] for p in entry.get('parameters', [])
                                  if p in PARAMS]
        algorithms.append(algo)

    return algorithms


def check_all_equivalences(H_user, user_oracles, library, z_var):
    """Check user's H(z) against all library entries.

    Returns list of match results sorted by strength (oracle > shift > LFT).
    """
    from equivalence import (check_oracle_equivalence,
                              check_shift_equivalence,
                              check_lft_equivalence,
                              build_block_m_hat)

    matches = []
    for algo in library:
        if algo['tf'] is None:
            continue

        # Library TFs use symbols('z'), but user TFs may use a different z symbol.
        # Substitute library's z with the user's z_var for comparison.
        H_lib = algo['tf'].subs(z, z_var)
        lib_oracles = algo.get('oracles', [])

        if user_oracles == lib_oracles:
            # Same oracles: try oracle equivalence
            result = check_oracle_equivalence(
                H_user, H_lib, z_var,
                lib_params=algo.get('param_symbols')
            )
            if result['match']:
                matches.append({
                    'algorithm': algo,
                    'type': 'oracle',
                    'details': result,
                })
                continue

            # Try shift equivalence (only for multi-oracle)
            if H_user.rows > 1:
                result = check_shift_equivalence(H_user, H_lib, z_var)
                if result['match']:
                    matches.append({
                        'algorithm': algo,
                        'type': 'shift',
                        'details': result,
                    })
                    continue

        elif len(user_oracles) == len(lib_oracles):
            # Different but same-count oracles: try LFT
            try:
                M_hat = build_block_m_hat(user_oracles, lib_oracles)
            except ValueError:
                M_hat = None
            if M_hat is not None:
                result = check_lft_equivalence(H_user, H_lib, M_hat, z_var)
                if result['match']:
                    matches.append({
                        'algorithm': algo,
                        'type': 'lft',
                        'details': result,
                    })

    # Sort: oracle first, then shift, then lft
    type_order = {'oracle': 0, 'shift': 1, 'lft': 2}
    matches.sort(key=lambda m: type_order[m['type']])

    return matches
