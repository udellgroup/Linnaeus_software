"""Load algorithm library and parse transfer functions into SymPy objects."""
import json
from sympy import symbols, Matrix
from sympy.parsing.sympy_parser import parse_expr, standard_transformations


z = symbols('z')

# All possible parameters used across library entries
PARAM_NAMES = [
    'alpha', 'beta', 'gamma', 'delta', 'eta', 'nu',
    'rho', 'sigma', 'tau', 'theta', 'lam',
    't', 's',
]
PARAMS = {}
for name in PARAM_NAMES:
    if name == 'lam':
        # Use Symbol('lambda') so latex() renders as \lambda.
        # 'lam' is used in JSON because 'lambda' is a Python keyword.
        PARAMS[name] = symbols('lambda')
    else:
        PARAMS[name] = symbols(name)


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


def _find_oracle_permutation(user_oracles, lib_oracles):
    """Find a permutation mapping user oracle order to library oracle order.

    Returns list perm where user_oracles[perm[i]] == lib_oracles[i],
    or None if no valid permutation exists (different oracle multisets).
    """
    if sorted(user_oracles) != sorted(lib_oracles):
        return None
    used = set()
    perm = []
    for lib_oracle in lib_oracles:
        for j, user_oracle in enumerate(user_oracles):
            if j not in used and user_oracle == lib_oracle:
                perm.append(j)
                used.add(j)
                break
        else:
            return None
    return perm


def _permute_tf(H, perm):
    """Permute rows and columns of a transfer function matrix.

    perm[i] = j means row/col i of the result comes from row/col j of H.
    """
    from sympy import zeros
    n = len(perm)
    H_perm = zeros(n, n)
    for i in range(n):
        for j in range(n):
            H_perm[i, j] = H[perm[i], perm[j]]
    return H_perm


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

        # Determine if oracles match (possibly up to permutation)
        if user_oracles == lib_oracles:
            H_check = H_user
        else:
            # Try permutation: same oracle types but different order
            perm = _find_oracle_permutation(user_oracles, lib_oracles)
            if perm is not None:
                H_check = _permute_tf(H_user, perm)
            else:
                H_check = None

        if H_check is not None:
            # Same oracles (possibly permuted): try oracle equivalence
            result = check_oracle_equivalence(
                H_check, H_lib, z_var,
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
            if H_check.rows > 1:
                result = check_shift_equivalence(
                    H_check, H_lib, z_var,
                    lib_params=algo.get('param_symbols')
                )
                if result['match']:
                    matches.append({
                        'algorithm': algo,
                        'type': 'shift',
                        'details': result,
                    })
                    continue

        if len(user_oracles) == len(lib_oracles) and user_oracles != lib_oracles:
            # Different oracle types but same count: try LFT.
            # Try all permutations of library oracles and shift vectors,
            # since equivalence may require permutation + LFT + shift.
            import itertools
            p = len(lib_oracles)
            MAX_SHIFT = 2
            tried_orderings = set()
            found_lft = False
            for perm_indices in itertools.permutations(range(p)):
                if found_lft:
                    break
                perm_lib_oracles = [lib_oracles[i] for i in perm_indices]
                key = tuple(perm_lib_oracles)
                if key in tried_orderings:
                    continue
                tried_orderings.add(key)
                try:
                    M_hat, internal_syms = build_block_m_hat(
                        user_oracles, perm_lib_oracles)
                except ValueError:
                    continue
                perm_list = list(perm_indices)
                H_lib_perm = _permute_tf(H_lib, perm_list) \
                    if perm_list != list(range(p)) else H_lib

                # Try LFT with each candidate shift vector applied to H_lib
                for shifts in itertools.product(
                        range(-MAX_SHIFT, MAX_SHIFT + 1), repeat=p - 1):
                    m = [0] + list(shifts)
                    # Apply shift: H_lib_shifted[i,j] = z^{m_i-m_j} * H_lib_perm[i,j]
                    from sympy import zeros as _zeros
                    H_lib_shifted = _zeros(p, p)
                    for i in range(p):
                        for j in range(p):
                            H_lib_shifted[i, j] = \
                                z_var**(m[i] - m[j]) * H_lib_perm[i, j]
                    result = check_lft_equivalence(
                        H_user, H_lib_shifted, M_hat, z_var,
                        lib_params=algo.get('param_symbols'),
                        internal_syms=internal_syms,
                    )
                    if result['match']:
                        # Normalize shift
                        min_m = min(m)
                        m_norm = [mi - min_m for mi in m]
                        if any(mi != 0 for mi in m_norm):
                            result['shift_vector'] = m_norm
                        matches.append({
                            'algorithm': algo,
                            'type': 'lft',
                            'details': result,
                        })
                        found_lft = True
                        break

    # Sort: oracle first, then shift, then lft
    type_order = {'oracle': 0, 'shift': 1, 'lft': 2}
    matches.sort(key=lambda m: type_order[m['type']])

    return matches
