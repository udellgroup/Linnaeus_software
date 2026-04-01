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

        # Parse characteristic polynomial for consensus entries
        cp_str = entry.get('charPoly', '')
        if cp_str:
            try:
                algo['char_poly'] = parse_expr(
                    cp_str, local_dict=local_dict,
                    transformations=standard_transformations)
            except Exception:
                algo['char_poly'] = None
        else:
            algo['char_poly'] = None

        # Parse parameter symbols for this entry
        algo['param_symbols'] = [PARAMS[p] for p in entry.get('parameters', [])
                                  if p in PARAMS]
        # Mark distributed algorithms and track their universal params
        algo['distributed'] = entry.get('distributed', False)
        if algo['distributed']:
            algo['universal_params'] = [PARAMS['lam']]
        else:
            algo['universal_params'] = []
        algorithms.append(algo)

    return algorithms


def _operator_class_counts(oracle_list, merge_conjugates=False):
    """Count oracles by operator class: grad, prox, proxstar, proj.

    If merge_conjugates=True, prox and proxstar are counted together
    (useful for Moreau-aware matching where prox_f <-> prox_fstar).
    """
    from collections import Counter
    counts = Counter()
    for o in oracle_list:
        op, func, conj = _decompose_oracle(o)
        if merge_conjugates and op == 'prox':
            counts['prox_any'] += 1
        elif op == 'prox' and conj:
            counts['proxstar'] += 1
        else:
            counts[op] += 1
    return counts


def _moreau_dual(oracle_name):
    """Return the Moreau dual of an oracle, or None if not applicable.

    Moreau identity: prox_f + prox_fstar = I, so prox_f <-> prox_fstar.
    """
    op, func, conj = _decompose_oracle(oracle_name)
    if op == 'prox':
        suffix = func + ('star' if not conj else '')
        return f'prox_{suffix}'
    return None


def _find_all_oracle_permutations(user_oracles, lib_oracles,
                                   allow_moreau=False):
    """Find all permutations mapping user oracle order to library oracle order.

    Yields lists perm where user_oracles[perm[i]] == lib_oracles[i].
    When oracle types repeat (e.g., two grad_f), there are multiple valid
    permutations; the correct one depends on the transfer function structure.

    If allow_moreau=True, also match Moreau-dual pairs (prox_X <-> prox_Xstar).
    """
    if not allow_moreau:
        if sorted(user_oracles) != sorted(lib_oracles):
            return

    n = len(lib_oracles)
    if len(user_oracles) != n:
        return

    def _compatible(user_o, lib_o):
        if user_o == lib_o:
            return True
        if allow_moreau:
            dual = _moreau_dual(user_o)
            if dual is not None and dual == lib_o:
                return True
        return False

    # Quick check: can all lib oracles be matched?
    if allow_moreau:
        from collections import Counter
        # For each lib oracle, at least one user oracle must be compatible
        for lo in lib_oracles:
            if not any(_compatible(uo, lo) for uo in user_oracles):
                return

    def _backtrack(pos, used, perm):
        if pos == n:
            yield list(perm)
            return
        for j in range(n):
            if j not in used and _compatible(user_oracles[j], lib_oracles[pos]):
                used.add(j)
                perm.append(j)
                yield from _backtrack(pos + 1, used, perm)
                perm.pop()
                used.discard(j)

    yield from _backtrack(0, set(), [])


def _reduce_tf_for_identity_oracles(H, proj_indices):
    """Reduce TF matrix by substituting identity for oracles at proj_indices.

    When P_C = identity (u_P = y_P), the projection oracle is a pass-through.
    Uses Schur complement: H_reduced = H_KK + H_KP * (I - H_PP)^{-1} * H_PK
    where K = kept (non-projection) indices, P = projection indices.

    Returns reduced Matrix, or None if the reduction is impossible
    (e.g. (I - H_PP) is singular, or all oracles are projections).
    """
    from sympy import eye, Matrix, cancel

    n = H.rows
    keep = [i for i in range(n) if i not in proj_indices]
    P = sorted(proj_indices)

    if not keep:
        return None  # All oracles are projections

    H_KK = Matrix([[H[i, j] for j in keep] for i in keep])
    H_KP = Matrix([[H[i, j] for j in P] for i in keep])
    H_PK = Matrix([[H[i, j] for j in keep] for i in P])
    H_PP = Matrix([[H[i, j] for j in P] for i in P])

    try:
        inv_term = (eye(len(P)) - H_PP).inv()
    except Exception:
        return None  # (I - H_PP) is singular

    H_reduced = H_KK + H_KP * inv_term * H_PK
    return H_reduced.applyfunc(lambda e: cancel(e))


def _decompose_oracle(name):
    """Decompose oracle name into (operator, function, is_conjugate).

    Examples:
        'grad_f'     -> ('grad', 'f', False)
        'prox_gstar' -> ('prox', 'g', True)
        'P_C'        -> ('proj', 'C', False)
    """
    if name == 'P_C':
        return ('proj', 'C', False)
    for prefix in ('prox_', 'grad_'):
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            if suffix.endswith('star'):
                return (prefix[:-1], suffix[:-4], True)
            return (prefix[:-1], suffix, False)
    return (name, '', False)


def _apply_func_renaming(oracle_list, mapping):
    """Apply function renaming to an oracle list.

    mapping: dict like {'h': 'f', 'f': 'h'}.
    Returns new list with renamed function suffixes.
    """
    result = []
    for name in oracle_list:
        op, func, conj = _decompose_oracle(name)
        if op == 'proj':
            result.append(name)
            continue
        new_func = mapping.get(func, func)
        suffix = new_func + ('star' if conj else '')
        result.append(f'{op}_{suffix}')
    return result


def _find_function_renamings(user_oracles, lib_oracles):
    """Find all consistent function renamings mapping user oracle types to lib.

    Yields (renamed_user_oracles, func_mapping) pairs where func_mapping
    is a dict like {'h': 'f', 'f': 'h'}.
    Only yields non-identity renamings.
    """
    from itertools import permutations

    user_ops = [_decompose_oracle(o) for o in user_oracles]
    lib_ops = [_decompose_oracle(o) for o in lib_oracles]

    # Get unique function names (excluding proj/C)
    user_funcs = sorted(set(func for op, func, conj in user_ops if op != 'proj'))
    lib_funcs = sorted(set(func for op, func, conj in lib_ops if op != 'proj'))

    if len(user_funcs) != len(lib_funcs):
        return

    # Check operator-class counts match (ignoring function names)
    from collections import Counter
    user_shapes = Counter((op, conj) for op, func, conj in user_ops)
    lib_shapes = Counter((op, conj) for op, func, conj in lib_ops)
    if user_shapes != lib_shapes:
        return

    # Try all bijections from user_funcs to lib_funcs
    for perm in permutations(lib_funcs):
        mapping = dict(zip(user_funcs, perm))
        # Skip identity
        if all(mapping.get(f, f) == f for f in user_funcs):
            continue
        renamed = _apply_func_renaming(user_oracles, mapping)
        if sorted(renamed) == sorted(lib_oracles):
            yield renamed, mapping


def _reduce_tf_for_zero_function(H, oracle_types, func_name):
    """Reduce TF by setting function func_name to zero.

    When func = 0:
      - grad_func -> 0 (dead oracle): remove row/col
      - prox_func -> identity (u = y): Schur complement
      - prox_funcstar -> 0 (dead oracle): remove row/col

    Returns (H_reduced, remaining_oracles) or None if reduction fails.
    """
    from sympy import Matrix, cancel

    identity_indices = []
    dead_indices = []

    for i, otype in enumerate(oracle_types):
        op, func, conj = _decompose_oracle(otype)
        if func != func_name:
            continue
        if op == 'prox' and not conj:
            identity_indices.append(i)  # prox_X -> identity
        else:
            dead_indices.append(i)  # grad_X, prox_Xstar -> zero

    if not identity_indices and not dead_indices:
        return None  # Function not used in this algorithm

    n = H.rows
    all_remove = set(dead_indices)

    # Step 1: Handle identity oracles via Schur complement
    H_work = H
    current_oracle_types = list(oracle_types)
    if identity_indices:
        H_work = _reduce_tf_for_identity_oracles(H_work, identity_indices)
        if H_work is None:
            return None
        # Remove identity indices from tracking
        keep_after_identity = [i for i in range(n) if i not in identity_indices]
        current_oracle_types = [oracle_types[i] for i in keep_after_identity]
        # Remap dead indices to new positions
        old_to_new = {old: new for new, old in enumerate(keep_after_identity)}
        dead_indices = [old_to_new[i] for i in dead_indices if i in old_to_new]

    # Step 2: Remove dead oracle rows/cols
    if dead_indices:
        n2 = H_work.rows
        keep = [i for i in range(n2) if i not in dead_indices]
        if not keep:
            return None  # All oracles removed
        H_work = Matrix([[H_work[i, j] for j in keep] for i in keep])
        H_work = H_work.applyfunc(lambda e: cancel(e))
        current_oracle_types = [current_oracle_types[i] for i in keep]

    return H_work, current_oracle_types


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


def _is_trivial_match(details, algo):
    """Reject matches where ALL library parameters map to zero.

    When every step-size / parameter of the library algorithm is zero, the
    algorithm degenerates and the match is meaningless.
    """
    from sympy import cancel
    params = details.get('params', {})
    lib_param_set = set(algo.get('param_symbols', []))
    if not lib_param_set:
        return False  # No params to check
    lib_vals = [v for k, v in params.items() if k in lib_param_set]
    if not lib_vals:
        return False
    return all(cancel(v) == 0 for v in lib_vals)


def _compute_gradf0_char_poly(equations, z_var):
    """Compute characteristic polynomial of an algorithm with oracle calls set to 0.

    Replaces all oracle calls (grad_f, prox_f, etc.) with 0, re-parses,
    and computes the characteristic polynomial of the resulting oracle-free system.
    """
    import re
    from parser import parse_equations, KNOWN_ORACLES
    from compute import compute_char_poly

    # Replace all oracle calls with 0
    zeroed = list(equations)
    for oracle in KNOWN_ORACLES:
        # Match oracle_name(...) with nested parens
        pattern = r'\b' + oracle + r'\([^)]*\)'
        zeroed = [re.sub(pattern, '0', eq) for eq in zeroed]

    parsed = parse_equations(zeroed)
    return compute_char_poly(parsed['state_vars'], parsed['z_equations'],
                             parsed['z_var'])


def check_all_equivalences(H_user, user_oracles, library, z_var,
                           user_distributed=False, user_universal_params=None,
                           user_has_projection=None,
                           user_char_poly=None, user_equations=None):
    """Check user's H(z) against all library entries.

    Args:
        user_distributed: True if the user's algorithm uses mixing matrix W.
        user_universal_params: list of Symbol objects that are universal
            (must match for ALL values, e.g. Symbol('lambda') for distributed).
        user_has_projection: True if the user's algorithm uses P_C oracle.
            Auto-detected from user_oracles if not provided.
        user_char_poly: SymPy expr for user's characteristic polynomial
            (for consensus algorithms with no oracles).
        user_equations: original equation strings (needed for grad_f=0 reduction).

    Returns list of match results sorted by strength (oracle > shift > LFT).
    """
    from equivalence import (check_oracle_equivalence,
                              check_shift_equivalence,
                              check_lft_equivalence,
                              build_block_m_hat)
    from sympy import cancel as _cancel, Poly, symbols as _symbols

    if user_universal_params is None:
        user_universal_params = []
    user_universal_set = set(user_universal_params)

    # Auto-detect projection from oracle types if not explicitly provided
    if user_has_projection is None:
        user_has_projection = any(o == 'P_C' for o in user_oracles)

    user_is_consensus = (len(user_oracles) == 0)

    # Pre-compute user-side function reductions (cache expensive Schur
    # complements so they aren't recomputed for every library entry).
    # Key: frozenset of zeroed function names → (H_reduced, oracle_list)
    _user_reductions = {}
    if H_user is not None and len(user_oracles) > 0:
        user_funcs = sorted(set(
            f for o in user_oracles
            for _, f, _ in [_decompose_oracle(o)]
            if f and f != 'C'))
        from itertools import combinations as _combs
        for n in range(1, len(user_funcs) + 1):
            for combo in _combs(user_funcs, n):
                H_r, o_r = H_user, list(user_oracles)
                ok = True
                for zf in combo:
                    if any(_decompose_oracle(o)[1] == zf for o in o_r):
                        res = _reduce_tf_for_zero_function(H_r, o_r, zf)
                        if res is None:
                            ok = False
                            break
                        H_r, o_r = res
                if ok:
                    _user_reductions[frozenset(combo)] = (H_r, o_r)

    matches = []
    for algo in library:
        lib_oracles = algo.get('oracles', [])
        lib_is_consensus = algo.get('consensus', False)
        lib_distributed = algo.get('distributed', False)

        # --- Consensus cross-category matching ---
        # One side has oracles, the other is consensus (no oracles).
        # Compare characteristic polynomials under grad_f=0 condition.
        if user_is_consensus != lib_is_consensus:
            if user_is_consensus and user_char_poly is not None:
                # User is consensus, library has oracles
                lib_eqs = algo.get('equations', [])
                if not lib_eqs:
                    continue
                try:
                    lib_cp = _compute_gradf0_char_poly(lib_eqs, z_var)
                except Exception:
                    continue
                user_cp = user_char_poly.subs(z, z_var)
                # Compare as polynomials in z with lambda as universal param
                diff = _cancel(user_cp - lib_cp.subs(z, z_var))
                if diff == 0:
                    matches.append({
                        'algorithm': algo,
                        'type': 'oracle',
                        'details': {
                            'match': True,
                            'params': {},
                            'user_params': {},
                            'free_params': [],
                            'condition_note': (
                                '\\text{Equivalent when } '
                                '\\nabla f = 0 \\text{ (trivial functions)}'),
                        },
                        'permuted': False,
                        'conditional': True,
                    })
            elif lib_is_consensus and algo.get('char_poly') is not None:
                # Library is consensus, user has oracles
                if user_equations is None:
                    continue
                try:
                    user_cp = _compute_gradf0_char_poly(
                        user_equations, z_var)
                except Exception:
                    continue
                lib_cp = algo['char_poly'].subs(z, z_var)
                diff = _cancel(user_cp - lib_cp)
                if diff == 0:
                    matches.append({
                        'algorithm': algo,
                        'type': 'oracle',
                        'details': {
                            'match': True,
                            'params': {},
                            'user_params': {},
                            'free_params': [],
                            'condition_note': (
                                '\\text{Equivalent when } '
                                '\\nabla f = 0 \\text{ (trivial functions)}'),
                        },
                        'permuted': False,
                        'conditional': True,
                    })
            continue  # Skip normal matching for consensus cross-category

        # Both consensus: compare char polys directly
        if user_is_consensus and lib_is_consensus:
            if user_char_poly is not None and algo.get('char_poly') is not None:
                user_cp = user_char_poly.subs(z, z_var)
                lib_cp = algo['char_poly'].subs(z, z_var)
                diff = _cancel(user_cp - lib_cp)
                if diff == 0:
                    matches.append({
                        'algorithm': algo,
                        'type': 'oracle',
                        'details': {
                            'match': True,
                            'params': {},
                            'user_params': {},
                            'free_params': [],
                        },
                        'permuted': False,
                        'conditional': False,
                    })
            continue

        if algo['tf'] is None:
            continue

        # Library TFs use symbols('z'), but user TFs may use a different z symbol.
        # Substitute library's z with the user's z_var for comparison.
        H_lib = algo['tf'].subs(z, z_var)
        lib_distributed = algo.get('distributed', False)

        # --- Detect cross-category situations ---
        # Distributed cross-category: one side uses W/L, the other doesn't.
        # Projection cross-category: at least one side uses P_C and the
        # oracle lists differ (different P_C counts or one side has none).
        lib_has_proj = any(o == 'P_C' for o in lib_oracles)
        dist_cross = (user_distributed != lib_distributed)
        proj_cross = ((user_has_projection or lib_has_proj)
                      and sorted(user_oracles) != sorted(lib_oracles))

        if dist_cross or proj_cross:
            # Build condition notes and apply reductions
            conditions = []
            H_u_cond = H_user
            H_l_cond = H_lib
            u_oracles_cond = list(user_oracles)
            l_oracles_cond = list(lib_oracles)

            # 1) Distributed cross-category: substitute lambda=0
            if dist_cross:
                lam_sym = PARAMS['lam']
                if user_distributed:
                    H_u_cond = H_u_cond.applyfunc(
                        lambda e: _cancel(e.subs(lam_sym, 0)))
                if lib_distributed:
                    H_l_cond = H_l_cond.applyfunc(
                        lambda e: _cancel(e.subs(lam_sym, 0)))
                conditions.append(
                    '\\lambda=0 \\text{ (trivial one-node graph)}')

            # 2) Projection cross-category: reduce P_C via Schur complement.
            #    Reduce both sides — handles cases where both have P_C but
            #    with different counts (e.g. Korpelevich 2×P_C vs Tseng 1×P_C).
            if proj_cross:
                if any(o == 'P_C' for o in u_oracles_cond):
                    proj_idx = [i for i, o in enumerate(u_oracles_cond)
                                if o == 'P_C']
                    H_reduced = _reduce_tf_for_identity_oracles(
                        H_u_cond, proj_idx)
                    if H_reduced is None:
                        continue
                    H_u_cond = H_reduced
                    u_oracles_cond = [o for o in u_oracles_cond if o != 'P_C']
                if any(o == 'P_C' for o in l_oracles_cond):
                    proj_idx = [i for i, o in enumerate(l_oracles_cond)
                                if o == 'P_C']
                    H_reduced = _reduce_tf_for_identity_oracles(
                        H_l_cond, proj_idx)
                    if H_reduced is None:
                        continue
                    H_l_cond = H_reduced
                    l_oracles_cond = [o for o in l_oracles_cond if o != 'P_C']
                conditions.append(
                    'P_C = I \\text{ (unconstrained)}')

            # Try oracle equivalence with reduced/substituted TFs,
            # including all valid oracle permutations.
            if sorted(u_oracles_cond) == sorted(l_oracles_cond):
                cond_perms = []
                if u_oracles_cond == l_oracles_cond:
                    cond_perms.append((None, False))
                for perm in _find_all_oracle_permutations(
                        u_oracles_cond, l_oracles_cond):
                    is_identity = (perm == list(range(len(perm))))
                    if not is_identity:
                        cond_perms.append((perm, True))

                for perm, permuted in cond_perms:
                    H_u_check = (_permute_tf(H_u_cond, perm)
                                 if perm else H_u_cond)
                    result = check_oracle_equivalence(
                        H_u_check, H_l_cond, z_var,
                        lib_params=algo.get('param_symbols')
                    )
                    if result['match']:
                        joiner = (' \\newline \\text{and } '
                                  if len(conditions) > 1
                                  else ' \\text{ and } ')
                        result['condition_note'] = (
                            '\\text{Equivalent when } '
                            + joiner.join(conditions))
                        matches.append({
                            'algorithm': algo,
                            'type': 'oracle',
                            'details': result,
                            'permuted': permuted,
                            'conditional': True,
                            'projected': proj_cross,
                        })
                        break  # First matching permutation suffices
            continue  # Skip normal equivalence for cross-category

        # Determine universal params for same-category comparison.
        if user_distributed and lib_distributed:
            universal_params = user_universal_set | set(algo.get('universal_params', []))
        else:
            universal_params = set()

        # Determine if oracles match (possibly up to permutation).
        # When oracle types repeat, there are multiple valid permutations;
        # try all of them since the correct one depends on the TF structure.
        perms_to_try = []
        if user_oracles == lib_oracles:
            perms_to_try.append((None, False))  # identity, not permuted
        if sorted(user_oracles) == sorted(lib_oracles):
            for perm in _find_all_oracle_permutations(user_oracles,
                                                       lib_oracles):
                is_identity = (perm == list(range(len(perm))))
                if not is_identity:
                    perms_to_try.append((perm, True))

        found = False
        for perm, permuted in perms_to_try:
            H_check = _permute_tf(H_user, perm) if perm else H_user

            # Same oracles (possibly permuted): try oracle equivalence
            result = check_oracle_equivalence(
                H_check, H_lib, z_var,
                lib_params=algo.get('param_symbols'),
                universal_params=universal_params,
            )
            if result['match']:
                matches.append({
                    'algorithm': algo,
                    'type': 'oracle',
                    'details': result,
                    'permuted': permuted,
                })
                found = True
                break

            # Try shift equivalence (only for multi-oracle)
            if H_check.rows > 1:
                result = check_shift_equivalence(
                    H_check, H_lib, z_var,
                    lib_params=algo.get('param_symbols'),
                    universal_params=universal_params,
                )
                if result['match']:
                    matches.append({
                        'algorithm': algo,
                        'type': 'shift',
                        'details': result,
                        'permuted': permuted,
                    })
                    found = True
                    break

        if found:
            continue

        # --- Function renaming (oracle swapping) ---
        # Try renaming functions (e.g., h->f) to match oracle types.
        # Pre-filter: operator class counts must match (renaming only
        # changes function names, not operator types).
        if (not found and len(user_oracles) == len(lib_oracles)
                and _operator_class_counts(user_oracles)
                    == _operator_class_counts(lib_oracles)):
            for renamed_oracles, func_mapping in _find_function_renamings(
                    user_oracles, lib_oracles):
                for perm in _find_all_oracle_permutations(
                        renamed_oracles, lib_oracles):
                    H_check = _permute_tf(H_user, perm) \
                        if perm != list(range(len(perm))) else H_user
                    # Try oracle equivalence
                    result = check_oracle_equivalence(
                        H_check, H_lib, z_var,
                        lib_params=algo.get('param_symbols'),
                        universal_params=universal_params,
                    )
                    if result['match']:
                        result['func_mapping'] = func_mapping
                        matches.append({
                            'algorithm': algo,
                            'type': 'oracle',
                            'details': result,
                            'permuted': True,
                        })
                        found = True
                        break
                    # Try shift equivalence
                    if H_check.rows > 1:
                        result = check_shift_equivalence(
                            H_check, H_lib, z_var,
                            lib_params=algo.get('param_symbols'),
                            universal_params=universal_params,
                        )
                        if result['match']:
                            result['func_mapping'] = func_mapping
                            matches.append({
                                'algorithm': algo,
                                'type': 'shift',
                                'details': result,
                                'permuted': True,
                            })
                            found = True
                            break
                if found:
                    break

        if found:
            continue

        # --- Moreau-dual oracle matching ---
        # Try matching prox_X <-> prox_Xstar (Moreau identity).
        # Pre-filter: operator counts must match when merging conjugates.
        if (not found and len(user_oracles) == len(lib_oracles)
                and _operator_class_counts(user_oracles, merge_conjugates=True)
                    == _operator_class_counts(lib_oracles, merge_conjugates=True)
                and _operator_class_counts(user_oracles)
                    != _operator_class_counts(lib_oracles)):
            # Build oracle variants: original + all function renamings
            moreau_variants = [(user_oracles, {})]
            u_ops = [_decompose_oracle(o) for o in user_oracles]
            l_ops = [_decompose_oracle(o) for o in lib_oracles]
            u_fns = sorted(set(f for _, f, _ in u_ops if f != 'C'))
            l_fns = sorted(set(f for _, f, _ in l_ops if f != 'C'))
            if len(u_fns) == len(l_fns):
                from itertools import permutations as _perms2
                for perm in _perms2(l_fns):
                    fmap = dict(zip(u_fns, perm))
                    if all(fmap.get(f, f) == f for f in u_fns):
                        continue
                    renamed = _apply_func_renaming(user_oracles, fmap)
                    moreau_variants.append((renamed, fmap))

            for u_orc_variant, func_mapping in moreau_variants:
                for perm in _find_all_oracle_permutations(
                        u_orc_variant, lib_oracles, allow_moreau=True):
                    # Identify Moreau dual pairs and real permutation
                    moreau_pairs = {}
                    for i in range(len(perm)):
                        u_o = u_orc_variant[perm[i]]
                        l_o = lib_oracles[i]
                        if u_o != l_o:
                            # This position uses Moreau duality
                            _, u_func, u_conj = _decompose_oracle(u_o)
                            _, l_func, l_conj = _decompose_oracle(l_o)
                            # Record: lib function = user function*
                            if u_conj and not l_conj:
                                moreau_pairs[l_func] = u_func + '^*'
                            elif not u_conj and l_conj:
                                moreau_pairs[l_func + '^*'] = u_func
                    if not moreau_pairs:
                        continue  # Pure permutation, already tried above
                    is_reordered = (perm != list(range(len(perm))))
                    H_check = _permute_tf(H_user, perm) if is_reordered \
                        else H_user

                    def _add_moreau_match(result, match_type):
                        if func_mapping:
                            result['func_mapping'] = dict(func_mapping)
                        # Merge Moreau pairs into func_mapping
                        fm = result.get('func_mapping', {})
                        fm.update(moreau_pairs)
                        result['func_mapping'] = fm
                        matches.append({
                            'algorithm': algo,
                            'type': match_type,
                            'details': result,
                            'permuted': is_reordered,
                        })

                    # Try oracle equivalence
                    result = check_oracle_equivalence(
                        H_check, H_lib, z_var,
                        lib_params=algo.get('param_symbols'),
                        universal_params=universal_params,
                    )
                    if result['match']:
                        _add_moreau_match(result, 'oracle')
                        found = True
                        break
                    # Try shift equivalence
                    if H_check.rows > 1:
                        result = check_shift_equivalence(
                            H_check, H_lib, z_var,
                            lib_params=algo.get('param_symbols'),
                            universal_params=universal_params,
                        )
                        if result['match']:
                            _add_moreau_match(result, 'shift')
                            found = True
                            break
                if found:
                    break

        if found:
            continue

        # --- Function zeroing conditional equivalence ---
        # Try setting one function to zero (f=0, g=0, or h=0) to match.
        if not found and user_equations is not None:
            # Collect function names from both sides
            all_funcs = set()
            for o in user_oracles + lib_oracles:
                _, func, _ = _decompose_oracle(o)
                if func and func != 'C':
                    all_funcs.add(func)

            # Helper to try matching reduced TFs (with renaming + permutation)
            def _try_match_reduced(H_u, u_orc, H_l, l_orc):
                """Try oracle equiv with optional renaming + permutation."""
                # Direct permutation
                if sorted(u_orc) == sorted(l_orc):
                    for perm in _find_all_oracle_permutations(u_orc, l_orc):
                        H_c = _permute_tf(H_u, perm) \
                            if perm != list(range(len(perm))) else H_u
                        r = check_oracle_equivalence(
                            H_c, H_l, z_var,
                            lib_params=algo.get('param_symbols'))
                        if r['match']:
                            return r
                # With function renaming
                for renamed, fmap in _find_function_renamings(u_orc, l_orc):
                    for perm in _find_all_oracle_permutations(
                            renamed, l_orc):
                        H_c = _permute_tf(H_u, perm) \
                            if perm != list(range(len(perm))) else H_u
                        r = check_oracle_equivalence(
                            H_c, H_l, z_var,
                            lib_params=algo.get('param_symbols'))
                        if r['match']:
                            r['func_mapping'] = fmap
                            return r
                return None

            # Try all non-empty subsets of functions to zero.
            # For each subset, try zeroing on: both sides, user only, lib only.
            # This handles cases where function names differ between sides
            # (e.g., user's grad_f plays the role of library's grad_h).
            from itertools import combinations
            sorted_funcs = sorted(all_funcs)

            def _try_zero_combo(zero_combo, zero_user, zero_lib):
                """Try zeroing a combo of functions on specified sides.
                Returns (result, zeroed_info) or (None, None).
                Uses pre-computed user reductions from _user_reductions cache.
                """
                # User side: use cache if zeroing user
                if zero_user:
                    u_key = frozenset(
                        zf for zf in zero_combo
                        if any(_decompose_oracle(o)[1] == zf
                               for o in user_oracles))
                    if u_key and u_key in _user_reductions:
                        H_u_r, u_o_r = _user_reductions[u_key]
                    elif u_key:
                        return None, None  # Pre-compute failed
                    else:
                        H_u_r, u_o_r = H_user, list(user_oracles)
                else:
                    H_u_r, u_o_r = H_user, list(user_oracles)

                # Library side: compute on the fly (varies per entry)
                H_l_r, l_o_r = H_lib, list(lib_oracles)
                if zero_lib:
                    for zf in zero_combo:
                        if any(_decompose_oracle(o)[1] == zf
                               for o in l_o_r):
                            res = _reduce_tf_for_zero_function(
                                H_l_r, l_o_r, zf)
                            if res is None:
                                return None, None
                            H_l_r, l_o_r = res

                # Track which side each function was actually zeroed on
                zeroed_info = []
                for zf in zero_combo:
                    u_had = any(_decompose_oracle(o)[1] == zf
                                for o in user_oracles)
                    l_had = any(_decompose_oracle(o)[1] == zf
                                for o in lib_oracles)
                    if u_had and l_had:
                        zeroed_info.append((zf, 'both'))
                    elif l_had:
                        zeroed_info.append((zf, 'lib'))
                    elif u_had:
                        zeroed_info.append((zf, 'user'))
                # Quick check: oracle counts must match after reduction
                if len(u_o_r) != len(l_o_r):
                    return None, None
                # Quick check: operator class counts must be compatible
                # (exact or Moreau-mergeable)
                if (_operator_class_counts(u_o_r) !=
                        _operator_class_counts(l_o_r) and
                    _operator_class_counts(u_o_r, True) !=
                        _operator_class_counts(l_o_r, True)):
                    return None, None
                r = _try_match_reduced(H_u_r, u_o_r, H_l_r, l_o_r)
                return r, zeroed_info

            for n_zero in range(1, len(sorted_funcs) + 1):
                if found:
                    break
                for zero_combo in combinations(sorted_funcs, n_zero):
                    result = None
                    zeroed_info = None
                    for zero_user, zero_lib in [(True, True),
                                                 (False, True),
                                                 (True, False)]:
                        result, zeroed_info = _try_zero_combo(
                            zero_combo, zero_user, zero_lib)
                        if result is not None:
                            break
                    if result is not None:
                        # Build condition note with side annotation
                        cond_parts = []
                        for zf, side in zeroed_info:
                            if side == 'lib':
                                cond_parts.append(
                                    zf + '_{\\text{lib}} = 0')
                            else:
                                cond_parts.append(zf + ' = 0')
                        joiner = (' \\text{ and } '
                                  if len(cond_parts) <= 2
                                  else ' \\newline \\text{and } ')
                        result['condition_note'] = (
                            '\\text{Equivalent when } '
                            + joiner.join(cond_parts))
                        matches.append({
                            'algorithm': algo,
                            'type': 'oracle',
                            'details': result,
                            'permuted': True,
                            'conditional': True,
                        })
                        found = True
                        break

        # Don't skip LFT if we only found conditional (zeroing) matches —
        # a non-conditional LFT match is better.
        if found and not any(
                m.get('conditional') and m['algorithm'] is algo
                for m in matches):
            continue

        if len(user_oracles) == len(lib_oracles) and user_oracles != lib_oracles:
            # Different oracle types but same count: try LFT.
            # Pre-filter: LFT can relate prox<->grad within the same function
            # but can't bridge completely different operator structures.
            # Skip if grad counts differ (LFT preserves grad count).
            u_grad_cnt = sum(1 for o in user_oracles
                             if _decompose_oracle(o)[0] == 'grad')
            l_grad_cnt = sum(1 for o in lib_oracles
                             if _decompose_oracle(o)[0] == 'grad')
            if u_grad_cnt != l_grad_cnt:
                continue  # Skip LFT entirely for this library entry

            import itertools
            p = len(lib_oracles)
            MAX_SHIFT = 2

            # Build list of (user_oracles_variant, func_mapping) to try:
            # original oracles + all function bijections (relaxed — we
            # don't require operator shapes to match since LFT can relate
            # different operator types like prox_gstar <-> prox_g).
            oracle_variants = [(user_oracles, {})]
            from itertools import permutations as _perms
            user_ops = [_decompose_oracle(o) for o in user_oracles]
            lib_ops = [_decompose_oracle(o) for o in lib_oracles]
            u_funcs = sorted(set(f for _, f, _ in user_ops if f != 'C'))
            l_funcs = sorted(set(f for _, f, _ in lib_ops if f != 'C'))
            if len(u_funcs) == len(l_funcs):
                for perm in _perms(l_funcs):
                    fmap = dict(zip(u_funcs, perm))
                    if all(fmap.get(f, f) == f for f in u_funcs):
                        continue  # skip identity
                    renamed = _apply_func_renaming(user_oracles, fmap)
                    oracle_variants.append((renamed, fmap))

            lft_candidates = []
            for u_oracles_variant, func_mapping in oracle_variants:
                tried_orderings = set()
                for perm_indices in itertools.permutations(range(p)):
                    perm_lib_oracles = [lib_oracles[i] for i in perm_indices]
                    key = tuple(perm_lib_oracles)
                    if key in tried_orderings:
                        continue
                    tried_orderings.add(key)
                    try:
                        M_hat, internal_syms = build_block_m_hat(
                            u_oracles_variant, perm_lib_oracles)
                    except ValueError:
                        continue
                    perm_list = list(perm_indices)
                    H_lib_perm = _permute_tf(H_lib, perm_list) \
                        if perm_list != list(range(p)) else H_lib

                    # Try LFT with each candidate shift vector
                    for shifts in itertools.product(
                            range(-MAX_SHIFT, MAX_SHIFT + 1), repeat=p - 1):
                        m = [0] + list(shifts)
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
                            universal_params=universal_params,
                        )
                        if result['match']:
                            min_m = min(m)
                            m_norm = [mi - min_m for mi in m]
                            if any(mi != 0 for mi in m_norm):
                                result['shift_vector'] = m_norm
                            if func_mapping:
                                result['func_mapping'] = func_mapping
                            n_zeros = sum(
                                1 for v in result.get('params', {}).values()
                                if v == 0
                            ) + sum(
                                1 for v in result.get('user_params', {}).values()
                                if v == 0
                            )
                            lft_candidates.append((n_zeros, {
                                'algorithm': algo,
                                'type': 'lft',
                                'details': result,
                                'permuted': perm_list != list(range(p)),
                            }))

            if lft_candidates:
                lft_candidates.sort(key=lambda x: x[0])
                matches.append(lft_candidates[0][1])

    # Sort: non-conditional before conditional, then oracle > shift > lft
    type_order = {'oracle': 0, 'shift': 1, 'lft': 2}
    matches.sort(key=lambda m: (
        1 if m.get('conditional') else 0,
        type_order[m['type']],
    ))

    # Filter out trivial matches (all library params zero)
    matches = [m for m in matches
               if not _is_trivial_match(m.get('details', {}), m['algorithm'])]

    # Remove conditional matches when a non-conditional match for the
    # same algorithm already exists (the conditional is subsumed).
    non_cond_algos = set(
        id(m['algorithm']) for m in matches if not m.get('conditional'))
    matches = [m for m in matches
               if not m.get('conditional')
               or id(m['algorithm']) not in non_cond_algos]

    return matches
