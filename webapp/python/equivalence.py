"""
Equivalence checking for transfer function matrices.

Supports three types of equivalence:
- Oracle equivalence: H1(z) == H2(z) (possibly with parametric matching)
- Shift equivalence: H1 = Diag(z^m) * H2 * Diag(z^{-m})
- LFT equivalence: [I | -H1] * M_hat * [H2; I] == 0
"""

from sympy import (
    Matrix, eye, zeros, symbols, cancel, simplify, Poly,
    solve, numer
)


def check_oracle_equivalence(H1, H2, z, lib_params=None):
    """
    Check if H1 and H2 are oracle-equivalent (identical transfer functions).

    If lib_params is provided, attempt parametric matching: solve for the
    library parameters so that H1 == H2 after substitution.

    Returns dict with 'match' (bool) and optionally 'params' (dict).
    """
    rows, cols = H1.shape
    if H2.shape != (rows, cols):
        return {'match': False}

    # Direct comparison (no free params)
    if lib_params is None or len(lib_params) == 0:
        for i in range(rows):
            for j in range(cols):
                diff = cancel(H1[i, j] - H2[i, j])
                if diff != 0:
                    return {'match': False}
        return {'match': True, 'params': {}}

    # Parametric matching: substitute library params with fresh symbols
    # so that shared names (e.g., user's alpha_rpp vs library's alpha_rpp)
    # don't collide. The fresh symbols are the unknowns we solve for.
    from sympy import Symbol, Dummy
    fresh_params = []
    sub_map = {}  # lib_param -> fresh_param
    reverse_map = {}  # fresh_param -> lib_param
    for p in lib_params:
        fresh = Dummy('_lib_' + p.name)
        sub_map[p] = fresh
        reverse_map[fresh] = p
        fresh_params.append(fresh)

    H2_fresh = H2.subs(sub_map)

    equations = []
    for i in range(rows):
        for j in range(cols):
            diff = cancel(H1[i, j] - H2_fresh[i, j])
            if diff == 0:
                continue
            num = numer(diff)
            # Extract coefficients as polynomial in z
            try:
                p = Poly(num, z)
                coeffs = p.all_coeffs()
            except Exception:
                coeffs = [num]
            equations.extend(coeffs)

    if not equations:
        return {'match': True, 'params': {}}

    # Collect all free symbols (user params) in the equations besides
    # fresh_params and z, so we can solve for them if needed.
    all_free = set()
    for eq in equations:
        all_free |= eq.free_symbols
    all_free.discard(z)
    user_params_in_eq = sorted(all_free - set(fresh_params), key=str)

    # Strategy: solve for lib params in terms of user params first.
    # If residual constraints remain on user params, solve those next.
    # This avoids SymPy picking the trivial (all-zero) solution.
    from sympy import Dummy as _Dummy, Integer

    fresh_set = set(fresh_params)

    # Step 1: Try solving for lib params only.
    try:
        lib_solutions = solve(equations, fresh_params, dict=True)
    except Exception:
        lib_solutions = []

    # Filter out solutions with z-dependent values
    lib_solutions = [
        s for s in lib_solutions
        if all(not v.has(z) for v in s.values())
    ]

    best_result = None

    for lib_sol in lib_solutions:
        # Substitute lib solution back to get residual constraints on
        # user params.
        residuals = []
        for eq in equations:
            r = cancel(eq.subs(lib_sol))
            if r != 0:
                try:
                    p = Poly(r, z)
                    residuals.extend(p.all_coeffs())
                except Exception:
                    residuals.append(r)

        user_sol = {}
        if residuals:
            # Solve residual constraints for user params
            try:
                user_solutions = solve(residuals, user_params_in_eq, dict=True)
            except Exception:
                user_solutions = []

            if not user_solutions:
                continue
            # Prefer non-trivial solutions (not all zeros)
            chosen = None
            for us in user_solutions:
                if not all(cancel(v) == 0 for v in us.values()):
                    chosen = us
                    break
            if chosen is None:
                chosen = user_solutions[0]
            user_sol = chosen

        # Build the full solution
        sol = {}
        sol.update(lib_sol)
        # Substitute user constraints into lib values
        for k in sol:
            sol[k] = cancel(sol[k].subs(user_sol))
        sol.update(user_sol)

        # Check for unsolved fresh Dummy symbols in solution values
        solved_set = set(sol.keys())
        free_dummies = fresh_set - solved_set
        for val in sol.values():
            free_dummies |= (val.free_symbols & fresh_set)
        free_dummies -= solved_set

        if free_dummies:
            zero_sub = {d: 0 for d in free_dummies}
            sol = {k: cancel(v.subs(zero_sub)) for k, v in sol.items()}
            for d in free_dummies:
                sol[d] = Integer(0)

        # Reject solutions containing z
        if any(v.has(z) for v in sol.values()):
            continue

        # Verify substitution works.
        valid = True
        for i in range(rows):
            for j in range(cols):
                diff = cancel(
                    H1[i, j].subs(sol) - H2_fresh[i, j].subs(sol)
                )
                if diff != 0:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        # Map back to original library parameter names for display.
        lib_solved = {}
        user_solved = {}
        for k, v in sol.items():
            if k in reverse_map:
                lib_solved[reverse_map[k]] = v
            else:
                user_solved[k] = v

        # Ensure ALL user params in the equation appear in the mapping.
        for up in user_params_in_eq:
            if up not in user_solved:
                # Check if a lib param equals this user param
                for lp, lv in lib_solved.items():
                    if lv == up:
                        user_solved[up] = lp
                        break
                else:
                    user_solved[up] = up

        result = {'match': True, 'params': lib_solved}
        if user_solved:
            result['user_params'] = user_solved
        return result

    # Step 2: If lib-only solving failed entirely, try solving for all
    # unknowns simultaneously (fallback for cases like q = t).
    if user_params_in_eq:
        all_unknowns = fresh_params + user_params_in_eq
        try:
            all_solutions = solve(equations, all_unknowns, dict=True)
        except Exception:
            all_solutions = []

        # Try each solution, preferring non-trivial ones
        non_trivial = []
        trivial = []
        for sol_candidate in all_solutions:
            if any(v.has(z) for v in sol_candidate.values()):
                continue
            if all(cancel(v) == 0 for v in sol_candidate.values()):
                trivial.append(sol_candidate)
            else:
                non_trivial.append(sol_candidate)

        for sol in non_trivial + trivial:
            sol = dict(sol)
            # Handle unsolved fresh dummies
            solved_set = set(sol.keys())
            free_dummies = fresh_set - solved_set
            for val in sol.values():
                free_dummies |= (val.free_symbols & fresh_set)
            free_dummies -= solved_set
            if free_dummies:
                zero_sub = {d: 0 for d in free_dummies}
                sol = {k: cancel(v.subs(zero_sub)) for k, v in sol.items()}
                for d in free_dummies:
                    sol[d] = Integer(0)

            if any(v.has(z) for v in sol.values()):
                continue

            # Verify
            valid = True
            for i in range(rows):
                for j in range(cols):
                    diff = cancel(
                        H1[i, j].subs(sol) - H2_fresh[i, j].subs(sol)
                    )
                    if diff != 0:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue

            lib_solved = {}
            user_solved = {}
            for k, v in sol.items():
                if k in reverse_map:
                    lib_solved[reverse_map[k]] = v
                else:
                    user_solved[k] = v

            for up in user_params_in_eq:
                if up not in user_solved:
                    for lp, lv in lib_solved.items():
                        if lv == up:
                            user_solved[up] = lp
                            break
                    else:
                        user_solved[up] = up

            result = {'match': True, 'params': lib_solved}
            if user_solved:
                result['user_params'] = user_solved
            return result

    return {'match': False}


def _extract_z_power(expr, z):
    """
    Check if expr equals z^k for some integer k.
    Returns k if so, None otherwise.

    Handles: z^n, 1/z^n, 1, z, etc.
    """
    expr = cancel(expr)

    if expr == 1:
        return 0
    if expr == z:
        return 1

    # Try as polynomial in z
    try:
        p = Poly(expr, z)
        monoms = p.monoms()
        if len(monoms) == 1 and p.nth(*monoms[0]) == 1:
            return monoms[0][0]
    except Exception:
        pass

    # Try reciprocal for negative powers
    try:
        p = Poly(1 / expr, z)
        monoms = p.monoms()
        if len(monoms) == 1 and p.nth(*monoms[0]) == 1:
            return -monoms[0][0]
    except Exception:
        pass

    return None


def check_shift_equivalence(H1, H2, z, lib_params=None):
    """
    Check if H1 = Diag(z^{m_i}) * H2 * Diag(z^{-m_j}), i.e.,
    H1[i,j] = z^{m_i - m_j} * H2[i,j].

    If the TFs contain free symbols, tries candidate shift vectors and
    solves for parameter values that make the shift relationship hold.

    Returns dict with 'match' (bool), 'shift_vector', and optionally 'params'.
    """
    rows, cols = H1.shape
    if H2.shape != (rows, cols) or rows != cols:
        return {'match': False, 'shift_vector': None}

    p = rows

    lib_param_set = set(lib_params) if lib_params else set()

    # Replace library params in H2 with fresh Dummy symbols so that shared
    # names (e.g., user's alpha vs library's alpha) can be solved independently.
    from sympy import Dummy
    fresh_map = {}      # lib_param -> Dummy
    reverse_map = {}    # Dummy -> lib_param
    for lp in lib_param_set:
        d = Dummy('_lib_' + lp.name)
        fresh_map[lp] = d
        reverse_map[d] = lp
    H2_fresh = H2.subs(fresh_map) if fresh_map else H2

    # Collect all free symbols (except z) across both TFs
    all_free = set()
    for i in range(p):
        for j in range(p):
            all_free |= H1[i, j].free_symbols
            all_free |= H2_fresh[i, j].free_symbols
    all_free.discard(z)

    if not all_free:
        # No free parameters — use exact check
        return _check_shift_exact(H1, H2, z, p)

    # Parametric: try candidate shift vectors and solve for all free symbols.
    # For p oracles, m[0]=0 (anchor), m[1],...,m[p-1] range over small integers.
    from sympy import Poly
    MAX_SHIFT = 3

    import itertools
    for shifts in itertools.product(range(-MAX_SHIFT, MAX_SHIFT + 1), repeat=p - 1):
        m = [0] + list(shifts)
        if all(mi == 0 for mi in m):
            continue  # Skip zero shift (that's oracle equivalence)

        # Build equations: H1[i,j] - z^{m_i - m_j} * H2_fresh[i,j] = 0
        equations = []
        for i in range(p):
            for j in range(p):
                shift_power = m[i] - m[j]
                diff = cancel(H1[i, j] - z**shift_power * H2_fresh[i, j])
                if diff == 0:
                    continue
                num = numer(diff)
                try:
                    poly = Poly(num, z)
                    equations.extend(poly.all_coeffs())
                except Exception:
                    equations.append(num)

        if not equations:
            # Trivially satisfied
            return {'match': True, 'shift_vector': m, 'params': {}}

        unknowns = sorted(all_free, key=str)
        try:
            solutions = solve(equations, unknowns, dict=True)
        except Exception:
            continue

        if not solutions:
            continue

        sol = solutions[0]

        # Verify no z in solution
        if any(v.has(z) for v in sol.values()):
            continue

        # Verify substitution works
        ok = True
        for i in range(p):
            for j in range(p):
                shift_power = m[i] - m[j]
                diff = cancel(
                    (H1[i, j] - z**shift_power * H2_fresh[i, j]).subs(sol)
                )
                if diff != 0:
                    ok = False
                    break
            if not ok:
                break

        if ok:
            # Normalize shift vector
            min_m = min(m)
            m = [mi - min_m for mi in m]
            # Map fresh dummies back to original lib param names
            lib_solved = {}
            user_solved = {}
            for k, v in sol.items():
                if k in reverse_map:
                    lib_solved[reverse_map[k]] = v
                else:
                    user_solved[k] = v
            result = {'match': True, 'shift_vector': m, 'params': lib_solved}
            if user_solved:
                result['user_params'] = user_solved
            return result

    return {'match': False, 'shift_vector': None}


def _check_shift_exact(H1, H2, z, p):
    """Exact (non-parametric) shift equivalence check."""
    # Check diagonals match
    for i in range(p):
        diff = cancel(H1[i, i] - H2[i, i])
        if diff != 0:
            return {'match': False, 'shift_vector': None}

    # Check sparsity pattern
    for i in range(p):
        for j in range(p):
            if (cancel(H1[i, j]) == 0) != (cancel(H2[i, j]) == 0):
                return {'match': False, 'shift_vector': None}

    # For nonzero off-diagonal entries, compute ratio and extract z-power.
    # If ratios still contain free parameters (user params), try to solve
    # for parameter values that make each ratio a pure z-power.
    b = {}
    extra_param_equations = []
    free_in_ratios = set()

    for i in range(p):
        for j in range(p):
            if i == j or cancel(H1[i, j]) == 0:
                continue
            ratio = cancel(H1[i, j] / H2[i, j])
            power = _extract_z_power(ratio, z)
            if power is not None:
                b[(i, j)] = power
                continue

            # Ratio has free params — try to find z-power by solving for them.
            # Express ratio as polynomial/polynomial in z, then try z^m for
            # small m: set ratio = z^m and collect coefficient equations.
            ratio_free = ratio.free_symbols - {z}
            if not ratio_free:
                return {'match': False, 'shift_vector': None}
            free_in_ratios |= ratio_free

            found = False
            for m_try in range(-p, p + 1):
                diff = cancel(ratio - z**m_try)
                if diff == 0:
                    b[(i, j)] = m_try
                    found = True
                    break
                num_diff = numer(diff)
                try:
                    poly_diff = Poly(num_diff, z)
                    coeffs = poly_diff.all_coeffs()
                except Exception:
                    coeffs = [num_diff]
                # Check if setting these coefficients to zero is consistent
                try:
                    trial_sol = solve(coeffs, list(ratio_free), dict=True)
                except Exception:
                    continue
                if trial_sol and all(not v.has(z) for v in trial_sol[0].values()):
                    extra_param_equations.extend(coeffs)
                    b[(i, j)] = m_try
                    found = True
                    break
            if not found:
                return {'match': False, 'shift_vector': None}

    # If off-diagonal constraints added equations for user params, solve them
    if extra_param_equations and free_in_ratios:
        try:
            extra_sol = solve(extra_param_equations, list(free_in_ratios), dict=True)
        except Exception:
            return {'match': False, 'shift_vector': None}
        if not extra_sol:
            return {'match': False, 'shift_vector': None}
        user_param_sol = extra_sol[0]
        # Verify no z in solution
        for val in user_param_sol.values():
            if val.has(z):
                return {'match': False, 'shift_vector': None}
        # Add user param values to param_solution and re-substitute
        param_solution.update(user_param_sol)
        # Verify the full substitution works for diagonals too
        H1_sub = H1.subs(user_param_sol)
        H2_sub = H2_resolved.subs(user_param_sol)
        for i in range(p):
            if cancel(H1_sub[i, i] - H2_sub[i, i]) != 0:
                return {'match': False, 'shift_vector': None}

    # Solve for consistent shift vector
    m = [None] * p
    m[0] = 0
    changed = True
    iterations = 0
    while changed and iterations < p * p:
        changed = False
        iterations += 1
        for (i, j), bij in b.items():
            if m[i] is not None and m[j] is None:
                m[j] = m[i] - bij
                changed = True
            elif m[j] is not None and m[i] is None:
                m[i] = m[j] + bij
                changed = True
            elif m[i] is not None and m[j] is not None:
                if m[i] - m[j] != bij:
                    return {'match': False, 'shift_vector': None}

    m = [mi if mi is not None else 0 for mi in m]
    min_m = min(m)
    m = [mi - min_m for mi in m]

    return {'match': True, 'shift_vector': m}


def check_lft_equivalence(H1, H2, M_hat, z, lib_params=None,
                          internal_syms=None):
    """
    Check LFT equivalence: [I | -H1] * M_hat * [H2; I] == 0.

    H1, H2 are p x p transfer function matrices.
    M_hat is a 2p x 2p transformation matrix (may contain internal
    oracle transformation symbols).
    internal_syms: set of symbols from M_hat that are internal (not
    user or library params).  When solved, their values are substituted
    into other params; when free, they are reported separately.

    Returns dict with 'match' (bool) and optionally 'params'.
    """
    p = H1.shape[0]

    # Replace library params in H2 with fresh Dummy symbols so that shared
    # names (e.g., user's alpha vs library's alpha) can be solved independently.
    from sympy import Dummy
    lib_param_set = set(lib_params) if lib_params else set()
    fresh_map = {}      # lib_param -> Dummy
    reverse_map = {}    # Dummy -> lib_param
    for lp in lib_param_set:
        d = Dummy('_lib_' + lp.name)
        fresh_map[lp] = d
        reverse_map[d] = lp
    H2_fresh = H2.subs(fresh_map) if fresh_map else H2

    # Compute LFT product (possibly with parametric solving)
    left = eye(p).row_join(-H1)
    right = H2_fresh.col_join(eye(p))
    product = left * M_hat * right

    # Collect all free symbols except z
    all_free = set()
    for i in range(product.rows):
        for j in range(product.cols):
            all_free |= product[i, j].free_symbols
    all_free.discard(z)

    if not all_free:
        # No free parameters — check if product is zero directly
        for i in range(product.rows):
            for j in range(product.cols):
                entry = cancel(product[i, j])
                if entry != 0:
                    entry = simplify(entry)
                    if entry != 0:
                        return {'match': False}
        return {'match': True}

    # Parametric: extract equations and solve for all free symbols
    from sympy import Poly
    equations = []
    for i in range(product.rows):
        for j in range(product.cols):
            entry = cancel(product[i, j])
            if entry == 0:
                continue
            num = numer(entry)
            try:
                poly = Poly(num, z)
                equations.extend(poly.all_coeffs())
            except Exception:
                equations.append(num)

    if not equations:
        return {'match': True, 'params': {}}

    unknowns = sorted(all_free, key=str)
    try:
        solutions = solve(equations, unknowns, dict=True)
    except Exception:
        return {'match': False}

    if not solutions:
        return {'match': False}

    sol = solutions[0]

    # Verify no z in solution values
    for val in sol.values():
        if val.has(z):
            return {'match': False}

    # Verify substitution makes product zero
    product_sub = product.subs(sol)
    for i in range(product_sub.rows):
        for j in range(product_sub.cols):
            entry = simplify(cancel(product_sub[i, j]))
            if entry != 0:
                return {'match': False}

    # Partition solution into lib params, user params, and internal
    # (oracle transformation) params.  Internal symbols that were solved
    # get substituted into other values; free internal symbols are
    # reported for display.
    internal = set(internal_syms) if internal_syms else set()
    solved_keys = set(sol.keys())

    # Internal symbols that were solved — substitute their values out.
    internal_sub = {k: v for k, v in sol.items() if k in internal}

    # Free internal parameters: internal symbols that the solver left
    # unconstrained.  User/lib params that are unsolved simply express
    # natural constraint relationships (e.g., sigma = 1/tau) and should
    # not be labelled "free".
    free_syms = sorted((internal - solved_keys) & all_free, key=str)

    # Collect display names of all solved non-internal params to detect
    # collisions with free param names.
    solved_display_names = set()
    for k in sol:
        if k in internal:
            continue
        orig = reverse_map.get(k, k)
        solved_display_names.add(orig.name)

    # Rename free params if their name collides with a solved param.
    # Use t_1, t_2, ... (incrementing subscript) to avoid duplicates.
    from sympy import Symbol
    all_used_names = set(solved_display_names)
    rename_map = {}  # old_sym -> new_sym (for substitution in values)
    free_display = []  # symbols with clean names for display
    for fs in free_syms:
        name = fs.name
        if name.startswith('_lib_'):
            name = name[5:]  # strip dummy prefix
        if name not in all_used_names:
            display_sym = Symbol(name)
            all_used_names.add(name)
        else:
            # Find an unused subscript
            idx = 1
            while f'{name}_{idx}' in all_used_names:
                idx += 1
            display_sym = Symbol(f'{name}_{idx}')
            all_used_names.add(display_sym.name)
        if display_sym != fs:
            rename_map[fs] = display_sym
        free_display.append(display_sym)

    # Combined substitution: solved internals + renames for free params
    value_sub = {**internal_sub, **rename_map}

    lib_solved = {}
    user_solved = {}
    for k, v in sol.items():
        if k in internal:
            continue  # don't expose internal symbols as params
        if value_sub:
            v = cancel(v.subs(value_sub))
        if k in reverse_map:
            lib_solved[reverse_map[k]] = v
        else:
            user_solved[k] = v
    result = {'match': True, 'params': lib_solved}
    if user_solved:
        result['user_params'] = user_solved
    if free_display:
        result['free_params'] = free_display
    return result

    # Non-parametric check
    left = eye(p).row_join(-H1)
    right = H2.col_join(eye(p))
    product = left * M_hat * right

    for i in range(product.rows):
        for j in range(product.cols):
            entry = cancel(product[i, j])
            if entry != 0:
                entry = simplify(entry)
                if entry != 0:
                    return {'match': False}

    return {'match': True}


# Oracle relation transformations (2x2 blocks)
# Maps (oracle_from, oracle_to) -> function(t) -> 2x2 Matrix
ORACLE_RELATIONS = {
    ('subgrad_f', 'subgrad_fstar'): lambda t: Matrix([[0, 1], [1, 0]]),
    ('subgrad_fstar', 'subgrad_f'): lambda t: Matrix([[0, 1], [1, 0]]),
    ('grad_f', 'prox_f'): lambda t: Matrix([[0, 1], [1/t, -1/t]]),
    ('prox_f', 'grad_f'): lambda t: Matrix([[0, t], [1, -1]]),
    ('grad_f', 'prox_fstar'): lambda t: Matrix([[t, -t], [0, 1]]),
    ('prox_fstar', 'grad_f'): lambda t: Matrix([[1/t, 1], [0, -1]]),
    ('subgrad_fstar', 'prox_f'): lambda t: Matrix([[1/t, -1/t], [0, 1]]),
    ('prox_f', 'subgrad_fstar'): lambda t: Matrix([[0, t], [1, -1]]),
    ('subgrad_fstar', 'prox_fstar'): lambda t: Matrix([[0, 1], [t, -t]]),
    ('prox_fstar', 'subgrad_fstar'): lambda t: Matrix([[0, 1/t], [1, -1]]),
    ('prox_f', 'prox_fstar'): lambda t: Matrix([[t, 0], [t, -t]]),
    ('prox_fstar', 'prox_f'): lambda t: Matrix([[1/t, 0], [1/t, -1/t]]),
    ('prox_g', 'prox_gstar'): lambda t: Matrix([[t, 0], [t, -t]]),
    ('prox_gstar', 'prox_g'): lambda t: Matrix([[1/t, 0], [1/t, -1/t]]),
}


def build_block_m_hat(oracles_1, oracles_2, params=None):
    """
    Build block-diagonal M_hat from oracle lists in STACKED ordering.

    The stacked ordering groups all oracle inputs (y) first, then all
    oracle outputs (u): (y_0, y_1, ..., u_0, u_1, ...).
    This matches the LFT formula [I | -H] * M * [H; I] = 0.

    oracles_1: list of oracle names for algorithm 1
    oracles_2: list of oracle names for algorithm 2
    params: dict of parameter symbols (e.g., {'t': t_symbol})

    Returns (M_hat, internal_syms) where M_hat is a 2p x 2p Matrix and
    internal_syms is the set of oracle-transformation symbols used
    internally (e.g., the Dummy 't').
    """
    if params is None:
        params = {}

    p = len(oracles_1)
    assert len(oracles_2) == p, "Oracle lists must have same length"

    from sympy import Dummy
    t = params.get('t', Dummy('t'))
    internal_syms = {t}

    blocks = []
    for i in range(p):
        o1 = oracles_1[i]
        o2 = oracles_2[i]
        if o1 == o2:
            blocks.append(eye(2))
        else:
            key = (o1, o2)
            if key not in ORACLE_RELATIONS:
                raise ValueError(
                    f"No known transformation from {o1} to {o2}"
                )
            blocks.append(ORACLE_RELATIONS[key](t))

    if p == 1:
        return blocks[0], internal_syms

    # Build in stacked ordering: [[diag(M_i[0,0]), diag(M_i[0,1])],
    #                              [diag(M_i[1,0]), diag(M_i[1,1])]]
    result = zeros(2 * p, 2 * p)
    for i, block in enumerate(blocks):
        result[i, i] = block[0, 0]          # y_i row, y_i col
        result[i, p + i] = block[0, 1]      # y_i row, u_i col
        result[p + i, i] = block[1, 0]      # u_i row, y_i col
        result[p + i, p + i] = block[1, 1]  # u_i row, u_i col
    return result, internal_syms
