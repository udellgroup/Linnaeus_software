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

    # Solve for fresh_params
    try:
        solution = solve(equations, fresh_params, dict=True)
    except Exception:
        return {'match': False}

    if not solution:
        return {'match': False}

    sol_fresh = solution[0]

    # Verify solution doesn't contain z
    for param, val in sol_fresh.items():
        if val.has(z):
            return {'match': False}

    # Check for unsolved fresh Dummy symbols in the solution values.
    # If the solver returned a parametric solution (e.g., alpha = user_alpha/(gamma+1)
    # where gamma is free), resolve by setting free Dummies to 0.
    fresh_set = set(fresh_params)
    solved_set = set(sol_fresh.keys())
    free_dummies = fresh_set - solved_set

    # Also check if solved values reference other fresh Dummies
    for val in sol_fresh.values():
        free_dummies |= (val.free_symbols & fresh_set)
    free_dummies -= solved_set  # only truly free ones

    if free_dummies:
        # Set free Dummies to 0 and re-substitute
        zero_sub = {d: 0 for d in free_dummies}
        sol_fresh = {k: v.subs(zero_sub) for k, v in sol_fresh.items()}
        # Add the zero-valued params explicitly
        for d in free_dummies:
            sol_fresh[d] = 0

    # Verify substitution works
    for i in range(rows):
        for j in range(cols):
            diff = cancel(H1[i, j] - H2_fresh[i, j].subs(sol_fresh))
            if diff != 0:
                return {'match': False}

    # Map back to original library parameter names for display
    sol = {reverse_map[k]: v for k, v in sol_fresh.items()}
    return {'match': True, 'params': sol}


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

    If lib_params is provided, first solve for library parameter values
    from diagonal constraints, then verify the shift structure.

    Returns dict with 'match' (bool), 'shift_vector', and optionally 'params'.
    """
    rows, cols = H1.shape
    if H2.shape != (rows, cols) or rows != cols:
        return {'match': False, 'shift_vector': None}

    p = rows

    # If library has free parameters, solve for them from diagonal constraints
    param_solution = {}
    H2_resolved = H2
    if lib_params and len(lib_params) > 0:
        from sympy import Dummy, Poly, solve as sym_solve
        fresh_params = []
        sub_map = {}
        reverse_map = {}
        for param in lib_params:
            fresh = Dummy('_lib_' + param.name)
            sub_map[param] = fresh
            reverse_map[fresh] = param
            fresh_params.append(fresh)

        H2_fresh = H2.subs(sub_map)

        # Solve diagonal constraints: H1[i,i] == H2_fresh[i,i]
        equations = []
        for i in range(p):
            diff = cancel(H1[i, i] - H2_fresh[i, i])
            if diff == 0:
                continue
            num = numer(diff)
            try:
                poly = Poly(num, z)
                equations.extend(poly.all_coeffs())
            except Exception:
                equations.append(num)

        if equations:
            try:
                solutions = sym_solve(equations, fresh_params, dict=True)
            except Exception:
                return {'match': False, 'shift_vector': None}
            if not solutions:
                return {'match': False, 'shift_vector': None}
            sol = solutions[0]
            # Verify no z in solution
            for val in sol.values():
                if val.has(z):
                    return {'match': False, 'shift_vector': None}
            # Handle free dummies (set to 0)
            fresh_set = set(fresh_params)
            free_dummies = fresh_set - set(sol.keys())
            for val in sol.values():
                free_dummies |= (val.free_symbols & fresh_set)
            free_dummies -= set(sol.keys())
            if free_dummies:
                zero_sub = {d: 0 for d in free_dummies}
                sol = {k: v.subs(zero_sub) for k, v in sol.items()}
                for d in free_dummies:
                    sol[d] = 0

            H2_resolved = H2_fresh.subs(sol)
            param_solution = {reverse_map[k]: v for k, v in sol.items()}
        else:
            H2_resolved = H2_fresh
            # All diag constraints trivially satisfied; params are free
            for fp in fresh_params:
                param_solution[reverse_map[fp]] = 0
            H2_resolved = H2_fresh.subs({fp: 0 for fp in fresh_params})

    # Check diagonals match (should be true after parametric solve)
    for i in range(p):
        diff = cancel(H1[i, i] - H2_resolved[i, i])
        if diff != 0:
            return {'match': False, 'shift_vector': None}

    # Check sparsity pattern matches
    for i in range(p):
        for j in range(p):
            h1_zero = (cancel(H1[i, j]) == 0)
            h2_zero = (cancel(H2_resolved[i, j]) == 0)
            if h1_zero != h2_zero:
                return {'match': False, 'shift_vector': None}

    # For nonzero off-diagonal entries, compute ratio and extract z-power
    b = {}
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if cancel(H1[i, j]) == 0:
                continue
            ratio = cancel(H1[i, j] / H2_resolved[i, j])
            power = _extract_z_power(ratio, z)
            if power is None:
                return {'match': False, 'shift_vector': None}
            b[(i, j)] = power

    # Solve for m_i - m_j = b[i,j] consistently
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

    if any(mi is None for mi in m):
        m = [mi if mi is not None else 0 for mi in m]

    min_m = min(m)
    m = [mi - min_m for mi in m]

    result = {'match': True, 'shift_vector': m}
    if param_solution:
        result['params'] = param_solution
    return result


def check_lft_equivalence(H1, H2, M_hat, z, lib_params=None):
    """
    Check LFT equivalence: [I | -H1] * M_hat * [H2; I] == 0.

    H1, H2 are p x p transfer function matrices.
    M_hat is a 2p x 2p transformation matrix (may contain symbol 't'
    for the oracle transformation parameter).

    If lib_params is provided, solve for library parameter values (and
    the oracle transformation parameter t) that make the LFT product zero.

    Returns dict with 'match' (bool) and optionally 'params'.
    """
    p = H1.shape[0]

    # Compute LFT product (possibly with parametric solving)
    left = eye(p).row_join(-H1)
    right = H2.col_join(eye(p))
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

    # Separate library params from user/oracle params
    lib_param_set = set(lib_params) if lib_params else set()
    params = {}
    for k, v in sol.items():
        if k in lib_param_set:
            params[k] = v

    return {'match': True, 'params': params}

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

    Returns a 2p x 2p Matrix.
    """
    if params is None:
        params = {}

    p = len(oracles_1)
    assert len(oracles_2) == p, "Oracle lists must have same length"

    t = params.get('t', symbols('t'))

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
        return blocks[0]

    # Build in stacked ordering: [[diag(M_i[0,0]), diag(M_i[0,1])],
    #                              [diag(M_i[1,0]), diag(M_i[1,1])]]
    result = zeros(2 * p, 2 * p)
    for i, block in enumerate(blocks):
        result[i, i] = block[0, 0]          # y_i row, y_i col
        result[i, p + i] = block[0, 1]      # y_i row, u_i col
        result[p + i, i] = block[1, 0]      # u_i row, y_i col
        result[p + i, p + i] = block[1, 1]  # u_i row, u_i col
    return result
