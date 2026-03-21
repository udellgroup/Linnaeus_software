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


def check_shift_equivalence(H1, H2, z):
    """
    Check if H1 = Diag(z^{m_i}) * H2 * Diag(z^{-m_j}), i.e.,
    H1[i,j] = z^{m_i - m_j} * H2[i,j].

    Returns dict with 'match' (bool) and 'shift_vector' (list of ints or None).
    """
    rows, cols = H1.shape
    if H2.shape != (rows, cols) or rows != cols:
        return {'match': False, 'shift_vector': None}

    p = rows

    # Check diagonals match
    for i in range(p):
        diff = cancel(H1[i, i] - H2[i, i])
        if diff != 0:
            return {'match': False, 'shift_vector': None}

    # Check sparsity pattern matches
    for i in range(p):
        for j in range(p):
            h1_zero = (cancel(H1[i, j]) == 0)
            h2_zero = (cancel(H2[i, j]) == 0)
            if h1_zero != h2_zero:
                return {'match': False, 'shift_vector': None}

    # For nonzero off-diagonal entries, compute ratio and extract z-power
    # b[i][j] should equal m_i - m_j
    b = {}
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if cancel(H1[i, j]) == 0:
                continue
            ratio = cancel(H1[i, j] / H2[i, j])
            power = _extract_z_power(ratio, z)
            if power is None:
                return {'match': False, 'shift_vector': None}
            b[(i, j)] = power

    # Solve for m_i - m_j = b[i,j] consistently
    # Use m[0] = 0 as anchor
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

    # Check all m values were determined (for connected components)
    if any(mi is None for mi in m):
        # For disconnected components, set undetermined to 0
        m = [mi if mi is not None else 0 for mi in m]

    # Normalize so min(m) = 0
    min_m = min(m)
    m = [mi - min_m for mi in m]

    # If all shifts are zero, it's just oracle equivalence, not shift equiv
    # But we still report it as a match
    return {'match': True, 'shift_vector': m}


def check_lft_equivalence(H1, H2, M_hat, z):
    """
    Check LFT equivalence: [I | -H1] * M_hat * [H2; I] == 0.

    H1, H2 are p x p transfer function matrices.
    M_hat is a 2p x 2p transformation matrix.

    Returns dict with 'match' (bool).
    """
    p = H1.shape[0]

    # left = [I | -H1], shape p x 2p
    left = eye(p).row_join(-H1)

    # right = [H2; I], shape 2p x p
    right = H2.col_join(eye(p))

    # product = left * M_hat * right, shape p x p
    product = left * M_hat * right

    # Check all entries are zero
    for i in range(product.rows):
        for j in range(product.cols):
            entry = cancel(product[i, j])
            if entry != 0:
                # Fall back to simplify for more aggressive simplification
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
    Build block-diagonal M_hat from oracle lists.

    oracles_1: list of oracle names for algorithm 1
    oracles_2: list of oracle names for algorithm 2
    params: dict of parameter symbols (e.g., {'t': t_symbol})

    For shared oracles (same name), use 2x2 identity block.
    For differing oracles, look up the transformation in ORACLE_RELATIONS.

    Returns a 2p x 2p block-diagonal Matrix.
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

    # Build block-diagonal matrix
    if p == 1:
        return blocks[0]

    result = zeros(2 * p, 2 * p)
    for i, block in enumerate(blocks):
        for r in range(2):
            for c in range(2):
                result[2*i + r, 2*i + c] = block[r, c]
    return result
