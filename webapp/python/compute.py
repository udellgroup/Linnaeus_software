"""Transfer function computation from z-domain linear equations."""
from sympy import Matrix, cancel, zeros


def compute_transfer_function(state_vars, oracle_inputs, oracle_outputs,
                               z_equations, z_var):
    """Compute H(z) from a system of z-domain linear equations.

    Args:
        state_vars: list of state variable names
        oracle_inputs: list of oracle input names (y1, y2, ...)
        oracle_outputs: list of oracle output names (u1, u2, ...)
        z_equations: list of dicts mapping variable name -> coefficient.
            Each dict: sum(coeff * var) = 0. All var names must be keys.
        z_var: the z Symbol

    Returns:
        SymPy Matrix H(z) of shape (len(oracle_inputs), len(oracle_outputs))
    """
    if not state_vars:
        raise ValueError("No state variables found. Check your equations.")
    if not oracle_outputs or not oracle_inputs:
        raise ValueError("No oracle calls found. Equations must reference at least one oracle.")
    if len(oracle_outputs) != len(oracle_inputs):
        raise ValueError(
            f"Mismatched oracle counts: {len(oracle_outputs)} outputs vs "
            f"{len(oracle_inputs)} inputs."
        )

    all_vars = state_vars + oracle_outputs + oracle_inputs
    n_state = len(state_vars)
    n_oracle = len(oracle_outputs)
    n_input = len(oracle_inputs)
    n_eq = len(z_equations)

    # Build coefficient matrix
    M = zeros(n_eq, len(all_vars))
    for i, eq in enumerate(z_equations):
        for j, var_name in enumerate(all_vars):
            M[i, j] = eq.get(var_name, 0)

    # Rearrange: L * [state; y] = -R * u
    state_cols = list(range(n_state))
    output_cols = list(range(n_state, n_state + n_oracle))
    input_cols = list(range(n_state + n_oracle, n_state + n_oracle + n_input))

    left_cols = state_cols + input_cols
    right_cols = output_cols

    L = M[:, left_cols]
    R = M[:, right_cols]

    if L.rows != L.cols:
        raise ValueError(
            f"System is {'underdetermined' if L.rows < L.cols else 'overdetermined'}: "
            f"{L.rows} equations for {L.cols} unknowns. "
            f"Check that your equations are consistent."
        )

    # Solve: x = L^{-1} * (-R)
    L_inv = L.inv()
    solution = L_inv * (-R)

    # Extract y rows (last n_input rows)
    H = solution[n_state:, :]

    # Simplify
    for i in range(H.rows):
        for j in range(H.cols):
            H[i, j] = cancel(H[i, j])

    return H
