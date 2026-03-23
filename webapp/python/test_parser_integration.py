"""Integration tests: parser output feeds into compute_transfer_function."""
import pytest
from sympy import symbols, cancel, Matrix


alpha, beta, t = symbols('alpha beta t')


def test_gradient_descent_transfer_function():
    from parser import parse_equations
    from compute import compute_transfer_function

    parsed = parse_equations(["x[k+1] = x[k] - alpha * grad_f(x[k])"])
    z = parsed['z_var']
    H = compute_transfer_function(
        parsed['state_vars'], parsed['oracle_inputs'],
        parsed['oracle_outputs'], parsed['z_equations'], z
    )
    # H(z) should be -alpha/(z - 1)
    expected = -alpha / (z - 1)
    assert cancel(H[0, 0] - expected) == 0, f"Got H(z) = {H[0,0]}, expected {expected}"


def test_heavy_ball_transfer_function():
    from parser import parse_equations
    from compute import compute_transfer_function

    parsed = parse_equations([
        "x[k+1] = x[k] - alpha * grad_f(x[k]) + beta * (x[k] - x[k-1])"
    ])
    z = parsed['z_var']
    H = compute_transfer_function(
        parsed['state_vars'], parsed['oracle_inputs'],
        parsed['oracle_outputs'], parsed['z_equations'], z
    )
    # H(z) for heavy ball: -alpha*z / (z^2 - (1+beta)*z + beta)
    expected = -alpha * z / (z**2 - (1 + beta) * z + beta)
    assert cancel(H[0, 0] - expected) == 0, f"Got H(z) = {H[0,0]}, expected {expected}"


def test_variable_named_z():
    """User variable named 'z' should not conflict with z-transform."""
    from parser import parse_equations
    from compute import compute_transfer_function

    parsed = parse_equations([
        "y[k] = x[k] + beta * (x[k] - x[k-1])",
        "z[k] = y[k]",
        "x[k+1] = z[k] - alpha * grad_f(x[k])",
    ])
    ztf = parsed['z_var']
    H = compute_transfer_function(
        parsed['state_vars'], parsed['oracle_inputs'],
        parsed['oracle_outputs'], parsed['z_equations'], ztf
    )
    # Should compute without error
    assert H.shape[0] >= 1
    assert H.shape[1] >= 1
