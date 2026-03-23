"""Tests for the equation parser."""
import pytest
from sympy import symbols

z = symbols('z')


def test_parse_gradient_descent():
    from parser import parse_equations
    result = parse_equations([
        "x[k+1] = x[k] - alpha * grad_f(x[k])"
    ])
    assert result['state_vars'] == ['x']
    assert result['oracle_inputs'] == ['y1']
    assert result['oracle_outputs'] == ['u1']
    assert result['oracle_types'] == ['grad_f']
    assert len(result['z_equations']) == 2  # 1 update + 1 oracle input


def test_parse_heavy_ball():
    from parser import parse_equations
    result = parse_equations([
        "x[k+1] = x[k] - alpha * grad_f(x[k]) + beta * (x[k] - x[k-1])"
    ])
    # x needs augmented state: x_curr and x_prev
    assert len(result['state_vars']) == 2
    assert result['oracle_types'] == ['grad_f']


def test_parse_douglas_rachford():
    from parser import parse_equations
    result = parse_equations([
        "x1[k+1] = prox_f(x3[k])",
        "x2[k+1] = prox_g(2 * x1[k+1] - x3[k])",
        "x3[k+1] = x3[k] + x2[k+1] - x1[k+1]",
    ])
    assert 'x3' in result['state_vars']
    assert len(result['oracle_types']) == 2
    assert 'prox_f' in result['oracle_types']
    assert 'prox_g' in result['oracle_types']


def test_parse_proximal_gradient():
    from parser import parse_equations
    result = parse_equations([
        "y[k] = x[k] - t * grad_f(x[k])",
        "x[k+1] = prox_g(y[k])",
    ])
    # y is intermediate but included in state_vars for the linear system solve
    assert 'y' in result['state_vars']
    assert len(result['oracle_types']) == 2


def test_parse_rejects_nonlinear():
    from parser import parse_equations
    with pytest.raises(ValueError, match="[Nn]on.?linear"):
        parse_equations(["x[k+1] = x[k] * x[k]"])


def test_parse_rejects_unknown_oracle():
    from parser import parse_equations
    with pytest.raises(ValueError, match="[Uu]nrecognized oracle"):
        parse_equations(["x[k+1] = gradient_f(x[k])"])


def test_parse_nesterov():
    from parser import parse_equations
    result = parse_equations([
        "y[k] = x[k] + beta * (x[k] - x[k-1])",
        "x[k+1] = y[k] - alpha * grad_f(y[k])",
    ])
    # y is intermediate but included in state_vars for the linear system solve
    # x needs augmented state for x[k-1]
    assert 'y' in result['state_vars']
    assert len(result['state_vars']) == 3  # x, y, x_prev
