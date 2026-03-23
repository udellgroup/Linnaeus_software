import pytest
from sympy import symbols, Matrix, cancel, zeros, Rational

z = symbols('z')

def test_gradient_descent_tf():
    """Gradient descent: x[k+1] = x[k] - alpha * grad_f(x[k])
    Transfer function should be H(z) = -alpha/(z-1)
    """
    from compute import compute_transfer_function
    alpha = symbols('alpha')
    result = compute_transfer_function(
        state_vars=['x'],
        oracle_inputs=['y1'],
        oracle_outputs=['u1'],
        z_equations=[
            {'x': z - 1, 'u1': alpha, 'y1': 0, 'const': 0},
            {'x': -1, 'u1': 0, 'y1': 1, 'const': 0},
        ],
        z_var=z
    )
    expected = Matrix([[-alpha/(z - 1)]])
    assert cancel(result - expected) == zeros(1)


def test_heavy_ball_tf():
    """Heavy ball: x[k+1] = x[k] - alpha*grad_f(x[k]) + beta*(x[k] - x[k-1])
    Transfer function: H(z) = -alpha*z/((z-1)*(z-beta))
    """
    from compute import compute_transfer_function
    alpha, beta = symbols('alpha beta')
    result = compute_transfer_function(
        state_vars=['x1', 'x2'],
        oracle_inputs=['y1'],
        oracle_outputs=['u1'],
        z_equations=[
            {'x1': z - 1 - beta, 'x2': beta, 'u1': alpha, 'y1': 0, 'const': 0},
            {'x1': -1, 'x2': z, 'u1': 0, 'y1': 0, 'const': 0},
            {'x1': -1, 'x2': 0, 'u1': 0, 'y1': 1, 'const': 0},
        ],
        z_var=z
    )
    expected = Matrix([[-alpha*z/((z - 1)*(z - beta))]])
    diff = cancel(result[0,0] - expected[0,0])
    assert diff == 0


def test_douglas_rachford_tf():
    """Douglas-Rachford with 2 oracles (prox_f, prox_g).
    Should produce a 2x2 transfer matrix.
    """
    from compute import compute_transfer_function
    result = compute_transfer_function(
        state_vars=['x3'],
        oracle_inputs=['y1', 'y2'],
        oracle_outputs=['u1', 'u2'],
        z_equations=[
            {'x3': -1, 'u1': 0, 'u2': 0, 'y1': 1, 'y2': 0, 'const': 0},
            {'x3': 1, 'u1': -2, 'u2': 0, 'y1': 0, 'y2': 1, 'const': 0},
            {'x3': z - 1, 'u1': 1, 'u2': -1, 'y1': 0, 'y2': 0, 'const': 0},
        ],
        z_var=z
    )
    assert result.shape == (2, 2)
    assert cancel(result[0,0] - (-1/(z-1))) == 0
    assert cancel(result[0,1] - 1/(z-1)) == 0
