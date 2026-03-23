import pytest
from sympy import symbols, Matrix, Rational, cancel

z = symbols('z')
alpha, beta, eta = symbols('alpha beta eta')


def test_oracle_equiv_identical():
    from equivalence import check_oracle_equivalence
    H1 = Matrix([[-alpha/(z - 1)]])
    H2 = Matrix([[-alpha/(z - 1)]])
    result = check_oracle_equivalence(H1, H2, z)
    assert result['match'] is True


def test_oracle_equiv_different():
    from equivalence import check_oracle_equivalence
    H1 = Matrix([[-alpha/(z - 1)]])
    H2 = Matrix([[-alpha*z/((z - 1)*(z - beta))]])
    result = check_oracle_equivalence(H1, H2, z)
    assert result['match'] is False


def test_oracle_equiv_parametric():
    from equivalence import check_oracle_equivalence
    H_user = Matrix([[Rational(-1, 10)/(z - 1)]])
    H_lib = Matrix([[-alpha/(z - 1)]])
    result = check_oracle_equivalence(H_user, H_lib, z, lib_params=[alpha])
    assert result['match'] is True
    assert result['params'][alpha] == Rational(1, 10)


def test_shift_equiv_dr_admm():
    from equivalence import check_shift_equivalence
    H_dr = Matrix([[-1/(z-1), 1/(z-1)],
                    [(2*z-1)/(z-1), -1/(z-1)]])
    H_admm = Matrix([[-1/(z-1), z/(z-1)],
                      [(2*z-1)/(z*(z-1)), -1/(z-1)]])
    result = check_shift_equivalence(H_dr, H_admm, z)
    assert result['match'] is True
    assert result['shift_vector'] is not None


def test_shift_equiv_not_matching():
    from equivalence import check_shift_equivalence
    H1 = Matrix([[-1/(z-1), 1/(z-1)],
                  [(2*z-1)/(z-1), -1/(z-1)]])
    H2 = Matrix([[0, 1/z],
                  [-1, 1/z]])
    result = check_shift_equivalence(H1, H2, z)
    assert result['match'] is False


def test_lft_equiv_prox_conjugate():
    from equivalence import check_lft_equivalence
    t = symbols('t')
    H_pg = Matrix([[0, 1/z], [-t, 1/z]])
    H_cpg = Matrix([[-t/(z-1), -t/(z-1)], [-z/(z-1), -1/(z-1)]])
    M_hat = Matrix([[1, 0, 0, 0],
                     [0, t, 0, 0],
                     [0, 0, 1, 0],
                     [0, t, 0, -t]])
    result = check_lft_equivalence(H_pg, H_cpg, M_hat, z)
    assert result['match'] is True
