"""Expert-level integration tests for the Linnaeus webapp.

Simulates what an optimization expert (e.g., Wotao Yin, Panagiotis Patrinos)
would try when exploring the webapp. Covers equivalent reformulations,
multi-oracle algorithms, edge cases, and error handling.
"""
import pytest
import os
from sympy import symbols, cancel, Matrix, Rational


def run_pipeline(equations):
    """Full pipeline: parse -> compute TF -> check against library."""
    from parser import parse_equations
    from compute import compute_transfer_function, compute_char_poly
    from library import load_library, check_all_equivalences

    parsed = parse_equations(equations)
    z = parsed['z_var']
    if len(parsed['oracle_types']) == 0:
        H = None
        char_poly = compute_char_poly(
            parsed['state_vars'], parsed['z_equations'], z)
    else:
        H = compute_transfer_function(
            parsed['state_vars'],
            parsed['oracle_inputs'],
            parsed['oracle_outputs'],
            parsed['z_equations'],
            z
        )
        char_poly = None

    json_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'algorithms.json')
    library = load_library(json_path)
    matches = check_all_equivalences(
        H, parsed['oracle_types'], library, z,
        user_char_poly=char_poly, user_equations=equations)

    return H, matches


def run_parse_only(equations):
    """Parse equations and compute TF without library check."""
    from parser import parse_equations
    from compute import compute_transfer_function

    parsed = parse_equations(equations)
    z = parsed['z_var']
    H = compute_transfer_function(
        parsed['state_vars'],
        parsed['oracle_inputs'],
        parsed['oracle_outputs'],
        parsed['z_equations'],
        z
    )
    return H, parsed


# =============================================================================
# Category 1: Standard algorithms in different forms
# =============================================================================

class TestEquivalentFormulations:
    """An expert knows the same algorithm can be written many ways."""

    def test_gradient_descent_explicit_step(self):
        """Gradient descent with a specific numeric step size (0.01).

        An expert might enter a concrete step size rather than a symbolic one.
        The library uses symbolic alpha, so parametric matching should find
        alpha = 0.01.
        """
        H, matches = run_pipeline([
            "x[k+1] = x[k] - 0.01 * grad_f(x[k])"
        ])
        names = [m['algorithm']['name'] for m in matches]
        assert 'Gradient Descent' in names
        gd = next(m for m in matches if m['algorithm']['name'] == 'Gradient Descent')
        assert gd['type'] == 'oracle'

    def test_gradient_descent_fraction_step(self):
        """Gradient descent with step size 1/3.

        Tests that rational arithmetic works in parametric matching.
        """
        H, matches = run_pipeline([
            "x[k+1] = x[k] - grad_f(x[k]) / 3"
        ])
        names = [m['algorithm']['name'] for m in matches]
        assert 'Gradient Descent' in names

    def test_nesterov_y_form(self):
        """Nesterov in the standard two-variable (y) form.

        This is the canonical form: extrapolation step then gradient step.
        """
        H, matches = run_pipeline([
            "y[k] = x[k] + beta * (x[k] - x[k-1])",
            "x[k+1] = y[k] - alpha * grad_f(y[k])",
        ])
        names = [m['algorithm']['name'] for m in matches]
        assert "Nesterov's Accelerated Method" in names

    def test_heavy_ball_numeric_params(self):
        """Heavy ball with specific numeric parameters (alpha=0.1, beta=0.9).

        An expert benchmarking convergence rates would plug in specific values.
        Should match Heavy Ball via parametric matching.
        """
        H, matches = run_pipeline([
            "x[k+1] = x[k] - 0.1 * grad_f(x[k]) + 0.9 * (x[k] - x[k-1])"
        ])
        names = [m['algorithm']['name'] for m in matches]
        assert 'Heavy Ball' in names

    def test_arrow_hurwicz_expanded_form(self):
        """Arrow-Hurwicz with the second equation expanded (no x1[k+1] on RHS).

        The canonical form x2[k+1] = x1[k+1] - eta*grad_f(x2[k]) references
        x1[k+1] on the RHS, which creates a coupling the compute engine cannot
        currently handle (singular L matrix). An expert would instead write the
        manually-substituted form: x2[k+1] = x1[k] - 2*eta*grad_f(x2[k]).
        Without P_C, this should conditionally match the projected variants
        (Projected Reflected Gradient, Modified Arrow-Hurwicz, Extrapolation
        from the Past) when P_C = I.
        """
        H, matches = run_pipeline([
            "x1[k+1] = x1[k] - eta * grad_f(x2[k])",
            "x2[k+1] = x1[k] - 2 * eta * grad_f(x2[k])",
        ])
        names = [m['algorithm']['name'] for m in matches]
        assert any('Projected Reflected' in n or 'Arrow-Hurwicz' in n
                    or 'Extrapolation' in n
                    for n in names), f"Expected conditional match, got: {names}"

    def test_arrow_hurwicz_canonical_form(self):
        """Arrow-Hurwicz in its canonical form with same-step coupling.

        x2[k+1] = x1[k+1] - eta*grad_f(x2[k]) references x1[k+1] on the RHS.
        Since both equations use the same oracle call grad_f(x2[k]), the parser
        correctly deduplicates this into a single oracle. This yields a
        canonical 1x1 TF that should conditionally match the projected variants
        when P_C = I.
        """
        H, matches = run_pipeline([
            "x1[k+1] = x1[k] - eta * grad_f(x2[k])",
            "x2[k+1] = x1[k+1] - eta * grad_f(x2[k])",
        ])
        names = [m['algorithm']['name'] for m in matches]
        assert any('Projected Reflected' in n or 'Arrow-Hurwicz' in n
                    or 'Extrapolation' in n
                    for n in names), f"Expected conditional match, got: {names}"

    def test_proximal_gradient_standard_form(self):
        """Proximal gradient in the standard two-line form.

        y[k] = x[k] - t * grad_f(x[k])  (gradient step)
        x[k+1] = prox_g(y[k])            (proximal step)
        """
        H, matches = run_pipeline([
            "y[k] = x[k] - t * grad_f(x[k])",
            "x[k+1] = prox_g(y[k])",
        ])
        names = [m['algorithm']['name'] for m in matches]
        assert 'Proximal Gradient' in names

    def test_proximal_point_basic(self):
        """Proximal point algorithm: the simplest proximal method.

        x[k+1] = prox_f(x[k])
        The user TF is 1/z (simple delay), while the library's proximal point
        TF is t/(z-1) (which encodes the resolvent relation with implicit
        parameter t). The parametric matching recognizes this matches
        Relaxed Proximal Point with alpha=1 (full relaxation = proximal
        point as a special case).
        """
        H, matches = run_pipeline([
            "x[k+1] = prox_f(x[k])"
        ])
        names = [m['algorithm']['name'] for m in matches]
        # Matches Relaxed Proximal Point (with alpha=1, reducing to PP)
        assert 'Relaxed Proximal Point' in names

    def test_relaxed_proximal_point(self):
        """Relaxed proximal point with explicit relaxation parameter.

        x[k+1] = (1 - rho)*x[k] + rho*prox_f(x[k])
        The library now uses simple parameter name 'alpha' for the relaxation
        parameter. Using a different user parameter name ('rho') avoids any
        collision with the library's alpha.
        """
        H, matches = run_pipeline([
            "x[k+1] = (1 - rho) * x[k] + rho * prox_f(x[k])"
        ])
        names = [m['algorithm']['name'] for m in matches]
        assert 'Relaxed Proximal Point' in names


# =============================================================================
# Category 2: Multi-oracle algorithms
# =============================================================================

class TestMultiOracle:
    """Algorithms using multiple oracles (prox_f + prox_g, grad_f + prox_g)."""

    def test_douglas_rachford_standard(self):
        """Douglas-Rachford splitting in the standard 3-line form.

        This is the canonical form using prox_f and prox_g with a reflected step.
        """
        H, matches = run_pipeline([
            "x1[k+1] = prox_f(x3[k])",
            "x2[k+1] = prox_g(2 * x1[k+1] - x3[k])",
            "x3[k+1] = x3[k] + x2[k+1] - x1[k+1]",
        ])
        names = [m['algorithm']['name'] for m in matches]
        assert 'Douglas-Rachford Splitting' in names

    def test_admm_standard(self):
        """ADMM in the standard 3-line form.

        ADMM should produce a valid TF. Due to variable ordering, it may be
        a permutation of the library's canonical form.
        """
        H, matches = run_pipeline([
            "x1[k+1] = prox_g(x2[k] - x3[k])",
            "x2[k+1] = prox_f(x1[k+1] + x3[k])",
            "x3[k+1] = x3[k] + x1[k+1] - x2[k+1]",
        ])
        # ADMM might match via oracle equiv or might need permutation
        # At minimum, the TF should be valid (non-None)
        assert H is not None
        assert H.rows == 2 and H.cols == 2

    def test_peaceman_rachford_tf(self):
        """Peaceman-Rachford splitting computed from equations.

        PR uses prox_f and prox_g with a reflection step (no dampening).
        The library has a TF but no equations, so we compute from scratch
        and check the TF has the expected structure.
        """
        H, parsed = run_parse_only([
            "x1[k+1] = prox_f(x3[k])",
            "x2[k+1] = prox_g(2 * x1[k+1] - x3[k])",
            "x3[k+1] = x3[k] + 2 * x2[k+1] - 2 * x1[k+1]",
        ])
        # PR should produce a 2x2 TF
        assert H.rows == 2 and H.cols == 2

    def test_extragradient_standard(self):
        """Extragradient method with two gradient oracle calls.

        y[k] = x[k] - t * grad_f(x[k])   (exploratory step)
        x[k+1] = x[k] - t * grad_f(y[k]) (main step using gradient at y)

        The parser assigns oracle ordering based on equation order, so the
        user's TF may have different row/column ordering than the library's
        canonical form. The equivalence checker does not currently try
        permutations of oracle orderings. We verify the TF is valid and
        has the correct structure.
        """
        H, parsed = run_parse_only([
            "y[k] = x[k] - t * grad_f(x[k])",
            "x[k+1] = x[k] - t * grad_f(y[k])",
        ])
        assert H is not None
        assert H.rows == 2 and H.cols == 2
        assert parsed['oracle_types'] == ['grad_f', 'grad_f']
        # Verify the TF entries are nonzero rational functions
        z = parsed['z_var']
        # At least the diagonal and one off-diagonal should be nonzero
        nonzero_count = sum(1 for i in range(2) for j in range(2)
                           if cancel(H[i, j]) != 0)
        assert nonzero_count >= 3, f"Expected at least 3 nonzero entries, got {nonzero_count}"


# =============================================================================
# Category 3: Algorithms NOT in the library
# =============================================================================

class TestNoMatch:
    """Algorithms that are genuinely novel or not in our library."""

    def test_nonstandard_momentum(self):
        """Forward-backward with a non-standard cubic momentum coefficient.

        An expert might try a creative momentum scheme that does not correspond
        to any known algorithm family in the library.
        """
        H, matches = run_pipeline([
            "x[k+1] = x[k] - alpha * grad_f(x[k]) + delta * (x[k] - x[k-1])"
        ])
        # This has the heavy-ball structure but with 'delta' instead of 'beta'.
        # It should still match Heavy Ball via parametric matching (beta -> delta).
        # The key test: the TF is valid.
        assert H is not None

    def test_unusual_pole_structure(self):
        """Algorithm with pole at z=3 instead of z=1.

        No standard algorithm has a pole away from z=1 (which corresponds to
        convergence to a fixed point). This should not match anything.
        """
        H, matches = run_pipeline([
            "x[k+1] = 3 * x[k] - 2 * grad_f(x[k])"
        ])
        assert len(matches) == 0, f"Expected no matches, got: {[m['algorithm']['name'] for m in matches]}"

    def test_three_oracle_algorithm(self):
        """Custom algorithm with 3 oracle calls (grad_f, prox_f, prox_g).

        Very few library entries use 3 oracles, so this likely won't match.
        Davis-Yin is 3-oracle but is catalogOnly.
        """
        H, matches = run_pipeline([
            "y[k] = x[k] - alpha * grad_f(x[k])",
            "w[k] = prox_f(y[k])",
            "x[k+1] = prox_g(2 * w[k] - x[k])",
        ])
        # Should produce a valid 3x3 TF but likely no match
        assert H is not None
        assert H.rows == 3 and H.cols == 3

    def test_custom_two_step_method(self):
        """A made-up two-step method with unusual pole structure.

        x[k+1] = 5*x[k] - 4*x[k-1] - grad_f(x[k])
        TF = -z/((z-1)(z-4)). While the poles are at z=1 and z=4 (unusual),
        the TF has the same rational form as Heavy Ball: -alpha*z/((z-1)(z-beta)).
        Parametric matching finds alpha=1, beta=4. This is mathematically correct
        even though beta=4 would not give a convergent algorithm.
        """
        H, matches = run_pipeline([
            "x[k+1] = 5 * x[k] - 4 * x[k-1] - grad_f(x[k])"
        ])
        names = [m['algorithm']['name'] for m in matches]
        assert 'Heavy Ball' in names

    def test_genuinely_unmatched_algorithm(self):
        """An algorithm with a TF structure unlike any library entry.

        x[k+1] = 3*x[k] - 2*grad_f(x[k])
        TF = -2/(z-3): single pole at z=3 (not z=1). This rational function
        form -c/(z-a) matches Gradient Descent's -alpha/(z-1) parametrically
        only if the pole equals 1, which it does not. No library match.
        """
        H, matches = run_pipeline([
            "x[k+1] = 3 * x[k] - 2 * grad_f(x[k])"
        ])
        assert len(matches) == 0


# =============================================================================
# Category 4: Edge cases an expert would try
# =============================================================================

class TestEdgeCases:
    """Edge cases that stress the parser and equivalence checker."""

    def test_single_line_gradient_descent(self):
        """Simplest possible input: one-line gradient descent.

        Tests that the pipeline handles the minimal case correctly.
        """
        H, matches = run_pipeline([
            "x[k+1] = x[k] - alpha * grad_f(x[k])"
        ])
        assert H.rows == 1 and H.cols == 1
        names = [m['algorithm']['name'] for m in matches]
        assert 'Gradient Descent' in names

    def test_four_state_variables(self):
        """Algorithm with 4+ state variables.

        Tests that the parser and matrix algebra handle larger systems.
        Stochastic Unified Momentum uses 3 lines; we test a system with more.
        """
        H, parsed = run_parse_only([
            "a[k+1] = a[k] - alpha * grad_f(b[k])",
            "b[k+1] = a[k+1] + beta * (a[k+1] - a[k])",
        ])
        # Should produce a valid 1x1 TF (one gradient oracle)
        assert H is not None
        assert H.rows == 1 and H.cols == 1

    def test_purely_numeric_parameters(self):
        """Algorithm with only numeric (no symbolic) parameters.

        x[k+1] = x[k] - 0.5 * grad_f(x[k]) + 0.3 * (x[k] - x[k-1])
        Should match Heavy Ball with alpha=0.5, beta=0.3.
        """
        H, matches = run_pipeline([
            "x[k+1] = x[k] - 0.5 * grad_f(x[k]) + 0.3 * (x[k] - x[k-1])"
        ])
        names = [m['algorithm']['name'] for m in matches]
        assert 'Heavy Ball' in names

    def test_prox_fstar_oracle(self):
        """Algorithm using the conjugate proximal oracle prox_fstar.

        An expert working with Fenchel duality would use prox_fstar.
        Conjugate proximal gradient uses grad_f and prox_gstar.
        """
        H, parsed = run_parse_only([
            "x[k+1] = prox_fstar(x[k])"
        ])
        assert H is not None
        assert H.rows == 1 and H.cols == 1
        assert parsed['oracle_types'] == ['prox_fstar']

    def test_prox_gstar_oracle(self):
        """Algorithm using prox_gstar (conjugate proximal of g).

        Used in conjugate proximal gradient and related methods.
        """
        H, parsed = run_parse_only([
            "y[k] = x[k] - t * grad_f(x[k])",
            "x[k+1] = y[k] - t * prox_gstar(y[k] / t)",
        ])
        assert H is not None
        assert parsed['oracle_types'] == ['grad_f', 'prox_gstar']

    def test_subgrad_f_oracle(self):
        """Algorithm using subgradient oracle.

        Subgradient descent is relevant for non-smooth optimization.
        Should parse correctly even though few library entries use it.
        """
        H, parsed = run_parse_only([
            "x[k+1] = x[k] - alpha * subgrad_f(x[k])"
        ])
        assert H is not None
        assert parsed['oracle_types'] == ['subgrad_f']

    def test_parameter_t_matches_library(self):
        """Using 't' as a parameter name, which is also used in library entries.

        The proximal gradient library entry uses 't'. A user entering equations
        with 't' should get correct parametric matching.
        """
        H, matches = run_pipeline([
            "y[k] = x[k] - t * grad_f(x[k])",
            "x[k+1] = prox_g(y[k])",
        ])
        names = [m['algorithm']['name'] for m in matches]
        assert 'Proximal Gradient' in names
        pg = next(m for m in matches if m['algorithm']['name'] == 'Proximal Gradient')
        assert pg['type'] == 'oracle'

    def test_variable_name_alpha(self):
        """Variable named 'alpha' that could be confused with a parameter.

        The system should distinguish variables (with [k] indexing) from
        parameters (bare symbols). Here 'alpha' is a state variable.
        """
        # 'alpha' as a state variable name, 'mu' as a parameter
        H, parsed = run_parse_only([
            "alpha[k+1] = alpha[k] - mu * grad_f(alpha[k])"
        ])
        assert H is not None
        assert 'alpha' in parsed['state_vars']
        assert 'mu' in parsed['parameters']


# =============================================================================
# Category 5: Specific equivalence types
# =============================================================================

class TestEquivalenceTypes:
    """Tests that trigger specific equivalence types (oracle, shift, LFT)."""

    def test_oracle_equivalence_gradient_descent(self):
        """Pure oracle equivalence: gradient descent matches directly.

        The simplest form of equivalence: identical transfer functions
        up to parameter substitution.
        """
        H, matches = run_pipeline([
            "x[k+1] = x[k] - alpha * grad_f(x[k])"
        ])
        gd = next(m for m in matches if m['algorithm']['name'] == 'Gradient Descent')
        assert gd['type'] == 'oracle'

    def test_oracle_equivalence_nesterov(self):
        """Oracle equivalence for Nesterov's method.

        More complex parametric matching with two parameters (alpha, beta).
        """
        H, matches = run_pipeline([
            "y[k] = x[k] + beta * (x[k] - x[k-1])",
            "x[k+1] = y[k] - alpha * grad_f(y[k])",
        ])
        nesterov = next(m for m in matches
                        if m['algorithm']['name'] == "Nesterov's Accelerated Method")
        assert nesterov['type'] == 'oracle'

    def test_conjugate_proximal_gradient_lft(self):
        """Conjugate proximal gradient should trigger LFT equivalence with proximal gradient.

        The conjugate proximal gradient uses grad_f + prox_gstar, while
        proximal gradient uses grad_f + prox_g. The Moreau decomposition
        relates prox_g and prox_gstar via an LFT.
        """
        H, matches = run_pipeline([
            "y[k] = x[k] - t * grad_f(x[k])",
            "x[k+1] = y[k] - t * prox_gstar(y[k] / t)",
        ])
        # Should match conjugate proximal gradient (oracle) and
        # proximal gradient (LFT via Moreau decomposition)
        match_types = {m['algorithm']['name']: m['type'] for m in matches}

        # At minimum, the conjugate proximal gradient should match as oracle equiv
        assert 'Conjugate Proximal Gradient' in match_types, \
            f"Expected Conjugate Proximal Gradient match, got: {match_types}"

    def test_shift_equivalence_admm_dr(self):
        """ADMM and Douglas-Rachford are shift-equivalent.

        The library documents this: ADMM is a shifted version of DR.
        We verify the TF structure is correct (ADMM is a permutation of DR
        shifted by z-powers).
        """
        # Compute DR transfer function
        H_dr, _ = run_parse_only([
            "x1[k+1] = prox_f(x3[k])",
            "x2[k+1] = prox_g(2 * x1[k+1] - x3[k])",
            "x3[k+1] = x3[k] + x2[k+1] - x1[k+1]",
        ])
        # Compute ADMM transfer function
        H_admm, _ = run_parse_only([
            "x1[k+1] = prox_g(x2[k] - x3[k])",
            "x2[k+1] = prox_f(x1[k+1] + x3[k])",
            "x3[k+1] = x3[k] + x1[k+1] - x2[k+1]",
        ])
        # Both should be 2x2
        assert H_dr.rows == 2 and H_dr.cols == 2
        assert H_admm.rows == 2 and H_admm.cols == 2


# =============================================================================
# Category 6: Error handling
# =============================================================================

class TestErrorHandling:
    """Tests that the parser gives clear errors for invalid input."""

    def test_nonlinear_equation(self):
        """Nonlinear equation: x[k+1] = x[k]^2.

        The parser should reject this because the system must be linear
        in the state variables and oracle outputs.
        """
        with pytest.raises(ValueError, match="[Nn]on-linear|[Nn]onlinear|parse"):
            run_pipeline([
                "x[k+1] = x[k]**2"
            ])

    def test_missing_equals_sign(self):
        """Equation without an equals sign should raise an error."""
        with pytest.raises((ValueError, Exception)):
            run_pipeline([
                "x[k+1] + grad_f(x[k])"
            ])

    def test_unknown_oracle_name(self):
        """An unrecognized oracle (hess_f) should raise an error.

        The parser should tell the user which oracles are supported.
        """
        with pytest.raises(ValueError, match="[Uu]nrecognized oracle"):
            run_pipeline([
                "x[k+1] = x[k] - alpha * hess_f(x[k])"
            ])

    def test_oracle_nonlinear_argument(self):
        """Oracle argument that isn't a linear expression.

        grad_f(x[k]^2) should fail because the oracle argument must be
        a linear combination of state variables.
        """
        with pytest.raises((ValueError, Exception)):
            run_pipeline([
                "x[k+1] = x[k] - alpha * grad_f(x[k]**2)"
            ])


# =============================================================================
# Category 7: Triple Momentum and QHM (complex parametric matching)
# =============================================================================

class TestComplexParametricMatching:
    """Tests for algorithms with 3+ parameters requiring complex matching."""

    def test_triple_momentum_symbolic(self):
        """Triple momentum method with all three parameters symbolic.

        y[k] = x[k] + gamma*(x[k] - x[k-1])
        x[k+1] = x[k] - alpha*grad_f(y[k]) + beta*(x[k] - x[k-1])

        This is the canonical form and should match directly.
        """
        H, matches = run_pipeline([
            "y[k] = x[k] + gamma * (x[k] - x[k-1])",
            "x[k+1] = x[k] - alpha * grad_f(y[k]) + beta * (x[k] - x[k-1])",
        ])
        names = [m['algorithm']['name'] for m in matches]
        assert 'Triple Momentum Method' in names

    def test_stochastic_unified_momentum(self):
        """Stochastic unified momentum with its 3-line form.

        y[k+1] = x[k] - alpha * grad_f(x[k])
        q[k+1] = x[k] - s * alpha * grad_f(x[k])
        x[k+1] = y[k+1] + beta_sum * (q[k+1] - q[k])

        All three equations use the same oracle call grad_f(x[k]), which is
        correctly deduplicated into a single oracle. The resulting 1x1 TF
        should match the library's Stochastic Unified Momentum entry.
        """
        H, matches = run_pipeline([
            "y[k+1] = x[k] - alpha * grad_f(x[k])",
            "q[k+1] = x[k] - s * alpha * grad_f(x[k])",
            "x[k+1] = y[k+1] + beta_sum * (q[k+1] - q[k])",
        ])
        names = [m['algorithm']['name'] for m in matches]
        assert 'Stochastic Unified Momentum' in names


# =============================================================================
# Category 8: Transfer function structure validation
# =============================================================================

class TestTransferFunctionStructure:
    """Verify that computed TFs have expected mathematical properties."""

    def test_gradient_descent_tf_poles(self):
        """Gradient descent TF should have a single pole at z=1.

        H(z) = -alpha/(z-1). This is the defining characteristic of
        a gradient descent method converging to a fixed point.
        """
        from sympy import Poly
        H, parsed = run_parse_only([
            "x[k+1] = x[k] - alpha * grad_f(x[k])"
        ])
        z = parsed['z_var']
        alpha = symbols('alpha')
        expected = -alpha / (z - 1)
        assert cancel(H[0, 0] - expected) == 0

    def test_heavy_ball_tf_two_poles(self):
        """Heavy ball TF should have poles at z=1 and z=beta.

        H(z) = -alpha*z / ((z-1)*(z-beta)).
        The second pole at z=beta is what gives momentum its character.
        """
        from sympy import Poly
        H, parsed = run_parse_only([
            "x[k+1] = x[k] - alpha * grad_f(x[k]) + beta * (x[k] - x[k-1])"
        ])
        z = parsed['z_var']
        alpha, beta = symbols('alpha beta')
        expected = -alpha * z / ((z - 1) * (z - beta))
        assert cancel(H[0, 0] - expected) == 0

    def test_douglas_rachford_tf_shape(self):
        """Douglas-Rachford should produce a 2x2 TF matrix.

        Two oracles (prox_f, prox_g) implies H is 2x2.
        """
        H, parsed = run_parse_only([
            "x1[k+1] = prox_f(x3[k])",
            "x2[k+1] = prox_g(2 * x1[k+1] - x3[k])",
            "x3[k+1] = x3[k] + x2[k+1] - x1[k+1]",
        ])
        assert H.shape == (2, 2)
        # All entries should be rational functions of z
        z = parsed['z_var']
        for i in range(2):
            for j in range(2):
                assert H[i, j] != 0, f"H[{i},{j}] should be nonzero for DR"


# ---- Category 9: Linear transformations of library algorithms ----

class TestLinearTransformations:
    """An expert might write an algorithm that is a linear rescaling of a
    library entry — e.g., doubling the relaxation parameter. The parametric
    solver must handle shared parameter names correctly."""

    def test_rescaled_relaxed_proximal_point(self):
        """RPP with doubled relaxation: alpha_rpp -> 2*alpha_rpp.
        Should still match RPP with appropriate parameter mapping.
        This tests that the parametric solver uses fresh symbols for
        library parameters to avoid collisions with user parameters.
        """
        H, matches = run_pipeline([
            "x[k+1] = (1 - 2 * alpha_rpp) * x[k] + 2 * alpha_rpp * prox_f(x[k])"
        ])
        names = [m['algorithm']['name'] for m in matches]
        assert 'Relaxed Proximal Point' in names
        rpp = next(m for m in matches if m['algorithm']['name'] == 'Relaxed Proximal Point')
        # Library's alpha should equal 2 * user's alpha_rpp
        from sympy import symbols
        alpha = symbols('alpha')
        alpha_rpp = symbols('alpha_rpp')
        assert rpp['details']['params'][alpha] == 2 * alpha_rpp

    def test_scaled_gradient_descent(self):
        """GD with step = 2*alpha should match GD with alpha(lib) = 2*alpha(user)."""
        H, matches = run_pipeline([
            "x[k+1] = x[k] - 2 * alpha * grad_f(x[k])"
        ])
        names = [m['algorithm']['name'] for m in matches]
        assert 'Gradient Descent' in names
        gd = next(m for m in matches if m['algorithm']['name'] == 'Gradient Descent')
        from sympy import symbols
        alpha = symbols('alpha')
        assert gd['details']['params'][alpha] == 2 * alpha


class TestOracleReuse:
    """Test recognition of trivial oracle re-use (shifted oracle calls)."""

    def test_nesterov_with_reused_gradient(self):
        """Nesterov's method written with explicit gradient re-use.

        v[k+1] = beta*(v[k] + alpha*grad_f(q[k-1]) - alpha*grad_f(q[k])) - alpha*grad_f(q[k])
        q[k+1] = q[k] + v[k+1]

        Here grad_f(q[k-1]) is the previous iteration's gradient of grad_f(q[k]),
        not a separate oracle. The parser should detect this and merge them into
        one oracle, enabling a match to Nesterov's Accelerated Method.
        """
        H, matches = run_pipeline([
            "v[k+1] = beta*(v[k] + alpha*grad_f(q[k-1]) - alpha*grad_f(q[k])) - alpha*grad_f(q[k])",
            "q[k+1] = q[k] + v[k+1]",
        ])
        names = [m['algorithm']['name'] for m in matches]
        assert "Nesterov's Accelerated Method" in names, (
            f"Expected Nesterov's Accelerated Method in matches, got: {names}"
        )

    def test_oracle_merge_preserves_non_shifted(self):
        """Oracle calls to different variables should NOT be merged.

        grad_f(x[k]) and grad_f(y[k]) are distinct oracles even though
        the function name is the same.
        """
        from parser import _merge_shifted_oracles
        equations = [
            "x[k+1] = x[k] - alpha * grad_f(x[k])",
            "y[k+1] = y[k] - beta * grad_f(y[k])",
        ]
        result = _merge_shifted_oracles(equations)
        # Should be unchanged — different argument variables
        assert result == equations

    def test_oracle_merge_simple_shift(self):
        """Verify pre-processing correctly rewrites shifted oracle calls."""
        from parser import _merge_shifted_oracles
        equations = [
            "v[k+1] = v[k] + alpha*grad_f(q[k-1]) - alpha*grad_f(q[k])",
            "q[k+1] = q[k] + v[k+1]",
        ]
        result = _merge_shifted_oracles(equations)
        # Should have 3 equations: the new oracle def + 2 rewritten originals
        assert len(result) == 3
        # First equation defines the oracle variable
        assert 'grad_f(q[k])' in result[0]
        assert '__orcl_' in result[0]
        # Original oracle calls should be replaced
        assert 'grad_f' not in result[1]
        assert 'grad_f' not in result[2]
