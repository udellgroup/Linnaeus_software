"""Regenerate tf_latex for all algorithms using polynomial-in-z formatting."""
import sys, json
sys.path.insert(0, '.')
from sympy import symbols, cancel, fraction, Poly, factor, latex, Add, Matrix
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

z = symbols('z')

param_names = [
    'alpha', 'beta', 'gamma', 'delta', 'eta', 'nu',
    'sigma', 'tau', 'theta', 'rho', 's', 't', 'mu', 'l',
]
local_dict = {'z': z, 'Matrix': Matrix}
for name in param_names:
    local_dict[name] = symbols(name)
local_dict['lam'] = symbols('lambda')


def poly_latex(expr, z):
    try:
        p = Poly(expr, z)
    except Exception:
        return latex(expr)
    deg = p.degree()
    if deg < 0:  # Poly(0, z) has degree -oo
        return '0'
    terms = []
    for i in range(deg, -1, -1):
        c = p.nth(i)
        if c == 0:
            continue
        c_factored = factor(c)
        if isinstance(c_factored, Add) and c_factored.could_extract_minus_sign():
            pos_part = factor(-c_factored)
            if isinstance(pos_part, Add):
                c_str = r'-\left(' + latex(pos_part) + r'\right)'
            else:
                c_str = '-' + latex(pos_part) if str(pos_part) != '1' else '-1'
            is_neg = True
        elif isinstance(c_factored, Add):
            c_str = latex(c_factored)
            is_neg = False
        else:
            c_str = latex(c_factored)
            is_neg = False

        if i == 0:
            terms.append(c_str)
        elif i >= 1:
            zs = 'z' if i == 1 else 'z^{' + str(i) + '}'
            if c_factored == 1:
                terms.append(zs)
            elif c_factored == -1:
                terms.append('-' + zs)
            elif is_neg:
                pos = factor(-c_factored)
                pos_str = latex(pos)
                if isinstance(pos, Add):
                    terms.append(r'-\left(' + pos_str + r'\right) ' + zs)
                else:
                    terms.append('-' + pos_str + ' ' + zs)
            else:
                needs_parens = isinstance(c_factored, Add)
                if needs_parens:
                    terms.append(r'\left(' + c_str + r'\right) ' + zs)
                else:
                    terms.append(c_str + ' ' + zs)
    if not terms:
        return '0'
    result = terms[0]
    for t in terms[1:]:
        if t.startswith('-'):
            result += ' ' + t
        else:
            result += ' + ' + t
    return result


def _factored_product_latex(expr):
    """Factor expr and render each factor as a polynomial in z.

    For a product like -2*alpha*(z-1), renders as ``-2 \\alpha (z - 1)``.
    Each factor that is a polynomial of degree >= 1 in z gets poly_latex
    treatment; scalar/constant factors are rendered with latex().
    """
    from sympy import Mul, Pow, Rational, Integer, Number

    expr_f = factor(expr)

    # Decompose into multiplicative factors
    if isinstance(expr_f, Mul):
        raw_factors = list(expr_f.args)
    else:
        raw_factors = [expr_f]

    # Separate: sign, numeric constants, parameter factors, z-polynomial factors
    sign = 1
    numerics = []
    params = []
    z_polys = []

    for f in raw_factors:
        # Handle negative one
        if f == -1:
            sign *= -1
            continue
        # Pow with negative exponent shouldn't appear (we're in num or den)
        # Check if factor involves z
        if f.has(z):
            z_polys.append(f)
        elif isinstance(f, (Integer, Rational, Number)):
            if f < 0:
                sign *= -1
                numerics.append(abs(f))
            else:
                numerics.append(f)
        else:
            params.append(f)

    # Build the LaTeX string
    parts = []

    # Sign
    if sign == -1:
        prefix = '-'
    else:
        prefix = ''

    # Numeric coefficient
    for n in numerics:
        if n != 1:
            parts.append(latex(n))

    # Parameter factors (alpha, beta, etc.)
    # Wrap multi-term expressions in parens (e.g., lambda - 1)
    for p in params:
        p_str = latex(p)
        if isinstance(p, Add):
            p_str = r'\left(' + p_str + r'\right)'
        parts.append(p_str)

    # z-polynomial factors — wrap in parens only when needed for grouping
    # (i.e., there are other factors to multiply with, or it has an exponent)
    total_factors = len(numerics) + len(params) + len(z_polys)
    for zp in z_polys:
        base = zp.base if isinstance(zp, Pow) else zp
        exp = zp.exp if isinstance(zp, Pow) else 1

        # Format the base as polynomial in z
        base_str = poly_latex(base, z)

        # Only wrap in parens if: (a) there are other factors or sign prefix,
        # or (b) the factor has an exponent (e.g., (z+λ-1)²)
        has_multiple_terms = ('+' in base_str or
                              base_str.count('-') > (1 if base_str.startswith('-') else 0))
        needs_parens = has_multiple_terms and (
            total_factors > 1 or sign == -1 or exp != 1
        )

        if needs_parens:
            base_str = r'\left(' + base_str + r'\right)'

        if exp != 1:
            parts.append(base_str + '^{' + latex(exp) + '}')
        else:
            parts.append(base_str)

    if not parts:
        return prefix + '1' if prefix else '1'

    result = prefix + ' '.join(parts)
    return result


def scalar_latex(expr):
    expr = cancel(expr)
    num, den = fraction(expr)
    if den == 1:
        return _factored_product_latex(num)
    return r'\frac{' + _factored_product_latex(num) + '}{' + _factored_product_latex(den) + '}'


def matrix_latex(m):
    entries = []
    for i in range(m.rows):
        row = []
        for j in range(m.cols):
            row.append(scalar_latex(m[i, j]))
        entries.append(row)
    lines = r' \\ '.join(' & '.join(r) for r in entries)
    return r'\begin{bmatrix} ' + lines + r' \end{bmatrix}'


def main():
    with open('../data/algorithms.json') as f:
        data = json.load(f)

    for algo in data:
        tf_str = algo.get('transferFunction')
        if not tf_str:
            continue
        try:
            tf = parse_expr(tf_str, local_dict=local_dict,
                            transformations=standard_transformations)
            if isinstance(tf, Matrix):
                algo['tf_latex'] = matrix_latex(tf)
            else:
                algo['tf_latex'] = scalar_latex(tf)
        except Exception as e:
            print(f"Error for {algo['id']}: {e}")

    with open('../data/algorithms.json', 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Verify a few
    for algo in data:
        print(f"{algo['id']}: {algo['tf_latex'][:120]}")


if __name__ == '__main__':
    main()
