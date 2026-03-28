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


def scalar_latex(expr):
    expr = cancel(expr)
    num, den = fraction(expr)
    if den == 1:
        return poly_latex(num, z)
    return r'\frac{' + poly_latex(num, z) + '}{' + poly_latex(den, z) + '}'


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
