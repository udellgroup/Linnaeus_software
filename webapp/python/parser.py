"""Parse user-entered iterative equations into z-domain format for compute.py."""
import re
from sympy import symbols, expand
from sympy.parsing.sympy_parser import parse_expr

# Known oracle types
KNOWN_ORACLES = {'grad_f', 'prox_f', 'prox_g', 'prox_fstar', 'prox_gstar', 'subgrad_f'}

# Regex patterns
VAR_REF_PATTERN = re.compile(r'(\w+)\[k([+-]\d+)?\]')
ORACLE_CALL_PATTERN = re.compile(r'(\w+)\(([^)]+)\)')


def _find_oracle_calls(eq_str):
    """Find known oracle calls in eq_str, handling arbitrarily nested parentheses.

    Returns list of (func_name, arg_str, orig_text).
    """
    results = []
    for oracle in KNOWN_ORACLES:
        for m in re.finditer(r'\b' + oracle + r'\(', eq_str):
            paren_start = m.end() - 1
            depth = 0
            j = paren_start
            while j < len(eq_str):
                if eq_str[j] == '(':
                    depth += 1
                elif eq_str[j] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if depth == 0:
                orig_text = eq_str[m.start():j + 1]
                arg_str = eq_str[paren_start + 1:j].strip()
                results.append((oracle, arg_str, orig_text))
    return results


MAX_EQUATIONS = 20  # Guard against pathological input

# Pattern for a simple variable reference as a full match (for oracle arg parsing)
SIMPLE_VAR_REF = re.compile(r'(\w+)\[k([+-]\d+)?\]$')


def _merge_shifted_oracles(equations):
    """Pre-process equations to merge shifted oracle calls into single oracles.

    Detects when the same oracle function is called at different time shifts of
    the same variable (e.g., grad_f(q[k]) and grad_f(q[k-1])) and rewrites the
    equations to use a single oracle with shifted output references.

    For example:
        grad_f(q[k-1]) and grad_f(q[k]) in the same system
    become:
        __orcl_1[k] = grad_f(q[k])   (new equation)
        ...with grad_f(q[k-1]) replaced by __orcl_1[k-1] in original equations
    """
    # Collect all oracle calls with parsed arguments
    # Key: (func_name, var_name) -> list of (offset, original_text)
    oracle_groups = {}

    for eq_str in equations:
        for func_name, arg_str, orig_text in _find_oracle_calls(eq_str):

            # Only merge when argument is a simple variable reference
            var_match = SIMPLE_VAR_REF.match(arg_str)
            if var_match:
                var_name = var_match.group(1)
                offset = int(var_match.group(2)) if var_match.group(2) else 0
                key = (func_name, var_name)
                if key not in oracle_groups:
                    oracle_groups[key] = []
                # Avoid duplicate entries for same original text
                if not any(o[1] == orig_text for o in oracle_groups[key]):
                    oracle_groups[key].append((offset, orig_text))

    # Find groups with multiple distinct offsets (shifted oracle calls)
    merges = {}  # original_text -> replacement_text
    extra_equations = []
    merge_counter = 0

    for (func_name, var_name), calls in oracle_groups.items():
        offsets = set(c[0] for c in calls)
        if len(offsets) <= 1:
            continue  # All calls at same offset, nothing to merge

        merge_counter += 1
        orcl_var = f'__orcl_{merge_counter}'

        # Base offset: prefer 0, otherwise use the maximum
        base_offset = 0 if 0 in offsets else max(offsets)

        # Format offset string for variable references
        def _offset_str(off):
            if off == 0:
                return '[k]'
            return f'[k{off:+d}]'

        # Add equation defining the oracle variable at base offset
        base_arg = f'{var_name}{_offset_str(base_offset)}'
        extra_equations.append(
            f'{orcl_var}{_offset_str(base_offset)} = {func_name}({base_arg})'
        )

        # Build replacement map for each call
        for offset, orig_text in calls:
            merges[orig_text] = f'{orcl_var}{_offset_str(offset)}'

    if not merges:
        return equations

    # Rewrite original equations (NOT the extra equations)
    new_equations = []
    for eq_str in equations:
        for orig, repl in merges.items():
            eq_str = eq_str.replace(orig, repl)
        new_equations.append(eq_str)

    return extra_equations + new_equations


def parse_equations(equations):
    """Parse iterative equations into z-domain format for compute_transfer_function.

    Args:
        equations: list of strings like "x[k+1] = x[k] - alpha * grad_f(x[k])"

    Returns:
        dict with keys: state_vars, oracle_inputs, oracle_outputs,
        oracle_types, z_equations, parameters
    """
    # Normalize 'lambda' -> 'lam': 'lambda' is a Python keyword and breaks
    # parse_expr; 'lam' is the canonical internal name (mapped to Symbol('lambda')
    # for correct LaTeX rendering, consistent with library.py).
    equations = [re.sub(r'\blambda\b', 'lam', eq) for eq in equations]

    # Pre-process: merge shifted oracle calls into single oracles
    equations = _merge_shifted_oracles(equations)

    if not equations:
        raise ValueError("No equations provided. Enter at least one update equation.")
    if len(equations) > MAX_EQUATIONS:
        raise ValueError(
            f"Too many equations ({len(equations)}). "
            f"Maximum supported is {MAX_EQUATIONS}."
        )
    for i, eq in enumerate(equations):
        if '=' not in eq:
            raise ValueError(
                f"Equation {i + 1} is missing '=': \"{eq}\""
            )
        parts = eq.split('=')
        if len(parts) < 2 or not parts[1].strip():
            raise ValueError(
                f"Equation {i + 1} has an empty right-hand side: \"{eq}\""
            )

    # Use an internal name for the z-transform variable to avoid conflicts
    # with user variables named 'z'.
    z = symbols('__ztf')

    # First pass: collect all variable references and oracle calls
    all_var_refs = {}  # var_name -> set of offsets
    oracle_calls = []  # list of (oracle_type, argument_str, original_text)
    oracle_calls_seen = set()  # track original_text to deduplicate
    parameters = set()

    # Check for unknown oracle-like calls (word_word(...) pattern)
    for eq_str in equations:
        for match in ORACLE_CALL_PATTERN.finditer(eq_str):
            func_name = match.group(1)
            # Skip if it's a known oracle
            if func_name in KNOWN_ORACLES:
                continue
            # Check if it looks like an oracle call (contains underscore
            # and the argument contains [k...])
            arg = match.group(2)
            if '_' in func_name and VAR_REF_PATTERN.search(arg):
                raise ValueError(
                    f"Unrecognized oracle '{func_name}'. "
                    f"Known oracles: {sorted(KNOWN_ORACLES)}"
                )

    for eq_str in equations:
        # Extract variable references
        for match in VAR_REF_PATTERN.finditer(eq_str):
            var_name = match.group(1)
            offset_str = match.group(2)
            offset = int(offset_str) if offset_str else 0
            if var_name not in all_var_refs:
                all_var_refs[var_name] = set()
            all_var_refs[var_name].add(offset)

        # Extract oracle calls (deduplicate by original text, since the same
        # oracle called on the same argument in multiple equations is one oracle)
        for func_name, arg_str, orig_text in _find_oracle_calls(eq_str):
            if orig_text not in oracle_calls_seen:
                oracle_calls.append((func_name, arg_str, orig_text))
                oracle_calls_seen.add(orig_text)

    # Classify variables
    # State: appears at [k+1] (LHS target) or at multiple offsets
    # Intermediate: appears only at [k] and never at [k+1] on LHS
    lhs_vars = set()
    for eq_str in equations:
        lhs = eq_str.split('=')[0].strip()
        m = VAR_REF_PATTERN.search(lhs)
        if m:
            lhs_vars.add(m.group(1))

    state_vars = []
    intermediate_vars = set()
    needs_prev = {}  # var_name -> True if [k-1] appears

    for var_name, offsets in all_var_refs.items():
        if 1 in offsets and (0 in offsets or -1 in offsets):
            # Appears at [k+1] and at [k] or [k-1] => state variable
            state_vars.append(var_name)
        elif 1 in offsets and len(offsets) == 1:
            # Only appears at [k+1], e.g. as intermediate that's used
            # in the same step. Check if it's used elsewhere.
            # If it appears on LHS at [k+1] and on RHS elsewhere at [k+1],
            # it's intermediate. If only on LHS, also intermediate.
            intermediate_vars.add(var_name)
        elif 0 in offsets and -1 not in offsets and 1 not in offsets:
            # Only at [k], likely intermediate
            intermediate_vars.add(var_name)
        elif 0 in offsets and -1 in offsets and 1 not in offsets:
            # e.g. y[k] = ... x[k-1] ... - intermediate with lookback
            intermediate_vars.add(var_name)
        else:
            state_vars.append(var_name)

        if -1 in offsets:
            needs_prev[var_name] = True

    # Sort state_vars for determinism
    state_vars.sort()

    # Include intermediate variables in augmented_state so they get proper
    # columns in the linear system (they'll be eliminated by the matrix solve)
    augmented_state = list(state_vars) + sorted(intermediate_vars)
    prev_map = {}  # var_name -> augmented_prev_name
    for var_name in sorted(needs_prev.keys()):
        if var_name in state_vars:
            prev_name = f'{var_name}_prev'
            augmented_state.append(prev_name)
            prev_map[var_name] = prev_name

    # Create oracle variable mappings
    oracle_types = []
    oracle_inputs = []
    oracle_outputs = []
    oracle_map = {}  # original_text -> (u_name, y_name)

    for i, (otype, arg_str, orig_text) in enumerate(oracle_calls, 1):
        u_name = f'u{i}'
        y_name = f'y{i}'
        oracle_types.append(otype)
        oracle_inputs.append(y_name)
        oracle_outputs.append(u_name)
        oracle_map[orig_text] = (u_name, y_name, arg_str)

    # Build symbol dict for parsing.
    # Map the internal z-transform symbol to 'z' in parsing context so that
    # z*var works correctly, but user variables named 'z' get their own symbol.
    all_syms = {}
    # Do NOT put z-transform in all_syms['z'] — reserve that for user vars.
    # Instead, use a temporary name during parsing and substitute later.
    _ztf_str = '__ztf'
    all_syms[_ztf_str] = z
    for var in augmented_state:
        all_syms[var] = symbols(var)
    for y in oracle_inputs:
        all_syms[y] = symbols(y)
    for u in oracle_outputs:
        all_syms[u] = symbols(u)

    # Collect all parameter names (symbols that aren't variables or z-transform)
    # We'll find them by scanning equations for non-variable, non-oracle tokens
    known_names = set(augmented_state) | set(oracle_inputs) | set(oracle_outputs)
    known_names.add('__ztf')
    known_names.update(KNOWN_ORACLES)

    for eq_str in equations:
        # Remove oracle calls temporarily
        cleaned = eq_str
        for orig_text in oracle_map:
            cleaned = cleaned.replace(orig_text, 'ORACLE_PLACEHOLDER')
        # Remove variable references
        cleaned = VAR_REF_PATTERN.sub('VAR_PLACEHOLDER', cleaned)
        # Find remaining identifiers
        tokens = re.findall(r'[a-zA-Z_]\w*', cleaned)
        for tok in tokens:
            if tok not in {'VAR_PLACEHOLDER', 'ORACLE_PLACEHOLDER', 'k'}:
                parameters.add(tok)
                all_syms[tok] = symbols('lambda' if tok == 'lam' else tok)

    # Also scan oracle arguments for parameters (e.g., tau in prox_f(x - tau*y))
    for orig_text, (u_name, y_name, arg_str) in oracle_map.items():
        cleaned_arg = VAR_REF_PATTERN.sub('VAR_PLACEHOLDER', arg_str)
        # Remove any nested oracle references (already replaced by oracle merging)
        for ot in oracle_map:
            cleaned_arg = cleaned_arg.replace(ot, 'ORACLE_PLACEHOLDER')
        tokens = re.findall(r'[a-zA-Z_]\w*', cleaned_arg)
        for tok in tokens:
            if (tok not in {'VAR_PLACEHOLDER', 'ORACLE_PLACEHOLDER', 'k'}
                    and tok not in known_names):
                parameters.add(tok)
                all_syms[tok] = symbols('lambda' if tok == 'lam' else tok)

    # Now process each equation into z-domain
    z_equations = []

    for eq_str in equations:
        lhs_str, rhs_str = eq_str.split('=', 1)
        lhs_str = lhs_str.strip()
        rhs_str = rhs_str.strip()

        # Convert both sides to z-domain symbolic expressions
        lhs_expr = _to_z_domain(lhs_str, z, augmented_state, prev_map,
                                intermediate_vars, oracle_map, all_syms)
        rhs_expr = _to_z_domain(rhs_str, z, augmented_state, prev_map,
                                intermediate_vars, oracle_map, all_syms)

        # Equation: lhs - rhs = 0
        eq_expr = expand(lhs_expr - rhs_expr)

        # Extract coefficients for all variables
        all_var_names = augmented_state + oracle_outputs + oracle_inputs
        coeff_dict = _extract_coefficients(eq_expr, all_var_names, all_syms)

        z_equations.append(coeff_dict)

    # Add oracle input equations: y_i - argument = 0
    for orig_text, (u_name, y_name, arg_str) in oracle_map.items():
        arg_expr = _to_z_domain(arg_str, z, augmented_state, prev_map,
                                intermediate_vars, oracle_map, all_syms)
        y_sym = all_syms[y_name]
        eq_expr = expand(y_sym - arg_expr)

        all_var_names = augmented_state + oracle_outputs + oracle_inputs
        coeff_dict = _extract_coefficients(eq_expr, all_var_names, all_syms)
        z_equations.append(coeff_dict)

    # Add augmented state equations: __ztf*x_prev = x  =>  __ztf*x_prev - x = 0
    for var_name, prev_name in prev_map.items():
        all_var_names = augmented_state + oracle_outputs + oracle_inputs
        coeff_dict = {v: 0 for v in all_var_names}
        coeff_dict[prev_name] = z  # z here is the __ztf symbol
        coeff_dict[var_name] = -1
        coeff_dict['const'] = 0
        z_equations.append(coeff_dict)

    # The z_equations use the internal __ztf symbol. The caller (compute.py)
    # needs to receive this same symbol as z_var.

    # Include all variables (including intermediates) in state_vars.
    # The matrix solve in compute.py will naturally eliminate intermediates.
    return {
        'state_vars': augmented_state,
        'oracle_inputs': oracle_inputs,
        'oracle_outputs': oracle_outputs,
        'oracle_types': oracle_types,
        'z_equations': z_equations,
        'parameters': sorted(parameters),
        'z_var': z,  # the internal z-transform symbol (may differ from Symbol('z'))
    }


def _to_z_domain(expr_str, z, augmented_state, prev_map, intermediate_vars,
                 oracle_map, all_syms):
    """Convert a string expression with [k+n] references to z-domain SymPy expr."""
    result = expr_str

    # Replace oracle calls with u variables
    for orig_text, (u_name, y_name, arg_str) in oracle_map.items():
        result = result.replace(orig_text, u_name)

    # Replace variable references with z-domain equivalents
    # Must process longer variable names first to avoid partial matches
    def replace_var_ref(match):
        var_name = match.group(1)
        offset_str = match.group(2)
        offset = int(offset_str) if offset_str else 0

        if offset == 1:
            return f'(__ztf*{var_name})'
        elif offset == 0:
            return var_name
        elif offset == -1:
            if var_name in prev_map:
                return prev_map[var_name]
            else:
                return f'({var_name}/__ztf)'
        else:
            raise ValueError(f"Unsupported offset {offset} for {var_name}")

    result = VAR_REF_PATTERN.sub(replace_var_ref, result)

    # Parse with SymPy
    try:
        expr = parse_expr(result, local_dict=all_syms)
    except Exception as e:
        raise ValueError(f"Failed to parse expression '{result}': {e}")

    return expr


def _extract_coefficients(expr, var_names, all_syms):
    """Extract linear coefficients from a SymPy expression.

    Returns dict mapping var_name -> coefficient, plus 'const' for the constant term.
    Raises ValueError if the expression is nonlinear in the variables.
    """
    expr = expand(expr)
    coeff_dict = {}
    remaining = expr

    for var_name in var_names:
        sym = all_syms[var_name]
        coeff = expr.coeff(sym)
        coeff_dict[var_name] = coeff
        remaining = remaining - coeff * sym

    remaining = expand(remaining)
    coeff_dict['const'] = remaining

    # Validate linearity: remaining should not contain any of our variables
    for var_name in var_names:
        sym = all_syms.get(var_name)
        if sym is None:
            continue
        if remaining.has(sym):
            raise ValueError(
                f"Non-linear term detected: expression contains '{var_name}' "
                f"in a non-linear way"
            )

    return coeff_dict
