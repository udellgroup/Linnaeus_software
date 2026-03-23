# Linnaeus Webapp — Architecture and Design Principles

## Overview

Linnaeus is a browser-based tool for detecting equivalence between iterative
optimization algorithms. It runs entirely client-side using Pyodide (Python in
WebAssembly) for symbolic computation and a vanilla JavaScript frontend for
rendering.

## Module Responsibilities

| Module | Role | Depends on |
|---|---|---|
| `parser.py` | Parse user equations into z-domain linear system | SymPy only |
| `compute.py` | Solve linear system to extract transfer function H(z) | SymPy only |
| `equivalence.py` | Check oracle, shift, and LFT equivalence between TFs | SymPy only |
| `library.py` | Load algorithm library; orchestrate equivalence search | All Python modules |
| `app.js` | Pyodide lifecycle, user input handling, tab switching | Pyodide, results.js, library-ui.js |
| `results.js` | Render equivalence results with KaTeX | KaTeX |
| `library-ui.js` | Render Browse Library tab and example chips | algorithms.json |

## Design Principles

1. **Single responsibility.** Each Python module handles one stage of the
   pipeline (parse, compute, compare). Each JS module handles one UI concern.

2. **Well-defined interfaces.** Python modules communicate via plain dicts and
   SymPy objects. JS modules communicate via function calls with JSON data.

3. **Python modules are independently testable.** No module requires browser
   APIs or Pyodide to run its tests.

4. **JS modules are pure rendering.** No symbolic computation happens in
   JavaScript; all math runs in Python via Pyodide.

5. **Defensive input validation.** User input is validated at the boundary
   (parser entry point) with clear, user-facing error messages.

6. **Security.** User input is never interpolated into code strings without
   sanitization. HTML output is escaped. Python code execution uses
   controlled data passing rather than string interpolation.

7. **No unnecessary global mutable state.** Module-level constants (like
   `KNOWN_ORACLES`, `ORACLE_RELATIONS`) are acceptable; mutable globals
   are avoided.

8. **Consistent naming.** Snake_case for Python, camelCase for JavaScript.
   Oracle variables use `u_i` (output) and `y_i` (input) following
   control theory convention.

9. **Performance.** The algorithm library is parsed once at startup and
   reused for all queries. SymPy `cancel()` is preferred over `simplify()`
   except as a fallback.

10. **DRY.** Shared constants (oracle names, parameter lists) are defined
    once and referenced everywhere.
