/* Linnaeus — Shared KaTeX rendering helpers */

/**
 * Convert an oracle name like "prox_f" to LaTeX like "\\text{prox}_f".
 */
function oracleToLatex(name) {
  const map = {
    'prox_f': '\\text{prox}_f',
    'prox_g': '\\text{prox}_g',
    'prox_h': '\\text{prox}_h',
    'prox_fstar': '\\text{prox}_{f^*}',
    'prox_gstar': '\\text{prox}_{g^*}',
    'prox_hstar': '\\text{prox}_{h^*}',
    'grad_f': '\\nabla f',
    'grad_g': '\\nabla g',
    'grad_h': '\\nabla h',
    'P_C': 'P_C',
  };
  return map[name] || '\\text{' + name.replace(/_/g, '\\_') + '}';
}

/**
 * Build the composition LaTeX: [matrix] \diamond \begin{bmatrix} oracles \end{bmatrix}
 * For distributed algorithms, appends: Linear oracle: L (eigenvalue λ)
 * @param {string} tfLatex - LaTeX for the transfer function matrix
 * @param {string[]} oracles - oracle name strings, e.g. ["prox_f", "prox_g"]
 * @param {boolean} [distributed] - whether this is a distributed algorithm
 * @returns {string} full composition LaTeX
 */
function compositionLatex(tfLatex, oracles, distributed) {
  let result = tfLatex;
  if (oracles && oracles.length > 0) {
    const oracleEntries = oracles.map(oracleToLatex).join(' \\\\ ');
    result += ' \\diamond \\begin{bmatrix} ' + oracleEntries + ' \\end{bmatrix}';
  }
  return result;
}

/**
 * Render a "Linear oracle: L (eigenvalue λ)" line as a separate KaTeX element.
 * @returns {HTMLElement|null} A div with the rendered line, or null if not distributed.
 */
function linearOracleElement(distributed) {
  if (!distributed) return null;
  const div = document.createElement('div');
  div.style.cssText = 'margin-top:0.3rem;font-size:0.85rem;color:#4a148c;';
  try {
    katex.render(
      '\\text{Linear oracle: } L \\text{ (eigenvalue } \\lambda \\text{)}',
      div, { throwOnError: false }
    );
  } catch (e) {
    div.textContent = 'Linear oracle: L (eigenvalue λ)';
  }
  return div;
}
