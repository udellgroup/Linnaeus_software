/* Linnaeus — Shared KaTeX rendering helpers */

/**
 * Convert an oracle name like "prox_f" to LaTeX like "\\text{prox}_f".
 */
function oracleToLatex(name) {
  const map = {
    'prox_f': '\\text{prox}_f',
    'prox_g': '\\text{prox}_g',
    'prox_fstar': '\\text{prox}_{f^*}',
    'prox_gstar': '\\text{prox}_{g^*}',
    'grad_f': '\\nabla f',
    'grad_g': '\\nabla g',
    'subgrad_f': '\\partial f',
    'subgrad_g': '\\partial g',
    'subgrad_fstar': '\\partial f^*',
    'subgrad_gstar': '\\partial g^*',
  };
  return map[name] || '\\text{' + name.replace(/_/g, '\\_') + '}';
}

/**
 * Build the composition LaTeX: [matrix] \diamond \begin{bmatrix} oracles \end{bmatrix}
 * @param {string} tfLatex - LaTeX for the transfer function matrix
 * @param {string[]} oracles - oracle name strings, e.g. ["prox_f", "prox_g"]
 * @returns {string} full composition LaTeX
 */
function compositionLatex(tfLatex, oracles) {
  if (!oracles || oracles.length === 0) {
    return tfLatex;
  }
  const oracleEntries = oracles.map(oracleToLatex).join(' \\\\ ');
  return tfLatex + ' \\diamond \\begin{bmatrix} ' + oracleEntries + ' \\end{bmatrix}';
}
