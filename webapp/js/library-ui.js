/* Linnaeus — Browse Library UI and example chips */

/**
 * Convert a SymPy-style string to approximate LaTeX.
 */
function sympy2Latex(s) {
  return s
    .replace(/\*\*/g, '^')
    .replace(/\*/g, ' \\cdot ')
    .replace(/\balpha\b/g, '\\alpha')
    .replace(/\bbeta\b/g, '\\beta')
    .replace(/\bgamma\b/g, '\\gamma')
    .replace(/\bdelta\b/g, '\\delta')
    .replace(/\beta\b/g, '\\eta')
    .replace(/\bnu\b/g, '\\nu')
    .replace(/\brho\b/g, '\\rho')
    .replace(/\bsigma\b/g, '\\sigma')
    .replace(/\btau\b/g, '\\tau')
    .replace(/\btheta\b/g, '\\theta')
    .replace(/\blam\b/g, '\\lambda');
}

/**
 * Load and render the library UI (Browse Library tab + example chips).
 * Called on page load, before Pyodide is ready.
 */
function loadLibraryUI() {
  'use strict';

  fetch('data/algorithms.json')
    .then(r => r.json())
    .then(data => {
      renderFilterChips(data);
      renderLibraryCards(data);
      renderExampleChips(data);
    })
    .catch(err => {
      console.error('Failed to load algorithms.json:', err);
    });
}

// ---- Filter chips ----

const FILTERS = [
  { label: 'All', value: 'all' },
  { label: 'One oracle', value: 'one-oracle' },
  { label: 'Two oracles', value: 'two-oracle' },
  { label: 'Three oracles', value: 'three-oracle' },
  { label: 'Projected', value: 'projected' },
  { label: 'Consensus', value: 'consensus' },
  { label: 'Distributed', value: 'distributed' },
];

let activeFilter = 'all';

const CATEGORY_DESCRIPTIONS = {
  'one-oracle': 'Algorithms using a single oracle: a gradient, proximal operator, or conjugate prox.',
  'two-oracle': 'Algorithms using two distinct oracles, e.g., <span class="ki" data-formula="\\nabla f"></span> &amp; <span class="ki" data-formula="\\text{prox}_g"></span> or <span class="ki" data-formula="\\text{prox}_f"></span> &amp; <span class="ki" data-formula="\\text{prox}_{g^*}"></span>.',
  'three-oracle': 'Algorithms using three distinct oracles, e.g., <span class="ki" data-formula="\\text{prox}_f"></span>, <span class="ki" data-formula="\\text{prox}_{g^*}"></span>, and <span class="ki" data-formula="\\nabla h"></span>.',
  projected: 'Algorithms for constrained optimization using the gradient <span class="ki" data-formula="\\nabla f"></span> and projection <span class="ki" data-formula="P_C"></span> onto a convex set <span class="ki" data-formula="C"></span>.',
  consensus: 'Consensus algorithms for agreement over a network, using only the mixing oracle <span class="ki" data-formula="W"></span> or Laplacian <span class="ki" data-formula="L"></span> with no objective function. States represent the aggregate states of all nodes.',
  distributed: 'Algorithms for distributed optimization over a network, using a linear mixing oracle <span class="ki" data-formula="W"></span> or graph Laplacian <span class="ki" data-formula="L"></span>. States represent the aggregate states of all nodes, and <span class="ki" data-formula="\\nabla f"></span> is the aggregated vector of local gradients evaluated at respective local states.',
};

function _renderInlineKatex(container) {
  container.querySelectorAll('.ki[data-formula]').forEach(el => {
    try {
      katex.render(el.getAttribute('data-formula'), el, { throwOnError: false });
    } catch (e) { /* ignore */ }
  });
}

function renderFilterChips(data) {
  const container = document.getElementById('library-controls');
  container.innerHTML = '';

  for (const f of FILTERS) {
    const chip = document.createElement('button');
    chip.className = 'filter-chip' + (f.value === activeFilter ? ' active' : '');
    chip.textContent = f.label;
    chip.addEventListener('click', () => {
      activeFilter = f.value;
      // Update active states
      container.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      applyFilter(data);
    });
    container.appendChild(chip);
  }

  // Category description div (inserted after chips container)
  let descDiv = document.getElementById('library-category-desc');
  if (!descDiv) {
    descDiv = document.createElement('div');
    descDiv.id = 'library-category-desc';
    descDiv.style.cssText =
      'margin-top:0.5rem;margin-bottom:0.75rem;font-size:0.85rem;color:#555;' +
      'line-height:1.5;display:none;';
    container.parentNode.insertBefore(descDiv, container.nextSibling);
  }
}

function _updateCategoryDesc() {
  const descDiv = document.getElementById('library-category-desc');
  if (!descDiv) return;
  const html = CATEGORY_DESCRIPTIONS[activeFilter];
  if (html) {
    descDiv.innerHTML = html;
    _renderInlineKatex(descDiv);
    descDiv.style.display = '';
  } else {
    descDiv.style.display = 'none';
  }
}

function applyFilter(data) {
  _updateCategoryDesc();
  const cards = document.querySelectorAll('.algo-card');
  cards.forEach((card, i) => {
    const algo = data[i];
    if (activeFilter === 'all'
        || (activeFilter === 'consensus' && algo.consensus)
        || (activeFilter === 'distributed' && algo.distributed && !algo.consensus)
        || (activeFilter !== 'distributed' && activeFilter !== 'consensus'
            && !algo.distributed && algo.oracleType === activeFilter)) {
      card.style.display = '';
    } else {
      card.style.display = 'none';
    }
  });
}

// ---- Library cards ----

function renderLibraryCards(data) {
  const grid = document.getElementById('library-grid');
  grid.innerHTML = '';

  for (const algo of data) {
    const card = document.createElement('div');
    const cardType = algo.consensus ? 'consensus' : (algo.distributed ? 'distributed' : (algo.oracleType || 'gradient'));
    card.className = 'algo-card ' + cardType;

    // Title: "Full Name (ShortName)" or just "Full Name"
    const title = document.createElement('div');
    title.className = 'algo-card-title';
    title.textContent = algo.shortName
      ? algo.name + ' (' + algo.shortName + ')'
      : algo.name;
    card.appendChild(title);

    // Citations with DOI links + BibTeX copy button
    const citations = algo.citations || [];
    if (citations.length > 0) {
      const meta = document.createElement('div');
      meta.className = 'algo-card-meta';
      citations.forEach((cit, idx) => {
        if (idx > 0) meta.appendChild(document.createTextNode(' / '));
        if (cit.doi) {
          const link = document.createElement('a');
          link.href = cit.doi;
          link.target = '_blank';
          link.rel = 'noopener noreferrer';
          link.className = 'citation-link';
          link.textContent = cit.label;
          meta.appendChild(link);
        } else {
          meta.appendChild(document.createTextNode(cit.label));
        }
      });
      if (algo.bibtex) {
        const btn = document.createElement('button');
        btn.className = 'bibtex-copy-btn';
        btn.title = 'Copy BibTeX';
        btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          navigator.clipboard.writeText(algo.bibtex).then(() => {
            btn.classList.add('copied');
            btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>';
            setTimeout(() => {
              btn.classList.remove('copied');
              btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
            }, 1500);
          });
        });
        meta.appendChild(btn);
      }
      card.appendChild(meta);
    }

    // Catalog-only badge
    if (algo.catalogOnly) {
      const badge = document.createElement('span');
      badge.style.cssText =
        'display:inline-block;margin-top:0.4rem;padding:0.15rem 0.5rem;' +
        'font-size:0.72rem;background:#e0e0e0;color:#666;border-radius:10px;';
      badge.textContent = 'catalog only';
      card.appendChild(badge);
    }

    // Distributed badge
    if (algo.distributed) {
      const distBadge = document.createElement('span');
      distBadge.style.cssText =
        'display:inline-block;margin-top:0.4rem;padding:0.15rem 0.5rem;' +
        'font-size:0.72rem;background:#e8daef;color:#4a148c;border-radius:10px;';
      distBadge.textContent = 'Distributed';
      card.appendChild(distBadge);
    }

    // Notes
    if (algo.notes) {
      const notesDiv = document.createElement('div');
      notesDiv.style.cssText =
        'margin-top:0.4rem;font-size:0.75rem;color:#666;font-style:italic;line-height:1.4;';
      notesDiv.textContent = algo.notes;
      card.appendChild(notesDiv);
    }

    // Equations
    if (algo.equations && algo.equations.length > 0) {
      const eqDiv = document.createElement('div');
      eqDiv.style.cssText =
        'margin-top:0.5rem;font-size:0.78rem;font-family:"SF Mono","Fira Code",Menlo,monospace;' +
        'color:#555;line-height:1.5;';
      for (const eq of algo.equations) {
        const line = document.createElement('div');
        line.textContent = eq;
        eqDiv.appendChild(line);
      }
      card.appendChild(eqDiv);
    }

    // Transfer function or characteristic polynomial
    if ((algo.tf_latex || algo.charPoly) && !algo.catalogOnly) {
      const tfDiv = document.createElement('div');
      tfDiv.className = 'transfer-function-display';
      tfDiv.style.marginTop = '0.6rem';

      const tfMath = document.createElement('div');
      let displayLatex;
      if (algo.consensus && algo.charPoly) {
        // Consensus: show reciprocal of char poly (like a TF)
        const cpLatex = sympy2Latex(algo.charPoly);
        displayLatex = '\\frac{1}{' + cpLatex + '}';
      } else {
        displayLatex = compositionLatex(algo.tf_latex, algo.oracles);
      }
      try {
        katex.render(displayLatex, tfMath, {
          throwOnError: false,
          displayMode: false,
        });
      } catch (e) {
        tfMath.textContent = displayLatex;
      }
      tfDiv.appendChild(tfMath);

      const libLinOracle = linearOracleElement(algo.distributed && !algo.consensus);
      if (libLinOracle) tfDiv.appendChild(libLinOracle);

      card.appendChild(tfDiv);
    }

    grid.appendChild(card);
  }
}

// ---- Example chips on the Check tab ----

function renderExampleChips(data) {
  const container = document.getElementById('example-chips');
  if (!container) return;
  container.innerHTML = '';

  const examples = data.filter(a => !a.catalogOnly && a.equations && a.equations.length > 0);

  for (const algo of examples) {
    const chip = document.createElement('button');
    chip.className = 'chip';
    chip.textContent = algo.shortName || algo.name;
    if (algo.shortName) {
      chip.title = algo.name;  // tooltip shows full name
    }
    chip.addEventListener('click', () => {
      // Load equations into textarea
      const textarea = document.getElementById('algo-input');
      textarea.value = algo.equations.join('\n');

      // Switch to Check tab
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      const checkTab = document.querySelector('.tab[data-tab="check"]');
      if (checkTab) checkTab.classList.add('active');
      const checkContent = document.getElementById('tab-check');
      if (checkContent) checkContent.classList.add('active');
    });
    container.appendChild(chip);
  }
}
