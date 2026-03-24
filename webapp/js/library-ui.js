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
  { label: '\u2207f', value: 'gradient' },
  { label: 'prox_f + prox_g', value: 'proximal' },
  { label: '\u2207f + prox_g', value: 'mixed' },
];

let activeFilter = 'all';

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
}

function applyFilter(data) {
  const cards = document.querySelectorAll('.algo-card');
  cards.forEach((card, i) => {
    const algo = data[i];
    if (activeFilter === 'all' || algo.oracleType === activeFilter) {
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
    card.className = 'algo-card ' + (algo.oracleType || 'gradient');

    // Title
    const title = document.createElement('div');
    title.className = 'algo-card-title';
    title.textContent = algo.name;
    card.appendChild(title);

    // Meta: oracle types + citation
    const meta = document.createElement('div');
    meta.className = 'algo-card-meta';
    const parts = [];
    if (algo.oracles && algo.oracles.length > 0) {
      parts.push(algo.oracles.join(', '));
    }
    if (algo.citation) {
      parts.push(algo.citation);
    }
    meta.textContent = parts.join(' \u2014 ');
    card.appendChild(meta);

    // Catalog-only badge
    if (algo.catalogOnly) {
      const badge = document.createElement('span');
      badge.style.cssText =
        'display:inline-block;margin-top:0.4rem;padding:0.15rem 0.5rem;' +
        'font-size:0.72rem;background:#e0e0e0;color:#666;border-radius:10px;';
      badge.textContent = 'catalog only';
      card.appendChild(badge);
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

    // Transfer function — rendered with KaTeX in display mode
    if (algo.tf_latex && !algo.catalogOnly) {
      const tfDiv = document.createElement('div');
      tfDiv.className = 'transfer-function-display';
      tfDiv.style.marginTop = '0.6rem';

      const tfLabel = document.createElement('div');
      tfLabel.style.cssText = 'font-size:0.75rem;color:#888;margin-bottom:0.25rem;';
      tfLabel.textContent = 'Transfer function';
      tfDiv.appendChild(tfLabel);

      const tfMath = document.createElement('div');
      try {
        katex.render('H(z) = ' + algo.tf_latex, tfMath, {
          throwOnError: false,
          displayMode: true,
        });
      } catch (e) {
        tfMath.textContent = 'H(z) = ' + algo.tf_latex;
      }
      tfDiv.appendChild(tfMath);
      card.appendChild(tfDiv);
    }

    // Known equivalences
    const equivs = algo.equivalences || {};
    const equivEntries = [];
    if (equivs.oracle && equivs.oracle.length > 0) {
      equivEntries.push('oracle: ' + equivs.oracle.join(', '));
    }
    if (equivs.shift && equivs.shift.length > 0) {
      equivEntries.push('shift: ' + equivs.shift.join(', '));
    }
    if (equivs.lft && equivs.lft.length > 0) {
      const lftNames = equivs.lft.map(e => (typeof e === 'object' ? e.id : e));
      equivEntries.push('LFT: ' + lftNames.join(', '));
    }
    if (equivEntries.length > 0) {
      const equivDiv = document.createElement('div');
      equivDiv.style.cssText = 'margin-top:0.4rem;font-size:0.75rem;color:#888;';
      equivDiv.textContent = 'Equivalences: ' + equivEntries.join('; ');
      card.appendChild(equivDiv);
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
    chip.textContent = algo.name;
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
