/* Linnaeus — Results display with KaTeX rendering */

/**
 * Display equivalence check results in the results panel.
 * @param {Object} result - { tf_latex, oracle_types, matches[] }
 */
function displayResults(result) {
  'use strict';

  const panel = document.getElementById('results-panel');
  panel.innerHTML = '<h2 class="panel-title">Results</h2>';

  // User's transfer function — composition notation
  const tfSection = document.createElement('div');
  tfSection.className = 'transfer-function-display';
  const tfLabel = document.createElement('div');
  tfLabel.style.cssText = 'font-size:0.8rem;color:#666;margin-bottom:0.5rem;font-weight:600;';
  tfLabel.textContent = 'Your algorithm';
  tfSection.appendChild(tfLabel);

  const tfMath = document.createElement('div');
  const userCompLatex = compositionLatex(result.tf_latex, result.oracle_types,
                                          result.user_distributed);
  try {
    katex.render(userCompLatex, tfMath, {
      throwOnError: false,
      displayMode: true,
    });
  } catch (e) {
    tfMath.textContent = userCompLatex;
  }
  tfSection.appendChild(tfMath);

  if (result.user_distributed) {
    const distLabel = document.createElement('span');
    distLabel.className = 'match-badge';
    distLabel.style.cssText = 'background:#bbdefb;color:#0d47a1;margin-top:0.3rem;display:inline-block;';
    distLabel.textContent = 'Distributed';
    tfSection.appendChild(distLabel);
  }

  panel.appendChild(tfSection);

  // Matches section
  if (!result.matches || result.matches.length === 0) {
    const noMatch = document.createElement('div');
    noMatch.style.cssText =
      'margin-top:1rem;padding:1rem;background:#fff3e0;border:1px solid #ffe0b2;' +
      'border-radius:6px;color:#e65100;font-size:0.9rem;';
    noMatch.textContent = 'No match found in the library.';
    panel.appendChild(noMatch);
    return;
  }

  const matchesHeader = document.createElement('h3');
  matchesHeader.style.cssText = 'margin-top:1.25rem;margin-bottom:0.75rem;font-size:1rem;color:#1a1a2e;';
  matchesHeader.textContent = 'Matches (' + result.matches.length + ')';
  panel.appendChild(matchesHeader);

  for (const match of result.matches) {
    const card = document.createElement('div');
    card.style.cssText =
      'margin-bottom:1rem;padding:1rem;background:#fafafa;border:1px solid #e0e0e0;border-radius:6px;';

    // Name
    const nameEl = document.createElement('div');
    nameEl.style.cssText = 'font-weight:600;font-size:0.95rem;margin-bottom:0.4rem;';
    nameEl.textContent = match.name;
    if (match.citation) {
      const citSpan = document.createElement('span');
      citSpan.style.cssText = 'font-weight:400;font-size:0.85rem;margin-left:0.4rem;';
      citSpan.appendChild(document.createTextNode('('));
      if (match.doi) {
        const link = document.createElement('a');
        link.href = match.doi;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.className = 'citation-link';
        link.textContent = match.citation;
        citSpan.appendChild(link);
      } else {
        citSpan.appendChild(document.createTextNode(match.citation));
      }
      citSpan.appendChild(document.createTextNode(')'));
      if (match.bibtex) {
        const btn = document.createElement('button');
        btn.className = 'bibtex-copy-btn';
        btn.title = 'Copy BibTeX';
        btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          navigator.clipboard.writeText(match.bibtex).then(() => {
            btn.classList.add('copied');
            btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>';
            setTimeout(() => {
              btn.classList.remove('copied');
              btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
            }, 1500);
          });
        });
        citSpan.appendChild(btn);
      }
      nameEl.appendChild(citSpan);
    }
    card.appendChild(nameEl);

    // Chips row
    const chipsRow = document.createElement('div');
    chipsRow.style.cssText = 'display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;';

    const badge = document.createElement('span');
    badge.className = 'match-badge';
    const badgeStyles = {
      oracle: 'background:#c8e6c9;color:#2e7d32;',
      shift: 'background:#fff9c4;color:#f57f17;',
      lft: 'background:#e1bee7;color:#6a1b9a;',
    };
    badge.style.cssText = badgeStyles[match.type] || '';
    const badgeLabels = {
      oracle: 'Exact match',
      shift: 'Shift equivalent',
      lft: 'LFT equivalent',
    };
    badge.textContent = badgeLabels[match.type] || match.type;
    chipsRow.appendChild(badge);

    if (match.permuted) {
      const permBadge = document.createElement('span');
      permBadge.className = 'match-badge';
      permBadge.style.cssText = 'background:#b3e5fc;color:#01579b;';
      permBadge.textContent = 'Permutation';
      chipsRow.appendChild(permBadge);
    }

    if (match.distributed) {
      const distBadge = document.createElement('span');
      distBadge.className = 'match-badge';
      distBadge.style.cssText = 'background:#bbdefb;color:#0d47a1;';
      distBadge.textContent = 'Distributed';
      chipsRow.appendChild(distBadge);
    }

    if (match.conditional) {
      const condBadge = document.createElement('span');
      condBadge.className = 'match-badge';
      condBadge.style.cssText = 'background:#fff3e0;color:#e65100;';
      condBadge.textContent = 'Conditional';
      chipsRow.appendChild(condBadge);
    }

    card.appendChild(chipsRow);

    // Library transfer function — composition notation
    if (match.lib_tf_latex) {
      const libMath = document.createElement('div');
      libMath.style.cssText = 'margin:0.5rem 0;';
      const libCompLatex = compositionLatex(match.lib_tf_latex, match.lib_oracles,
                                            match.distributed);
      try {
        katex.render(libCompLatex, libMath, {
          throwOnError: false,
          displayMode: true,
        });
      } catch (e) {
        libMath.textContent = libCompLatex;
      }
      card.appendChild(libMath);
    }

    // Parameter mapping — displayed as a set of equations.
    // When a symbol name appears in both lib and user params,
    // add _{\text{lib}} subscript to the lib one.
    const hasLibParams = match.params && Object.keys(match.params).length > 0;
    const hasUserParams = match.user_params && Object.keys(match.user_params).length > 0;
    if (hasLibParams || hasUserParams) {
      const libEntries = hasLibParams ? Object.entries(match.params) : [];
      const userEntries = hasUserParams ? Object.entries(match.user_params) : [];

      // Always subscript lib param keys with _{\text{lib}} to distinguish
      // them from user params, and substitute lib param names in values too.
      const libKeys = new Set(libEntries.map(([k]) => k));

      const equations = [];
      for (const [k, v] of libEntries) {
        const displayKey = k + '_{\\text{lib}}';
        // Substitute lib param names in the value with subscripted version,
        // but NOT the param that matches this key (to avoid alpha_lib = alpha_lib).
        let displayVal = v;
        for (const lk of libKeys) {
          if (lk === k) continue;  // don't subscript same-name param in value
          const re = new RegExp('(?<![a-zA-Z_])' + lk.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '(?![a-zA-Z_])', 'g');
          displayVal = displayVal.replace(re, lk + '_{\\text{lib}}');
        }
        equations.push([displayKey, displayVal]);
      }
      for (const [k, v] of userEntries) {
        let displayVal = v;
        for (const lk of libKeys) {
          const re = new RegExp('(?<![a-zA-Z_])' + lk.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '(?![a-zA-Z_])', 'g');
          displayVal = displayVal.replace(re, lk + '_{\\text{lib}}');
        }
        equations.push([k, displayVal]);
      }

      // Filter out trivial identity equations (x = x) — not informative.
      const nonTrivialEqs = equations.filter(([k, v]) => k !== v);

      if (nonTrivialEqs.length > 0) {
        const paramDiv = document.createElement('div');
        paramDiv.style.cssText =
          'margin-top:0.5rem;padding:0.5rem 0.75rem;background:#f0f4ff;' +
          'border:1px solid #c5cae9;border-radius:4px;';
        const paramLabel = document.createElement('div');
        paramLabel.style.cssText = 'font-size:0.78rem;color:#666;margin-bottom:0.25rem;font-weight:600;';
        paramLabel.textContent = 'Parameter mapping:';
        paramDiv.appendChild(paramLabel);

        const paramMath = document.createElement('div');
        let paramLatex = nonTrivialEqs.map(([k, v]) => k + ' = ' + v).join(', \\quad ');

        // Append free parameter annotation if present
        if (match.free_params && match.free_params.length > 0) {
          const freeList = match.free_params.join(', ');
          paramLatex += ' \\qquad (' + freeList + ' \\text{ free})';
        }

        try {
          katex.render(paramLatex, paramMath, { throwOnError: false, displayMode: false });
        } catch (e) {
          paramMath.textContent = nonTrivialEqs.map(([k, v]) => k + ' = ' + v).join(', ');
        }
        paramDiv.appendChild(paramMath);
        card.appendChild(paramDiv);
      }
    }

    // Shift vector
    if (match.shift_vector && match.shift_vector.some(v => v !== 0)) {
      const shiftDiv = document.createElement('div');
      shiftDiv.style.cssText = 'font-size:0.85rem;color:#444;margin-top:0.5rem;';
      const shiftLabel = document.createElement('strong');
      shiftLabel.textContent = 'Shift vector: ';
      shiftDiv.appendChild(shiftLabel);
      shiftDiv.appendChild(document.createTextNode('[' + match.shift_vector.join(', ') + ']'));
      card.appendChild(shiftDiv);
    }

    // Explanation text
    const explanation = document.createElement('div');
    explanation.style.cssText = 'font-size:0.82rem;color:#666;margin-top:0.5rem;font-style:italic;';
    const explanations = {
      oracle:
        'Produces identical iterates for matching parameter values.',
      shift:
        'Same iterates as this entry, offset by a fixed index.',
      lft:
        'Equivalent via a linear fractional transformation of the oracle.',
    };
    explanation.textContent = explanations[match.type] || '';
    card.appendChild(explanation);

    // Conditional match note (e.g., "Equivalent when λ=1")
    if (match.condition_note) {
      const condDiv = document.createElement('div');
      condDiv.style.cssText =
        'font-size:0.82rem;margin-top:0.3rem;';
      try {
        katex.render(
          '\\textit{' + match.condition_note + '}',
          condDiv,
          { throwOnError: false }
        );
      } catch (e) {
        condDiv.textContent = match.condition_note;
      }
      card.appendChild(condDiv);
    }

    panel.appendChild(card);
  }
}
