/* Linnaeus — Results display with KaTeX rendering */

/**
 * Display equivalence check results in the results panel.
 * @param {Object} result - { tf_latex, oracle_types, matches[] }
 */
function displayResults(result) {
  'use strict';

  const panel = document.getElementById('results-panel');
  panel.innerHTML = '<h2 class="panel-title">Results</h2>';

  // User's transfer function
  const tfSection = document.createElement('div');
  tfSection.className = 'transfer-function-display';
  const tfLabel = document.createElement('div');
  tfLabel.style.cssText = 'font-size:0.8rem;color:#666;margin-bottom:0.5rem;font-weight:600;';
  tfLabel.textContent = 'Your transfer function';
  tfSection.appendChild(tfLabel);

  const tfMath = document.createElement('div');
  try {
    katex.render('H(z) = ' + result.tf_latex, tfMath, {
      throwOnError: false,
      displayMode: true,
    });
  } catch (e) {
    tfMath.textContent = 'H(z) = ' + result.tf_latex;
  }
  tfSection.appendChild(tfMath);

  // Oracle types
  if (result.oracle_types && result.oracle_types.length > 0) {
    const oracleInfo = document.createElement('div');
    oracleInfo.style.cssText = 'font-size:0.8rem;color:#666;margin-top:0.5rem;';
    oracleInfo.textContent = 'Oracles: ' + result.oracle_types.join(', ');
    tfSection.appendChild(oracleInfo);
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

    // Badge + name row
    const header = document.createElement('div');
    header.style.cssText = 'display:flex;align-items:center;gap:0.75rem;margin-bottom:0.5rem;';

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
    header.appendChild(badge);

    const nameEl = document.createElement('span');
    nameEl.style.cssText = 'font-weight:600;font-size:0.95rem;';
    nameEl.textContent = match.name;
    if (match.citation) {
      nameEl.textContent += ' (' + match.citation + ')';
    }
    header.appendChild(nameEl);

    card.appendChild(header);

    // Library transfer function
    if (match.lib_tf_latex) {
      const libTf = document.createElement('div');
      libTf.style.cssText = 'margin:0.5rem 0;';
      const libLabel = document.createElement('div');
      libLabel.style.cssText = 'font-size:0.78rem;color:#888;margin-bottom:0.25rem;';
      libLabel.textContent = 'Library transfer function';
      libTf.appendChild(libLabel);

      const libMath = document.createElement('div');
      try {
        katex.render('H_{\\text{lib}}(z) = ' + match.lib_tf_latex, libMath, {
          throwOnError: false,
          displayMode: true,
        });
      } catch (e) {
        libMath.textContent = match.lib_tf_latex;
      }
      libTf.appendChild(libMath);
      card.appendChild(libTf);
    }

    // Parameter mapping — show both library and user param constraints.
    // When a symbol name appears in both, add _{\text{lib}} subscript to the lib one.
    const hasLibParams = match.params && Object.keys(match.params).length > 0;
    const hasUserParams = match.user_params && Object.keys(match.user_params).length > 0;
    if (hasLibParams || hasUserParams) {
      const libEntries = hasLibParams ? Object.entries(match.params) : [];
      const userEntries = hasUserParams ? Object.entries(match.user_params) : [];

      // Detect name collisions between lib and user param keys
      const libKeys = new Set(libEntries.map(([k]) => k));
      const userKeys = new Set(userEntries.map(([k]) => k));
      const collisions = new Set([...libKeys].filter(k => userKeys.has(k)));

      // Build display entries: [displayKey, value] pairs.
      // Only add _{\text{lib}} subscript when there's a name collision.
      const allDisplay = [];
      for (const [k, v] of libEntries) {
        const displayKey = collisions.has(k) ? k + '_{\\text{lib}}' : k;
        allDisplay.push([displayKey, v]);
      }
      for (const [k, v] of userEntries) {
        allDisplay.push([k, v]);
      }

      const paramDiv = document.createElement('div');
      paramDiv.style.cssText =
        'margin-top:0.5rem;padding:0.5rem 0.75rem;background:#f0f4ff;' +
        'border:1px solid #c5cae9;border-radius:4px;';
      const paramLabel = document.createElement('div');
      paramLabel.style.cssText = 'font-size:0.78rem;color:#666;margin-bottom:0.25rem;font-weight:600;';
      paramLabel.textContent = 'Parameter mapping:';
      paramDiv.appendChild(paramLabel);

      const paramMath = document.createElement('div');
      const keys = allDisplay.map(([k]) => k).join(', ');
      const vals = allDisplay.map(([, v]) => v).join(', ');
      let paramLatex = '(' + keys + ') \\leftarrow (' + vals + ')';

      // Append free parameter annotation if present
      if (match.free_params && match.free_params.length > 0) {
        const freeList = match.free_params.join(', ');
        paramLatex += ' \\qquad (' + freeList + ' \\text{ free})';
      }

      try {
        katex.render(paramLatex, paramMath, { throwOnError: false, displayMode: false });
      } catch (e) {
        paramMath.textContent = '(' + keys + ') <- (' + vals + ')';
      }
      paramDiv.appendChild(paramMath);
      card.appendChild(paramDiv);
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

    panel.appendChild(card);
  }
}
