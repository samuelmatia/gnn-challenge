const els = {
  search: document.getElementById('search'),
  typeFilter: document.getElementById('typeFilter'),
  sortBy: document.getElementById('sortBy'),
  tbody: document.querySelector('#leaderboard tbody'),
  statusMessage: document.getElementById('statusMessage'),
  emptyState: document.getElementById('emptyState'),
  statTotal: document.getElementById('statTotal'),
  statTopF1: document.getElementById('statTopF1'),
  statVisible: document.getElementById('statVisible')
};

function setStatus(message, isError = false) {
  els.statusMessage.textContent = message;
  els.statusMessage.classList.toggle('error', isError);
}

async function loadCSV() {
  const response = await fetch('leaderboard.csv');
  if (!response.ok) {
    throw new Error('Failed to load leaderboard.csv');
  }
  const text = await response.text();
  return parseCSV(text);
}

function parseCSV(text) {
  const rows = [];
  let current = '';
  let row = [];
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      row.push(current);
      current = '';
    } else if ((char === '\n' || char === '\r') && !inQuotes) {
      if (char === '\r' && next === '\n') {
        i += 1;
      }
      row.push(current);
      rows.push(row);
      row = [];
      current = '';
    } else {
      current += char;
    }
  }

  if (current.length > 0 || row.length > 0) {
    row.push(current);
    rows.push(row);
  }

  if (!rows.length) {
    return [];
  }

  const headers = rows.shift().map(h => h.trim());
  return rows
    .filter(values => values.some(value => value.trim() !== ''))
    .map(values => {
      const item = {};
      headers.forEach((header, idx) => {
        item[header] = (values[idx] || '').trim();
      });
      return item;
    });
}

function formatMetric(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(2) : '-';
}

function normalizeType(type) {
  const value = (type || 'unknown').toLowerCase();
  if (value === 'human' || value === 'llm' || value === 'hybrid') {
    return value;
  }
  return 'unknown';
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function renderTable(rows) {
  els.tbody.innerHTML = '';

  rows.forEach(row => {
    const rank = Number(row.rank);
    const rankClass = Number.isFinite(rank) && rank > 0 && rank <= 3 ? `rank-${rank}` : '';
    const type = normalizeType(row.model_type);
    const submitter = row.submitter ? escapeHtml(row.submitter) : '';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><span class="rank-pill ${rankClass}">${escapeHtml(row.rank || '-')}</span></td>
      <td>${escapeHtml(row.team || '-')}</td>
      <td>${escapeHtml(row.run_id || '-')}</td>
      <td>${escapeHtml(row.model || '-')}</td>
      <td><span class="type-pill type-${type}">${type}</span></td>
      <td class="metric">${formatMetric(row.f1_score)}</td>
      <td class="metric">${formatMetric(row.accuracy)}</td>
      <td class="metric">${formatMetric(row.precision)}</td>
      <td class="metric">${formatMetric(row.recall)}</td>
      <td>${escapeHtml(row.submission_date || '-')}</td>
      <td>
        ${submitter
          ? `<a class="submitter-link" href="https://github.com/${submitter}" target="_blank" rel="noopener noreferrer">${submitter}</a>`
          : '-'}
      </td>
    `;
    els.tbody.appendChild(tr);
  });

  els.emptyState.hidden = rows.length > 0;
}

function updateStats(allRows, visibleRows) {
  const topF1 = allRows.reduce((best, row) => {
    const f1 = Number(row.f1_score);
    return Number.isFinite(f1) ? Math.max(best, f1) : best;
  }, Number.NEGATIVE_INFINITY);

  els.statTotal.textContent = String(allRows.length);
  els.statTopF1.textContent = Number.isFinite(topF1) ? topF1.toFixed(2) : '-';
  els.statVisible.textContent = String(visibleRows.length);
}

function applyFilters(rows) {
  const query = els.search.value.toLowerCase().trim();
  const type = els.typeFilter.value;
  const sortBy = els.sortBy.value;

  const filtered = rows.filter(row => {
    const haystack = [row.team, row.run_id, row.model, row.model_type, row.submitter]
      .map(value => (value || '').toLowerCase())
      .join(' ');

    const matchesText = haystack.includes(query);
    const rowType = normalizeType(row.model_type);
    const matchesType = type === 'all' || rowType === type;

    return matchesText && matchesType;
  });

  filtered.sort((a, b) => {
    if (sortBy === 'submission_date') {
      return (b.submission_date || '').localeCompare(a.submission_date || '');
    }
    return Number(b[sortBy]) - Number(a[sortBy]);
  });

  renderTable(filtered);
  updateStats(rows, filtered);
  setStatus(`Showing ${filtered.length} of ${rows.length} submissions.`);
}

(async () => {
  try {
    setStatus('Loading leaderboard...');
    const rows = await loadCSV();
    applyFilters(rows);

    els.search.addEventListener('input', () => applyFilters(rows));
    els.typeFilter.addEventListener('change', () => applyFilters(rows));
    els.sortBy.addEventListener('change', () => applyFilters(rows));
  } catch (error) {
    console.error(error);
    setStatus('Unable to load leaderboard data. Check leaderboard.csv availability.', true);
    els.emptyState.hidden = false;
    els.emptyState.textContent = 'Leaderboard data could not be loaded.';
  }
})();
