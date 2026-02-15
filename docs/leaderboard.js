const els = {
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
  if (window.location.protocol === 'file:') {
    throw new Error('Cannot fetch CSV from file://. Serve docs over HTTP.');
  }

  const candidates = [
    'leaderboard.csv',
    './leaderboard.csv',
    '/gnn-challenge/leaderboard.csv',
    'docs/leaderboard.csv'
  ];

  for (const path of candidates) {
    try {
      const response = await fetch(path, { cache: 'no-store' });
      if (response.ok) {
        return parseCSV(await response.text());
      }
    } catch (_) {
      // Try next candidate.
    }
  }

  throw new Error('Failed to load leaderboard.csv from known paths.');
}

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  if (!lines.length) {
    return [];
  }

  const headers = lines.shift().split(',').map(h => h.trim());
  return lines
    .filter(Boolean)
    .map(line => {
      const values = line.split(',');
      const row = {};
      headers.forEach((header, idx) => {
        row[header] = (values[idx] || '').trim();
      });
      return row;
    });
}

function formatMetric(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(4) : '-';
}

function renderTable(rows) {
  els.tbody.innerHTML = '';

  rows.forEach(row => {
    const rank = Number(row.rank);
    const rankClass = Number.isFinite(rank) && rank > 0 && rank <= 3 ? `rank-${rank}` : '';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><span class="rank-pill ${rankClass}">${row.rank || '-'}</span></td>
      <td>${row.team || '-'}</td>
      <td class="metric">${formatMetric(row.f1_score)}</td>
      <td class="metric">${formatMetric(row.accuracy)}</td>
      <td class="metric">${formatMetric(row.precision)}</td>
      <td class="metric">${formatMetric(row.recall)}</td>
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
  els.statTopF1.textContent = Number.isFinite(topF1) ? topF1.toFixed(4) : '-';
  els.statVisible.textContent = String(visibleRows.length);
}

function sortRows(rows) {
  const sortBy = els.sortBy.value;
  const sorted = [...rows];
  sorted.sort((a, b) => Number(b[sortBy]) - Number(a[sortBy]));
  return sorted;
}

function applySort(rows) {
  const sorted = sortRows(rows);
  renderTable(sorted);
  updateStats(rows, sorted);
  setStatus(`Showing ${sorted.length} leaderboard rows.`);
}

(async () => {
  try {
    setStatus('Loading leaderboard...');
    const rows = await loadCSV();
    applySort(rows);

    els.sortBy.addEventListener('change', () => applySort(rows));
  } catch (error) {
    console.error(error);
    const message = window.location.protocol === 'file:'
      ? 'Open this page through a local server (not file://) to load leaderboard.csv.'
      : 'Unable to load leaderboard data. Hard refresh and verify leaderboard.csv is deployed.';
    setStatus(message, true);
    els.emptyState.hidden = false;
    els.emptyState.textContent = 'Leaderboard data could not be loaded.';
  }
})();
