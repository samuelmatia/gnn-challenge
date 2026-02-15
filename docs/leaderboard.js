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
  const response = await fetch('leaderboard.csv');
  if (!response.ok) {
    throw new Error('Failed to load leaderboard.csv');
  }
  return parseCSV(await response.text());
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
    setStatus('Unable to load leaderboard data. Check leaderboard.csv availability.', true);
    els.emptyState.hidden = false;
    els.emptyState.textContent = 'Leaderboard data could not be loaded.';
  }
})();
