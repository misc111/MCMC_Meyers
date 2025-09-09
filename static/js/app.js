// Minimal JS to drive the UI, SSE live trace, and simple charts

async function fetchJSON(url, opts = {}) {
  const resp = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${txt}`);
  }
  return await resp.json();
}

function fmt(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return '';
  if (Math.abs(x) >= 1000) return x.toFixed(0);
  if (Math.abs(x) >= 1) return x.toFixed(2);
  return x.toExponential(2);
}

function buildTriangleTable(el, C) {
  const W = C.length;
  const D = C[0].length;
  const head = document.createElement('thead');
  const trh = document.createElement('tr');
  trh.appendChild(Object.assign(document.createElement('th'), { innerText: 'AY \\ Dev' }));
  for (let d = 0; d < D; d++) {
    trh.appendChild(Object.assign(document.createElement('th'), { innerText: d + 1 }));
  }
  head.appendChild(trh);

  const body = document.createElement('tbody');
  for (let w = 0; w < W; w++) {
    const tr = document.createElement('tr');
    const th = document.createElement('th');
    th.className = 'ay';
    th.innerText = `${w + 1}`;
    tr.appendChild(th);
    for (let d = 0; d < D; d++) {
      const td = document.createElement('td');
      const val = C[w][d];
      const observed = (val !== null) && !Number.isNaN(val);
      td.className = observed ? 'obs' : 'miss';
      td.innerText = observed ? fmt(val) : '—';
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }

  el.innerHTML = '';
  el.appendChild(head);
  el.appendChild(body);
}

// Simple canvas line chart
function drawLine(canvas, xs, ys, opts = {}) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0b0f14';
  ctx.fillRect(0, 0, W, H);

  if (ys.length === 0) return;
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const pad = 24;
  const ix = (x) => pad + (W - 2 * pad) * (x - xMin) / Math.max(1e-9, (xMax - xMin));
  const iy = (y) => H - pad - (H - 2 * pad) * (y - yMin) / Math.max(1e-9, (yMax - yMin));

  // Axes
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, H - pad); ctx.lineTo(W - pad, H - pad); ctx.stroke();

  // Path
  ctx.strokeStyle = '#58a6ff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(ix(xs[0]), iy(ys[0]));
  for (let i = 1; i < ys.length; i++) ctx.lineTo(ix(xs[i]), iy(ys[i]));
  ctx.stroke();
}

// Simple histogram
function drawHist(canvas, bins, counts) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0b0f14';
  ctx.fillRect(0, 0, W, H);
  if (!bins || bins.length === 0) return;
  const maxC = Math.max(...counts, 1);
  const pad = 24;
  const bw = (W - 2 * pad) / bins.length;
  for (let i = 0; i < bins.length; i++) {
    const h = (H - 2 * pad) * (counts[i] / maxC);
    ctx.fillStyle = '#3fb950';
    ctx.fillRect(pad + i * bw, H - pad - h, Math.max(1, bw - 1), h);
  }
  // Axes
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, H - pad); ctx.lineTo(W - pad, H - pad); ctx.stroke();
}

// State for live traces (populated after data load)
const TRACE = { ready: false, xs: [], es: null, W: 0, D: 0, logELR: [], alpha: [], beta: [], sigma: [] };
const CURRENT_RUN = { iterations: 0, chains: 1 };

function buildTraceDashboard(W, D) {
  TRACE.ready = true;
  TRACE.W = W; TRACE.D = D;
  TRACE.xs = []; TRACE.logELR = [];
  TRACE.alpha = Array.from({ length: W }, () => []);
  TRACE.beta = Array.from({ length: D }, () => []);
  TRACE.sigma = Array.from({ length: D }, () => []);

  const aGrid = document.getElementById('trace-alpha-grid');
  const bGrid = document.getElementById('trace-beta-grid');
  const sGrid = document.getElementById('trace-sigma-grid');
  if (!aGrid || !bGrid || !sGrid) return;
  aGrid.innerHTML = ''; bGrid.innerHTML = ''; sGrid.innerHTML = '';

  for (let w = 0; w < W; w++) {
    const wrap = document.createElement('div'); wrap.className = 'mini';
    const title = document.createElement('div'); title.className = 'mini-title'; title.innerText = `α_${w+1}`;
    const canvas = document.createElement('canvas'); canvas.id = `trace-alpha-${w}`; canvas.width = 200; canvas.height = 110;
    wrap.appendChild(title); wrap.appendChild(canvas); aGrid.appendChild(wrap);
  }
  for (let d = 0; d < D; d++) {
    const wrap = document.createElement('div'); wrap.className = 'mini';
    const title = document.createElement('div'); title.className = 'mini-title'; title.innerText = `β_${d+1}`;
    const canvas = document.createElement('canvas'); canvas.id = `trace-beta-${d}`; canvas.width = 200; canvas.height = 110;
    wrap.appendChild(title); wrap.appendChild(canvas); bGrid.appendChild(wrap);
  }
  for (let d = 0; d < D; d++) {
    const wrap = document.createElement('div'); wrap.className = 'mini';
    const title = document.createElement('div'); title.className = 'mini-title'; title.innerText = `σ_${d+1}`;
    const canvas = document.createElement('canvas'); canvas.id = `trace-sigma-${d}`; canvas.width = 200; canvas.height = 110;
    wrap.appendChild(title); wrap.appendChild(canvas); sGrid.appendChild(wrap);
  }
}

async function loadData() {
  const data = await fetchJSON('/api/data');
  document.getElementById('dims').innerText = `W=${data.W}, D=${data.D}`;
  buildTriangleTable(document.getElementById('triangle-table'), data.C);
  buildTraceDashboard(data.W, data.D);
}

async function useSample() {
  await fetchJSON('/api/use_sample', { method: 'POST' });
  await loadData();
}

function openSSE() {
  if (TRACE.es) TRACE.es.close();
  TRACE.xs = []; TRACE.logELR = [];
  TRACE.alpha = Array.from({ length: TRACE.W }, () => []);
  TRACE.beta = Array.from({ length: TRACE.D }, () => []);
  TRACE.sigma = Array.from({ length: TRACE.D }, () => []);
  const es = new EventSource('/api/stream/mcmc');
  TRACE.es = es;
  es.onmessage = (ev) => {
    if (!ev.data) return;
    try {
      const msg = JSON.parse(ev.data);
      if (msg.type === 'progress') {
        const it = msg.iter;
        TRACE.xs.push(it);
        // Progress bar
        const chains = CURRENT_RUN.chains || 1;
        const iters = CURRENT_RUN.iterations || 1;
        const total = chains * iters;
        const curr = (Math.max(0, (msg.chain || 1) - 1) * iters) + it;
        const prog = document.getElementById('mcmc-progress');
        if (prog) { prog.max = total; prog.value = Math.min(total, curr); }
        const tb = msg.traceBundle || {};
        // logELR
        if (typeof tb.logELR === 'number') {
          TRACE.logELR.push(tb.logELR);
          const c = document.getElementById('trace-logELR');
          if (c) drawLine(c, TRACE.xs, TRACE.logELR);
        }
        // alpha / beta / sigma arrays
        if (Array.isArray(tb.alpha)) {
          for (let w = 0; w < Math.min(TRACE.W, tb.alpha.length); w++) {
            TRACE.alpha[w].push(tb.alpha[w]);
            const c = document.getElementById(`trace-alpha-${w}`);
            if (c) drawLine(c, TRACE.xs, TRACE.alpha[w]);
          }
        }
        if (Array.isArray(tb.beta)) {
          for (let d = 0; d < Math.min(TRACE.D, tb.beta.length); d++) {
            TRACE.beta[d].push(tb.beta[d]);
            const c = document.getElementById(`trace-beta-${d}`);
            if (c) drawLine(c, TRACE.xs, TRACE.beta[d]);
          }
        }
        if (Array.isArray(tb.sigma)) {
          for (let d = 0; d < Math.min(TRACE.D, tb.sigma.length); d++) {
            TRACE.sigma[d].push(tb.sigma[d]);
            const c = document.getElementById(`trace-sigma-${d}`);
            if (c) drawLine(c, TRACE.xs, TRACE.sigma[d]);
          }
        }
        document.getElementById('trace-meta').innerText = `iter=${it}, chain=${msg.chain}`;
        const t = msg.tau; if (t) document.getElementById('tau-info').innerText = `τ α=${t.alpha.toFixed(3)} β=${t.beta.toFixed(3)} a=${t.a.toFixed(3)} logELR=${t.logELR.toFixed(3)}`;
      } else if (msg.type === 'chain_start') {
        document.getElementById('trace-meta').innerText = `Chain ${msg.chain} started`;
      } else if (msg.type === 'chain_done') {
        const chains = CURRENT_RUN.chains || 1;
        const iters = CURRENT_RUN.iterations || 1;
        const prog = document.getElementById('mcmc-progress');
        if (prog) { prog.max = chains * iters; prog.value = Math.min(chains * iters, (msg.chain || 1) * iters); }
        document.getElementById('trace-meta').innerText = `Chain ${msg.chain} done; draws=${msg.draws}`;
      } else if (msg.type === 'done') {
        document.getElementById('trace-meta').innerText = `MCMC complete`;
        es.close(); TRACE.es = null;
      }
    } catch {}
  };
}

async function startMCMC() {
  const payload = {
    iterations: Number(document.getElementById('iterations').value),
    burn_in: Number(document.getElementById('burn_in').value),
    thin: Number(document.getElementById('thin').value),
    adapt: Number(document.getElementById('adapt').value),
    chains: Number(document.getElementById('chains').value),
    update_every: 10,
  };
  CURRENT_RUN.iterations = payload.iterations;
  CURRENT_RUN.chains = payload.chains;
  const prog = document.getElementById('mcmc-progress');
  if (prog) { prog.max = payload.iterations * payload.chains; prog.value = 0; }
  await fetchJSON('/api/start_mcmc', { method: 'POST', body: JSON.stringify(payload) });
  openSSE();
}

async function stopMCMC() {
  try { await fetchJSON('/api/stop_mcmc', { method: 'POST' }); } catch {}
  if (TRACE.es) { TRACE.es.close(); TRACE.es = null; }
}

async function simulate() {
  const n = Number(document.getElementById('use_draws').value);
  const res = await fetchJSON('/api/simulate', { method: 'POST', body: JSON.stringify({ use_draws: n }) });
  const sum = res.summary;
  document.getElementById('sim-summary').innerText = `Ultimate mean=${fmt(sum.ultimate.mean)} p05=${fmt(sum.ultimate.p05)} p50=${fmt(sum.ultimate.p50)} p95=${fmt(sum.ultimate.p95)} | Reserve mean=${fmt(sum.reserve.mean)}`;
  const kind = document.getElementById('hist-kind').value;
  const h = res.hist[kind];
  drawHist(document.getElementById('hist-canvas'), h.bins, h.counts);
  document.getElementById('sample-index').value = res.first_index ?? 0;
}

async function loadSimTriangle() {
  const idx = Number(document.getElementById('sample-index').value);
  const data = await fetchJSON(`/api/sim_sample?i=${idx}`);
  buildTriangleTable(document.getElementById('sim-triangle-table'), data.triangle);
  // Render parameters
  const p = data.params || {};
  const fmtArr = (arr, k) => (Array.isArray(arr) ? `${k}: [${arr.map(v => Number(v).toFixed(3)).join(', ')}]` : `${k}: —`);
  const txt = [
    `logELR: ${p.logELR !== undefined ? Number(p.logELR).toFixed(4) : '—'}`,
    fmtArr(p.alpha, 'alpha'),
    fmtArr(p.beta, 'beta'),
    fmtArr(p.sigma, 'sigma'),
  ].join('\n');
  const el = document.getElementById('params-content');
  if (el) el.textContent = txt;
}

function onHistKindChange() {
  // Re-fetch summary to redraw current hist, if available
  fetchJSON('/api/simulate', { method: 'POST', body: JSON.stringify({ use_draws: Number(document.getElementById('use_draws').value) }) })
    .then((res) => {
      const kind = document.getElementById('hist-kind').value;
      const h = res.hist[kind];
      drawHist(document.getElementById('hist-canvas'), h.bins, h.counts);
    })
    .catch(() => {});
}

window.addEventListener('DOMContentLoaded', async () => {
  document.getElementById('load-sample').addEventListener('click', useSample);
  document.getElementById('run-mcmc').addEventListener('click', startMCMC);
  document.getElementById('stop-mcmc').addEventListener('click', stopMCMC);
  document.getElementById('simulate').addEventListener('click', simulate);
  document.getElementById('load-sample-tri').addEventListener('click', loadSimTriangle);
  document.getElementById('hist-kind').addEventListener('change', onHistKindChange);
  try {
    await loadData();
  } catch (e) {
    document.getElementById('dims').innerText = `Failed to load data: ${e}`;
  }
});
