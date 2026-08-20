// Dashboard polling logic. Poll, not WebSocket -- the underlying data
// changes on a nightly/hourly cadence at most, so push adds real
// complexity (a persistent connection to manage, reconnect logic) for
// no benefit over a 2s poll, matching the reference dashboard's own
// refresh rate. API_BASE/fetchJson now live in core/api.js (Phase F5 --
// every panel shares one client instead of each redeclaring it).

const POLL_INTERVAL_MS = 2000;

function formatMoney(n) {
  if (n === null || n === undefined) return "--";
  const sign = n < 0 ? "-" : "";
  return `${sign}₹${Math.abs(n).toLocaleString("en-IN", { maximumFractionDigits: 2 })}`;
}

function severityClass(sev) {
  return { critical: "sev-critical", high: "sev-high", notable: "sev-notable", ok: "sev-ok" }[sev] || "muted";
}

async function updateClock() {
  document.getElementById("clock").textContent = new Date().toLocaleTimeString("en-IN", { hour12: false });
}

async function updateHealth() {
  const el = document.getElementById("db-status");
  const dataEl = document.getElementById("latest-data");
  try {
    const health = await fetchJson("/api/health");
    el.textContent = `connected (v${health.version})`;
    el.className = "";
    dataEl.textContent = `data: ${health.latest_candle_date || "none"}`;
  } catch (e) {
    el.textContent = "API unreachable";
    el.className = "sev-critical";
  }
}

async function updateSummary() {
  const el = document.getElementById("summary");
  try {
    const s = await fetchJson("/api/summary");
    if (s.n_positions === 0) {
      el.innerHTML = `<div class="empty-state">No positions ingested yet.</div>`;
      return;
    }
    const netClass = s.total_net_pnl >= 0 ? "sev-ok" : "sev-critical";
    el.innerHTML = `
      <div class="row"><span>positions</span><span>${s.n_positions}</span></div>
      <div class="row"><span>net P&amp;L</span><span class="${netClass}">${formatMoney(s.total_net_pnl)}</span></div>
      <div class="row"><span>charges</span><span>${formatMoney(s.total_charges)}</span></div>
      <div class="row"><span>win rate</span><span>${(s.win_rate * 100).toFixed(0)}%</span></div>
    `;
  } catch (e) {
    el.innerHTML = `<div class="empty-state">unavailable</div>`;
  }
}

async function updateDoshas() {
  const el = document.getElementById("doshas");
  try {
    const { doshas } = await fetchJson("/api/doshas");
    const notable = doshas.filter((d) => d.severity !== "ok");
    if (notable.length === 0) {
      el.innerHTML = `<div class="empty-state">No findings above baseline.</div>`;
      return;
    }
    el.innerHTML = notable
      .map((d) => `<div class="row"><span class="${severityClass(d.severity)}">&#9679; ${d.metric_name}</span><span>${Number(d.metric_value).toFixed(3)}</span></div>`)
      .join("");
  } catch (e) {
    el.innerHTML = `<div class="empty-state">unavailable</div>`;
  }
}

async function updateHarness() {
  const el = document.getElementById("harness");
  try {
    const { jobs } = await fetchJson("/api/harness");
    if (jobs.length === 0) {
      el.innerHTML = `<div class="empty-state">No jobs run yet.</div>`;
      return;
    }
    el.innerHTML = jobs
      .map((j) => `<div class="row"><span>${j.job_name}</span><span class="status-${j.status}">${j.status}</span></div>`)
      .join("");
  } catch (e) {
    el.innerHTML = `<div class="empty-state">unavailable</div>`;
  }
}

async function updateRegistry() {
  const el = document.getElementById("registry");
  try {
    const { models } = await fetchJson("/api/registry");
    if (models.length === 0) {
      el.innerHTML = `<div class="empty-state">No models registered yet.</div>`;
      return;
    }
    el.innerHTML = models
      .slice(0, 10)
      .map((m) => `<div class="row"><span>h${m.horizon_bars} ${m.lifecycle_state}</span><span>${m.gate_decision || "pending"}</span></div>`)
      .join("");
  } catch (e) {
    el.innerHTML = `<div class="empty-state">unavailable</div>`;
  }
}

async function updateAgentRuns() {
  const el = document.getElementById("agent-runs");
  try {
    const data = await fetchJson("/api/agent-runs?limit=10");
    if (data.agent_runs.length === 0) {
      el.innerHTML = `<div class="empty-state">No agent calls yet.</div>`;
      return;
    }
    const warn = data.n_unverified > 0 ? `<div class="row sev-high"><span>unverified numbers</span><span>${data.n_unverified}</span></div>` : "";
    el.innerHTML = warn + data.agent_runs
      .slice(0, 8)
      .map((r) => `<div class="row"><span>${r.skill_name || "(none)"}</span><span class="${r.status === 'ok' ? '' : 'sev-high'}">${r.status}</span></div>`)
      .join("");
  } catch (e) {
    el.innerHTML = `<div class="empty-state">unavailable</div>`;
  }
}

async function refresh() {
  await Promise.all([
    updateHealth(), updateSummary(), updateDoshas(),
    updateHarness(), updateRegistry(), updateAgentRuns(),
  ]);
}

updateClock();
setInterval(updateClock, 1000);
refresh();
setInterval(refresh, POLL_INTERVAL_MS);
