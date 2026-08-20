// Statements & Kundli panel (Phase F5). File picker -> statement-ingest
// -> kundli, via F1's job API. The numbers (diagnostics, counterfactuals,
// summary) are read from the SAME endpoints the Dashboard tab already
// uses -- Kundli's own persistence writes to those same tables, so no
// new read endpoint is needed. The narrative paragraph isn't a separate
// endpoint either: it's printed to stdout by `stocksense kundli` and
// shows up naturally in the job console's live log when access is
// granted (or is simply absent from that log when declined -- graceful
// degradation is what "numbers always, narrative only if authorized"
// looks like here, enforced server-side by agent/access.py, not by this
// panel hiding anything).

(function () {
  async function refreshFileList() {
    const select = document.getElementById("statement-file-select");
    try {
      const { files } = await fetchJson("/api/statements-folder");
      select.innerHTML = files.length
        ? files.map((f) => `<option value="statements/${f}">${f}</option>`).join("")
        : `<option value="">(no files found in statements/)</option>`;
    } catch (e) {
      select.innerHTML = `<option value="">(unavailable)</option>`;
    }
  }

  async function ingestSelected() {
    const select = document.getElementById("statement-file-select");
    const statusEl = document.getElementById("status-ingest");
    if (!select.value) {
      statusEl.textContent = "no file selected";
      statusEl.className = "sev-high";
      return;
    }
    statusEl.textContent = "starting...";
    statusEl.className = "muted";
    try {
      await triggerJob("statement-ingest", { file: select.value });
      statusEl.textContent = "running -- see job console";
      statusEl.className = "status-running";
    } catch (e) {
      statusEl.textContent = `failed: ${e.message}`;
      statusEl.className = "sev-critical";
    }
  }

  async function runKundli() {
    const statusEl = document.getElementById("status-kundli");
    statusEl.textContent = "starting...";
    statusEl.className = "muted";
    try {
      await triggerJob("kundli", {});
      statusEl.textContent = "running -- narrative (if authorized) streams in the job console below";
      statusEl.className = "status-running";
    } catch (e) {
      statusEl.textContent = `failed: ${e.message}`;
      statusEl.className = "sev-critical";
    }
  }

  function severityClass(sev) {
    return { critical: "sev-critical", high: "sev-high", notable: "sev-notable", ok: "sev-ok" }[sev] || "muted";
  }

  async function refreshKundliNumbers() {
    const summaryEl = document.getElementById("kundli-summary");
    const doshasEl = document.getElementById("kundli-doshas");
    const cfEl = document.getElementById("kundli-counterfactuals");

    try {
      const s = await fetchJson("/api/summary");
      summaryEl.innerHTML = s.n_positions === 0
        ? `<div class="empty-state">No positions ingested yet.</div>`
        : `
          <div class="row"><span>positions</span><span>${s.n_positions}</span></div>
          <div class="row"><span>net P&amp;L</span><span class="${s.total_net_pnl >= 0 ? "sev-ok" : "sev-critical"}">₹${s.total_net_pnl.toLocaleString("en-IN", { maximumFractionDigits: 2 })}</span></div>
          <div class="row"><span>gross P&amp;L</span><span>₹${s.total_gross_pnl.toLocaleString("en-IN", { maximumFractionDigits: 2 })}</span></div>
          <div class="row"><span>charges</span><span>₹${s.total_charges.toLocaleString("en-IN", { maximumFractionDigits: 2 })}</span></div>
          <div class="row"><span>win rate</span><span>${(s.win_rate * 100).toFixed(1)}%</span></div>
        `;
    } catch (e) {
      summaryEl.innerHTML = `<div class="empty-state">unavailable</div>`;
    }

    try {
      const { doshas } = await fetchJson("/api/doshas");
      doshasEl.innerHTML = doshas.length
        ? doshas.map((d) => `<div class="row"><span class="${severityClass(d.severity)}">● ${d.metric_name}</span><span>${Number(d.metric_value).toFixed(4)} ${d.metric_unit || ""}</span></div>`).join("")
        : `<div class="empty-state">No diagnostics run yet.</div>`;
    } catch (e) {
      doshasEl.innerHTML = `<div class="empty-state">unavailable</div>`;
    }

    try {
      const { counterfactuals } = await fetchJson("/api/counterfactuals");
      cfEl.innerHTML = counterfactuals.length
        ? counterfactuals.map((c) => `<div class="row"><span>${c.scenario_name}</span><span class="${c.delta_pnl >= 0 ? "sev-ok" : "sev-critical"}">Δ₹${c.delta_pnl.toLocaleString("en-IN", { maximumFractionDigits: 2 })} (${c.n_trades_affected} trades)</span></div>`).join("")
        : `<div class="empty-state">No counterfactuals run yet.</div>`;
    } catch (e) {
      cfEl.innerHTML = `<div class="empty-state">unavailable</div>`;
    }
  }

  function init() {
    document.getElementById("btn-refresh-files").addEventListener("click", refreshFileList);
    document.getElementById("btn-ingest").addEventListener("click", ingestSelected);
    document.getElementById("btn-kundli").addEventListener("click", runKundli);
    refreshFileList();
    refreshKundliNumbers();
  }

  document.addEventListener("job-finished", refreshKundliNumbers);
  onTabActivate("statements", () => { refreshFileList(); refreshKundliNumbers(); });
  document.addEventListener("DOMContentLoaded", init);
})();
