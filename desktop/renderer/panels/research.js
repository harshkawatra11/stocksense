// Research panel (Phase G/Track C). The missing surface the prior audit
// found: train-candidate/reconcile were in the job allowlist but had no
// button anywhere, GET /api/positions and the full registry existed but
// nothing read them, and there was no way to see a verdict document
// (e.g. research/verdict_intraday.md) without a terminal.

(function () {
  function severityRowsFromRegistry(models) {
    if (models.length === 0) return `<div class="empty-state">No models registered yet.</div>`;
    return models
      .map((m) => `
        <div class="row">
          <span>${m.model_id ? m.model_id.slice(0, 8) : "?"} · h${m.horizon_bars} · ${m.lifecycle_state}</span>
          <span class="${m.gate_decision === "promote" ? "sev-ok" : m.gate_decision === "reject" ? "sev-critical" : "muted"}">${m.gate_decision || "pending"}</span>
        </div>
      `)
      .join("");
  }

  async function refreshRegistry() {
    const el = document.getElementById("research-registry");
    try {
      const { models } = await fetchJson("/api/registry");
      // Full list, unlike the Dashboard's own truncated 10-row widget --
      // this tab is where you actually inspect the registry.
      el.innerHTML = severityRowsFromRegistry(models);
    } catch (e) {
      el.innerHTML = `<div class="empty-state">${e.message}</div>`;
    }
  }

  async function refreshPositions() {
    const el = document.getElementById("research-positions");
    try {
      const { positions, total } = await fetchJson("/api/positions?limit=50");
      if (positions.length === 0) {
        el.innerHTML = `<div class="empty-state">No positions ingested yet -- run Kundli from the STATEMENTS tab.</div>`;
        return;
      }
      const rows = positions
        .map((p) => {
          const cls = p.net_pnl >= 0 ? "sev-ok" : "sev-critical";
          return `<div class="row"><span>${p.symbol} · ${p.open_date}</span><span class="${cls}">₹${Number(p.net_pnl).toFixed(2)}</span></div>`;
        })
        .join("");
      el.innerHTML = `<div class="usage-label">showing ${positions.length} of ${total}</div>` + rows;
    } catch (e) {
      el.innerHTML = `<div class="empty-state">${e.message}</div>`;
    }
  }

  async function refreshDocList() {
    const select = document.getElementById("research-doc-select");
    try {
      const { docs } = await fetchJson("/api/research/docs");
      select.innerHTML = docs.length
        ? docs.map((d) => `<option value="${d}">${d}</option>`).join("")
        : `<option value="">(no research docs found)</option>`;
    } catch (e) {
      select.innerHTML = `<option value="">(${e.message})</option>`;
    }
  }

  async function openSelectedDoc() {
    const select = document.getElementById("research-doc-select");
    const viewer = document.getElementById("research-doc-content");
    if (!select.value) return;
    viewer.textContent = "loading...";
    try {
      const { content } = await fetchJson(`/api/research/doc/${encodeURIComponent(select.value)}`);
      viewer.textContent = content;
    } catch (e) {
      viewer.textContent = `could not load: ${e.message}`;
    }
  }

  async function runTrainCandidate() {
    const statusEl = document.getElementById("status-train-candidate");
    statusEl.textContent = "starting...";
    statusEl.className = "muted";
    try {
      await triggerJob("train-candidate", {
        horizon: document.getElementById("train-horizon").value,
        top_n: document.getElementById("train-top-n").value,
        cost_bps: document.getElementById("train-cost-bps").value,
      });
      statusEl.textContent = "running -- see job console below";
      statusEl.className = "status-running";
    } catch (e) {
      statusEl.textContent = `failed to start: ${e.message}`;
      statusEl.className = "sev-critical";
    }
  }

  async function runReconcile() {
    const statusEl = document.getElementById("status-reconcile");
    statusEl.textContent = "starting...";
    statusEl.className = "muted";
    try {
      await triggerJob("reconcile", {
        horizon: document.getElementById("reconcile-horizon").value,
        lifecycle: document.getElementById("reconcile-lifecycle").value,
      });
      statusEl.textContent = "running -- see job console below";
      statusEl.className = "status-running";
    } catch (e) {
      statusEl.textContent = `failed to start: ${e.message}`;
      statusEl.className = "sev-critical";
    }
  }

  async function askCorpus() {
    const input = document.getElementById("ask-question");
    const answerEl = document.getElementById("ask-answer");
    const question = input.value.trim();
    if (!question) return;

    answerEl.innerHTML = `<div class="empty-state">asking...</div>`;
    try {
      // Check access first so a decline reads as a clear instruction,
      // not a silently empty answer -- rag.agent.ask() still enforces
      // this server-side (agent.claude_cli.invoke's single gate); this
      // is purely a friendlier message for the common case.
      const auth = await fetchJson("/api/claude/auth");
      if (!auth.access_granted) {
        answerEl.innerHTML = `<div class="empty-state">Claude access not authorized -- grant it in the CLAUDE tab first.</div>`;
        return;
      }

      const result = await postJson("/api/ask", { question });
      if (result.n_chunks_retrieved === 0) {
        answerEl.innerHTML = `<div class="empty-state">${result.answer}</div>`;
        return;
      }
      const citations = result.citations
        .map((c) => `<div class="row"><span>[${c.index}]</span><span>${c.title} (${c.source_ref})</span></div>`)
        .join("");
      answerEl.innerHTML = `<div class="doc-viewer">${result.answer}</div>${citations}`;
    } catch (e) {
      answerEl.innerHTML = `<div class="empty-state">${e.message}</div>`;
    }
  }

  function init() {
    document.getElementById("btn-train-candidate").addEventListener("click", runTrainCandidate);
    document.getElementById("btn-reconcile").addEventListener("click", runReconcile);
    document.getElementById("btn-ask").addEventListener("click", askCorpus);
    document.getElementById("btn-load-doc").addEventListener("click", openSelectedDoc);
    refreshRegistry();
    refreshPositions();
    refreshDocList();
  }

  document.addEventListener("job-finished", () => { refreshRegistry(); refreshPositions(); });
  onTabActivate("research", () => { refreshRegistry(); refreshPositions(); refreshDocList(); });
  document.addEventListener("DOMContentLoaded", init);
})();
