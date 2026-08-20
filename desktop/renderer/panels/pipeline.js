// Data & Backfills panel (Phase F5). Triggers backfill-nse-archive,
// backfill-corporate-actions, backfill-intraday, index-corpus via F1's
// job API. Progress comes from the job console's live stream, which
// parses the "PROGRESS: n/total (pct%)" lines cli/main.py's backfill
// loops now print (see _emit_progress in cli/main.py).

(function () {
  function today() {
    return new Date().toISOString().slice(0, 10);
  }

  function progressBar(pct) {
    const filled = Math.round(pct / 5);
    return "█".repeat(filled) + "░".repeat(20 - filled);
  }

  async function runBackfill(command, params, statusElId) {
    const statusEl = document.getElementById(statusElId);
    statusEl.textContent = "starting...";
    statusEl.className = "muted";
    try {
      await triggerJob(command, params);
      statusEl.textContent = "running -- see job console below";
      statusEl.className = "status-running";
    } catch (e) {
      statusEl.textContent = `failed to start: ${e.message}`;
      statusEl.className = "sev-critical";
    }
  }

  function wireForm(formId, command, buildParams, statusElId) {
    const form = document.getElementById(formId);
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      runBackfill(command, buildParams(new FormData(form)), statusElId);
    });
  }

  function init() {
    document.getElementById("nse-archive-end").value = today();
    document.getElementById("ca-end").value = today();
    document.getElementById("intraday-end").value = today();

    wireForm("form-nse-archive", "backfill-nse-archive", (fd) => ({
      start: fd.get("start"), end: fd.get("end"), kind: fd.get("kind"),
    }), "status-nse-archive");

    wireForm("form-ca", "backfill-corporate-actions", (fd) => ({
      start: fd.get("start"), end: fd.get("end"),
    }), "status-ca");

    wireForm("form-intraday", "backfill-intraday", (fd) => ({
      start: fd.get("start"), end: fd.get("end"), top_n: fd.get("top_n"),
    }), "status-intraday");

    document.getElementById("btn-index-corpus").addEventListener("click", () => {
      runBackfill("index-corpus", {}, "status-index-corpus");
    });

    refreshJobHistory();
  }

  async function refreshJobHistory() {
    const el = document.getElementById("pipeline-job-history");
    try {
      const { jobs } = await fetchJson("/api/jobs?limit=15");
      const pipelineJobs = jobs.filter((j) =>
        ["backfill-nse-archive", "backfill-corporate-actions", "backfill-intraday", "index-corpus"].includes(j.command)
      );
      if (pipelineJobs.length === 0) {
        el.innerHTML = `<div class="empty-state">No pipeline jobs run yet.</div>`;
        return;
      }
      el.innerHTML = pipelineJobs
        .map((j) => `<div class="row"><span>${j.command} · ${new Date(j.started_at).toLocaleString("en-IN")}</span><span class="status-${j.status}">${j.status}</span></div>`)
        .join("");
    } catch (e) {
      el.innerHTML = `<div class="empty-state">unavailable</div>`;
    }
  }

  document.addEventListener("job-finished", refreshJobHistory);
  onTabActivate("pipeline", refreshJobHistory);
  document.addEventListener("DOMContentLoaded", init);
})();
