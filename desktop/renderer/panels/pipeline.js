// Data & Backfills panel (Phase F5). Triggers backfill-nse-archive,
// backfill-corporate-actions, backfill-intraday, index-corpus via F1's
// job API. Progress comes from the job console's live stream, which
// parses the "PROGRESS: n/total (pct%)" lines cli/main.py's backfill
// loops print -- job-console.js renders that as an in-place progress
// bar (Phase G/A2; this panel used to define its own progressBar()
// helper and never call it, since the parsing needs to live where the
// stream is actually consumed).

(function () {
  function today() {
    return new Date().toISOString().slice(0, 10);
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
      // Phase G/A1: rows are clickable -- re-opens that job's persisted
      // log (live buffer if still running, the on-disk file if it's
      // finished) in the console drawer. Before this, a job's output was
      // only ever visible while it was the MOST RECENTLY triggered one.
      el.innerHTML = pipelineJobs
        .map((j) => `<div class="row row-clickable" data-job-id="${j.job_id}" data-command="${j.command}"><span>${j.command} · ${new Date(j.started_at).toLocaleString("en-IN")}</span><span class="status-${j.status}">${j.status}</span></div>`)
        .join("");
      el.querySelectorAll(".row-clickable").forEach((row) => {
        row.addEventListener("click", () => viewJobLog(row.dataset.jobId, row.dataset.command));
      });
    } catch (e) {
      el.innerHTML = `<div class="empty-state">${e.message}</div>`;
    }
  }

  document.addEventListener("job-finished", refreshJobHistory);
  onTabActivate("pipeline", refreshJobHistory);
  document.addEventListener("DOMContentLoaded", init);
})();
