// Foreman panel (Phase F5) -- MANUAL CONTROL ONLY. This exposes the
// knobs (goal text, which model/effort the planner and self-assessment
// use) a future auto-switching policy would eventually drive; it does
// not decide anything on its own. That auto-policy (Opus<->Sonnet
// switching, effort scaling, plan/auto mode) is explicitly separate,
// later work per the plan -- this panel is the "manual gearbox," not
// the automatic transmission.

(function () {
  const MODELS = ["opus", "sonnet", "haiku"];
  const EFFORTS = ["low", "medium", "high", "xhigh", "max"];

  function optionsHtml(values, current) {
    return values.map((v) => `<option value="${v}"${v === current ? " selected" : ""}>${v}</option>`).join("");
  }

  async function loadModelSettings() {
    const { settings } = await fetchJson("/api/settings");
    const plannerModel = document.getElementById("foreman-planner-model");
    const plannerEffort = document.getElementById("foreman-planner-effort");
    const assessModel = document.getElementById("foreman-assess-model");
    const assessEffort = document.getElementById("foreman-assess-effort");
    plannerModel.innerHTML = optionsHtml(MODELS, settings.planner_model || "opus");
    plannerEffort.innerHTML = optionsHtml(EFFORTS, settings.planner_effort || "low");
    assessModel.innerHTML = optionsHtml(MODELS, settings.assess_model || "opus");
    assessEffort.innerHTML = optionsHtml(EFFORTS, settings.assess_effort || "low");
  }

  async function saveModelSettings() {
    const statusEl = document.getElementById("status-foreman-settings");
    try {
      await putJson("/api/settings", {
        planner_model: document.getElementById("foreman-planner-model").value,
        planner_effort: document.getElementById("foreman-planner-effort").value,
        assess_model: document.getElementById("foreman-assess-model").value,
        assess_effort: document.getElementById("foreman-assess-effort").value,
      });
      statusEl.textContent = "saved -- takes effect on the next run";
      statusEl.className = "sev-ok";
    } catch (e) {
      statusEl.textContent = `save failed: ${e.message}`;
      statusEl.className = "sev-critical";
    }
  }

  async function runForemanGoal() {
    const goal = document.getElementById("foreman-goal-input").value.trim();
    const statusEl = document.getElementById("status-foreman-run");
    if (!goal) {
      statusEl.textContent = "enter a goal first";
      statusEl.className = "sev-high";
      return;
    }
    statusEl.textContent = "starting...";
    statusEl.className = "muted";
    try {
      await triggerJob("foreman-run", { goal });
      statusEl.textContent = "running -- see job console below";
      statusEl.className = "status-running";
    } catch (e) {
      statusEl.textContent = `failed to start: ${e.message}`;
      statusEl.className = "sev-critical";
    }
  }

  async function runAssess() {
    const statusEl = document.getElementById("status-foreman-assess");
    statusEl.textContent = "starting...";
    statusEl.className = "muted";
    try {
      await triggerJob("foreman-assess", {});
      statusEl.textContent = "running -- see job console below";
      statusEl.className = "status-running";
    } catch (e) {
      statusEl.textContent = `failed: ${e.message}`;
      statusEl.className = "sev-critical";
    }
  }

  async function refreshStatus() {
    const el = document.getElementById("foreman-status");
    try {
      const { jobs } = await fetchJson("/api/jobs?limit=15");
      const foremanJobs = jobs.filter((j) => ["foreman-run", "foreman-assess"].includes(j.command));
      if (foremanJobs.length === 0) {
        el.innerHTML = `<div class="empty-state">No Foreman runs yet.</div>`;
        return;
      }
      // Phase G/A1: clickable rows re-open that run's persisted log --
      // same treatment as pipeline.js's job history.
      el.innerHTML = foremanJobs
        .map((j) => `<div class="row row-clickable" data-job-id="${j.job_id}" data-command="${j.command}"><span>${j.command} · ${new Date(j.started_at).toLocaleString("en-IN")}</span><span class="status-${j.status}">${j.status}</span></div>`)
        .join("");
      el.querySelectorAll(".row-clickable").forEach((row) => {
        row.addEventListener("click", () => viewJobLog(row.dataset.jobId, row.dataset.command));
      });
    } catch (e) {
      el.innerHTML = `<div class="empty-state">${e.message}</div>`;
    }
  }

  function init() {
    loadModelSettings();
    document.getElementById("btn-save-foreman-settings").addEventListener("click", saveModelSettings);
    document.getElementById("btn-foreman-run").addEventListener("click", runForemanGoal);
    document.getElementById("btn-foreman-assess").addEventListener("click", runAssess);
    refreshStatus();
  }

  document.addEventListener("job-finished", refreshStatus);
  onTabActivate("foreman", () => { loadModelSettings(); refreshStatus(); });
  document.addEventListener("DOMContentLoaded", init);
})();
