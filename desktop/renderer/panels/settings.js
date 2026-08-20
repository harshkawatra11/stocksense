// Settings panel (Phase F4/F5). Every core/config.py Settings field is
// editable here, grouped for scanability. Writes go to .env (see
// server/app.py's put_app_settings) -- pydantic-settings re-reads it
// fresh on every call, so a save takes effect on the NEXT pipeline
// invocation with no server restart. Secret fields (Upstox credentials)
// are shown masked and are write-only from this form: leaving one blank
// on save means "don't change it," never "clear it."

(function () {
  const SECRET_FIELDS = new Set(["upstox_api_key", "upstox_api_secret", "upstox_access_token"]);
  const BOOL_FIELDS = new Set(["use_point_in_time_universe"]);
  const LIST_FIELDS = new Set(["cost_grid_bps", "horizon_grid", "top_n_grid"]);

  const GROUPS = [
    {
      title: "DATA & UNIVERSE",
      fields: ["price_source", "return_basis", "use_point_in_time_universe",
        "min_history_days", "min_price_inr", "min_avg_daily_turnover_inr"],
    },
    {
      title: "CLAUDE ROLES (model/effort per role)",
      fields: ["planner_model", "planner_effort", "codegen_model", "codegen_effort",
        "assess_model", "assess_effort", "kundli_narrative_model", "kundli_narrative_effort"],
    },
    {
      title: "FOREMAN",
      fields: ["foreman_max_invocations_per_day"],
    },
    {
      title: "RESEARCH SWEEP GRIDS",
      fields: ["cost_grid_bps", "horizon_grid", "top_n_grid", "random_seed"],
    },
    {
      title: "UPSTOX CREDENTIALS (write-only -- blank = unchanged)",
      fields: ["upstox_api_key", "upstox_api_secret", "upstox_access_token"],
    },
  ];

  function fieldInputHtml(name, value) {
    if (BOOL_FIELDS.has(name)) {
      return `<input type="checkbox" id="setting-${name}" ${value ? "checked" : ""} />`;
    }
    if (SECRET_FIELDS.has(name)) {
      return `<input type="password" id="setting-${name}" placeholder="${value || "(not set)"}" autocomplete="off" />`;
    }
    const displayValue = LIST_FIELDS.has(name) && Array.isArray(value) ? value.join(",") : value;
    return `<input type="text" id="setting-${name}" value="${displayValue ?? ""}" />`;
  }

  async function loadSettings() {
    const container = document.getElementById("settings-form");
    try {
      const { settings } = await fetchJson("/api/settings");
      container.innerHTML = GROUPS.map((group) => `
        <div class="panel-title">${group.title}</div>
        ${group.fields.map((f) => `
          <div class="row">
            <span>${f}</span>
            <span>${fieldInputHtml(f, settings[f])}</span>
          </div>
        `).join("")}
      `).join("");
    } catch (e) {
      container.innerHTML = `<div class="empty-state">unavailable: ${e.message}</div>`;
    }
  }

  async function saveSettings() {
    const statusEl = document.getElementById("status-settings-save");
    const updates = {};
    for (const group of GROUPS) {
      for (const name of group.fields) {
        const el = document.getElementById(`setting-${name}`);
        if (!el) continue;
        if (BOOL_FIELDS.has(name)) {
          updates[name] = el.checked;
        } else if (SECRET_FIELDS.has(name)) {
          if (el.value.trim()) updates[name] = el.value.trim(); // blank = leave unchanged, don't send
        } else {
          updates[name] = el.value.trim();
        }
      }
    }
    statusEl.textContent = "saving...";
    statusEl.className = "muted";
    try {
      await putJson("/api/settings", updates);
      statusEl.textContent = "saved -- takes effect on the next run, no restart needed";
      statusEl.className = "sev-ok";
      loadSettings(); // secret fields re-render masked, confirming what actually landed
    } catch (e) {
      statusEl.textContent = `save failed: ${e.message}`;
      statusEl.className = "sev-critical";
    }
  }

  function init() {
    document.getElementById("btn-save-settings").addEventListener("click", saveSettings);
    loadSettings();
  }

  onTabActivate("settings", loadSettings);
  document.addEventListener("DOMContentLoaded", init);
})();
