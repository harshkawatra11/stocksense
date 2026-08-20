// Claude Connection panel (Phase F2/F3/F5). Auth status, the explicit
// Authorize/Decline control (never a silent default), and the usage
// gauge -- MEASURED from local session logs, not an official quota
// (see agent/usage_tracker.py; Anthropic publishes no such API for
// Pro/Max plans). Labeled that way in the UI itself, not just in code
// comments, per the plan's explicit requirement.

(function () {
  function fmtTokens(n) {
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
    if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
    return String(n);
  }

  function bar(current, total, width = 24) {
    if (!total) return "░".repeat(width);
    const filled = Math.min(width, Math.round((current / total) * width));
    return "█".repeat(filled) + "░".repeat(width - filled);
  }

  async function refreshAuth() {
    const el = document.getElementById("claude-auth-status");
    const authorizeBtn = document.getElementById("btn-claude-authorize");
    const declineBtn = document.getElementById("btn-claude-decline");
    try {
      const status = await fetchJson("/api/claude/auth");
      el.innerHTML = `
        <div class="row"><span>logged in</span><span class="${status.logged_in ? "sev-ok" : "sev-critical"}">${status.logged_in ? "yes" : "no"}</span></div>
        <div class="row"><span>account</span><span>${status.email || "--"}</span></div>
        <div class="row"><span>plan</span><span>${status.plan || "--"}</span></div>
        <div class="row"><span>StockSense access</span><span class="${status.access_granted ? "sev-ok" : "sev-high"}">${status.access_granted ? "AUTHORIZED" : "not authorized"}</span></div>
        ${status.raw_error ? `<div class="row sev-high"><span>note</span><span>${status.raw_error}</span></div>` : ""}
      `;
      authorizeBtn.disabled = status.access_granted || !status.logged_in;
      declineBtn.disabled = !status.access_granted;
    } catch (e) {
      el.innerHTML = `<div class="empty-state">unavailable</div>`;
    }
    updateHeaderGauge();
  }

  async function authorize() {
    const statusEl = document.getElementById("status-claude-authorize");
    try {
      await postJson("/api/claude/authorize");
      statusEl.textContent = "";
      refreshAuth();
    } catch (e) {
      statusEl.textContent = `authorize failed: ${e.message}`;
      statusEl.className = "sev-critical";
    }
  }

  async function decline() {
    await postJson("/api/claude/decline");
    refreshAuth();
  }

  async function refreshUsage() {
    const el = document.getElementById("claude-usage-detail");
    try {
      const u = await fetchJson("/api/claude/usage");
      const modelMixHtml = Object.entries(u.window_7d.model_mix)
        .map(([m, pct]) => `<div class="row"><span>${m}</span><span>${(pct * 100).toFixed(0)}%</span></div>`)
        .join("") || `<div class="empty-state">no usage recorded</div>`;

      el.innerHTML = `
        <div class="usage-label">(measured from local session logs -- not an official quota)</div>
        <div class="row"><span>5h window</span><span>${bar(u.window_5h.total_tokens, u.soft_alarm_tokens_5h || u.window_5h.total_tokens || 1)}</span></div>
        <div class="row"><span></span><span>${fmtTokens(u.window_5h.total_tokens)} tok · ${u.window_5h.message_count} msgs</span></div>
        <div class="row"><span>7d window</span><span>${fmtTokens(u.window_7d.total_tokens)} tok · ${u.window_7d.message_count} msgs</span></div>
        <div class="row ${u.soft_alarm_tripped ? "sev-critical" : ""}"><span>soft alarm (5h)</span><span>${u.soft_alarm_tokens_5h ? fmtTokens(u.soft_alarm_tokens_5h) + (u.soft_alarm_tripped ? " -- TRIPPED" : "") : "not set"}</span></div>
        <div class="panel-title" style="margin-top:8px">MODEL MIX (7d)</div>
        ${modelMixHtml}
      `;
    } catch (e) {
      el.innerHTML = `<div class="empty-state">unavailable</div>`;
    }
  }

  async function saveSoftAlarm() {
    const input = document.getElementById("soft-alarm-input");
    const statusEl = document.getElementById("status-soft-alarm");
    const tokens = input.value ? parseInt(input.value, 10) : null;
    try {
      await putQuery("/api/claude/usage/soft-alarm", { tokens_5h: tokens });
      statusEl.textContent = "saved";
      statusEl.className = "sev-ok";
      refreshUsage();
    } catch (e) {
      statusEl.textContent = `failed: ${e.message}`;
      statusEl.className = "sev-critical";
    }
  }

  async function updateHeaderGauge() {
    const el = document.getElementById("usage-gauge");
    try {
      const u = await fetchJson("/api/claude/usage");
      const tripped = u.soft_alarm_tripped;
      el.textContent = `claude: ${fmtTokens(u.window_5h.total_tokens)}/5h`;
      el.className = tripped ? "sev-critical" : "";
    } catch (e) {
      el.textContent = "claude: --";
    }
  }

  function init() {
    document.getElementById("btn-claude-authorize").addEventListener("click", authorize);
    document.getElementById("btn-claude-decline").addEventListener("click", decline);
    document.getElementById("btn-save-soft-alarm").addEventListener("click", saveSoftAlarm);
    refreshAuth();
    refreshUsage();
    updateHeaderGauge();
    setInterval(updateHeaderGauge, 30000); // header gauge stays live even off this tab
  }

  onTabActivate("claude", () => { refreshAuth(); refreshUsage(); });
  document.addEventListener("DOMContentLoaded", init);
})();
