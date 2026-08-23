// Daily Brief panel (Phase G5). Renders /api/brief's top-N recommendation,
// yesterday's graded revision, and cost/sizing notes. The optional
// capital input on this page is deliberately never sent to the server --
// /api/brief has no capital parameter at all (see test_server_brief.py's
// test_brief_never_accepts_or_returns_a_capital_field) -- it only scales
// the weights already returned into an indicative rupee figure, purely
// in this tab. That is the whole point of Phase G's capital-agnostic
// design invariant: the system computes weights and percentages; a
// viewer's own capital is theirs to type in or not.

(function () {
  function formatPct(w) {
    if (w === null || w === undefined) return "--";
    return `${(w * 100).toFixed(1)}%`;
  }

  function formatMove(predicted, confidence) {
    if (predicted === null || predicted === undefined) return "expected move: not yet available";
    const pct = (predicted * 100).toFixed(2);
    if (confidence === null || confidence === undefined) {
      return `expected move: ${pct}% (no confidence band yet)`;
    }
    const band = (confidence * 100).toFixed(2);
    return `expected move: ${pct}% ± ${band}%`;
  }

  function renderPicks(picks, capital) {
    if (picks.length === 0) {
      return `<div class="empty-state">No picks in the latest run.</div>`;
    }
    return picks
      .map((p) => {
        const rupees = capital ? formatMoney(capital * p.weight) : null;
        const priceLine = p.last_close !== null && p.last_close !== undefined
          ? `last close ₹${Number(p.last_close).toLocaleString("en-IN")}`
          : "last close unavailable";
        return `
          <div class="row">
            <span>${p.symbol} · rank ${p.rank ?? "?"} · ${formatPct(p.weight)}${rupees ? ` · ${rupees}` : ""}</span>
            <span class="muted">${priceLine}</span>
          </div>
          <div class="usage-label">${formatMove(p.predicted_return, p.confidence)}</div>
        `;
      })
      .join("");
  }

  function renderMoves(actions, nextRebalanceInTradingDays) {
    // Phase H4: actions reflect the LAST actual rebalance point (every
    // `horizon` trading days), never re-ranked into a fresh move list
    // every day -- that daily-churn behavior is exactly what turned
    // the user's own real account gross-positive into net-negative
    // (docs/STATUS.md's Phase E4 entry). next_rebalance_in_trading_days
    // is 0 only when today's own run IS the rebalance point.
    const dueLine = nextRebalanceInTradingDays === 0
      ? `<div class="usage-label sev-ok">Today IS a rebalance point -- the moves below are new as of today.</div>`
      : `<div class="usage-label">Next rebalance in ${nextRebalanceInTradingDays} trading day${nextRebalanceInTradingDays === 1 ? "" : "s"}. The moves below are still the ones from the last rebalance point -- nothing new to act on today, by design (re-ranking daily would churn on noise the backtest never paid for).</div>`;

    if (!actions || actions.length === 0) {
      return dueLine + `<div class="empty-state">No moves -- current holdings already match target.</div>`;
    }
    const actionClass = { enter: "sev-ok", add: "sev-ok", exit: "sev-critical", trim: "sev-critical" };
    const rows = actions
      .map((a) => {
        const cls = actionClass[a.action] || "muted";
        const costPct = (a.estimated_cost_fraction * 100).toFixed(3);
        return `<div class="row"><span>${a.action.toUpperCase()} ${a.symbol}</span><span class="${cls}">${formatPct(a.target_weight)} target · cost ${costPct}% of position</span></div>`;
      })
      .join("");
    return dueLine + rows;
  }

  function renderRevision(rows) {
    if (rows.length === 0) {
      return `<div class="empty-state">No graded predictions yet -- run RECONCILE from the RESEARCH tab, then check back once horizons have matured.</div>`;
    }
    return rows
      .map((r) => {
        const cls = r.direction_correct === true ? "sev-ok" : r.direction_correct === false ? "sev-critical" : "muted";
        const predicted = r.predicted_return !== null ? `${(r.predicted_return * 100).toFixed(2)}%` : "--";
        const actual = r.actual_return !== null ? `${(r.actual_return * 100).toFixed(2)}%` : "--";
        return `<div class="row"><span>${r.symbol} · ${r.as_of_date}</span><span class="${cls}">predicted ${predicted} → actual ${actual}</span></div>`;
      })
      .join("");
  }

  async function refreshBrief() {
    const horizon = document.getElementById("brief-horizon").value || 10;
    const capitalRaw = document.getElementById("brief-capital").value;
    const capital = capitalRaw ? Number(capitalRaw) : null;

    const statusEl = document.getElementById("brief-status");
    const picksEl = document.getElementById("brief-picks");
    const movesEl = document.getElementById("brief-moves");
    const revisionEl = document.getElementById("brief-revision");
    const notesEl = document.getElementById("brief-notes");

    statusEl.textContent = "loading...";
    statusEl.className = "muted";

    try {
      const body = await fetchJson(`/api/brief?horizon=${encodeURIComponent(horizon)}`);

      if (body.status === "no_live_model") {
        statusEl.textContent = `No live model yet for horizon=${horizon}. Train and promote one from the RESEARCH tab first.`;
        statusEl.className = "sev-critical";
        picksEl.innerHTML = `<div class="empty-state">Nothing to show yet.</div>`;
        movesEl.innerHTML = "";
        revisionEl.innerHTML = "";
        notesEl.innerHTML = "";
        return;
      }
      if (body.status === "no_predictions") {
        statusEl.textContent = `A live model exists (${body.model_id.slice(0, 12)}...) but nothing has been recorded yet. Run RECONCILE from the RESEARCH tab.`;
        statusEl.className = "sev-critical";
        picksEl.innerHTML = `<div class="empty-state">Nothing recorded yet.</div>`;
        movesEl.innerHTML = "";
        revisionEl.innerHTML = "";
        notesEl.innerHTML = "";
        return;
      }

      statusEl.textContent = `As of ${body.as_of_date} · model ${body.model_id.slice(0, 12)}... · horizon ${body.horizon_bars}b`;
      statusEl.className = "muted";

      picksEl.innerHTML = renderPicks(body.picks, capital);
      movesEl.innerHTML = renderMoves(body.actions, body.next_rebalance_in_trading_days);
      revisionEl.innerHTML = renderRevision(body.yesterdays_revision);

      const minCapLine = body.min_capital_for_full_positions_inr !== null
        ? `Minimum capital for every pick to hold at least 1 whole share: ${formatMoney(body.min_capital_for_full_positions_inr)}. Below this, at least one pick would round to zero shares.`
        : `Minimum-capital figure unavailable (no recent price data for these symbols).`;
      notesEl.innerHTML = `
        <div class="usage-label">${minCapLine}</div>
        <div class="usage-label">Round-trip cost for this holding period (equity delivery): ${body.round_trip_cost_bps_equity_delivery.toFixed(2)} bps -- measured EXACTLY the same at any position size (zero brokerage on delivery at discount brokers; every other charge scales with trade value). There is no "minimum ticket size" to clear costs on this track.</div>
      `;
    } catch (e) {
      statusEl.textContent = `could not load: ${e.message}`;
      statusEl.className = "sev-critical";
    }
  }

  function init() {
    document.getElementById("btn-brief-refresh").addEventListener("click", refreshBrief);
  }

  onTabActivate("brief", refreshBrief);
  document.addEventListener("DOMContentLoaded", init);
})();
