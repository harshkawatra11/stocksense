"use client";
import { useLabActions } from "@/lib/ml/actions";
import {
  evidencePct,
  EXPLORATORY_LABEL,
  FOOTER,
  headerNote,
  inr,
  rows as historyRows,
  splitPicks,
  type HistoryRow,
  type Pick,
  type PicksReply,
} from "@/lib/picks";
import { usePanel } from "@/components/of/usePanel";

const f2 = (x: number | null | undefined) => (x == null ? "–" : x.toFixed(2));

export function PickCard({ p, all, exploratory }: { p: Pick; all: Pick[]; exploratory?: boolean }) {
  const { openChart } = useLabActions();
  return (
    <article className={`pk-card${exploratory ? " expl" : ""}`} data-sym={p.sym}>
      <h3>
        <span>
          {p.sym} <small>{p.side >= 0 ? "long" : "short"}</small>
        </span>
        {exploratory && <span className="pk-label">{EXPLORATORY_LABEL}</span>}
      </h3>
      <div>
        {p.strategies.map((s) => (
          <span className="pk-chip" key={s}>
            {s}
          </span>
        ))}
      </div>
      {!exploratory && (
        <div className="pk-ev" title={`evidence ${f2(p.evidence)}`}>
          <i style={{ width: `${evidencePct(p, all)}%` }} />
        </div>
      )}
      <div>
        Entry ref {f2(p.entry_ref)} · stop {f2(p.stop)} · target {p.target == null ? "–" : f2(p.target)}
      </div>
      <div>
        {p.qty > 0 ? (
          <>
            Paper qty <b>{p.qty}</b> ({inr(p.notional_inr)}, risk {inr(p.risk_inr)})
          </>
        ) : (
          <span className="of-note">No paper size (capital unavailable)</span>
        )}
      </div>
      <div className="of-note">
        {exploratory ? "Exploratory stats" : "Best trial"}: net Sharpe at ₹5L {f2(p.net_sharpe_5l)}
        {!exploratory && (
          <>
            {" "}
            · DSR {f2(p.dsr)} · PBO {f2(p.pbo)} · {p.trades_oos} OOS trades
          </>
        )}
      </div>
      {p.reason && <div className="of-note">{p.reason}</div>}
      <button onClick={() => openChart(p.sym)} style={{ marginTop: 6 }}>
        Open chart
      </button>
    </article>
  );
}

function Track({ hist }: { hist: HistoryRow[] }) {
  return (
    <section className="st-sec" id="track">
      <h2>Track record</h2>
      {!hist.length ? (
        <div className="of-note">No past picks yet.</div>
      ) : (
        <table>
          <thead>
            <tr>
              <th>As of</th>
              <th>Symbol</th>
              <th>Status</th>
              <th>R</th>
              <th>Net %</th>
            </tr>
          </thead>
          <tbody>
            {hist.map((h, i) => (
              <tr key={`${h.asof}${h.sym}${i}`}>
                <td>{h.asof}</td>
                <td>{h.sym}</td>
                <td>{h.status}</td>
                <td className={h.r == null ? "" : h.r >= 0 ? "up" : "down"}>{f2(h.r)}</td>
                <td className={h.net_pct == null ? "" : h.net_pct >= 0 ? "up" : "down"}>{f2(h.net_pct)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

export function PicksPage() {
  const picks = usePanel<PicksReply>("/api/picks");
  const hist = usePanel<HistoryRow[] | { rows?: HistoryRow[] }>("/api/picks/history?days=30");
  const d = picks.data;
  const { gated, exploratory } = d ? splitPicks(d) : { gated: [], exploratory: [] };
  return (
    <main className="pk-page">
      <h1>Daily picks</h1>
      {picks.loading && <div className="of-note">Loading…</div>}
      {picks.error && <div className="of-note">Not available: {picks.error}</div>}
      {d && (
        <>
          <div className="pk-strip">
            <span>
              As of <b>{d.asof}</b>
            </span>
            <span>
              Strategies passing <b>{d.n_strategies_passing}</b> of {d.n_trials_total} trials
            </span>
            {headerNote(d) && <span role="note">{headerNote(d)}</span>}
          </div>
          <section className="pk-sec" id="gated">
            <h2>Gate-passing picks</h2>
            {gated.length ? (
              <div className="pk-cards">
                {gated.map((p) => (
                  <PickCard key={p.sym} p={p} all={gated} />
                ))}
              </div>
            ) : (
              <div className="of-note">None today.</div>
            )}
          </section>
          <section className="pk-sec" id="exploratory">
            <h2>Exploratory</h2>
            <div className="pk-label" style={{ display: "inline-block", marginBottom: 8 }}>
              {EXPLORATORY_LABEL}
            </div>
            <div className="pk-cards">
              {exploratory.map((p) => (
                <PickCard key={p.sym} p={p} all={exploratory} exploratory />
              ))}
            </div>
          </section>
        </>
      )}
      <Track hist={historyRows(hist.data)} />
      <p className="pk-foot">{FOOTER}</p>
    </main>
  );
}
