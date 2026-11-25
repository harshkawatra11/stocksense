"use client";
import { useEffect, useMemo, useState } from "react";
import { API_BASE, getJSON } from "@/lib/api";
import {
  gateOf,
  pivot,
  rColor,
  tierSentence,
  tierSummary,
  type LibRow,
  type MatrixReply,
} from "@/lib/strategies";
import { usePanel } from "@/components/of/usePanel";

const f2 = (x: number | null) => (x == null ? "–" : x.toFixed(2));

export function TierVsEvidence({ rows }: { rows: LibRow[] }) {
  const stats = tierSummary(rows);
  const mx = Math.max(0.01, ...stats.map((s) => Math.abs(s.medianSharpe ?? 0)));
  return (
    <section className="st-sec" id="tiers">
      <h2>Tier list vs evidence</h2>
      <table>
        <thead>
          <tr>
            <th>Tier</th>
            <th>Strategies</th>
            <th>Tested</th>
            <th>Pass the gate</th>
            <th>Median net Sharpe at ₹5L</th>
            <th style={{ width: "30%" }} />
          </tr>
        </thead>
        <tbody>
          {stats.map((s) => (
            <tr key={s.tier} data-tier={s.tier}>
              <td>{s.tier}</td>
              <td>{s.n}</td>
              <td>{s.tested}</td>
              <td>{s.passing}</td>
              <td>{f2(s.medianSharpe)}</td>
              <td>
                {s.medianSharpe != null && (
                  <div className="st-tierbar" style={{ width: `${(Math.abs(s.medianSharpe) / mx) * 100}%`, background: s.medianSharpe < 0 ? "#d6455d" : "#2563eb" }} />
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p>{tierSentence(stats)}</p>
    </section>
  );
}

function Library({ rows }: { rows: LibRow[] }) {
  const [text, setText] = useState("");
  const [tier, setTier] = useState("all");
  const [ver, setVer] = useState("all");
  const shown = useMemo(
    () =>
      rows.filter((r) => {
        const g = gateOf(r.gated);
        if (tier !== "all" && r.tier !== tier) return false;
        if (ver === "pass" && !g.pass) return false;
        if (ver === "fail" && !(g.tested && !g.pass)) return false;
        if (ver === "untested" && g.tested) return false;
        const t = text.trim().toLowerCase();
        return !t || `${r.name} ${r.key} ${r.source} ${r.kind}`.toLowerCase().includes(t);
      }),
    [rows, text, tier, ver],
  );
  return (
    <section className="st-sec" id="library">
      <h2>Strategy library</h2>
      <div className="of-bar">
        <input placeholder="Filter by name, source, kind" value={text} onChange={(e) => setText(e.target.value)} aria-label="Filter" />
        <select value={tier} onChange={(e) => setTier(e.target.value)} aria-label="Tier">
          {["all", "F", "C", "B", "none"].map((t) => (
            <option key={t} value={t}>
              {t === "all" ? "all tiers" : t === "none" ? "no tier" : `tier ${t}`}
            </option>
          ))}
        </select>
        <select value={ver} onChange={(e) => setVer(e.target.value)} aria-label="Verdict">
          {["all", "pass", "fail", "untested"].map((t) => (
            <option key={t} value={t}>
              {t === "all" ? "any verdict" : t}
            </option>
          ))}
        </select>
        <span className="of-note">
          {shown.length} of {rows.length}
        </span>
      </div>
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Source</th>
            <th>Tier</th>
            <th>Kind</th>
            <th>TF</th>
            <th>Gate</th>
            <th>Lines</th>
            <th>Net Sharpe 5L</th>
            <th>2× costs</th>
            <th>OOS trades</th>
            <th>Positive years</th>
            <th>Eval</th>
          </tr>
        </thead>
        <tbody>
          {shown.map((r) => {
            const g = gateOf(r.gated);
            return (
              <tr key={r.key} title={r.description}>
                <td>{r.name}</td>
                <td>{r.source}</td>
                <td>{r.tier}</td>
                <td>{r.kind}</td>
                <td>{r.timeframe}</td>
                <td className={g.pass ? "up" : g.tested ? "down" : ""}>{g.verdict ?? "not tested"}</td>
                <td>{g.lines == null ? "–" : `${g.lines}/9`}</td>
                <td>{f2(g.sharpe5)}</td>
                <td>{f2(g.sharpe2x)}</td>
                <td>{g.trades ?? "–"}</td>
                <td>{g.posYears ?? "–"}</td>
                <td>{g.tested ? <a href={`${API_BASE}/api/lab/strategies?key=${encodeURIComponent(r.key)}`}>eval</a> : "–"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}

function MatrixSection() {
  const [input, setInput] = useState("");
  const [sym, setSym] = useState<string | null>(null);
  const [hits, setHits] = useState<string[]>([]);
  const m = usePanel<MatrixReply>(sym ? `/api/lab/matrix?sym=${encodeURIComponent(sym)}&tf=1d` : null);

  useEffect(() => {
    const q = input.trim();
    if (q.length < 2) return;
    const c = new AbortController();
    getJSON<{ type: string; sym?: string }[]>(`/api/search?q=${encodeURIComponent(q)}`, c.signal).then(
      (r) => !c.signal.aborted && setHits(r.flatMap((x) => (x.type === "stock" && x.sym ? [x.sym] : [])).slice(0, 8)),
      () => {},
    );
    return () => c.abort();
  }, [input]);

  const pv = m.data ? pivot(m.data.rows) : null;
  return (
    <section className="st-sec" id="matrix">
      <h2>Per-stock matrix</h2>
      <form
        className="of-bar"
        onSubmit={(e) => {
          e.preventDefault();
          if (input.trim()) setSym(input.trim().toUpperCase());
        }}
      >
        <input list="st-hits" placeholder="Symbol, e.g. RELIANCE" value={input} onChange={(e) => setInput(e.target.value)} aria-label="Symbol" />
        <datalist id="st-hits">
          {hits.map((h) => (
            <option key={h} value={h} />
          ))}
        </datalist>
        <button type="submit">Show</button>
        <span className="of-note">Average net R per trade, strategies × exits, daily bars.</span>
      </form>
      {m.loading && <div className="of-note">Loading…</div>}
      {m.error && <div className="of-note">Not available: {m.error}</div>}
      {m.data && pv && (
        <>
          <div className="st-warn" role="note">
            Multiple-testing warning: this table holds {m.data.n_tests} tests on one stock. Under no skill at all, the best of {m.data.n_tests} would
            show an annualised Sharpe near {m.data.expected_max_sharpe_null.toFixed(2)}. A bright cell here is not evidence.
          </div>
          {!pv.strats.length ? (
            <div className="of-note">No rows: not enough daily history for {m.data.sym}.</div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Strategy</th>
                  {pv.exits.map((x) => (
                    <th key={x}>{x}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {pv.strats.map((s) => (
                  <tr key={s.strat}>
                    <td>{s.name}</td>
                    {pv.exits.map((x) => {
                      const c = s.cells[x];
                      return (
                        <td key={x} data-r={c?.avgR ?? ""} style={{ background: rColor(c?.avgR, c?.n) }} title={c ? `${c.n} trades` : ""}>
                          {c && c.n ? c.avgR?.toFixed(2) : ""}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
    </section>
  );
}

export function StrategiesPage() {
  const lib = usePanel<LibRow[]>("/api/lab/strategies");
  return (
    <main className="st-page">
      <h1>Strategies</h1>
      {lib.loading && <div className="of-note">Loading…</div>}
      {lib.error && <div className="of-note">Not available: {lib.error}</div>}
      {lib.data && (
        <>
          <Library rows={lib.data} />
          <TierVsEvidence rows={lib.data} />
        </>
      )}
      <MatrixSection />
    </main>
  );
}
