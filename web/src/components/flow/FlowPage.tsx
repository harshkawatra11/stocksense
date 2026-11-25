"use client";
// Money Flow page (/flow): where money moved, descriptively. Never a forecast.
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { usePoll } from "@/lib/api";
import { useLabActions } from "@/lib/ml/actions";
import { cls, pc, rf, rs, rsg } from "@/lib/ml/format";
import { COLORS } from "@/lib/ml/sector/horizon";
import { followTick, inPreopenWindow, preopenView, scatterAxes, scatterPoints, scatterXY, sparkPoints, treemap } from "@/lib/flow/charts";
import { QUADS, quadColor, quadStyle } from "@/lib/flow/quad";
import { COLUMNS, filterStocks, gammaBadge, sortRows, zeroGammaDistancePct, type SortKey } from "@/lib/flow/table";
import {
  rowsOf,
  type Deal,
  type FlowMarket,
  type FlowStock,
  type IndexGamma,
  type OiRow,
  type ParticipantDay,
  type Preopen,
  type SectorFlow,
  type Wrapped,
} from "@/lib/flow/types";

export const DESCRIPTIVE_FOOT =
  "Descriptive: shows where money moved; whether it predicts returns is tested on the Strategies page (m-flow).";

export function Foot() {
  return <div className="fl-foot">{DESCRIPTIVE_FOOT}</div>;
}

const num = (v: number | null | undefined, d = 2): string => (v == null || !isFinite(v) ? "–" : v.toFixed(d));

export function QuadChip({ label }: { label: string | null | undefined }) {
  const q = quadStyle(label);
  if (!q) return <span className="muted">–</span>;
  return (
    <span className="fl-chip" data-quad={q.label} style={{ color: q.fg, background: q.bg }}>
      {q.label}
    </span>
  );
}

function CashCard({ title, p }: { title: string; p: FlowMarket["fii"] | undefined }) {
  return (
    <div className="fl-card" id={"c-" + title.toLowerCase()}>
      <div className="fl-h">{title} cash</div>
      <div className={"fl-big mono " + cls(p?.cash_net)}>{p?.cash_net == null ? "–" : rsg(p.cash_net, 0) + " cr"}</div>
      <div className="fl-sub">today</div>
      <div className={"fl-mid mono " + cls(p?.cash_net_5d)}>{p?.cash_net_5d == null ? "–" : rsg(p.cash_net_5d, 0) + " cr"}</div>
      <div className="fl-sub">last 5 sessions</div>
      <Foot />
    </div>
  );
}

function Spark({ values, color }: { values: (number | null)[]; color: string }) {
  return (
    <svg width="90" height="22" viewBox="0 0 90 22" className="fl-spark">
      <polyline fill="none" stroke={color} strokeWidth="1.5" points={sparkPoints(values, 90, 22)} />
    </svg>
  );
}

function RatioCard({ rows, m }: { rows: ParticipantDay[]; m: FlowMarket | null }) {
  const items: [string, keyof ParticipantDay, FlowMarket["fii"] | undefined, string][] = [
    ["FII", "fii_ratio", m?.fii, "#2563eb"],
    ["DII", "dii_ratio", m?.dii, "#22a05a"],
    ["Pro", "pro_ratio", m?.pro, "#c27c0e"],
    ["Client", "client_ratio", m?.client, "#6b7487"],
  ];
  return (
    <div className="fl-card" id="c-ratios">
      <div className="fl-h">Index-futures long ratio · 60 days</div>
      {items.map(([name, key, p, col]) => (
        <div className="fl-rrow" key={name}>
          <span className="fl-rn">{name}</span>
          <span className="mono">{p?.idx_fut_ratio == null ? "–" : (p.idx_fut_ratio * 100).toFixed(0) + "%"}</span>
          <Spark values={rows.map((r) => r[key] as number | null)} color={col} />
        </div>
      ))}
      <Foot />
    </div>
  );
}

export function GammaCard({ und, g }: { und: string; g: IndexGamma | null | undefined }) {
  const b = gammaBadge(g);
  const dist = zeroGammaDistancePct(g);
  return (
    <div className="fl-card" id={"g-" + und}>
      <div className="fl-h">{und} gamma</div>
      <span className={"fl-badge " + b.kind} data-kind={b.kind}>{b.text}</span>
      {g && (
        <div className="fl-kv mono">
          <span>Spot</span><b>{rf(g.spot, 0)}</b>
          <span>Zero-gamma</span><b>{rf(g.zero_gamma, 0)}{dist != null && <i className={cls(dist)}> ({pc(dist, 1)})</i>}</b>
          <span>Call wall</span><b>{rf(g.call_wall, 0)}</b>
          <span>Put wall</span><b>{rf(g.put_wall, 0)}</b>
          <span>Max pain</span><b>{rf(g.max_pain, 0)}</b>
          <span>PCR (OI)</span><b>{num(g.pcr_oi)}</b>
        </div>
      )}
      <Foot />
    </div>
  );
}

function VixCard({ m }: { m: FlowMarket | null }) {
  return (
    <div className="fl-card" id="c-vix">
      <div className="fl-h">VIX and breadth</div>
      <div className="fl-big mono">{m?.vix ? num(m.vix.close) : "–"}</div>
      <div className={"fl-sub " + cls(m?.vix?.chg1)}>{m?.vix?.chg1 != null ? `${m.vix.chg1 > 0 ? "+" : ""}${m.vix.chg1.toFixed(2)} on the day` : "India VIX"}</div>
      <div className="fl-mid mono">
        <span className="up">{m?.breadth.adv ?? "–"}</span> / <span className="down">{m?.breadth.dec ?? "–"}</span>
      </div>
      <div className="fl-sub">F&amp;O stocks up / down</div>
      <Foot />
    </div>
  );
}

function useIstWindow(): boolean {
  const [on, setOn] = useState(() => inPreopenWindow(new Date()));
  useEffect(() => {
    const id = setInterval(() => setOn(inPreopenWindow(new Date())), 30000);
    return () => clearInterval(id);
  }, []);
  return on;
}

export function PreopenBoard({ p }: { p: Preopen | null }) {
  const v = preopenView(p);
  return (
    <div className="fl-card fl-wide" id="c-preopen">
      <div className="fl-h">Pre-open board</div>
      {v.kind === "before" && <div className="fl-empty">The board opens at 09:00. Nothing to show yet.</div>}
      {v.kind === "live" && (
        <>
          <div className="fl-sub">09:00–09:15 · live indicative open, largest buy/sell imbalance first</div>
          {v.rows.length === 0 ? (
            <div className="fl-empty">No rows yet.</div>
          ) : (
            <table className="fl-tbl" id="pre-live">
              <thead><tr><th>Symbol</th><th>IEP gap %</th><th>Imbalance</th><th>Total qty</th></tr></thead>
              <tbody>
                {v.rows.slice(0, 25).map((r) => (
                  <tr key={r.symbol}>
                    <td className="fl-sym">{r.symbol}</td>
                    <td className={cls(r.iep_gap_pct)}>{pc(r.iep_gap_pct)}</td>
                    <td>
                      <span className="fl-imb"><i style={{ width: Math.min(100, Math.abs(r.imb ?? 0) * 100) + "%", background: (r.imb ?? 0) >= 0 ? "var(--up)" : "var(--down)" }} /></span>{" "}
                      <span className="mono">{r.imb == null ? "–" : (r.imb > 0 ? "+" : "") + r.imb.toFixed(2)}</span>
                    </td>
                    <td className="mono">{rf(r.total_qty)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
      {v.kind === "after" && (
        <>
          <div className="fl-sub">Pre-open versus the first 15 minutes · ✓ the price kept the direction of the indicative gap</div>
          {v.rows.length === 0 ? (
            <div className="fl-empty">No pre-open rows were captured today.</div>
          ) : (
            <table className="fl-tbl" id="pre-after">
              <thead><tr><th>Symbol</th><th>IEP gap %</th><th>Imbalance</th><th>Follow-through</th></tr></thead>
              <tbody>
                {v.rows.slice(0, 25).map((r) => (
                  <tr key={r.symbol}>
                    <td className="fl-sym">{r.symbol}</td>
                    <td className={cls(r.iep_gap_pct)}>{pc(r.iep_gap_pct)}</td>
                    <td className="mono">{r.imb == null ? "–" : (r.imb > 0 ? "+" : "") + r.imb.toFixed(2)}</td>
                    <td className={"fl-tick " + (r.follow_through === true ? "up" : r.follow_through === false ? "down" : "muted")}>{followTick(r.follow_through)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
      <Foot />
    </div>
  );
}

export function FlowTable({ rows, onOpen }: { rows: FlowStock[]; onOpen: (sym: string) => void }) {
  const [key, setKey] = useState<SortKey>("flow_score");
  const [dir, setDir] = useState<"asc" | "desc">("desc");
  const [quad, setQuad] = useState("all");
  const [sector, setSector] = useState("all");
  const sectors = useMemo(() => Array.from(new Set(rows.map((r) => r.sector))).sort(), [rows]);
  const view = useMemo(() => sortRows(filterStocks(rows, quad, sector), key, dir), [rows, quad, sector, key, dir]);
  const head = (k: SortKey, text?: boolean) => {
    if (k === key) setDir(dir === "desc" ? "asc" : "desc");
    else {
      setKey(k);
      setDir(text ? "asc" : "desc");
    }
  };
  return (
    <div className="fl-card fl-wide" id="c-flow">
      <div className="fl-hrow">
        <div className="fl-h">Flow table · every F&amp;O stock</div>
        <span className="fl-filters">
          <label>OI quad <select id="fquad" value={quad} onChange={(e) => setQuad(e.target.value)}>
            <option value="all">all</option>
            {QUADS.map((q) => <option key={q.label} value={q.label}>{q.label}</option>)}
          </select></label>
          <label>Sector <select id="fsector" value={sector} onChange={(e) => setSector(e.target.value)}>
            <option value="all">all</option>
            {sectors.map((s) => <option key={s} value={s}>{s}</option>)}
          </select></label>
        </span>
      </div>
      <div className="fl-scroll">
        <table className="fl-tbl" id="flowtbl">
          <thead>
            <tr>
              {COLUMNS.map((c) => (
                <th key={c.key} onClick={() => head(c.key, c.text)} data-key={c.key} aria-sort={c.key === key ? (dir === "asc" ? "ascending" : "descending") : "none"} className={c.key === key ? "on" : ""}>
                  {c.label}{c.key === key ? (dir === "asc" ? " ▲" : " ▼") : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {view.map((r) => (
              <tr key={r.sym} onClick={() => onOpen(r.sym)} data-sym={r.sym}>
                <td className="fl-sym">{r.sym}</td>
                <td className="muted">{r.sector}</td>
                <td className="mono">{rs(r.ltp)}</td>
                <td className={"mono " + cls(r.pct)}>{pc(r.pct)}</td>
                <td className="mono">{num(r.disp)}</td>
                <td className="mono">{num(r.rvol)}</td>
                <td><QuadChip label={r.oi_quad_label} /></td>
                <td className={"mono " + cls(r.oi_chg_pct)}>{pc(r.oi_chg_pct, 1)}</td>
                <td className="mono">{num(r.deliv_z20_prev)}</td>
                <td className="mono">{num(r.depth_imb5)}</td>
                <td className="mono">{num(r.buy_sell_ratio)}</td>
                <td className="mono">{num(r.cvd_slope_30m)}</td>
                <td className="mono"><b>{num(r.flow_score, 3)}</b></td>
              </tr>
            ))}
            {view.length === 0 && <tr><td colSpan={COLUMNS.length} className="fl-empty">No stock matches (or the tape is not running).</td></tr>}
          </tbody>
        </table>
      </div>
      <Foot />
    </div>
  );
}

export function SectorGrid({ rows }: { rows: SectorFlow[] }) {
  const rects = useMemo(() => treemap(rows.map((r) => r.turnover_share ?? 0)), [rows]);
  return (
    <div className="fl-card fl-wide" id="c-sectors">
      <div className="fl-h">Sector flow · area = share of turnover, colour = flow direction</div>
      {rows.length === 0 ? (
        <div className="fl-empty">No sector data yet.</div>
      ) : (
        <div className="fl-tree" id="sectree">
          {rects.map((t) => {
            const s = rows[t.i];
            return (
              <div key={s.name} className="fl-tile" data-sector={s.name} style={{ left: t.x + "%", top: t.y + "%", width: t.w + "%", height: t.h + "%" }}>
                <div className="fl-tin" style={{ background: COLORS((s.flow_dir ?? 0) * 3) }}>
                  <b>{s.name}</b>
                  <span className="mono">{s.flow_dir == null ? "–" : (s.flow_dir > 0 ? "+" : "") + s.flow_dir.toFixed(2)}</span>
                  <span className="fl-rot" title={`rotation: ${s.rotation}`}>{s.rotation === "in" ? "↑ in" : s.rotation === "out" ? "↓ out" : ""}</span>
                  <small>{s.up} up · {s.down} down{Object.entries(s.oi_quads).map(([q, n]) => ` · ${n} ${q}`).join("")}</small>
                </div>
              </div>
            );
          })}
        </div>
      )}
      <Foot />
    </div>
  );
}

export function OiScatter({ rows }: { rows: OiRow[] }) {
  const pts = useMemo(() => scatterPoints(rows), [rows]);
  const ax = useMemo(() => scatterAxes(pts), [pts]);
  const [hover, setHover] = useState<string | null>(null);
  const box = { W: 560, H: 340, pad: 28 };
  const [cx, cy] = scatterXY({ x: 0, y: 0 }, ax, box);
  const corner: [string, number, number, "start" | "end"][] = [
    ["long buildup", box.W - box.pad - 4, box.pad + 10, "end"],
    ["short covering", box.W - box.pad - 4, box.H - box.pad - 4, "end"],
    ["long unwinding", box.pad + 4, box.H - box.pad - 4, "start"],
    ["short buildup", box.pad + 4, box.pad + 10, "start"],
  ];
  return (
    <div className="fl-card" id="c-scatter">
      <div className="fl-hrow"><div className="fl-h">OI quadrants · price Δ% (x) against OI Δ% (y)</div><span className="mono fl-hov">{hover ?? ""}</span></div>
      <svg viewBox={`0 0 ${box.W} ${box.H}`} className="fl-scat" role="img" aria-label="OI quadrant scatter">
        <line x1={box.pad} x2={box.W - box.pad} y1={cy} y2={cy} stroke="#c8ced8" />
        <line x1={cx} x2={cx} y1={box.pad} y2={box.H - box.pad} stroke="#c8ced8" />
        {corner.map(([l, x, y, a]) => (
          <text key={l} x={x} y={y} textAnchor={a} fontSize="10" fontWeight="700" fill={quadColor(l)}>{l}</text>
        ))}
        {pts.map((p) => {
          const [x, y] = scatterXY(p, ax, box);
          return (
            <circle key={p.sym} className="fl-dot" cx={x} cy={y} r="3.6" fill={quadColor(p.label)} fillOpacity=".8" onMouseEnter={() => setHover(`${p.sym}  ${pc(p.x)} price · ${pc(p.y, 1)} OI`)} onMouseLeave={() => setHover(null)}>
              <title>{p.sym}</title>
            </circle>
          );
        })}
        <text x={box.W - box.pad} y={cy - 4} textAnchor="end" fontSize="9" fill="#8a93a6">+{ax.xmax}%</text>
        <text x={box.pad} y={cy - 4} fontSize="9" fill="#8a93a6">-{ax.xmax}%</text>
        <text x={cx + 4} y={box.pad - 2} fontSize="9" fill="#8a93a6">OI +{ax.ymax}%</text>
        <text x={cx + 4} y={box.H - box.pad + 12} fontSize="9" fill="#8a93a6">OI -{ax.ymax}%</text>
      </svg>
      <Foot />
    </div>
  );
}

export function DealsList({ rows }: { rows: Deal[] }) {
  return (
    <div className="fl-card" id="c-deals">
      <div className="fl-h">Bulk and block deals · last 5 sessions</div>
      {rows.length === 0 ? (
        <div className="fl-empty">No deals in the window.</div>
      ) : (
        <div className="fl-scroll" style={{ maxHeight: 300 }}>
          <table className="fl-tbl" id="deals">
            <thead><tr><th>Date</th><th>Symbol</th><th>Buy cr</th><th>Sell cr</th><th>Net cr</th><th>Deals</th><th>Top client</th></tr></thead>
            <tbody>
              {rows.map((d) => (
                <tr key={d.date + d.symbol} className={d.flagged ? "fl-flag" : ""}>
                  <td className="mono">{d.date}</td>
                  <td className="fl-sym">{d.symbol}</td>
                  <td className="mono">{rf(d.buy_value_cr, 1)}</td>
                  <td className="mono">{rf(d.sell_value_cr, 1)}</td>
                  <td className={"mono " + cls(d.net_value_cr)}>{rsg(d.net_value_cr, 1)}</td>
                  <td className="mono">{d.n_deals}</td>
                  <td>{d.top_client ?? "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <Foot />
    </div>
  );
}

export function FlowPage() {
  const { openChart } = useLabActions();
  const inWindow = useIstWindow();
  const market = usePoll<FlowMarket>("/api/flow/market", 30000);
  const parts = usePoll<Wrapped<ParticipantDay>>("/api/flow/participants?days=60", 300000);
  const stocks = usePoll<Wrapped<FlowStock>>("/api/flow/stocks", 15000);
  const pre = usePoll<Preopen>("/api/flow/preopen", inWindow ? 15000 : 120000);
  const sectors = usePoll<Wrapped<SectorFlow>>("/api/flow/sectors?range=today", 30000);
  const oi = usePoll<Wrapped<OiRow>>("/api/flow/oi?scope=stocks", 60000);
  const deals = usePoll<Wrapped<Deal>>("/api/flow/deals?days=5", 300000);
  const m = market.data;
  return (
    <div className="flow-page">
      <div className="fl-title">
        <h1>Money Flow</h1>
        <div className="fl-sub">
          Where money moved: participants, option positioning, pre-open, F&amp;O flow and sector rotation.
          {m?.asof ? ` End-of-day data as of ${m.asof}.` : ""} <Link href="/flow/options/?und=NIFTY">Options positioning →</Link>
        </div>
      </div>
      <div className="fl-strip" id="strip">
        <CashCard title="FII" p={m?.fii} />
        <CashCard title="DII" p={m?.dii} />
        <RatioCard rows={rowsOf(parts.data)} m={m} />
        <GammaCard und="NIFTY" g={m?.indices?.NIFTY} />
        <GammaCard und="BANKNIFTY" g={m?.indices?.BANKNIFTY} />
        <VixCard m={m} />
      </div>
      <div className="fl-body">
        <PreopenBoard p={pre.data} />
        <FlowTable rows={rowsOf(stocks.data)} onOpen={openChart} />
        <SectorGrid rows={rowsOf(sectors.data)} />
        <div className="fl-two">
          <OiScatter rows={rowsOf(oi.data)} />
          <DealsList rows={rowsOf(deals.data)} />
        </div>
      </div>
    </div>
  );
}
