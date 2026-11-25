"use client";
// Swing Lab: port of Market Lab's lab.html with StockSense additions (filters, Gate column, multiple-testing note).
import { useEffect, useMemo, useRef, useState } from "react";
import { usePoll } from "@/lib/api";
import { useLabActions } from "@/lib/ml/actions";
import { sg } from "@/lib/ml/format";
import {
  barMax,
  barWidth,
  boardRows,
  filterRows,
  gateKind,
  gateText,
  multipleTestingNote,
  SOURCE_OPTIONS,
  TIER_OPTIONS,
  type SourceFilter,
  type TierFilter,
} from "@/lib/ml/lab/board";
import { drawEquity, hoverIndex, tooltipText, type EquityData } from "@/lib/ml/lab/equity";
import type { RegistryRow, SwingCandidate, SwingData } from "@/lib/ml/lab/types";

const IST = "Asia/Kolkata";
const tm = (t: number | null | undefined): string =>
  t == null ? "–" : new Date(t * 1000).toLocaleDateString("en-IN", { timeZone: IST, day: "2-digit", month: "short" });
const n1 = (v: number): number => +v.toFixed(1);

function GateChip({ gated, exploratory }: { gated: string | null | undefined; exploratory?: boolean }) {
  const k = gateKind(gated);
  const style =
    k === "pass"
      ? { background: "#e8f6ee", color: "#17603a" }
      : k === "fail"
        ? { background: "#fdecef", color: "#9b1c2e" }
        : { background: "#f1f3f6", color: "#6b7385" };
  return (
    <span className="pill" style={style} data-gate={k}>
      {k === "none" ? "NOT YET TESTED" : `GATE ${gateText(gated)}`}
      {exploratory ? " · EXPLORATORY" : ""}
    </span>
  );
}

function Verdict({ c, asof }: { c: SwingCandidate; asof: string }) {
  const m = c.metrics;
  if (m && m["CAGR%"] != null && c.ew && c.nifty) {
    return (
      <>
        <b>Your backtest on your own data:</b> {c.label} made {m["CAGR%"]}% a year with a worst drawdown of {m["maxDD%"]}%, against{" "}
        {c.ew["CAGR%"]}% for the equal-weight universe and {c.nifty["CAGR%"]}% for Nifty, after delivery costs. A backtest is not a
        promise: survivorship bias (today&apos;s Nifty 500 list is used for the past) flatters every long test, and live results
        will differ. <GateChip gated={c.gated} exploratory={c.exploratory} />
      </>
    );
  }
  return (
    <>
      <b>Exploratory candidate</b> (as of {asof}): {c.label}.{" "}
      {m && m.n != null
        ? `${m.n} trades, ${m.win}% won, average ${n1(m.avgR ?? 0)}R, ${n1(m.net_pct ?? 0)}% total net after StockSense costs. `
        : "No trades yet. "}
      A backtest is not a promise: survivorship bias flatters every long test. <GateChip gated={c.gated} exploratory={c.exploratory} />
    </>
  );
}

function Cmp({ c }: { c: SwingCandidate }) {
  const m = c.metrics;
  if (!m || m["CAGR%"] == null || !c.ew || !c.nifty) return null;
  const yrs = Object.keys(m["years%"] ?? {});
  const row = (n: string, x: typeof c.ew) =>
    x && (
      <tr key={n}>
        <td>{n}</td>
        <td>{x["CAGR%"]}%</td>
        <td className="down">{x["maxDD%"]}%</td>
        <td>{x.Sharpe}</td>
        {yrs.map((y) => {
          const v = x["years%"][y];
          return (
            <td key={y} className={v != null && v >= 0 ? "up" : "down"}>
              {v == null ? "–" : sg(v) + "%"}
            </td>
          );
        })}
      </tr>
    );
  return (
    <div className="tw">
      <table className="cmp" id="swcmp">
        <thead>
          <tr>
            <th></th>
            <th>CAGR</th>
            <th>Max DD</th>
            <th>Sharpe</th>
            {yrs.map((y) => (
              <th key={y}>
                {y}
                {y === "2022" ? "*" : y === "2026" ? " YTD" : ""}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {row("Swing system", m as typeof c.ew)}
          {row("Equal-weight universe", c.ew)}
          {row("Nifty 50", c.nifty)}
        </tbody>
      </table>
    </div>
  );
}

function EquityChart({ c }: { c: SwingCandidate }) {
  const wrap = useRef<HTMLDivElement>(null);
  const cv = useRef<HTMLCanvasElement>(null);
  const tip = useRef<HTMLDivElement>(null);
  const hover = useRef<number | null>(null);
  const geom = useRef<[number, number, number, number] | null>(null);
  const data = useRef<EquityData | null>(null);
  const has = !!(c.curve && c.curve_ew && c.curve_nifty && c.curve.length > 1);

  useEffect(() => {
    data.current = has
      ? { lines: [c.curve!, c.curve_ew!, c.curve_nifty!], regime: c.regime ?? [], t0: performance.now() }
      : null;
  }, [c, has]);

  useEffect(() => {
    if (!has) return;
    let raf = 0;
    const frame = () => {
      const w = wrap.current;
      const el = cv.current;
      if (w && el) geom.current = drawEquity(el, w.clientWidth, w.clientHeight, window.devicePixelRatio || 1, data.current, hover.current, performance.now());
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [has]);

  if (!has) return <div className="note">The equity curve is not part of this scan build; the table above is the summary.</div>;

  const onMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const d = data.current;
    const g = geom.current;
    const t = tip.current;
    if (!d || !g || !t) return;
    const r = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - r.left;
    const i = hoverIndex(mx, g[2], g[3]);
    hover.current = i;
    t.style.display = "block";
    t.textContent = tooltipText(d, i);
    t.style.left = Math.max(0, Math.min(r.width - t.offsetWidth - 4, mx + 10)) + "px";
    t.style.top = "4px";
  };
  const onLeave = () => {
    hover.current = null;
    if (tip.current) tip.current.style.display = "none";
  };
  return (
    <div id="swc" ref={wrap}>
      <canvas id="swcv" ref={cv} onMouseMove={onMove} onMouseLeave={onLeave} />
      <div id="swtip" ref={tip} />
    </div>
  );
}

function Why({ c }: { c: SwingCandidate }) {
  if (!c.regime || !c.skipped || !c.metrics) return null;
  const off = c.regime.filter((r) => !r[1]).length;
  return (
    <div className="why" id="why">
      <b>Why the line goes flat:</b> those are cash periods, not missing data. When Nifty closes under its 200-day average the
      system stops buying and lets open trades finish, so the account sits in cash (grey bands: {off} of {c.regime.length} days,
      whenever Nifty was under its 200-day average). The {c.skipped.n} VCP breakouts it skipped in those periods averaged only{" "}
      {sg(c.skipped.avg)}% each after costs. *2022 = Nov–Dec only.
    </div>
  );
}

export function LabSection() {
  const { labShow } = useLabActions();
  const swing = usePoll<SwingData>("/api/lab/swing", 30000);
  const reg = usePoll<RegistryRow[]>("/api/lab/strategies", 300000);
  const SWD = swing.data && !swing.data.error && swing.data.candidate ? swing.data : null;
  const [source, setSource] = useState<SourceFilter>("all");
  const [tier, setTier] = useState<TierFilter>("all");
  const root = useRef<HTMLDivElement>(null);
  const prev = useRef("");

  // Market Lab flashed the three radar lists whenever the live block changed after the first load.
  const liveKey = SWD?.live ? JSON.stringify(SWD.live) : "";
  useEffect(() => {
    const first = !prev.current;
    if (!liveKey || liveKey === prev.current) return;
    prev.current = liveKey;
    if (first) return;
    const lists = Array.from(root.current?.querySelectorAll(".list") ?? []);
    lists.forEach((l) => l.classList.add("flash"));
    const id = setTimeout(() => lists.forEach((l) => l.classList.remove("flash")), 1000);
    return () => clearTimeout(id);
  }, [liveKey]);

  const registry = useMemo(
    () => (Array.isArray(reg.data) ? new Map(reg.data.map((r) => [r.key, r])) : undefined),
    [reg.data],
  );
  const all = useMemo(() => (SWD ? boardRows(SWD.strategies, registry) : []), [SWD, registry]);
  const rows = useMemo(() => filterRows(all, source, tier), [all, source, tier]);

  if (!SWD) {
    return (
      <main ref={root}>
        <div className="top">
          <div>
            <h1>
              SWING <span>LAB</span>
            </h1>
          </div>
        </div>
        <div className="verdict" id="verdict">
          {swing.data?.error ?? (swing.error ? "The lab is not answering." : "Loading…")}
        </div>
      </main>
    );
  }

  const c = SWD.candidate;
  const L = SWD.live;
  const mx = barMax(rows);
  const maxd = L ? Math.max(5, ...L.forming.map((x) => x.dist_pct ?? 0)) : 5;
  const show = (sym: string, strat?: string) => labShow(sym, strat || "candidate", "1d");
  const bb = (v: number | null) =>
    v == null ? (
      "–"
    ) : (
      <>
        <span className="bar" style={{ width: barWidth(v, mx), background: v >= 0 ? "var(--up)" : "var(--down)" }} />
        <span className={v >= 0 ? "up" : "down"}>{sg(v)}%</span>
      </>
    );
  const filt = L && !L.mkt_up ? <span className="pill r">FILTER OFF</span> : null;
  const P = SWD.profile ?? {};
  const topOf = (k: string) => (P[k] ?? [])[0];
  const parts = ["Sector", "Company size", "Strength vs Nifty (3 months)", "Base width"].map(topOf).filter(Boolean).map((r) => r!.k);
  const B = SWD.best_stocks ?? [];
  const bmx = Math.max(...B.map((x) => x.avg), 1);

  return (
    <main ref={root}>
      <div className="top">
        <div>
          <h1>
            SWING <span>LAB</span>
          </h1>
          <div className="sub">
            Holds of 2–40 days in delivery, tested on 5 years of NSE stocks after every cost. Click any stock on this page: its
            chart opens on the daily view with every past trade, the setup right now and its risk:reward.
          </div>
        </div>
        <div className="badges" id="badges">
          {(
            [
              ["Stocks", SWD.stocks],
              ["Trades simulated", (SWD.trades / 1e5).toFixed(1) + " lakh"],
              ["Strategies tested", SWD.list?.length ?? 0],
              ["Working", Object.keys(SWD.strategies).length],
              ["Cost / round trip", SWD.cost_pct != null ? SWD.cost_pct + "%" : "–"],
            ] as [string, string | number][]
          ).map(([k, v]) => (
            <span className="badge" key={k}>
              {k} <b>{v}</b>
            </span>
          ))}
        </div>
      </div>
      <div className="verdict" id="verdict">
        <Verdict c={c} asof={SWD.asof} />
      </div>

      <div className="grid">
        <div className="card">
          <div className="h">
            The system · after delivery costs (<span id="swcost">{SWD.cost_pct ?? "–"}</span>% a round trip)
          </div>
          <h3 id="swname">{c.label}</h3>
          <Cmp c={c} />
          <div style={{ marginTop: 9 }}>
            <span className="lg"><i style={{ background: "#2563eb" }}></i>Swing system</span>
            <span className="lg"><i style={{ background: "#9aa3b2" }}></i>Equal-weight universe</span>
            <span className="lg"><i style={{ background: "#d97706" }}></i>Nifty 50</span>
            <span className="lg"><i className="c"></i>In cash: Nifty under its 200-DMA</span>
          </div>
          <EquityChart c={c} />
          <Why c={c} />
        </div>
        <div className="card">
          <div className="h">Today</div>
          <div id="swmk">
            {L ? (
              <div className={"mk " + (L.mkt_up ? "on" : "off")}>
                <span className="d"></span>
                <span>
                  <b>Market filter {L.mkt_up ? "ON: new buys allowed" : "OFF: no new buys"}</b>
                  <br />
                  Nifty {L.nifty.toLocaleString("en-IN")} vs 200-DMA {L.nifty_200.toLocaleString("en-IN")} (
                  {sg(+((L.nifty / L.nifty_200 - 1) * 100).toFixed(1))}%)
                </span>
              </div>
            ) : (
              <div className="empty">Scanning the universe…</div>
            )}
          </div>
          <div className="h" style={{ marginTop: 12 }}>The rules</div>
          <ol className="rules" id="swrules">
            {c.rules.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ol>
        </div>
      </div>

      <div className="sec">
        STOCKS IN RADAR{" "}
        <small id="radt">
          {L
            ? "· scanned " +
              new Date(L.time * 1000).toLocaleString("en-IN", { timeZone: IST, day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }) +
              " IST · refreshed after every close"
            : ""}
        </small>
      </div>
      <div className="g3">
        <div className="card">
          <div className="hrow"><div className="h">Setup ready · waiting for the trigger</div><span className="mono muted" id="nform">{L?.forming.length ?? 0}</span></div>
          <div className="muted" style={{ fontSize: 11, marginTop: 2 }}>Uptrend, near the 52-week high, tight 10-day base. A close above the trigger on 1.3× volume is the entry.</div>
          <div className="list" id="form">
            {L && L.forming.length ? (
              L.forming.map((x) => (
                <div className="row" key={x.sym + x.strat} data-s={x.sym} onClick={() => show(x.sym)}>
                  <span className="s">{x.sym} <span className="pill b">READY</span>{filt}</span>
                  <span className="r">buy &gt; ₹{x.trigger}</span>
                  <span className="m">now ₹{x.close} · {x.dist_pct}% to trigger · SL ₹{x.stop} · T ₹{x.target} · risk {x.risk_pct}% · 1:3 · RS {sg(x.rs63)}</span>
                  <div className="dist"><i style={{ width: Math.max(4, 100 - ((x.dist_pct ?? 0) / maxd) * 100) + "%" }} /></div>
                </div>
              ))
            ) : (
              <div className="empty">No stock is in a ready VCP base right now. In a weak market (Nifty under its 200-DMA) very few stocks hold near their highs.</div>
            )}
          </div>
        </div>
        <div className="card">
          <div className="hrow"><div className="h">Triggered · open trades</div><span className="mono muted" id="ntrig">{L ? `${L.signals.length} triggered · ${L.open.length} open` : ""}</span></div>
          <div className="muted" style={{ fontSize: 11, marginTop: 2 }}>Fired at the last close (buy at the next open), trades running, and trades of the last two weeks.</div>
          <div className="list" id="trig">
            {L && L.signals.length + L.open.length + (L.recent?.length ?? 0) > 0 ? (
              <>
                {L.signals.map((x) => (
                  <div className="row live" key={"s" + x.sym + x.strat} data-s={x.sym} onClick={() => show(x.sym)}>
                    <span className="s">{x.sym} <span className="pill b">TRIGGERED</span>{filt}</span>
                    <span className="r">buy ~₹{x.p_in}</span>
                    <span className="m">at the next open · SL ₹{x.stop} · T ₹{x.target} · RS {sg(x.rs63)}</span>
                  </div>
                ))}
                {L.open.map((x) => {
                  const p = n1(((x.p_out ?? x.close ?? x.p_in) / x.p_in - 1) * 100);
                  return (
                    <div className="row" key={"o" + x.sym + x.strat} data-s={x.sym} onClick={() => show(x.sym)}>
                      <span className="s">{x.sym} <span className="pill g">OPEN</span></span>
                      <span className={"r " + (p >= 0 ? "up" : "down")}>{sg(p)}%</span>
                      <span className="m">since {tm(x.t_in)} · in ₹{x.p_in} · now ₹{x.p_out ?? x.close} · SL ₹{x.stop} · T ₹{x.target}</span>
                    </div>
                  );
                })}
                {(L.recent ?? []).map((x) => (
                  <div className="row" key={"r" + x.sym + x.strat + x.t_in} data-s={x.sym} onClick={() => show(x.sym)}>
                    <span className="s">{x.sym} <span className="pill n">CLOSED</span></span>
                    <span className={"r " + ((x.pct ?? 0) >= 0 ? "up" : "down")}>{sg(x.pct)}%</span>
                    <span className="m">{tm(x.t_in)} → {tm(x.t_out)} · ₹{x.p_in} → ₹{x.p_out} · {sg(x.R)}R</span>
                  </div>
                ))}
              </>
            ) : (
              <div className="empty">Nothing triggered or open. With the market filter off the system is in cash.</div>
            )}
          </div>
        </div>
        <div className="card">
          <div className="hrow"><div className="h">Stocks where it worked best</div><span className="mono muted" id="nbest">{B.length}</span></div>
          <div className="muted" style={{ fontSize: 11, marginTop: 2 }}>The system&apos;s own trades since 2022, 4+ trades each, average net per trade. Few trades each: a watchlist, not a promise.</div>
          <div className="list" id="best">
            {B.map((x) => (
              <div className="row" key={x.sym} data-s={x.sym} onClick={() => show(x.sym)}>
                <span className="s">{x.sym} <span className="muted" style={{ fontWeight: 500, fontSize: 10.5 }}>{x.sector}</span></span>
                <span className="r up">{sg(x.avg)}% / trade</span>
                <span className="m">{x.n} trades · {x.win}% won · total {sg(x.tot)}%</span>
                <div className="dist"><i style={{ width: (x.avg / bmx) * 100 + "%", background: "var(--up)" }} /></div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="sec">WHAT KIND OF STOCKS IT WORKS IN <small>average net per trade of the system&apos;s trades, by type</small></div>
      <div className="prof" id="prof">
        {Object.entries(P).map(([t, prow], j) => {
          const pmx = Math.max(...prow.map((r) => Math.abs(r.avg)), 1);
          return (
            <div className="card" key={t} style={{ animationDelay: 0.05 * j + "s" }}>
              <div className="h">{t}</div>
              {prow.slice(0, 8).map((r, i) => (
                <div className="pb" key={r.k}>
                  <span className="k" title={r.k}>{r.k}</span>
                  <span className={"v " + (r.avg >= 0 ? "up" : "down")}>{sg(r.avg)}%</span>
                  <div className="bar">
                    <i style={{ width: (Math.abs(r.avg) / pmx) * 100 + "%", background: i === 0 ? "var(--up)" : r.avg >= 0 ? "#7cc79c" : "var(--down)", animationDelay: 0.04 * i + "s" }} />
                  </div>
                  <span className="n">{r.n} trades · {r.win}% won</span>
                </div>
              ))}
            </div>
          );
        })}
      </div>
      <div className="note" id="profnote">
        {parts.length ? (
          <>
            <b>Stock selection, in short:</b> it worked best in {parts.join(" · ")}. Prefer ₹10 crore+ traded a day: the lowest-turnover stocks look best on paper, but real fills there are worse than this test assumes.
          </>
        ) : (
          <>No stock-type breakdown yet: it needs 4+ trades per bucket.</>
        )}
      </div>

      <div className="sec">SWING LEADERBOARD <small>every strategy with its edge over the universe, in and out of sample</small></div>
      <div className="card tw" style={{ marginTop: 10 }}>
        <div className="hrow" id="lbfilters" style={{ justifyContent: "flex-start", gap: 14, flexWrap: "wrap", marginBottom: 6 }}>
          <label className="muted">
            Source:{" "}
            <select id="fsrc" value={source} onChange={(e) => setSource(e.target.value as SourceFilter)}>
              {SOURCE_OPTIONS.map((o) => (
                <option key={o.v} value={o.v}>{o.label}</option>
              ))}
            </select>
          </label>
          <label className="muted">
            Tier:{" "}
            <select id="ftier" value={tier} onChange={(e) => setTier(e.target.value as TierFilter)}>
              {TIER_OPTIONS.map((o) => (
                <option key={o} value={o}>{o}</option>
              ))}
            </select>
          </label>
        </div>
        <table className="lb" id="swlb">
          <thead>
            <tr>
              <th>Strategy</th><th>Exit</th><th>Trades</th><th>Win %</th><th>Avg hold</th><th>Net / trade</th>
              <th>Edge vs universe</th><th>Out-of-sample edge</th><th>Years</th><th>Gate</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.k} data-s={r.key} onClick={() => show("RELIANCE", r.key)}>
                <td>{r.key === c.strat || r.k === c.strat ? "★ " : ""}{r.k}</td>
                <td>{r.exit_name}</td>
                <td>{r.n.toLocaleString("en-IN")}</td>
                <td>{r.win}</td>
                <td>{r.days}d</td>
                <td className={r.net >= 0 ? "up" : "down"}>{sg(r.net)}%</td>
                <td>{bb(r.excess)}</td>
                <td>{bb(r.OOS_exc)}</td>
                <td>
                  <span className="yb">
                    {Object.entries(r.years).map(([y, v]) => (
                      <i key={y} title={`${y}: ${sg(v)}% a trade`} style={{ background: v >= 0 ? "var(--up)" : "var(--down)", opacity: 0.35 + Math.min(0.65, Math.abs(v) / 4) }} />
                    ))}
                  </span>
                </td>
                <td data-gate={gateKind(r.gated)}><GateChip gated={r.gated} /></td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="note" id="mtnote">{multipleTestingNote(rows.length)}</div>
        <div className="note" id="removed">
          ★ = the system. &quot;Edge vs universe&quot; = the trade&apos;s return minus the equal-weight universe over the same days, which strips out the bull run and the survivorship bias. Out-of-sample = trades from July 2024 on. Click a row to see its trades on a chart.
          <br />
          <b>Removed</b> (no edge over the universe, or it vanished out of sample): {SWD.removed.join(", ") || "none"}.
        </div>
      </div>

      <div className="find" id="swrob">
        {(SWD.robust ?? []).map(([t, p], i) => (
          <div className="f" key={t} style={{ animationDelay: 0.05 * i + "s" }}>
            <span className="n">{String(i + 1).padStart(2, "0")}</span>
            <b>{t}</b>
            <p>{p}</p>
          </div>
        ))}
      </div>
      <div className="note" id="foot">
        Research built {SWD.built.replace("T", " ")} · NSE daily bars, Nifty 500 and F&amp;O lists · entry at the next day&apos;s open, stop checked before target, gaps through the stop fill at the open · exploratory, net of StockSense costs · a research tool, not investment advice.
      </div>
    </main>
  );
}
