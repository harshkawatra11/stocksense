// Port of smart.html trades-on-the-chart code (lines 683-835) plus the StockSense additions of card M6.2:
// strategy dropdown grouped by source, the EXPLORATORY badge and the gate chip.
import { esc } from "../format";
import { isIdx, type Marker } from "./chart";
import type { Candle, GatedVerdict, SetupState, StrategyRow, Trade, TradesResponse } from "./types";

export const INDEX_EXITS: [string, string][] = [
  ["rr15", "Target 1.5R"],
  ["rr2", "Target 2R"],
  ["rr3", "Target 3R"],
  ["trail", "Trailing stop"],
  ["eod", "Hold to 15:15"],
];
const INDEX_EXIT_NAME: Record<string, string> = {
  rr15: "target 1.5R",
  rr2: "target 2R",
  rr3: "target 3R",
  trail: "trailing stop",
  eod: "hold to 15:15",
};

/** Price levels of live trades and a forming setup, so the price scale keeps stop and target in view. */
export function tlevels(TR: TradesResponse | null, sym: string | null, tradesOn: boolean): number[] {
  if (!tradesOn || !TR || TR.sym !== sym) return [];
  const lv: (number | null | undefined)[] = [];
  (TR.trades || []).filter((t) => t.status !== "CLOSED").forEach((t) => lv.push(t.stop, t.target ?? t.p_in, t.p_in));
  if (TR.state && TR.state.state === "FORMING") lv.push(TR.state.stop, TR.state.target);
  return lv.filter((v): v is number => v != null && isFinite(v));
}

export function tradeMarkers(TR: TradesResponse | null, candles: Candle[], sym: string | null, tradesOn: boolean): Marker[] {
  const m: Marker[] = [];
  if (!(tradesOn && TR && TR.trades && TR.sym === sym)) return m;
  const have = new Set(candles.map((c) => c.time));
  TR.trades.forEach((t) => {
    if (have.has(t.t_in))
      m.push({ time: t.t_in, position: t.side > 0 ? "belowBar" : "aboveBar", color: "#2563eb", shape: t.side > 0 ? "arrowUp" : "arrowDown", text: t.side > 0 ? "BUY" : "SELL" });
    if (t.status === "CLOSED" && t.t_out != null && have.has(t.t_out))
      m.push({
        time: t.t_out,
        position: t.side > 0 ? "aboveBar" : "belowBar",
        color: (t.R ?? 0) > 0 ? "#22a05a" : "#dc3f54",
        shape: "circle",
        text: ((t.R ?? 0) > 0 ? "+" : "") + t.R + "R",
      });
  });
  return m;
}

export function rrbar(entry: number, stop: number, target: number | null | undefined): string {
  const risk = Math.abs(entry - stop);
  const rew = target != null ? Math.abs(target - entry) : null;
  if (!rew) return "";
  const tot = risk + rew;
  return `<div class="rr"><i style="width:${(risk / tot) * 100}%;background:#dc2626"></i><i style="width:${(rew / tot) * 100}%;background:#16a34a"></i></div>
    <div class="rrl"><span class="down">Risk ₹${risk.toFixed(2)} (${((risk / entry) * 100).toFixed(1)}%)</span><b>1 : ${(rew / risk).toFixed(1)}</b><span class="up">Reward ₹${rew.toFixed(2)} (${((rew / entry) * 100).toFixed(1)}%)</span></div>`;
}

export function tstate(st: SetupState | null | undefined): string {
  if (!st) return "";
  const ok = (b: boolean | undefined, t: string) => `<span class="${b ? "up" : "down"}">${b ? "✓" : "✗"} ${t}</span>`;
  const mk = st.mkt_up ? '<span class="up">Market filter ON</span>' : '<span class="down">Market filter OFF (Nifty under 200-DMA): wait</span>';
  if (st.state === "FORMING")
    return `<div class="st forming"><b>SETUP READY · waiting for the trigger</b><div>Buy above <b>₹${st.trigger}</b> (${st.dist_pct}% away) on volume ≥ 1.3× average</div>
    <div>Stop ₹${st.stop} · Target ₹${st.target}</div>${rrbar(st.trigger as number, st.stop as number, st.target)}<div style="margin-top:3px">${mk}</div></div>`;
  if (st.state === "SIGNAL")
    return `<div class="st sig"><b>TRIGGERED at the last close · buy at the next open</b><div>~₹${st.p_in} · Stop ₹${st.stop} · Target ₹${st.target}</div>${rrbar(st.p_in as number, st.stop as number, st.target)}<div style="margin-top:3px">${mk}</div></div>`;
  if (st.state === "OPEN")
    return `<div class="st open"><b>OPEN TRADE</b><div>Entry ₹${st.p_in} · Stop ₹${st.stop} · Target ₹${st.target}</div>${rrbar(st.p_in as number, st.stop as number, st.target)}</div>`;
  return `<div class="st none"><b>No setup right now</b><div class="chk">${ok(st.trend, "Uptrend (200/50/20)")}${ok(st.near_high, `Near 52-week high (${st.from_high}%)`)}${ok(st.tight, "Tight 10-day base")}</div><div>${mk}</div></div>`;
}

// ---- StockSense additions: exploratory badge and the gate chip ----
export function gateChipHTML(g: GatedVerdict | null | undefined): string {
  const base = "display:inline-block;font:700 9.5px Inter,sans-serif;letter-spacing:.4px;border-radius:4px;padding:1px 6px;color:#fff;";
  const n = g && g.lines_passed != null ? g.lines_passed : null;
  if (g && g.verdict === "PASS")
    return `<span class="gate pass" style="${base}background:#16a34a">GATE PASS ${n ?? 9}/9</span>`;
  if (g && g.verdict === "FAIL")
    return `<span class="gate fail" style="${base}background:#dc2626">GATE FAIL ${n ?? 0}/9</span>`;
  return `<span class="gate none" style="${base}background:#8a93a6">NOT YET TESTED</span>`;
}
export const exploratoryBadgeHTML = (): string =>
  `<span class="expl" style="display:inline-block;font:700 9.5px Inter,sans-serif;letter-spacing:.4px;border-radius:4px;padding:1px 6px;background:#fff6dc;color:#8a5a00;border:1px solid #f1d58a">EXPLORATORY · net of StockSense costs</span>`;

export function tpanelHTML(TR: TradesResponse | null, sym: string | null, tradesOn: boolean): string {
  const idx = isIdx(sym);
  if (!tradesOn)
    return `<div class="tt"><span>${idx ? "NIFTY INTRADAY" : "SWING SYSTEM"}</span><span>off</span></div>
      <button class="ton">▶ Show ${idx ? "intraday" : "swing"} trades &amp; risk:reward</button>`;
  // a note or error reply (index on 1d, lab down) has no sym; Market Lab's own check would leave it on "Loading…"
  const bare = !!TR && TR.sym == null && !!(TR.note || TR.error);
  if (!TR || (TR.sym !== sym && !bare)) return '<div class="tt"><span>TRADES</span></div><div class="muted" style="font-size:11px;margin-top:4px">Loading…</div>';
  const head = `<div class="tt"><span>${TR.swing ? "SWING SYSTEM" : "NIFTY INTRADAY"}</span><span>${TR.cost_pct ? "after " + TR.cost_pct + "% costs" : ""}</span></div>`;
  const stamp =
    "exploratory" in TR || "gated" in TR
      ? `<div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:4px">${TR.exploratory ? exploratoryBadgeHTML() : ""}${gateChipHTML(TR.gated)}</div>`
      : "";
  if (TR.note || TR.error)
    return head + stamp + `<div class="muted" style="font-size:11px;margin-top:4px">${esc(TR.note || TR.error)}</div>`;
  const s = TR.summary || {};
  const list = (TR.trades || []).map((t, i) => [t, i] as [Trade, number]).reverse();
  const tm = (t: number) =>
    new Date(t * 1000).toLocaleString(
      "en-IN",
      TR.swing
        ? { timeZone: "Asia/Kolkata", day: "2-digit", month: "short", year: "2-digit" }
        : { timeZone: "Asia/Kolkata", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" },
    );
  const hold = TR.days_held && TR.days_held.length ? Math.round(TR.days_held.reduce((a, b) => a + b, 0) / TR.days_held.length) : null;
  const sg = (v: number | undefined) => ((v || 0) > 0 ? "+" : "");
  return (
    head +
    stamp +
    `<div class="tn">${TR.candidate ? "★ " : ""}${esc(TR.strat)} <span class="muted">· ${TR.swing ? "exit " + esc(TR.exit_name) : INDEX_EXIT_NAME[TR.exit ?? ""] ?? esc(TR.exit)}</span></div>
    ${TR.swing ? tstate(TR.state) : ""}
    <div class="ts"><div>Trades<b>${s.n || 0}</b></div><div>Win rate<b>${s.win != null ? s.win + "%" : "–"}</b></div><div>Net<b class="${(s.totR || 0) >= 0 ? "up" : "down"}">${sg(s.totR)}${s.totR || 0}R</b></div>
      ${TR.swing ? `<div>Sum of trades<b class="${(s.pct || 0) >= 0 ? "up" : "down"}">${sg(s.pct)}${s.pct || 0}%</b></div><div>Avg / trade<b class="${(s.avgR || 0) >= 0 ? "up" : "down"}">${s.avgR != null ? (s.avgR > 0 ? "+" : "") + s.avgR + "R" : "–"}</b></div><div>Avg hold<b>${hold != null ? hold + "d" : "–"}</b></div>` : ""}</div>
    <div class="tl">${list.map(([t, i]) => `<div class="tr ${t.status !== "CLOSED" ? "live" : ""}" data-i="${i}"><span>${tm(t.t_in)} <b class="${t.side > 0 ? "up" : "down"}">${t.side > 0 ? "LONG" : "SHORT"}</b></span><span>${t.status === "CLOSED" ? `${t.p_in} → ${t.p_out} <b class="${(t.R ?? 0) > 0 ? "up" : "down"}">${(t.R ?? 0) > 0 ? "+" : ""}${t.R}R${t.pct != null ? ` ${t.pct > 0 ? "+" : ""}${t.pct}%` : ""}</b>` : `<b style="color:#2563eb">${esc(t.status)}</b> ${t.p_in} · SL ${t.stop}`}</span></div>`).join("") || '<div class="muted" style="font-size:11px;margin:4px 0">No trades for this strategy in the history.</div>'}</div>
    ${TR.swing ? `<div class="warn" style="background:#e8f6ee;color:#17603a">Backtest of fixed rules on your own data, after delivery costs. Past results do not predict future returns. Not a recommendation: decide for yourself.</div>` : `<div class="warn">Nifty intraday: paper trading only. It was about break-even over 3 years after costs.</div>`}`
  );
}

// ---- the box drawn on the chart ----
export interface TradeDrawEnv {
  TR: TradesResponse;
  sym: string | null;
  tradesOn: boolean;
  hot: Trade | null;
  t0: number;
  now: number;
  pw: number;
  lastCandleTime: number | null;
  timeToX: (t: number) => number | null;
  priceToY: (p: number) => number | null;
}

export function drawTrades(x: CanvasRenderingContext2D, env: TradeDrawEnv): void {
  const { TR, pw, now, hot } = env;
  if (!env.tradesOn || !TR.trades || TR.sym !== env.sym) return;
  const grow = Math.min(1, (now - env.t0) / 520);
  const ease = 1 - Math.pow(1 - grow, 3);
  const st = TR.state;
  if (st && st.state === "FORMING") {
    (
      [
        [st.trigger, "#2563eb", `BUY ABOVE ${st.trigger} · ${st.dist_pct}% away`],
        [st.stop, "#dc2626", `STOP ${st.stop}`],
        [st.target, "#16a34a", `TARGET ${st.target} (3R)`],
      ] as [number | undefined, string, string][]
    ).forEach(([p, col, lab]) => {
      if (p == null) return;
      const y = env.priceToY(p);
      if (y == null) return;
      x.setLineDash([6, 4]);
      x.strokeStyle = col;
      x.lineWidth = 1.2;
      x.beginPath();
      x.moveTo(pw * 0.55, y);
      x.lineTo(pw, y);
      x.stroke();
      x.setLineDash([]);
      x.font = "700 10px JetBrains Mono, monospace";
      const w = x.measureText(lab).width + 10;
      x.globalAlpha = 0.6 + 0.4 * Math.abs(Math.sin(now / 500));
      x.fillStyle = col;
      x.fillRect(pw * 0.55, y - 15, w, 14);
      x.globalAlpha = 1;
      x.fillStyle = "#fff";
      x.textBaseline = "middle";
      x.fillText(lab, pw * 0.55 + 5, y - 8);
    });
  }
  TR.trades.forEach((t) => {
    const open = t.status !== "CLOSED";
    let a = env.timeToX(t.t_in);
    let b = open ? pw - 2 : t.t_out != null ? env.timeToX(t.t_out) : null;
    if (t.status === "SIGNAL") {
      const xl = env.lastCandleTime != null ? env.timeToX(env.lastCandleTime) : null;
      a = xl != null ? Math.min(xl + 6, pw - 70) : pw - 70;
    }
    if (a == null || b == null) return;
    b = a + Math.max(open ? 60 : 16, b - a) * ease;
    const yE = env.priceToY(t.p_in);
    const yS = env.priceToY(t.stop);
    const yT = t.target != null ? env.priceToY(t.target) : null;
    if (yE == null || yS == null) return;
    const isHot = hot === t;
    x.globalAlpha = open ? 0.6 + 0.4 * Math.abs(Math.sin(now / 420)) : isHot ? 1 : 0.82;
    if (yT != null) {
      x.fillStyle = isHot ? "rgba(34,160,90,.30)" : "rgba(34,160,90,.17)";
      x.fillRect(a, Math.min(yE, yT), b - a, Math.abs(yT - yE));
    }
    x.fillStyle = isHot ? "rgba(220,63,84,.30)" : "rgba(220,63,84,.17)";
    x.fillRect(a, Math.min(yE, yS), b - a, Math.abs(yS - yE));
    x.lineWidth = 1;
    x.setLineDash([]);
    x.strokeStyle = "#475569";
    x.beginPath();
    x.moveTo(a, Math.round(yE) + 0.5);
    x.lineTo(b, Math.round(yE) + 0.5);
    x.stroke();
    const R = t.R ?? 0;
    if (!open && t.p_out != null) {
      const yX = env.priceToY(t.p_out);
      if (yX != null) {
        x.setLineDash([3, 3]);
        x.strokeStyle = R > 0 ? "#15803d" : "#b91c1c";
        x.beginPath();
        x.moveTo(a, yE);
        x.lineTo(b, yX);
        x.stroke();
        x.setLineDash([]);
        x.fillStyle = x.strokeStyle;
        x.beginPath();
        x.arc(b, yX, 3, 0, 7);
        x.fill();
      }
    }
    const lab = open
      ? `${t.status === "SIGNAL" ? "SIGNAL" : "LIVE"} ${t.side > 0 ? "LONG" : "SHORT"} · SL ${t.stop}${t.target ? " · T " + t.target : ""}`
      : `${t.side > 0 ? "LONG" : "SHORT"} ${R > 0 ? "+" : ""}${t.R}R${t.pct != null ? " · " + (t.pct > 0 ? "+" : "") + t.pct + "%" : ""}`;
    x.font = "700 10px JetBrains Mono, monospace";
    const w = x.measureText(lab).width + 10;
    const top = Math.max(16, Math.min(yE, yS, yT ?? yE) - 4);
    const lx = Math.max(2, Math.min(a, pw - w - 4));
    x.fillStyle = open ? "#2563eb" : R > 0 ? "#16a34a" : "#dc2626";
    x.beginPath();
    if (x.roundRect) x.roundRect(lx, top - 14, w, 14, 4);
    else x.rect(lx, top - 14, w, 14);
    x.fill();
    x.fillStyle = "#fff";
    x.textBaseline = "middle";
    x.fillText(lab, lx + 5, top - 7);
    if (open) {
      x.font = "700 9.5px JetBrains Mono, monospace";
      (
        [
          [yT, "T " + t.target, "#15803d"],
          [yE, "E " + t.p_in, "#334155"],
          [yS, "SL " + t.stop, "#b91c1c"],
        ] as [number | null, string, string][]
      ).forEach(([y, s, col]) => {
        if (y == null) return;
        const tw = x.measureText(s).width + 8;
        const tx = Math.max(a + 2, b - tw - 2);
        x.fillStyle = "rgba(255,255,255,.9)";
        x.fillRect(tx, y - 7, tw, 13);
        x.fillStyle = col;
        x.fillText(s, tx + 4, y);
      });
    }
    x.globalAlpha = 1;
  });
}

// ---- strategy dropdown ----
export const SOURCE_GROUPS: { id: string; label: string; match: (r: StrategyRow) => boolean }[] = [
  { id: "mls", label: "Market Lab swing", match: (r) => r.source === "marketlab_swing" },
  { id: "mli", label: "Market Lab intraday", match: (r) => r.source === "marketlab_intraday" },
  { id: "tF", label: "Tier F", match: (r) => r.source === "tier" && r.tier === "F" },
  { id: "tC", label: "Tier C", match: (r) => r.source === "tier" && r.tier === "C" },
  { id: "tB", label: "Tier B", match: (r) => r.source === "tier" && r.tier === "B" },
  { id: "k151", label: "151 Trading Strategies", match: (r) => r.source === "k151" },
  { id: "flow", label: "Flow", match: (r) => r.source === "flow" },
];

/** Strategies that can draw on a chart of timeframe `tf`: daily rules on a daily chart, the rest on intraday charts. */
export function strategiesForTf(rows: StrategyRow[], tf: string): StrategyRow[] {
  return rows.filter((r) => (tf === "1d") === ((r.timeframe ?? "1d") === "1d"));
}

export interface StrategyGroup {
  label: string;
  items: StrategyRow[];
}
export function groupStrategies(rows: StrategyRow[]): StrategyGroup[] {
  const used = new Set<StrategyRow>();
  const out: StrategyGroup[] = [];
  for (const g of SOURCE_GROUPS) {
    const items = rows.filter((r) => g.match(r));
    items.forEach((r) => used.add(r));
    if (items.length) out.push({ label: g.label, items });
  }
  const rest = rows.filter((r) => !used.has(r));
  if (rest.length) out.push({ label: "Other", items: rest });
  return out;
}

export const candidateLabel = (idx: boolean): string => (idx ? "★ Nifty intraday · Index ORB" : "★ Swing system · VCP");
