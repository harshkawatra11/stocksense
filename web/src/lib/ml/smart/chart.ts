// Port of smart.html's pop-up chart helpers (lines 592-681): live candle merge, layers, zones, level panel.
import { esc, pc } from "../format";
import { px2 } from "./tiles";
import type { Candle, ChartRead, QuoteData, Tf, ZoneRow } from "./types";

export const IDX = new Set(["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "MIDCPNIFTY"]);
export const isIdx = (s: string | null | undefined): boolean => IDX.has(s ?? "") || String(s || "").startsWith("^");

export type LayerKey = "ob" | "fvg" | "structure" | "liquidity" | "vwap" | "trades";
export type Layers = Record<LayerKey, boolean>;
/** Layers on when a chart opens; trades are off until the Trades button (or the Lab) turns them on. */
export const DEFAULT_LAYERS: Layers = { ob: true, fvg: true, structure: true, liquidity: true, vwap: true, trades: false };
export const TF_BUTTONS: [Tf, string][] = [
  ["1m", "1m"],
  ["5m", "5m"],
  ["15m", "15m"],
  ["60m", "1H"],
  ["1d", "D"],
];
export const LAYER_BUTTONS: [LayerKey, string][] = [
  ["ob", "Order block"],
  ["fvg", "Fair value gaps"],
  ["structure", "Structure"],
  ["liquidity", "Liquidity"],
  ["vwap", "VWAP"],
  ["trades", "Trades"],
];
export const TF_SECONDS: Record<string, number> = { "1m": 60, "5m": 300, "15m": 900, "60m": 3600 };
export const VISIBLE_BARS: Record<Tf, number> = { "1m": 220, "5m": 150, "15m": 130, "60m": 120, "1d": 200 };

/** The visible logical range a freshly loaded chart opens at (smart.html line 628). */
export const visibleRange = (n: number, tf: Tf): { from: number; to: number } => ({
  from: Math.max(0, n - VISIBLE_BARS[tf]),
  to: n + 6,
});

export function priceHTML(ltp: number | null | undefined, prev: number | null | undefined): string {
  if (ltp == null) return "";
  const p = prev ? (ltp / prev - 1) * 100 : null;
  return `₹${px2(Number(ltp))} · <span class="${p != null && p >= 0 ? "up" : "down"}">${pc(p)}</span>`;
}

const istDay = (sec: number): string => new Date(sec * 1000).toLocaleDateString("en-CA", { timeZone: "Asia/Kolkata" });

/** Start time of the candle a quote belongs to (smart.html liveCandle). */
export function candleTime(tf: Tf, qTime: number, lastTime: number): number {
  if (tf === "1d") {
    const day = istDay(qTime);
    return day === istDay(lastTime) ? lastTime : Date.parse(day + "T03:45:00Z") / 1000;
  }
  const sec = TF_SECONDS[tf];
  const off = 13500 % sec;
  return Math.floor((qTime - off) / sec) * sec + off;
}

/** Fold a quote into the last candle (or open a new one) and return the candle to draw. Mutates `candles`. */
export function applyQuote(candles: Candle[], q: QuoteData, tf: Tf): Candle | null {
  if (!candles.length || !q.ltp) return null;
  const last = candles[candles.length - 1];
  const t = candleTime(tf, q.time, last.time);
  if (t > last.time) candles.push({ time: t, open: q.ltp, high: q.ltp, low: q.ltp, close: q.ltp });
  else {
    last.high = Math.max(last.high, q.ltp);
    last.low = Math.min(last.low, q.ltp);
    last.close = q.ltp;
  }
  return candles[candles.length - 1];
}

export interface Marker {
  time: number;
  position: "belowBar" | "aboveBar";
  color: string;
  shape: "arrowUp" | "arrowDown" | "circle";
  text: string;
}

export function structureMarkers(read: ChartRead | null | undefined, on: boolean): Marker[] {
  const ch = (read || ({} as ChartRead)).chart || {};
  return on
    ? (ch.events || []).map((e) => ({
        time: e.time,
        position: e.dir === "bull" ? "belowBar" : "aboveBar",
        color: e.dir === "bull" ? "#2f9e5f" : "#d6455d",
        shape: e.dir === "bull" ? "arrowUp" : "arrowDown",
        text: e.type,
      }))
    : [];
}

export const LIQ_LINES: [keyof NonNullable<ChartRead["liquidity"]>, string, string][] = [
  ["pdh", "PDH", "#c05621"],
  ["pdl", "PDL", "#c05621"],
  ["or_hi", "Opening range", "#6b7385"],
  ["or_lo", "", "#6b7385"],
];

export interface PriceLineSpec {
  price: number;
  color: string;
  title: string;
}
export function liquidityLines(read: ChartRead | null | undefined, on: boolean): PriceLineSpec[] {
  const lq = (read || ({} as ChartRead)).liquidity || {};
  if (!on) return [];
  const out: PriceLineSpec[] = [];
  for (const [k, title, color] of LIQ_LINES) {
    const v = lq[k];
    if (typeof v === "number") out.push({ price: v, color, title });
  }
  return out;
}

export interface ZoneDraw {
  fill: string;
  stroke: string;
  label: string;
  z: ZoneRow;
}
/** The order-block and FVG rectangles of the overlay canvas, as data (smart.html over()). */
export function zoneSpecs(read: ChartRead | null | undefined, lay: Pick<Layers, "ob" | "fvg">): ZoneDraw[] {
  const ch = (read || ({} as ChartRead)).chart || {};
  const out: ZoneDraw[] = [];
  if (lay.ob)
    for (const z of ch.obs || [])
      out.push({
        z,
        fill: z.dir > 0 ? "rgba(47,158,95,.14)" : "rgba(214,69,93,.14)",
        stroke: z.dir > 0 ? "#2f9e5f" : "#d6455d",
        label: `Order Block ${z.lo}–${z.hi}`,
      });
  if (lay.fvg) for (const z of ch.fvgs || []) out.push({ z, fill: "rgba(124,92,230,.14)", stroke: "#7c5ce6", label: `FVG ${z.lo}–${z.hi} · ${z.status}` });
  return out;
}

export function drawZones(
  x: CanvasRenderingContext2D,
  specs: ZoneDraw[],
  pw: number,
  timeToX: (t: number) => number | null,
  priceToY: (p: number) => number | null,
): void {
  for (const { z, fill, stroke, label } of specs) {
    let a = z.time != null ? timeToX(z.time) : null;
    if (a == null) a = 0;
    const y1 = priceToY(z.hi);
    const y2 = priceToY(z.lo);
    if (y1 == null || y2 == null) continue;
    x.fillStyle = fill;
    x.fillRect(a, y1, pw - a, y2 - y1);
    x.strokeStyle = stroke;
    x.strokeRect(a + 0.5, y1 + 0.5, pw - a, y2 - y1);
    x.fillStyle = stroke;
    x.font = "600 10px JetBrains Mono";
    x.fillText(label, Math.max(4, a + 4), y1 - 3);
  }
}

/** The right-hand panel of zones, structure, liquidity and order flow (smart.html levels()). */
export function levelsHTML(r: ChartRead | null | undefined): string {
  if (!r) return '<div class="muted">Not enough bars for a read.</div>';
  const st = r.structure.last;
  const lq = r.liquidity || {};
  const z = (zs: ZoneRow[]) =>
    zs.length
      ? zs
          .map(
            (q) =>
              `<div>${q.dir > 0 ? "Bull" : "Bear"} ${q.lo}–${q.hi} <span class="muted">· ${q.tested != null ? (q.tested ? "tested" : "price hasn't come back yet") : esc(q.status)} · ${Math.abs(q.dist_pct)}% ${q.dist_pct < 0 ? "below" : q.dist_pct > 0 ? "above" : "at price"}</span></div>`,
          )
          .join("")
      : '<div class="muted">none open</div>';
  return `<h4>STRUCTURE</h4>${st ? `<div class="${st.dir === "bull" ? "up" : "down"}"><b>${esc(st.type)} ${st.dir === "bull" ? "up" : "down"} · ${esc(st.time)}</b></div><div class="muted">Closed ${st.dir === "bull" ? "above" : "below"} the ${esc(st.level)} swing ${st.dir === "bull" ? "high" : "low"} · ${r.structure.breaks_today} break${r.structure.breaks_today === 1 ? "" : "s"} today</div>` : '<div class="muted">no break yet</div>'}
    <h4>ZONES · ORDER BLOCKS</h4>${z(r.order_blocks)}<h4>ZONES · FAIR VALUE GAPS</h4>${z(r.fvgs)}
    <h4>LIQUIDITY</h4>${lq.pdh != null ? `<div>PDH ${lq.pdh} · PDL ${lq.pdl}</div>` : ""}<div class="muted">${lq.swept && lq.swept.length ? "Swept today: " + esc(lq.swept.join(", ")) : "Not swept today"}</div>${lq.or_hi != null ? `<div>Opening range ${lq.or_lo}–${lq.or_hi}</div>` : ""}
    <h4>ORDER FLOW</h4>${r.buyers_pct != null ? `<div>Buyers ${r.buyers_pct}% <span class="muted">(volume on up candles today)</span></div>` : '<div class="muted">no volume for an index</div>'}${r.vwap ? `<div>VWAP ${r.vwap} · price ${(r.price ?? 0) > r.vwap ? "above" : "below"}</div>` : ""}`;
}

export function eventsHTML(r: ChartRead | null | undefined): string {
  if (!r) return "";
  return (
    '<span style="border:none;padding-left:0" class="muted">TODAY</span>' +
    (r.events || []).map((e) => `<span>${esc(e.time)} ${esc(e.text)}</span>`).join("")
  );
}
