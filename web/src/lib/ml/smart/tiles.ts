// Port of smart.html tiles (lines 238-270) plus the StockSense flow mini-card (card M6.2, addition 3).
import { esc, pc } from "../format";
import type { FlowCard, SmartStock, StructInfo } from "./types";

/** Rank ranges of the four tile columns, left to right: ranks 13-20, 8-12, 4-7, 1-3. */
export const COLS: [number, number][] = [
  [13, 20],
  [8, 12],
  [4, 7],
  [1, 3],
];
export const COL_WIDTHS = [1, 1.05, 1.25, 1.6];

/** Ranks shown in column `i`; the weakest sit at the bottom, mirroring the strongest at the top. */
export function columnRanks(side: "bull" | "bear", i: number): number[] {
  const [a, b] = COLS[i];
  const ranks: number[] = [];
  for (let r = a; r <= b; r++) ranks.push(r);
  if (side === "bear") ranks.reverse();
  return ranks;
}

export const stars = (disp: number): number => Math.max(1, Math.min(5, Math.ceil(Math.abs(disp) / 0.8)));

export const px2 = (v: number): string => v.toLocaleString("en-IN", { minimumFractionDigits: 2 });

export function structTag(st: StructInfo | null | undefined): string {
  return st
    ? `<span class="tag ${esc(st.dir)} ${st.type === "CHoCH" ? "choch" : ""}">${esc(st.type)} ${st.dir === "bull" ? "↑" : "↓"}</span>`
    : "";
}

const QUAD_UP = /long buildup|short covering/i;
const QUAD_DOWN = /short buildup|long unwinding/i;

/** Descriptive flow mini-card: OI quad chip, relative volume, delivery z-score, flow score. Empty when the row has none. */
export function flowMiniHTML(f: FlowCard | null | undefined): string {
  if (!f) return "";
  const n = (v: number | null | undefined, d = 1) => (v == null || !isFinite(v) ? "–" : v.toFixed(d));
  const quad = f.oi_quad_label ?? f.oi_quad;
  const qc = quad && QUAD_UP.test(quad) ? "bull" : quad && QUAD_DOWN.test(quad) ? "bear" : "";
  const chip = quad ? `<span class="tag ${qc}">${esc(quad)}</span> ` : "";
  return `<div class="flowm" data-flow="1" style="margin-top:3px">FLOW ${chip}rvol <b>${n(f.rvol)}×</b> · deliv z <b>${n(f.deliv_z20_prev)}</b> · score <b>${n(f.flow_score, 2)}</b></div>`;
}

export function tileHTML(s: SmartStock | undefined, big: boolean): string {
  if (!s) return `<div class="tile" style="visibility:hidden"></div>`;
  const r = s.read || {};
  const st = r.struct;
  const side = s.side;
  const nStars = stars(s.disp);
  const tag = structTag(st);
  const where = (d: number) => (d < 0 ? "below" : "above");
  const det = big
    ? `<div class="det">ST ${st ? `<b class="${st.dir === "bull" ? "up" : "down"}">${esc(st.type)} ${st.dir === "bull" ? "↑" : "↓"}</b> ${esc(st.time)}` : "–"}${r.ob ? `<br>OB ${Math.abs(r.ob.dist_pct)}% ${where(r.ob.dist_pct)}` : ""}${r.fvg ? `<br>FVG ${Math.abs(r.fvg.dist_pct)}% ${where(r.fvg.dist_pct)}` : ""}<br>LIQ ${s.liq && s.liq.length ? `<span style="color:var(--gold)">${esc(s.liq.join(" + "))} swept</span>` : "not swept"}${flowMiniHTML(s.flow)}</div>`
    : "";
  return `<div class="tile ${side} ${big ? "big" : ""}" data-s="${esc(s.sym)}" data-k="t:${esc(s.sym)}" data-v="${s.ltp}">
    <canvas></canvas><div class="lab">${big ? `<span class="logo" style="color:${side === "bull" ? "var(--up)" : "var(--down)"}">${esc(s.sym[0])}</span>` : ""}<b style="display:inline">${esc(s.sym)}</b><br><span class="${s.pct >= 0 ? "up" : "down"}">${pc(s.pct)}</span>${big ? `<br><span class="stars ${side === "bull" ? "up" : "down"}">${"●".repeat(nStars)}<span style="opacity:.25">${"●".repeat(5 - nStars)}</span></span>` : `<br>${tag}`}</div>
    <span class="rk ${side === "bull" ? "up" : "down"}">${String(s.rank).padStart(2, "0")}</span>${big || s.rank <= 7 ? `<span class="px" ${big ? 'style="bottom:46%"' : ""}>${px2(s.ltp)}</span>` : ""}${det}</div>`;
}

/** The inner HTML of one half (bull or bear): four columns of tiles. */
export function halfHTML(side: "bull" | "bear", list: string[], by: Record<string, SmartStock>): string {
  const tot = COL_WIDTHS.reduce((a, b) => a + b);
  return COLS.map(
    (_, i) =>
      `<div class="col" style="width:${(COL_WIDTHS[i] / tot) * 100}%">${columnRanks(side, i)
        .map((r) => tileHTML(by[list[r - 1]], i === 3))
        .join("")}</div>`,
  ).join("");
}

export function chipsHTML(sectors: { name: string; pct: number; up: number; down: number }[], room: number): string {
  return sectors
    .slice(0, room)
    .map(
      (s) => `<div class="chip" data-sec="${esc(s.name)}"><div class="h"><span>${esc(s.name)}</span><span class="${s.pct >= 0 ? "up" : "down"}">${pc(s.pct)}</span></div>
    <div class="seg">${"<i></i>".repeat(Math.min(s.up, 24))}${'<i class="d"></i>'.repeat(Math.min(s.down, 24))}</div></div>`,
    )
    .join("");
}
