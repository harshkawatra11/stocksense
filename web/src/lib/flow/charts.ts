// Pure geometry for the Money Flow page: sparkline points, OI scatter axes, sector treemap, pre-open board.
import type { OiRow, Preopen, PreopenRow } from "./types";

/** SVG polyline `points` for a series scaled into w x h with 2 px padding; null/NaN values are skipped. */
export function sparkPoints(values: (number | null | undefined)[], w: number, h: number): string {
  const v = values.map((x, i) => [i, x] as const).filter((p): p is readonly [number, number] => p[1] != null && isFinite(p[1]));
  if (v.length < 2) return "";
  const lo = Math.min(...v.map((p) => p[1]));
  const hi = Math.max(...v.map((p) => p[1]));
  const n = values.length - 1;
  return v
    .map(([i, x]) => `${((i / n) * (w - 4) + 2).toFixed(1)},${(h - 2 - ((x - lo) / (hi - lo || 1)) * (h - 4)).toFixed(1)}`)
    .join(" ");
}

export interface ScatterPoint {
  sym: string;
  x: number;
  y: number;
  label: string | null;
}

/** One dot per stock future: x = price change %, y = OI change %; rows missing either value are dropped. */
export function scatterPoints(rows: OiRow[]): ScatterPoint[] {
  return rows
    .filter((r) => r.price_chg_pct != null && r.oi_chg_pct != null && isFinite(r.price_chg_pct) && isFinite(r.oi_chg_pct))
    .map((r) => ({ sym: r.symbol, x: r.price_chg_pct as number, y: r.oi_chg_pct as number, label: r.quad_label }));
}

export interface ScatterAxes {
  xmax: number;
  ymax: number;
}

/** Symmetric axes around zero: the largest absolute value rounded up to a whole percent, at least 1. */
export function scatterAxes(pts: { x: number; y: number }[]): ScatterAxes {
  const m = (k: "x" | "y") => Math.max(1, Math.ceil(Math.max(0, ...pts.map((p) => Math.abs(p[k])))));
  return { xmax: m("x"), ymax: m("y") };
}

export interface ScatterBox {
  W: number;
  H: number;
  pad: number;
}

/** Pixel position of a point: zero sits in the middle, +x to the right, +y up. */
export function scatterXY(p: { x: number; y: number }, ax: ScatterAxes, b: ScatterBox): [number, number] {
  const iw = b.W - 2 * b.pad;
  const ih = b.H - 2 * b.pad;
  return [b.pad + ((p.x / ax.xmax + 1) / 2) * iw, b.pad + (1 - (p.y / ax.ymax + 1) / 2) * ih];
}

export interface Rect {
  i: number;
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Binary-split treemap of `values` (descending order expected) in a 100 x 100 box; each rect's area is exactly
 *  proportional to its value. Non-positive values get a zero-area rect. */
export function treemap(values: number[]): Rect[] {
  const out: Rect[] = [];
  const idx = values.map((v, i) => ({ i, v: v > 0 ? v : 0 })).sort((a, b) => b.v - a.v);
  const place = (items: { i: number; v: number }[], x: number, y: number, w: number, h: number) => {
    if (!items.length) return;
    if (items.length === 1) {
      out.push({ i: items[0].i, x, y, w, h });
      return;
    }
    const total = items.reduce((s, t) => s + t.v, 0);
    if (total <= 0) {
      items.forEach((t) => out.push({ i: t.i, x, y, w: 0, h: 0 }));
      return;
    }
    let k = 1;
    let acc = items[0].v;
    while (k < items.length - 1 && acc + items[k].v / 2 < total / 2) acc += items[k++].v;
    const a = items.slice(0, k);
    const b = items.slice(k);
    const r = acc / total;
    if (w >= h) {
      place(a, x, y, w * r, h);
      place(b, x + w * r, y, w - w * r, h);
    } else {
      place(a, x, y, w, h * r);
      place(b, x, y + h * r, w, h - h * r);
    }
  };
  place(idx, 0, 0, 100, 100);
  return out.sort((p, q) => p.i - q.i);
}

export type PreopenView =
  | { kind: "before" }
  | { kind: "live"; rows: PreopenRow[] }
  | { kind: "after"; rows: PreopenRow[] };

/** Board phase switch: before 09:00 nothing, 09:00-09:15 the live table sorted by |imbalance|, after 09:15 the
 *  pre-open versus first-15-minutes comparison (follow-through per row). */
export function preopenView(p: Preopen | null | undefined): PreopenView {
  if (!p || p.phase === "before") return { kind: "before" };
  const byImb = [...p.rows].sort((a, b) => Math.abs(b.imb ?? 0) - Math.abs(a.imb ?? 0));
  return p.phase === "live" ? { kind: "live", rows: byImb } : { kind: "after", rows: byImb };
}

/** Tick for a follow-through value: held (true), faded (false) or unknown (null). */
export const followTick = (v: boolean | null | undefined): string => (v === true ? "✓" : v === false ? "✗" : "–");

/** Minutes after midnight in Asia/Kolkata. */
export function istMinutes(d: Date): number {
  const p = new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).formatToParts(d);
  const g = (t: string) => Number(p.find((x) => x.type === t)?.value ?? 0);
  return g("hour") * 60 + g("minute");
}

/** True from 09:00 to 09:15 IST, the window in which the pre-open board is polled every 15 s. */
export const inPreopenWindow = (d: Date): boolean => {
  const m = istMinutes(d);
  return m >= 540 && m <= 555;
};
