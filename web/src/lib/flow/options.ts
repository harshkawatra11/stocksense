// Pure geometry for the options positioning page: strike window, mirrored OI bars, level markers, GEX, IV smile.
import type { OptionStrike, OptionsReply } from "./types";

/** The `k` strikes either side of the strike nearest to spot (all strikes when fewer); input sorted ascending. */
export function windowStrikes(strikes: OptionStrike[], spot: number | null, k = 25): OptionStrike[] {
  if (strikes.length <= 2 * k + 1 || spot == null) return strikes.length <= 2 * k + 1 ? strikes : strikes.slice(0, 2 * k + 1);
  let at = 0;
  strikes.forEach((s, i) => {
    if (Math.abs(s.strike - spot) < Math.abs(strikes[at].strike - spot)) at = i;
  });
  const from = Math.max(0, Math.min(strikes.length - 2 * k - 1, at - k));
  return strikes.slice(from, from + 2 * k + 1);
}

/** Largest open interest on either side (the common scale of the call, put and delta-OI bars); 0 when empty. */
export function oiMax(strikes: OptionStrike[]): number {
  return Math.max(0, ...strikes.flatMap((s) => [Math.abs(s.call_oi ?? 0), Math.abs(s.put_oi ?? 0), Math.abs(s.call_doi ?? 0), Math.abs(s.put_doi ?? 0)]));
}

/** Bar length in pixels: `half` pixels at the maximum, 0 for a missing value or a zero scale. */
export const barLen = (v: number | null | undefined, max: number, half: number): number =>
  v == null || max <= 0 ? 0 : (Math.abs(v) / max) * half;

export interface LevelPos {
  y: number;
  /** True when the level lies outside the strike range and was pinned to the nearest edge. */
  clamped: boolean;
}

/** Vertical position of a price level in a ladder whose rows run from the highest strike (top) to the lowest, each
 *  `rowH` tall; a price between two strikes is linearly interpolated, so a level at a strike sits mid-row. */
export function levelY(price: number | null | undefined, strikes: OptionStrike[], rowH: number): LevelPos | null {
  if (price == null || !isFinite(price) || strikes.length === 0) return null;
  const n = strikes.length;
  const lo = strikes[0].strike;
  const hi = strikes[n - 1].strike;
  const row = (i: number) => (n - 1 - i) * rowH + rowH / 2; // ascending index -> top-down row
  if (n === 1 || hi === lo) return { y: row(0), clamped: price !== lo };
  if (price <= lo) return { y: row(0), clamped: price < lo };
  if (price >= hi) return { y: row(n - 1), clamped: price > hi };
  let i = 0;
  while (strikes[i + 1].strike < price) i++;
  const f = (price - strikes[i].strike) / (strikes[i + 1].strike - strikes[i].strike);
  return { y: row(i) + (row(i + 1) - row(i)) * f, clamped: false };
}

/** Cumulative GEX from the lowest strike upwards (crore); missing values count as 0. */
export function cumulativeGex(strikes: OptionStrike[]): number[] {
  let acc = 0;
  return strikes.map((s) => (acc += s.gex_cr ?? 0));
}

export interface Scale {
  lo: number;
  hi: number;
}

/** Linear pixel position of v in [lo, hi] mapped onto [a, b]. */
export const lin = (v: number, s: Scale, a: number, b: number): number => a + ((v - s.lo) / (s.hi - s.lo || 1)) * (b - a);

/** Implied vol in percent: the chain may hold a fraction (0.15) or percent (15); values up to 3 are read as fractions. */
export const ivPct = (v: number): number => (v <= 3 ? v * 100 : v);

/** IV smile polylines: call and put IV (percent, see ivPct) against strike; a gap (missing IV) starts a new segment. */
export function smileSegments(strikes: OptionStrike[], key: "call_iv" | "put_iv", x: (k: number) => number, y: (iv: number) => number): string[] {
  const segs: string[] = [];
  let cur: string[] = [];
  for (const s of strikes) {
    const iv = s[key];
    if (iv == null || !isFinite(iv) || iv <= 0) {
      if (cur.length) segs.push(cur.join(" "));
      cur = [];
    } else cur.push(`${x(s.strike).toFixed(1)},${y(ivPct(iv)).toFixed(1)}`);
  }
  if (cur.length) segs.push(cur.join(" "));
  return segs;
}

export interface HistPoint {
  time: string;
  value: number;
}

/** One history series for lightweight-charts: ascending unique dates, missing values dropped. */
export function historySeries(history: OptionsReply["history"], key: "gex_total_cr" | "pcr_oi"): HistPoint[] {
  const m = new Map<string, number>();
  for (const h of history) {
    const v = h[key];
    if (v != null && isFinite(v)) m.set(h.date, v);
  }
  return Array.from(m, ([time, value]) => ({ time, value })).sort((a, b) => (a.time < b.time ? -1 : 1));
}
