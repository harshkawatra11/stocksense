// Pure drawing helpers for the order-flow canvases (docs/PLAN-M.md 5.6): colours, scaling and grids, no DOM.
import type { FpBar, FpRow, HeatReply, TpoProfile, VolumeProfile } from "./types";

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

/** Footprint cell background: green for net buying, red for net selling, alpha grows with |delta| / maxAbs. */
export function footprintCellColor(delta: number, maxAbs: number): string {
  if (!delta || !(maxAbs > 0)) return "rgba(120,130,150,0.06)";
  const a = 0.1 + 0.6 * clamp01(Math.abs(delta) / maxAbs);
  return delta > 0 ? `rgba(47,158,95,${a.toFixed(3)})` : `rgba(214,69,93,${a.toFixed(3)})`;
}

export const maxAbsDelta = (bars: FpBar[]): number =>
  bars.reduce((m, b) => b.rows.reduce((mm, r) => Math.max(mm, Math.abs(r.delta)), m), 0);

/** Footprint cell label "sell×buy", rounded to whole shares. */
export const cellText = (r: FpRow): string => `${Math.round(r.sell)}×${Math.round(r.buy)}`;

/** True when `price` lies inside a stacked-imbalance run of the bar (these cells get an outline). */
export function inStack(b: FpBar, price: number): "buy" | "sell" | null {
  for (const s of b.stacked_imbalances) {
    const lo = Math.min(s.from_price, s.to_price),
      hi = Math.max(s.from_price, s.to_price);
    if (price >= lo - 1e-9 && price <= hi + 1e-9) return s.side;
  }
  return null;
}

/** Linear price -> y (top = high). */
export function priceY(price: number, lo: number, hi: number, h: number, pad = 0): number {
  if (hi <= lo) return h / 2;
  return pad + (1 - (price - lo) / (hi - lo)) * (h - 2 * pad);
}

/** Histogram scaling: bar length in px for each bin, the largest bin gets `width`. */
export function profileWidths(vol: number[], width: number): number[] {
  const m = Math.max(0, ...vol);
  return vol.map((v) => (m > 0 ? (v / m) * width : 0));
}

/** Index of the bin that holds `price` (bins are lower edges of equal width `bin`), or -1. */
export function binIndex(p: VolumeProfile | TpoProfile, price: number): number {
  const i = Math.floor((price - p.bins[0]) / p.bin + 1e-9);
  return i >= 0 && i < p.bins.length ? i : -1;
}

/** Whether a bin lies in the value area (val <= lower edge and upper edge <= vah). */
export const inValueArea = (p: VolumeProfile | TpoProfile, i: number): boolean =>
  p.bins[i] >= p.val - 1e-9 && p.bins[i] + p.bin <= p.vah + 1e-9;

export interface TpoRow {
  price: number; // lower edge
  letters: string;
  single: boolean;
  ib: boolean;
  poc: boolean;
  va: boolean;
}
/** The TPO letter grid, highest price first. */
export function tpoGrid(t: TpoProfile): TpoRow[] {
  const singles = new Set(t.single_prints.map((x) => Math.round(x * 1e6)));
  const rows: TpoRow[] = t.bins.map((p, i) => ({
    price: p,
    letters: t.letters[i] ?? "",
    single: singles.has(Math.round(p * 1e6)),
    ib: p + t.bin > t.ib_low + 1e-9 && p < t.ib_high - 1e-9,
    poc: t.poc_tpo >= p - 1e-9 && t.poc_tpo < p + t.bin - 1e-9,
    va: inValueArea(t, i),
  }));
  return rows.reverse();
}

// Heat map ramp: five fixed stops, dark violet to yellow (viridis-like).
export const HEAT_STOPS: [number, number, number][] = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37],
];
/** t in [0,1] -> "rgb(r,g,b)" linearly between the five stops. */
export function heatColor(t: number): string {
  const x = clamp01(t) * (HEAT_STOPS.length - 1);
  const i = Math.min(Math.floor(x), HEAT_STOPS.length - 2);
  const f = x - i;
  const [a, b] = [HEAT_STOPS[i], HEAT_STOPS[i + 1]];
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",")})`;
}
export const heatMax = (h: HeatReply): number => Math.max(0, ...(h.qty ?? []).map((r) => Math.max(0, ...r)));

/** Method label for a panel title, e.g. "Delta (BVC estimate from 1-minute bars)". */
export const panelTitle = (name: string, method?: string | null): string => (method ? `${name} (${method})` : name);

export const learnHref = (slug: string): string => `/learn/?topic=${slug}`;
