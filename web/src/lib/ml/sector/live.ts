// Port of sector.html live code (lines 562-595) and the watchlist (lines 547-558).
import type { HorizonData, SectorData, StockData, TickData } from "./types";
import type { View } from "./route";

export type AnyData = HorizonData | SectorData | StockData;

/** Symbols whose price is on screen: the first 14 of each group, or the stock and its peers (sector.html onScreen). */
export function onScreen(view: View, d: AnyData | null): string[] {
  if (!d) return [];
  if (view === "sector") return (d as SectorData).groups.flatMap((g) => g.stocks.slice(0, 14).map((s) => s.sym));
  if (view === "stock") {
    const s = d as StockData;
    return [s.sym, ...s.peers.map((p) => p.sym)];
  }
  return [];
}

/** Fold a tick into the horizon data in place (sector values and NIFTY). */
export function applyTickToHorizon(d: HorizonData, t: TickData): void {
  d.sectors.forEach((s) => {
    const x = t.sectors[s.name];
    if (x && x.v != null) s.v = x.v;
  });
  if (t.nifty.v != null) d.nifty.v = t.nifty.v;
}

// ---- watchlist: localStorage key sm_watch, every access wrapped in try/catch ----
export const WATCH_KEY = "sm_watch";

export function watchList(): string[] {
  try {
    const v: unknown = JSON.parse(localStorage.getItem(WATCH_KEY) || "[]");
    return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];
  } catch {
    return [];
  }
}

/** Add or remove `sym`; returns whether it is on the list afterwards. */
export function watchToggle(sym: string): boolean {
  const w = watchList();
  const i = w.indexOf(sym);
  if (i >= 0) w.splice(i, 1);
  else w.push(sym);
  try {
    localStorage.setItem(WATCH_KEY, JSON.stringify(w));
  } catch {
    /* private window or blocked storage: the star just does not persist */
  }
  return i < 0;
}
