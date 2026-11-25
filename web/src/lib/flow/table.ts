// Sorting and filtering of the flow table, and the gamma-regime badge.
import type { FlowStock, IndexGamma } from "./types";

export type SortKey = keyof Pick<
  FlowStock,
  "sym" | "sector" | "ltp" | "pct" | "disp" | "rvol" | "oi_quad_label" | "oi_chg_pct" | "deliv_z20_prev" | "depth_imb5" | "buy_sell_ratio" | "cvd_slope_30m" | "flow_score"
>;

export const COLUMNS: { key: SortKey; label: string; text?: boolean }[] = [
  { key: "sym", label: "Symbol", text: true },
  { key: "sector", label: "Sector", text: true },
  { key: "ltp", label: "LTP" },
  { key: "pct", label: "% chg" },
  { key: "disp", label: "Disp" },
  { key: "rvol", label: "RVOL" },
  { key: "oi_quad_label", label: "OI quad", text: true },
  { key: "oi_chg_pct", label: "OI Δ%" },
  { key: "deliv_z20_prev", label: "Deliv z (prev)" },
  { key: "depth_imb5", label: "Depth imb" },
  { key: "buy_sell_ratio", label: "Buy/Sell" },
  { key: "cvd_slope_30m", label: "CVD slope" },
  { key: "flow_score", label: "Flow score" },
];

/** Stable sort by one column; missing values are always last, whatever the direction. */
export function sortRows(rows: FlowStock[], key: SortKey, dir: "asc" | "desc"): FlowStock[] {
  const s = dir === "asc" ? 1 : -1;
  return rows
    .map((r, i) => ({ r, i }))
    .sort((a, b) => {
      const x = a.r[key];
      const y = b.r[key];
      if (x == null && y == null) return a.i - b.i;
      if (x == null) return 1;
      if (y == null) return -1;
      const c = typeof x === "string" || typeof y === "string" ? String(x).localeCompare(String(y)) : (x as number) - (y as number);
      return c ? c * s : a.i - b.i;
    })
    .map((t) => t.r);
}

export function filterStocks(rows: FlowStock[], quad: string, sector: string): FlowStock[] {
  return rows.filter((r) => (quad === "all" || r.oi_quad_label === quad) && (sector === "all" || r.sector === sector));
}

export interface GammaBadge {
  text: string;
  kind: "pos" | "neg" | "none";
}

/** Regime badge text of an index gamma card. */
export function gammaBadge(g: IndexGamma | null | undefined): GammaBadge {
  if (!g || g.regime == null) return { text: "NO OPTION DATA", kind: "none" };
  return g.regime === "positive"
    ? { text: "POSITIVE GAMMA · range / mean-reverting", kind: "pos" }
    : { text: "NEGATIVE GAMMA · trending / volatile", kind: "neg" };
}

/** Distance of the zero-gamma level from spot, in percent of spot; null when either is missing. */
export function zeroGammaDistancePct(g: IndexGamma | null | undefined): number | null {
  return g && g.zero_gamma != null && g.spot ? (g.zero_gamma / g.spot - 1) * 100 : null;
}
