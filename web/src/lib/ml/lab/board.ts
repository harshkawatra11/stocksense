// Port of lab.html board() plus StockSense additions (source/tier filters, Gate column, multiple-testing note).
// Ported from market-lab web/lab.html (MIT, ba3f771); changes: typed, filters and gate added, note computed client-side.
import type { RegistryRow, SwingStrategy } from "./types";

export type SourceFilter = "all" | "marketlab_swing" | "tier" | "k151" | "flow";
export type TierFilter = "all" | "F" | "C" | "B";

export const SOURCE_OPTIONS: { v: SourceFilter; label: string }[] = [
  { v: "all", label: "all" },
  { v: "marketlab_swing", label: "Market Lab swing" },
  { v: "tier", label: "Tier" },
  { v: "k151", label: "151" },
  { v: "flow", label: "Flow" },
];
export const TIER_OPTIONS: TierFilter[] = ["all", "F", "C", "B"];

/** Trading days assumed behind the leaderboard's trades (five years of daily bars); input to the multiple-testing note. */
export const LAB_N_OBS = 1260;

export interface BoardRow extends SwingStrategy {
  /** Display name (the key of `strategies` in the reply). */
  k: string;
  source: string;
  tier: string;
}

/** Source and tier of a registry key: from the registry row when known, else from the key prefix. */
export function classify(key: string, registry?: Map<string, RegistryRow>): { source: string; tier: string } {
  const r = registry?.get(key);
  if (r) return { source: r.source, tier: r.tier };
  const m = /^tier_([fcb])_/.exec(key);
  if (m) return { source: "tier", tier: m[1].toUpperCase() };
  if (key.startsWith("ml_swing_")) return { source: "marketlab_swing", tier: "none" };
  if (key.startsWith("k151_")) return { source: "k151", tier: "none" };
  if (key.startsWith("flow_")) return { source: "flow", tier: "none" };
  return { source: "other", tier: "none" };
}

/** Rows sorted by `excess` descending (the leaderboard order), each with its source and tier. */
export function boardRows(strategies: Record<string, SwingStrategy>, registry?: Map<string, RegistryRow>): BoardRow[] {
  return Object.entries(strategies)
    .map(([k, v]) => ({ k, ...v, ...classify(v.key, registry) }))
    .sort((a, b) => b.excess - a.excess);
}

export function filterRows(rows: BoardRow[], source: SourceFilter, tier: TierFilter): BoardRow[] {
  return rows.filter((r) => (source === "all" || r.source === source) && (tier === "all" || r.tier === tier));
}

/** Scale of the edge bars: the largest absolute edge, at least 0.3 (lab.html `mx`). */
export function barMax(rows: BoardRow[]): number {
  return Math.max(0.3, ...rows.map((r) => Math.abs(r.excess)), ...rows.map((r) => Math.abs(r.OOS_exc ?? 0)));
}

/** Pixel width of an edge bar: 50 px at the maximum. */
export const barWidth = (v: number, mx: number): number => (Math.abs(v) / mx) * 50;

export type GateKind = "pass" | "fail" | "none";
export function gateKind(g: string | null | undefined): GateKind {
  return g === "PASS" ? "pass" : g ? "fail" : "none";
}
export const gateText = (g: string | null | undefined): string => (g === "PASS" ? "PASS" : g ? "FAIL" : "not tested");

/** Inverse of the standard normal CDF (Acklam's rational approximation, relative error below 1.2e-9). */
export function invNormCdf(p: number): number {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const lo = 0.02425;
  if (p <= 0 || p >= 1) return p <= 0 ? -Infinity : Infinity;
  if (p < lo) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - lo) return -invNormCdf(1 - p);
  const q = p - 0.5;
  const r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** Expected best annualised Sharpe of nTests independent no-skill strategies over nObs days; same formula as
 *  explore.expected_max_sharpe_null (Bailey and Lopez de Prado 2014). 0 for a single test. */
export function expectedMaxSharpeNull(nTests: number, nObs: number): number {
  if (nTests <= 1 || nObs <= 0) return 0;
  const g = 0.5772156649015329;
  const z = (1 - g) * invNormCdf(1 - 1 / nTests) + g * invNormCdf(1 - 1 / (nTests * Math.E));
  return Math.sqrt(252 / nObs) * z;
}

export function multipleTestingNote(n: number, nObs = LAB_N_OBS): string {
  const x = expectedMaxSharpeNull(n, nObs);
  return `${n} strategy×exit combinations shown; the best of ${n} random strategies would show Sharpe ≈ ${x.toFixed(2)} by luck (expected_max_sharpe_null). A good-looking row here is a hypothesis, not evidence: only the gate verdict counts.`;
}
