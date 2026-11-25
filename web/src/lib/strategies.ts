// Pure helpers for the /strategies page: gate verdict normalising, tier-vs-evidence summary, matrix pivot and colours.

export interface GateObj {
  verdict?: string | null;
  lines_passed?: number | null;
  net_sharpe_5l?: number | null;
  sharpe_2x?: number | null;
  trades_oos?: number | null;
  positive_years?: number | null;
  trial_id?: string | null;
}
/** /api/lab/strategies `gated` is a verdict string today and may become an object later. */
export type Gated = string | GateObj | null | undefined;

export interface LibRow {
  key: string;
  name: string;
  source: string;
  tier: string;
  timeframe: string;
  kind: string;
  direction?: string;
  exits?: string[];
  description?: string;
  reference?: string;
  gated?: Gated;
}
export interface Gate {
  tested: boolean;
  pass: boolean;
  verdict: string | null;
  lines: number | null;
  sharpe5: number | null;
  sharpe2x: number | null;
  trades: number | null;
  posYears: number | null;
}

export function gateOf(g: Gated): Gate {
  const o: GateObj = typeof g === "string" ? { verdict: g } : (g ?? {});
  const verdict = o.verdict ? String(o.verdict).toUpperCase() : null;
  return {
    tested: verdict !== null,
    pass: verdict === "PASS",
    verdict,
    lines: o.lines_passed ?? null,
    sharpe5: o.net_sharpe_5l ?? null,
    sharpe2x: o.sharpe_2x ?? null,
    trades: o.trades_oos ?? null,
    posYears: o.positive_years ?? null,
  };
}

export const median = (xs: number[]): number | null => {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b),
    m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

export interface TierStat {
  tier: string;
  n: number;
  tested: number;
  passing: number;
  medianSharpe: number | null;
}
export const TIERS = ["F", "C", "B"] as const;

/** One row per tier F, C, B: strategy count, how many have a gate verdict, how many pass, median net Sharpe at 5 lakh. */
export function tierSummary(rows: LibRow[]): TierStat[] {
  return TIERS.map((t) => {
    const mine = rows.filter((r) => r.tier === t);
    const gates = mine.map((r) => gateOf(r.gated));
    const sh = gates.flatMap((g) => (g.sharpe5 == null ? [] : [g.sharpe5]));
    return {
      tier: t,
      n: mine.length,
      tested: gates.filter((g) => g.tested).length,
      passing: gates.filter((g) => g.pass).length,
      medianSharpe: median(sh),
    };
  });
}

/** "Claimed order F < C < B; measured order …" (ascending median Sharpe) or an honest "not yet measured". */
export function tierSentence(stats: TierStat[]): string {
  const claimed = "Claimed order F < C < B";
  const m = stats.filter((s) => s.medianSharpe != null);
  if (m.length < stats.length) return `${claimed}; measured order not available yet (${m.length} of ${stats.length} tiers have a net Sharpe).`;
  const order = [...m].sort((a, b) => (a.medianSharpe as number) - (b.medianSharpe as number));
  return `${claimed}; measured order ${order.map((s) => s.tier).join(" < ")}.`;
}

export interface MatrixRow {
  strat: string;
  name: string;
  tier: string;
  exit: string;
  n: number;
  avgR: number | null;
  gated?: Gated;
}
export interface MatrixReply {
  sym: string;
  tf: string;
  n_tests: number;
  expected_max_sharpe_null: number;
  rows: MatrixRow[];
}
export interface Pivot {
  exits: string[];
  strats: { strat: string; name: string; cells: Record<string, MatrixRow> }[];
}
/** Strategies x exits table from the flat matrix rows (order of first appearance). */
export function pivot(rows: MatrixRow[]): Pivot {
  const exits: string[] = [];
  const map = new Map<string, { strat: string; name: string; cells: Record<string, MatrixRow> }>();
  for (const r of rows) {
    if (!exits.includes(r.exit)) exits.push(r.exit);
    if (!map.has(r.strat)) map.set(r.strat, { strat: r.strat, name: r.name, cells: {} });
    map.get(r.strat)!.cells[r.exit] = r;
  }
  return { exits, strats: [...map.values()] };
}

/** Cell background for an average net R: red below 0, green above, saturating at +-0.5 R; grey when missing or no trades. */
export function rColor(avgR: number | null | undefined, n = 1): string {
  if (avgR == null || !n) return "#f1f3f6";
  const a = Math.min(Math.abs(avgR) / 0.5, 1) * 0.65;
  return avgR >= 0 ? `rgba(47,158,95,${a.toFixed(2)})` : `rgba(214,69,93,${a.toFixed(2)})`;
}
