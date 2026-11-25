// Types and pure helpers for /picks (docs/PLAN-M.md M5.9). Capital and broker details never appear in the reply.

export interface Pick {
  sym: string;
  side: number;
  entry_ref: number;
  stop: number;
  target: number | null;
  strategies: string[];
  evidence: number;
  best_trial: string;
  net_sharpe_5l: number;
  dsr: number;
  pbo: number;
  trades_oos: number;
  qty: number;
  notional_inr: number;
  risk_inr: number;
  reason: string;
}
export interface PicksReply {
  asof: string;
  gated: Pick[];
  exploratory: Pick[];
  n_strategies_passing: number;
  n_trials_total: number;
  note: string;
}
export interface HistoryRow {
  asof: string;
  sym: string;
  strat?: string;
  status: string;
  entry?: number | null;
  exit?: number | null;
  r: number | null;
  net_pct: number | null;
}

export const EXPLORATORY_LABEL = "EXPLORATORY — not validated";
export const FOOTER = "Backtests decide which strategies may pick; past results do not guarantee future returns.";

/** Evidence bar width (0 to 100) relative to the strongest gated pick. */
export function evidencePct(p: Pick, all: Pick[]): number {
  const m = Math.max(0, ...all.map((x) => x.evidence));
  return m > 0 ? Math.round((Math.max(0, p.evidence) / m) * 100) : 0;
}

/** The two lists as received; a missing list is empty. They are rendered in separate sections and never merged. */
export const splitPicks = (r: PicksReply): { gated: Pick[]; exploratory: Pick[] } => ({
  gated: r.gated ?? [],
  exploratory: r.exploratory ?? [],
});

/** The header note: the reply's own note, or the fixed no-pass sentence when nothing passed and none was given. */
export function headerNote(r: PicksReply): string {
  if (r.note) return r.note;
  if (!(r.gated ?? []).length)
    return `No strategy has passed the gate yet (n_trials_total = ${r.n_trials_total}). Exploratory candidates below are not evidence.`;
  return "";
}

export const rows = (h: HistoryRow[] | { rows?: HistoryRow[] } | null | undefined): HistoryRow[] =>
  Array.isArray(h) ? h : (h?.rows ?? []);

export const inr = (x: number): string => "₹" + Math.round(x).toLocaleString("en-IN");
