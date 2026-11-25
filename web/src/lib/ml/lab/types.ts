// Shapes of /api/lab/swing (src/stocksense/strategies/scan.py summary) and /api/lab/strategies.
// Market Lab's richer fields (curve, regime, ew, nifty, robust) are optional: StockSense's scan may not carry them.

export interface SwingStrategy {
  key: string;
  exit: string;
  exit_name: string;
  n: number;
  win: number;
  net: number;
  excess: number;
  days: number;
  IS: number | null;
  OOS: number | null;
  OOS_exc: number | null;
  years: Record<string, number>;
  gated: string | null;
}

export interface MlMetrics {
  "CAGR%": number;
  "maxDD%": number;
  Sharpe: number;
  "years%": Record<string, number | null>;
}

export interface OwnMetrics {
  n: number;
  win: number;
  avgR: number;
  totR: number;
  net_pct: number;
}

export type CurvePoint = [string, number];

export interface SwingCandidate {
  strat: string;
  exit: string;
  k: number;
  label: string;
  rules: string[];
  gated: string | null;
  exploratory: boolean;
  metrics: (Partial<MlMetrics> & Partial<OwnMetrics>) | null;
  ew?: MlMetrics;
  nifty?: MlMetrics;
  curve?: CurvePoint[];
  curve_ew?: CurvePoint[];
  curve_nifty?: CurvePoint[];
  /** [date, 1 = market filter on, 0 = cash] */
  regime?: [string, number][];
  skipped?: { n: number; avg: number };
}

export interface LiveRow {
  sym: string;
  strat: string;
  state: string;
  t_in: number | null;
  p_in: number;
  p_out?: number;
  stop: number;
  target: number | null;
  rs63: number | null;
  dist_pct: number | null;
  risk_pct: number | null;
  trigger: number | null;
  close: number | null;
  strat_name: string;
  pct?: number;
  R?: number;
  t_out?: number;
}

export interface SwingLive {
  time: number;
  asof: string;
  nifty: number;
  nifty_200: number;
  mkt_up: boolean;
  signals: LiveRow[];
  open: LiveRow[];
  forming: LiveRow[];
  recent?: LiveRow[];
}

export interface ProfileRow {
  k: string;
  n: number;
  win: number;
  avg: number;
}

export interface BestStock {
  sym: string;
  sector: string;
  n: number;
  win: number;
  avg: number;
  tot: number;
}

export interface SwingData {
  error?: string;
  built: string;
  asof: string;
  cost_pct: number | null;
  stocks: number;
  trades: number;
  strategies: Record<string, SwingStrategy>;
  removed: string[];
  profile: Record<string, ProfileRow[]>;
  best_stocks: BestStock[];
  candidate: SwingCandidate;
  live: SwingLive | null;
  list: string[];
  exits: Record<string, string>;
  robust?: [string, string][];
  exploratory: boolean;
}

export interface RegistryRow {
  key: string;
  name: string;
  source: string;
  tier: string;
  timeframe: string;
  gated: string | null;
}
