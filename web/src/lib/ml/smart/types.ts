// Shapes of /api/sm/*, /api/lab/trades and /api/research as the Smart Money section reads them.
// Everything the backend may leave out is optional: Market Lab's code guarded each with `|| {}`.

export type Bar = [number, number, number, number]; // open, high, low, close
export type PathPt = [number, number]; // minute since 09:15, % from the open

export interface StructInfo {
  type: string; // "BOS" | "CHoCH"
  dir: "bull" | "bear";
  time: string;
  level?: number | string;
}
export interface DistInfo {
  dist_pct: number;
}
export interface SmartRead {
  struct?: StructInfo | null;
  ob?: DistInfo | null;
  fvg?: DistInfo | null;
  ev?: [number, ...unknown[]][];
}

// The optional flow mini-card that /api/sm/live rows carry from M3.7 on.
export interface FlowCard {
  disp?: number | null;
  rvol?: number | null;
  oi_quad?: string | null;
  oi_quad_label?: string | null;
  oi_chg_pct?: number | null;
  deliv_z20_prev?: number | null;
  flow_score?: number | null;
}

export interface SmartStock {
  sym: string;
  name?: string;
  sector: string;
  ltp: number;
  pct: number;
  from_open: number;
  disp: number;
  path?: PathPt[];
  value_cr?: number | null;
  ffmc?: number | null;
  bars?: Bar[] | null;
  pdh?: number | null;
  pdl?: number | null;
  liq?: string[] | null;
  read?: SmartRead | null;
  trail: number | null;
  rank: number;
  side: "bull" | "bear";
  flow?: FlowCard | null;
}
export interface SmartIndex {
  ltp: number;
  pct: number;
  from_open: number;
  path?: PathPt[];
  bars?: Bar[] | null;
  read?: (SmartRead & { liq?: { swept?: string[] } }) | null;
  expiry?: string | null;
  delay?: number | null;
}
export interface SmartSector {
  name: string;
  pct: number;
  up: number;
  down: number;
  n: number;
}
export type IndexKey = "NIFTY" | "BANKNIFTY" | "SENSEX";
export interface SmartLive {
  time: string;
  market: string;
  errors: unknown;
  indices: Partial<Record<IndexKey, SmartIndex>>;
  stocks: SmartStock[];
  sectors: SmartSector[];
  bull: string[];
  bear: string[];
}

export type Hover = { type: "stock" | "sector"; key: string } | null;

// ---- the pop-up chart ----
export type Tf = "1m" | "5m" | "15m" | "60m" | "1d";
export interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}
export interface ZoneRow {
  dir: number;
  lo: number;
  hi: number;
  dist_pct: number;
  tested?: boolean | null;
  status?: string;
  time?: number;
}
export interface ChartRead {
  chart?: {
    vwap?: { time: number; value: number }[];
    events?: { time: number; dir: string; type: string }[];
    obs?: ZoneRow[];
    fvgs?: ZoneRow[];
  };
  liquidity?: { pdh?: number | null; pdl?: number | null; or_hi?: number | null; or_lo?: number | null; swept?: string[] };
  structure: { last?: { type: string; dir: string; time: string; level: number | string } | null; breaks_today: number };
  order_blocks: ZoneRow[];
  fvgs: ZoneRow[];
  events?: { time: string; text: string }[];
  buyers_pct?: number | null;
  vwap?: number | null;
  price?: number;
}
export interface ChartData {
  sym: string;
  tf: Tf;
  ltp: number | null;
  prev: number | null;
  candles: Candle[];
  read: ChartRead | null;
  error?: string;
}
export interface QuoteData {
  ltp?: number | null;
  time: number;
  prev?: number | null;
}

// ---- trades (Market Lab shape + StockSense additions) ----
export interface GatedVerdict {
  verdict?: string | null; // "PASS" | "FAIL"
  lines_passed?: number | null;
  net_sharpe_5l?: number | null;
  sharpe_2x?: number | null;
  trial_id?: string | null;
}
export interface Trade {
  t_in: number;
  t_out?: number | null;
  p_in: number;
  p_out?: number | null;
  stop: number;
  target?: number | null;
  side: number;
  status: string; // CLOSED | OPEN | SIGNAL
  R?: number;
  pct?: number | null;
}
export interface SetupState {
  state: string; // FORMING | SIGNAL | OPEN | other
  trigger?: number;
  stop?: number;
  target?: number;
  p_in?: number;
  dist_pct?: number;
  mkt_up?: boolean;
  trend?: boolean;
  near_high?: boolean;
  from_high?: number;
  tight?: boolean;
}
export interface TradesResponse {
  sym?: string;
  strat?: string;
  exit?: string;
  exit_name?: string;
  swing?: boolean;
  candidate?: boolean;
  cost_pct?: number | null;
  trades: Trade[];
  state?: SetupState | null;
  summary?: { n?: number; win?: number | null; totR?: number; pct?: number; avgR?: number | null };
  days_held?: number[];
  note?: string;
  error?: string;
  // StockSense additions (M2.6)
  exploratory?: boolean;
  costs_model?: string;
  gated?: GatedVerdict | null;
}
export interface StrategyRow {
  key: string;
  name: string;
  source: string;
  tier?: string;
  timeframe?: string;
  exits?: string[];
  gated?: GatedVerdict | null;
}

// ---- research note ----
export interface ResearchNote {
  sym: string;
  name?: string;
  built: number;
  macro?: string;
  sector?: string;
  industry?: string;
  basic?: string;
  index?: string;
  what?: string;
  summary: string[];
  wiki?: { text: string; url?: string } | null;
  mcap_cr?: number | null;
  pe?: number | null;
  sector_pe?: number | null;
  delivery?: number | null;
  high52?: number | null;
  low52?: number | null;
  ltp?: number | null;
  results: { label: string; income?: number | null; profit?: number | null }[];
  ttm?: {
    income?: number | null;
    profit?: number | null;
    margin?: number | null;
    norm?: number | string | null;
    g_inc?: number | null;
    g_pat?: number | null;
    profit_q?: number;
    n_q?: number;
  };
  balance: { de?: number | null; as_of?: string | null };
  financial?: boolean;
  peers: { sym: string; name?: string; mcap_cr?: number | null; pe?: number | null }[];
  sector_strength?: { sector?: string; sector_3m?: number | null; nifty_3m?: number | null };
  economy: {
    name: string;
    why: string;
    factor?: string;
    effect: number;
    push: number | null;
    reading?: { chg3m?: number | null } | null;
  }[];
  tech?: { ret1y?: number | null; rs63?: number | null };
  scores?: Record<string, number | null>;
  weights: Record<string, number>;
  total?: number | null;
  grade?: string;
  swot?: Record<string, string[]>;
  sources: string[];
  error?: string;
}
