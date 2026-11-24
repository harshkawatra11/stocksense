// Response shapes of the read-only API (docs/PLAN-M.md key trees, M2.3 to M3.7).
// Routes whose trees are not yet fixed in the plan are typed loosely and are tightened by the card that renders them.

export interface FeedStatus {
  interval_s: number;
  last_ok: string | null;
  errors: number;
  last_err: string | null;
}
export interface StatusResponse {
  ready: boolean;
  feeds: Record<string, FeedStatus>;
  catchup: Record<string, unknown>;
  time: string;
  market: string;
  classified?: number;
  history?: number;
  progress?: unknown;
}
export interface HealthResponse {
  ok: boolean;
  version: string;
}

export interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// /api/sm/live
export interface SmIndex {
  ltp: number | null;
  pct: number | null;
  from_open: number | null;
  path?: number[];
  bars?: Candle[];
  read?: unknown;
  expiry?: string | null;
  delay?: number | null;
}
export interface SmStock {
  sym: string;
  name: string;
  sector: string;
  ltp: number | null;
  pct: number | null;
  from_open: number | null;
  disp: number | null;
  path?: number[];
  value_cr: number | null;
  ffmc: number | null;
  bars?: Candle[];
  pdh?: number | null;
  pdl?: number | null;
  liq?: ("PDH" | "PDL")[];
  read?: unknown;
  trail: number | null;
  rank: number;
  side: "bull" | "bear";
  flow?: Record<string, unknown>;
}
export interface SmSector {
  name: string;
  pct: number | null;
  up: number;
  down: number;
  n: number;
}
export interface SmLive {
  time: string;
  market: string;
  errors: unknown;
  indices: Record<"NIFTY" | "BANKNIFTY" | "SENSEX", SmIndex>;
  stocks: SmStock[];
  sectors: SmSector[];
  bull: string[];
  bear: string[];
}
// /api/sm/chart
export interface SmChart {
  sym: string;
  tf: "1m" | "5m" | "15m" | "60m" | "1d";
  ltp: number | null;
  prev: number | null;
  candles: Candle[];
  read: unknown;
}

// Sector Map, research, lab, flow, order flow, strategies, picks: loose until the card that renders them.
export type Json = Record<string, unknown>;
export type HorizonResponse = Json;
export type SectorResponse = Json;
export type StockResponse = Json;
export type SearchResponse = Json;
export type ResearchResponse = Json;
export type LabResponse = Json;
export type FlowResponse = Json;
export type OrderFlowResponse = Json;
export type StrategiesResponse = Json;
export type PicksResponse = Json;

export interface LearnNote {
  slug: string;
  title: string;
}
