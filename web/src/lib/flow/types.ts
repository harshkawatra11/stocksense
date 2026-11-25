// Shapes of the /api/flow/* replies (src/stocksense/api/routes_flow.py, flow/engine.py). List routes are wrapped:
// {descriptive: true, rows: [...]}, never bare lists.

export interface Wrapped<T> {
  descriptive: boolean;
  rows: T[];
}

export interface Participant {
  cash_net: number | null;
  cash_net_5d: number | null;
  idx_fut_ratio: number | null;
  ratio_chg5: number | null;
}

export interface IndexGamma {
  spot: number | null;
  pcr_oi: number | null;
  gex_total_cr: number | null;
  regime: "positive" | "negative" | null;
  zero_gamma: number | null;
  call_wall: number | null;
  put_wall: number | null;
  max_pain: number | null;
  atm_iv: number | null;
}

export interface FlowMarket {
  descriptive: boolean;
  asof: string | null;
  live: boolean;
  fii: Participant;
  dii: Participant;
  pro: Participant;
  client: Participant;
  indices: Record<string, IndexGamma | null>;
  vix: { close: number; chg1: number | null } | null;
  breadth: { adv: number; dec: number };
  preopen: { adv: number; dec: number; nifty_iep: number | null } | null;
}

export interface ParticipantDay {
  date: string;
  fii_ratio: number | null;
  dii_ratio: number | null;
  pro_ratio: number | null;
  client_ratio: number | null;
  fii_cash_net: number | null;
  dii_cash_net: number | null;
}

export interface FlowStock {
  sym: string;
  name: string;
  sector: string;
  ltp: number | null;
  pct: number | null;
  disp: number | null;
  rvol: number | null;
  oi_quad: number | null;
  oi_quad_label: string | null;
  oi_chg_pct: number | null;
  deliv_z20_prev: number | null;
  depth_imb5: number | null;
  buy_sell_ratio: number | null;
  preopen_imb: number | null;
  cvd_slope_30m: number | null;
  flow_score: number | null;
}

export interface OiRow {
  symbol: string;
  quad_label: string | null;
  oi_chg_pct: number | null;
  price_chg_pct: number | null;
  basis_pct: number | null;
  rollover_pct: number | null;
}

export interface PreopenRow {
  symbol: string;
  iep: number | null;
  iep_gap_pct: number | null;
  imb: number | null;
  total_qty: number | null;
  follow_through?: boolean | null;
}

export interface Preopen {
  descriptive: boolean;
  phase: "before" | "live" | "after";
  nifty: { iep: number | null; gap_pct: number | null } | null;
  rows: PreopenRow[];
  breadth: { adv: number; dec: number };
}

export interface SectorFlow {
  name: string;
  flow_dir: number | null;
  turnover_share: number | null;
  share_ratio: number | null;
  rotation: "in" | "out" | "neutral";
  up: number;
  down: number;
  oi_quads: Record<string, number>;
  deliv_spikes: number;
}

export interface Deal {
  date: string;
  symbol: string;
  buy_value_cr: number | null;
  sell_value_cr: number | null;
  net_value_cr: number | null;
  n_deals: number;
  top_client: string | null;
  flagged: boolean;
}

export interface OptionStrike {
  strike: number;
  call_oi: number | null;
  put_oi: number | null;
  call_doi: number | null;
  put_doi: number | null;
  call_iv: number | null;
  put_iv: number | null;
  gex_cr: number | null;
}

export interface OptionTotals {
  gex_total_cr: number | null;
  zero_gamma: number | null;
  call_wall: number | null;
  put_wall: number | null;
  max_pain: number | null;
  pcr_oi: number | null;
  pcr_doi: number | null;
}

export interface OptionsReply {
  descriptive: boolean;
  und: string;
  spot: number | null;
  expiry: string | null;
  expiries: string[];
  strikes: OptionStrike[];
  totals: OptionTotals;
  history: { date: string; gex_total_cr: number | null; pcr_oi: number | null; spot: number | null }[];
}

/** Rows of a wrapped list reply; tolerates a bare array and a missing reply. */
export function rowsOf<T>(d: Wrapped<T> | T[] | null | undefined): T[] {
  return Array.isArray(d) ? d : (d?.rows ?? []);
}
