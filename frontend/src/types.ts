export interface Signal {
  id?: number
  ticker: string
  price: number
  signal: 'BUY' | 'SELL' | 'HOLD' | 'SKIP'
  confidence: number
  stop_loss?: number
  target?: number
  ml_confidence?: number
  kronos_confidence?: number
  slm_confidence?: number
  claude_confidence?: number
  ml_reasoning?: string
  kronos_reasoning?: string
  slm_reasoning?: string
  claude_reasoning?: string
  combined_reasoning?: string
  fired_at?: string
  status?: string
  actual_close?: number
  stock_name?: string
  sector?: string
}

export interface TerminalLine {
  ts: string
  text: string
  color: 'green' | 'red' | 'yellow' | 'grey' | 'white'
  ticker?: string
}

export interface PortfolioItem {
  ticker: string
  name: string
  sector: string
  quantity: number
  avg_price: number
  current_price: number
  pnl_pct: number
  pnl_abs: number
  buy_date: string
}

export interface Learning {
  id: number
  learning_date: string
  learning_type: string
  ticker?: string
  title: string
  body: string
  tags?: string[]
}

export interface LiveSignal {
  id: number
  ticker: string
  name?: string
  sector?: string
  timeframe: string
  horizon_days?: number
  price_at_signal: number
  target_price?: number
  stop_loss?: number
  final_confidence: number
  affordable?: boolean
  shares_affordable?: number
  macro_sector_score?: number
  target_eta_days?: number
  expected_move_pct?: number
  fired_at?: string
}

export interface Account {
  cash_available: number
  cash_reserve: number
  deployable_note?: string
}

export interface ActivityEvent {
  id: number
  event_type: 'SUGGESTED' | 'RATED' | 'BOUGHT' | 'SOLD' | 'REANALYZED' | 'NOTE'
    | 'AUTO_BUY' | 'AUTO_SELL' | 'AUTO_PASS' | 'PARAM_CHANGE' | 'RETRAIN' | 'JOB_RUN'
  ticker?: string
  name?: string
  rating?: 'LIKE' | 'DISLIKE'
  note?: string
  payload?: Record<string, unknown>
  created_at: string
}

export interface PositionReview {
  ticker: string
  days_elapsed?: number
  eta_days?: number
  entry_price?: number
  target_price?: number
  current_price?: number
  progress_pct?: number
  status: string
  verdict: string
  reasoning?: string
  reviewed_at: string
}

export interface DataStatus {
  latest_date: string | null
  age_hours: number | null
  source: 'live_intraday' | 'eod_daily'
  is_live: boolean
  label: string
}

export interface TradingMode {
  mode: 'PAPER' | 'LIVE'
  reason: string
  resolved_count: number
  span_days: number
  rolling_accuracy: number | null
  gate: { min_days: number; min_resolved: number; min_accuracy: number }
}

export interface BrainParam {
  param_name: string
  value: number
  min_value: number
  max_value: number
  updated_at: string
  updated_by: string
  reason: string
}

export interface BrainParamChange {
  param_name: string
  old_value: number | null
  new_value: number
  changed_by: string
  reason: string
  changed_at: string
}

export interface JobRun {
  job_id: string
  started_at: string
  finished_at: string | null
  status: 'running' | 'ok' | 'error'
  summary: string
  error: string | null
  cadence?: string
}

export interface EquityPoint {
  date: string
  cash: number
  market_value: number
  equity: number
}

export interface Decision {
  id: number
  ticker: string
  action: 'BUY' | 'SELL' | 'PASS' | 'SKIP'
  quantity: number
  price: number | null
  cash_after: number | null
  rationale: string
  outcome: string | null
  pnl: number | null
  decided_at: string
  resolved_at: string | null
  timeframe?: string
  final_confidence?: number
}

export interface BrainStatus {
  autonomous: boolean
  mode: TradingMode
  data: DataStatus
  params: BrainParam[]
  jobs: JobRun[]
  account: {
    cash_available: number
    cash_reserve: number
    positions: number
    market_value: number
    equity: number
  }
  today: { buys: number; sells: number; passes: number; realized_pnl: number }
  model: { lgbm_latest_mtime: string | null }
  groww_feed: { configured: boolean }
}

export type Tab = 'watchlist' | 'portfolio' | 'brain' | 'charts' | 'market' | 'intelligence' | 'live' | 'logs'
