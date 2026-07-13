// ---- LLM provider setup (first-run modal) ----
export interface SynthCliInfo { installed: boolean; label: string; path: string | null }

export interface ProviderStatus {
  is_configured: boolean
  active: {
    macro: { mode: string; model: string; base_url?: string }
    synthesis: { backend: string; fast_model: string; deep_model: string }
  }
  synthesis_clis: Record<string, SynthCliInfo>
  macro: { endpoint_reachable: boolean; has_cloud_key: boolean }
}

export interface ProviderOptions {
  synthesis: { backend: string; label: string; install: string; login: string }[]
  macro: { mode: string; label: string; install: string; note: string }[]
}

export interface Signal {
  id?: number
  ticker: string
  price: number
  signal: 'BUY' | 'SELL' | 'HOLD' | 'SKIP'
  confidence: number
  stop_loss?: number
  target?: number
  ml_confidence?: number
  slm_confidence?: number
  claude_confidence?: number
  ml_reasoning?: string
  slm_reasoning?: string
  claude_reasoning?: string
  combined_reasoning?: string
  fired_at?: string
  status?: string
  actual_close?: number
  stock_name?: string
  sector?: string
  components_json?: ComponentStatuses | null
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
  /** "live" when live_* fields come from the realtime quote cache; "eod" = last daily close. */
  price_source?: 'live' | 'eod'
  live_ltp?: number | null
  live_pnl?: number | null
  live_pnl_pct?: number | null
  /** Unix seconds timestamp of the live_ltp tick. */
  live_ts?: number | null
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
  components_json?: ComponentStatuses | null
  /** Live LTP as of the last WS tick, joined in on the client (not persisted). */
  live_ltp?: number | null
  /** Unix seconds timestamp of the live_ltp tick. */
  live_ts?: number | null
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

// ---- Stage 0 truth layer ----
export interface ComponentStatus {
  status: 'ok' | 'degraded' | 'unavailable'
  detail: string
  source?: string   // llm_synthesis: claude|codex|gemini|skipped; macro: ollama_local|ollama_cloud|neutral_fallback
  stale?: boolean   // lightgbm only
}

export interface ComponentStatuses {
  lightgbm: ComponentStatus
  llm_synthesis: ComponentStatus
  macro: ComponentStatus
  /** Scheduler process heartbeat, derived from job_runs MAX(started_at). Optional: only present on newer backends. */
  scheduler?: ComponentStatus
  /** Upstox live WS quote feed connection + tick-staleness. Optional: only present on newer backends. */
  quote_feed?: ComponentStatus
}

export interface SystemHealth {
  generated_at: string
  components: ComponentStatuses
  data_freshness: DataStatus
  angel_one: { tripped: boolean | null; last_error: string | null; retry_in_seconds: number | null }
  lightgbm_model: { exists: boolean; mtime: string | null; age_hours: number | null }
}

export type Tab = 'watchlist' | 'portfolio' | 'brain' | 'charts' | 'market' | 'intelligence' | 'live' | 'logs'

// ---- Human-confirmed trade queue (sandbox order rehearsal) ----
export interface PendingConfirmation {
  id: number
  signal_id: number | null
  ticker: string
  action: 'BUY' | 'SELL'
  quantity: number
  price: number
  reasoning: string | null
  status: 'PENDING' | 'APPROVED' | 'REJECTED' | 'EXPIRED'
  created_at: string
  resolved_at: string | null
  order_id: string | null
  execution_status: string | null
  execution_detail: string | null
  is_sandbox: boolean
  is_stale?: boolean
}
