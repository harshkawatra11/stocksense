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

export type Tab = 'watchlist' | 'portfolio' | 'orders' | 'charts' | 'market' | 'intelligence'
