import type { PortfolioItem } from '../types'

const BASE = '/api'

export async function fetchPortfolio(): Promise<PortfolioItem[]> {
  const r = await fetch(`${BASE}/portfolio`)
  if (!r.ok) throw new Error('Failed to fetch portfolio')
  const d = await r.json()
  return Array.isArray(d) ? d : []
}

export async function fetchPnlSummary(): Promise<{ total_pnl: number; total_pct: number; day_pnl: number }> {
  const r = await fetch(`${BASE}/portfolio/pnl`)
  // A silent {0,0,0} fallback here is indistinguishable from a genuinely flat
  // day on a live-trading dashboard — throw instead so the caller can show an
  // explicit error state rather than a misleading "no P&L" reading.
  if (!r.ok) throw new Error('Failed to fetch P&L summary')
  return r.json()
}
