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
  if (!r.ok) return { total_pnl: 0, total_pct: 0, day_pnl: 0 }
  return r.json()
}
