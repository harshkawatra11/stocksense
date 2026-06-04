import { useEffect, useState } from 'react'
import { fetchPortfolio, fetchPnlSummary } from '../api/portfolio'
import type { PortfolioItem } from '../types'

export function usePortfolio() {
  const [items, setItems] = useState<PortfolioItem[]>([])
  const [pnl, setPnl] = useState({ total_pnl: 0, total_pct: 0, day_pnl: 0 })
  const [loading, setLoading] = useState(false)

  const load = async () => {
    setLoading(true)
    try {
      const [portfolio, summary] = await Promise.all([fetchPortfolio(), fetchPnlSummary()])
      setItems(portfolio)
      setPnl(summary)
    } catch {}
    setLoading(false)
  }

  useEffect(() => { load() }, [])

  return { items, pnl, loading, refresh: load }
}
