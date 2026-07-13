import { useEffect, useMemo, useState } from 'react'
import type { PortfolioItem } from '../../types'
import { useRealtimePrices } from '../../hooks/useRealtimePrices'
import { useMarketStatus } from '../../hooks/useMarketStatus'
import FlashPrice from '../FlashPrice'
import PriceAgeDot from '../PriceAgeDot'

export default function Portfolio() {
  const [items, setItems] = useState<PortfolioItem[]>([])
  // Distinguishes "fetch failed" from "genuinely no holdings" — without this,
  // a backend outage silently rendered the same empty-state row as an
  // actually-empty portfolio, with no indication anything was wrong.
  const [error, setError] = useState<string | null>(null)
  const { prices: livePrices, connected: wsConnected } = useRealtimePrices()
  const marketStatus = useMarketStatus()
  const marketOpen = marketStatus.state === 'open'

  useEffect(() => {
    fetch('/api/portfolio/')
      .then(r => {
        if (!r.ok) throw new Error(`Portfolio fetch failed (${r.status})`)
        return r.json()
      })
      .then(d => { setItems(Array.isArray(d) ? d : []); setError(null) })
      .catch(e => setError(e instanceof Error ? e.message : 'Failed to load portfolio'))
  }, [])

  // Per-position current price: prefer the ticking WS quote (main.py subscribes
  // held tickers on the feed), then the server-joined live_ltp, then EOD close.
  const rows = useMemo(() =>
    items.map(item => {
      const wsTick = livePrices[item.ticker]
      const ltp = wsTick?.ltp ?? item.live_ltp ?? null
      const isLive = ltp != null
      const current = ltp ?? item.current_price
      const ts = wsTick?.ts ?? item.live_ts ?? null
      const pnlAbs = (current - item.avg_price) * item.quantity
      const pnlPct = ((current - item.avg_price) / item.avg_price) * 100
      return { item, current, isLive, ts, pnlAbs, pnlPct }
    }), [items, livePrices])

  // Summary recomputed from the same per-row prices so cards tick with the table.
  const summary = useMemo(() => {
    if (rows.length === 0) return null
    let invested = 0, currentVal = 0
    let anyLive = false
    for (const r of rows) {
      invested += r.item.avg_price * r.item.quantity
      currentVal += r.current * r.item.quantity
      if (r.isLive) anyLive = true
    }
    return {
      invested, currentVal,
      pnl: currentVal - invested,
      pnlPct: (currentVal - invested) / (invested + 1e-8) * 100,
      anyLive,
    }
  }, [rows])

  const asOfNote = !marketOpen ? 'as of 15:30 close' : null

  return (
    <div className="p-6">
      {summary && (
        <div className="grid grid-cols-4 gap-4 mb-6">
          {[
            { label: 'Invested', value: `₹${summary.invested.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`, color: 'text-text-primary', watch: null as number | null },
            { label: 'Current', value: `₹${summary.currentVal.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`, color: 'text-text-primary', watch: summary.currentVal },
            { label: 'P&L', value: `₹${summary.pnl.toFixed(0)}`, color: summary.pnl >= 0 ? 'text-green' : 'text-red', watch: summary.pnl },
            { label: 'P&L %', value: `${summary.pnlPct.toFixed(2)}%`, color: summary.pnlPct >= 0 ? 'text-green' : 'text-red', watch: summary.pnlPct },
          ].map(c => (
            <div key={c.label} className="bg-bg-card border border-border rounded p-4">
              <p className="text-xs text-text-secondary mb-1">
                {c.label}
                {c.label === 'Current' && (
                  <span className="ml-1.5 text-[10px] opacity-70">
                    {summary.anyLive && marketOpen ? 'live' : asOfNote ?? 'EOD'}
                  </span>
                )}
              </p>
              <p className={`text-lg font-semibold ${c.color}`}>
                {c.watch != null
                  ? <FlashPrice value={Math.round(c.watch * 100)} className="">{c.value}</FlashPrice>
                  : c.value}
              </p>
            </div>
          ))}
        </div>
      )}

      <div className="bg-bg-card border border-border rounded overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-bg-hover">
            <tr className="text-text-secondary text-xs">
              {['Ticker', 'Qty', 'Avg Price', `Current${asOfNote ? ` (${asOfNote})` : ''}`, 'P&L', 'P&L %', 'Sector'].map(h => (
                <th key={h} className="text-left px-4 py-3">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {error ? (
              <tr><td colSpan={7} className="px-4 py-8 text-center text-red text-xs">
                Failed to load portfolio: {error}
              </td></tr>
            ) : rows.length === 0 ? (
              <tr><td colSpan={7} className="px-4 py-8 text-center text-text-secondary text-xs">
                No holdings. Add positions via API: POST /api/portfolio/add
              </td></tr>
            ) : rows.map(({ item, current, isLive, ts, pnlAbs, pnlPct }) => (
              <tr key={item.ticker} className="border-t border-border hover:bg-bg-hover">
                <td className="px-4 py-3 font-medium text-white">{item.ticker}</td>
                <td className="px-4 py-3 text-text-primary">{item.quantity}</td>
                <td className="px-4 py-3 text-text-primary">₹{item.avg_price.toFixed(1)}</td>
                <td className="px-4 py-3 text-text-primary">
                  <span className="flex items-center gap-1.5">
                    {isLive && <PriceAgeDot ts={ts} connected={wsConnected} marketOpen={marketOpen} />}
                    <FlashPrice value={current} className="">₹{current.toFixed(1)}</FlashPrice>
                    {!isLive && <span className="text-[10px] text-text-secondary opacity-60">EOD</span>}
                  </span>
                </td>
                <td className={`px-4 py-3 ${pnlAbs >= 0 ? 'text-green' : 'text-red'}`}>
                  <FlashPrice value={Math.round(pnlAbs)} className="">₹{pnlAbs.toFixed(0)}</FlashPrice>
                </td>
                <td className={`px-4 py-3 ${pnlPct >= 0 ? 'text-green' : 'text-red'}`}>
                  {pnlPct.toFixed(2)}%
                </td>
                <td className="px-4 py-3 text-text-secondary text-xs">{item.sector}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
