import { useEffect, useState } from 'react'
import type { PortfolioItem } from '../../types'

export default function Portfolio() {
  const [items, setItems] = useState<PortfolioItem[]>([])
  const [pnl, setPnl] = useState<{ total_invested: number; total_current: number; total_pnl: number; total_pnl_pct: number } | null>(null)

  useEffect(() => {
    fetch('/api/portfolio/').then(r => r.json()).then(d => setItems(Array.isArray(d) ? d : []))
    fetch('/api/portfolio/pnl').then(r => r.json()).then(setPnl)
  }, [])

  return (
    <div className="p-6">
      {pnl && (
        <div className="grid grid-cols-4 gap-4 mb-6">
          {[
            { label: 'Invested', value: `₹${pnl.total_invested.toLocaleString('en-IN')}`, color: 'text-text-primary' },
            { label: 'Current', value: `₹${pnl.total_current.toLocaleString('en-IN')}`, color: 'text-text-primary' },
            { label: 'P&L', value: `₹${pnl.total_pnl.toFixed(0)}`, color: pnl.total_pnl >= 0 ? 'text-green' : 'text-red' },
            { label: 'P&L %', value: `${pnl.total_pnl_pct.toFixed(2)}%`, color: pnl.total_pnl_pct >= 0 ? 'text-green' : 'text-red' },
          ].map(c => (
            <div key={c.label} className="bg-bg-card border border-border rounded p-4">
              <p className="text-xs text-text-secondary mb-1">{c.label}</p>
              <p className={`text-lg font-semibold ${c.color}`}>{c.value}</p>
            </div>
          ))}
        </div>
      )}

      <div className="bg-bg-card border border-border rounded overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-bg-hover">
            <tr className="text-text-secondary text-xs">
              {['Ticker', 'Qty', 'Avg Price', 'Current', 'P&L', 'P&L %', 'Sector'].map(h => (
                <th key={h} className="text-left px-4 py-3">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {items.length === 0 ? (
              <tr><td colSpan={7} className="px-4 py-8 text-center text-text-secondary text-xs">
                No holdings. Add positions via API: POST /api/portfolio/add
              </td></tr>
            ) : items.map(item => (
              <tr key={item.ticker} className="border-t border-border hover:bg-bg-hover">
                <td className="px-4 py-3 font-medium text-white">{item.ticker}</td>
                <td className="px-4 py-3 text-text-primary">{item.quantity}</td>
                <td className="px-4 py-3 text-text-primary">₹{item.avg_price.toFixed(1)}</td>
                <td className="px-4 py-3 text-text-primary">₹{item.current_price.toFixed(1)}</td>
                <td className={`px-4 py-3 ${item.pnl_abs >= 0 ? 'text-green' : 'text-red'}`}>
                  ₹{item.pnl_abs.toFixed(0)}
                </td>
                <td className={`px-4 py-3 ${item.pnl_pct >= 0 ? 'text-green' : 'text-red'}`}>
                  {item.pnl_pct.toFixed(2)}%
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
