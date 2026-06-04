import { useEffect, useState } from 'react'
import { fetchRecentSignals } from '../../api/signals'
import type { Signal } from '../../types'

export default function Watchlist() {
  const [signals, setSignals] = useState<Signal[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    fetchRecentSignals(100)
      .then(setSignals)
      .catch(() => setSignals([]))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div className="px-4 py-3 border-b border-border bg-bg-card flex-shrink-0">
        <span className="text-sm font-medium text-white">Watchlist</span>
        <p className="text-xs text-text-secondary mt-0.5">Recent signals — live SmartAPI feed in Phase 2</p>
      </div>

      <div className="flex-1 overflow-y-auto">
        {loading && <p className="text-xs text-text-secondary px-4 py-3">Loading...</p>}
        {!loading && signals.length === 0 && (
          <p className="text-xs text-text-secondary px-4 py-3">No signals yet. Run the pipeline from the Intelligence tab.</p>
        )}
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-bg-card">
            <tr className="text-text-secondary border-b border-border">
              <th className="text-left px-4 py-2">Ticker</th>
              <th className="text-left px-2 py-2">Signal</th>
              <th className="text-right px-2 py-2">Price</th>
              <th className="text-right px-2 py-2">Target</th>
              <th className="text-right px-4 py-2">Confidence</th>
            </tr>
          </thead>
          <tbody>
            {signals.map((s, i) => (
              <tr key={i} className="border-b border-border hover:bg-bg-hover transition-colors">
                <td className="px-4 py-2 font-medium text-white">{s.ticker}</td>
                <td className={`px-2 py-2 font-medium ${s.signal === 'BUY' ? 'text-green' : s.signal === 'SELL' ? 'text-red' : 'text-yellow'}`}>
                  {s.signal === 'BUY' ? '▲' : s.signal === 'SELL' ? '▼' : '—'} {s.signal}
                </td>
                <td className="px-2 py-2 text-right text-text-primary">₹{s.price?.toFixed(1)}</td>
                <td className="px-2 py-2 text-right text-green">₹{s.target?.toFixed(1) || '—'}</td>
                <td className="px-4 py-2 text-right text-text-secondary">{((s.confidence || 0) * 100).toFixed(0)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
