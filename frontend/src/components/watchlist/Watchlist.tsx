import { useEffect, useState } from 'react'
import { TrendingUp, TrendingDown, Minus, RefreshCw } from 'lucide-react'

interface SignalRow {
  id: number
  ticker: string
  signal_type: string
  price_at_signal: number
  target_price?: number
  stop_loss?: number
  final_confidence: number
  fired_at: string
  stock_name?: string
  sector?: string
}

export default function Watchlist() {
  const [signals, setSignals] = useState<SignalRow[]>([])
  const [loading, setLoading] = useState(false)
  const [filter, setFilter] = useState<'ALL' | 'BUY' | 'SELL'>('ALL')

  function load() {
    setLoading(true)
    fetch('/api/signals/recent?limit=200')
      .then(r => r.json())
      .then(d => setSignals(Array.isArray(d) ? d : []))
      .catch(() => setSignals([]))
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  const filtered = filter === 'ALL' ? signals : signals.filter(s => s.signal_type === filter)

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div className="px-4 py-3 border-b border-border bg-bg-card flex-shrink-0 flex items-center gap-3">
        <span className="text-sm font-medium text-white">Recent Signals</span>
        <div className="flex gap-1 ml-4">
          {(['ALL', 'BUY', 'SELL'] as const).map(f => (
            <button key={f} onClick={() => setFilter(f)}
              className={`text-xs px-2 py-0.5 rounded ${filter === f
                ? f === 'BUY' ? 'bg-green text-bg-primary font-medium'
                  : f === 'SELL' ? 'bg-red text-white font-medium'
                  : 'bg-accent text-white font-medium'
                : 'bg-bg-hover text-text-secondary hover:text-text-primary'}`}>
              {f}
            </button>
          ))}
        </div>
        <button onClick={load} className="ml-auto text-text-secondary hover:text-text-primary">
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
        </button>
        <span className="text-xs text-text-secondary">{filtered.length} signals</span>
      </div>

      <div className="flex-1 overflow-y-auto">
        {!loading && filtered.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-text-secondary">
            <TrendingUp size={32} className="opacity-30" />
            <p className="text-sm">No signals yet.</p>
            <p className="text-xs">Go to Intelligence → Run Pipeline to generate signals.</p>
          </div>
        )}

        {filtered.length > 0 && (
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-bg-card border-b border-border z-10">
              <tr className="text-text-secondary">
                <th className="text-left px-4 py-2">Ticker</th>
                <th className="text-left px-2 py-2">Sector</th>
                <th className="text-left px-2 py-2">Signal</th>
                <th className="text-right px-2 py-2">Price</th>
                <th className="text-right px-2 py-2">Target</th>
                <th className="text-right px-2 py-2">Stop</th>
                <th className="text-right px-2 py-2">Conf</th>
                <th className="text-right px-4 py-2">Date</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((s) => {
                const isBuy = s.signal_type === 'BUY'
                const isSell = s.signal_type === 'SELL'
                return (
                  <tr key={s.id} className="border-b border-border hover:bg-bg-hover transition-colors">
                    <td className="px-4 py-2 font-medium text-white">{s.ticker}</td>
                    <td className="px-2 py-2 text-text-secondary">{s.sector || '—'}</td>
                    <td className={`px-2 py-2 font-medium flex items-center gap-1 ${isBuy ? 'text-green' : isSell ? 'text-red' : 'text-yellow'}`}>
                      {isBuy ? <TrendingUp size={11} /> : isSell ? <TrendingDown size={11} /> : <Minus size={11} />}
                      {s.signal_type}
                    </td>
                    <td className="px-2 py-2 text-right text-text-primary">
                      ₹{s.price_at_signal?.toLocaleString('en-IN', { minimumFractionDigits: 1, maximumFractionDigits: 1 })}
                    </td>
                    <td className="px-2 py-2 text-right text-green">
                      {s.target_price ? `₹${s.target_price.toFixed(1)}` : '—'}
                    </td>
                    <td className="px-2 py-2 text-right text-red">
                      {s.stop_loss ? `₹${s.stop_loss.toFixed(1)}` : '—'}
                    </td>
                    <td className="px-2 py-2 text-right">
                      <span className={`${(s.final_confidence || 0) >= 0.7 ? 'text-green' : (s.final_confidence || 0) >= 0.6 ? 'text-yellow' : 'text-text-secondary'}`}>
                        {((s.final_confidence || 0) * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td className="px-4 py-2 text-right text-text-secondary">
                      {s.fired_at ? new Date(s.fired_at).toLocaleDateString('en-IN') : '—'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
