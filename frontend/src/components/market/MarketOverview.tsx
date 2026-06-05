import { useEffect, useState } from 'react'
import { TrendingUp, TrendingDown, RefreshCw } from 'lucide-react'

interface SignalRow {
  ticker: string
  signal_type: string
  price_at_signal: number
  target_price?: number
  stop_loss?: number
  final_confidence: number
  date: string
  sector?: string
}

interface SectorRow { sector: string; total: number; buys: number; sells: number; avg_conf: number }
interface MoverRow { ticker: string; close: number; chg_pct: number; mover_type: string }
interface Stats { total_buys: number; total_sells: number; total_holds: number; total: number; avg_conf: number }

interface Overview {
  top_buy_signals: SignalRow[]
  top_sell_signals: SignalRow[]
  sector_summary: SectorRow[]
  top_movers: MoverRow[]
  stats: Stats
  period_days: number
}

export default function MarketOverview() {
  const [data, setData] = useState<Overview | null>(null)
  const [loading, setLoading] = useState(false)
  const [tab, setTab] = useState<'signals' | 'sectors' | 'movers'>('signals')

  function load() {
    setLoading(true)
    fetch('/api/market/overview')
      .then(r => r.json())
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  const gainers = data?.top_movers.filter(m => m.mover_type === 'gainer') || []
  const losers  = data?.top_movers.filter(m => m.mover_type === 'loser') || []

  return (
    <div className="h-full flex flex-col overflow-hidden bg-bg-primary">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border bg-bg-card flex-shrink-0 flex items-center gap-4">
        <span className="text-sm font-medium text-white">Market Overview</span>
        <span className="text-xs text-text-secondary">Last 7 days</span>

        {/* Stats pills */}
        {data?.stats && (
          <div className="flex gap-3 ml-4">
            <span className="text-xs px-2 py-0.5 rounded bg-bg-hover text-green">
              ▲ {data.stats.total_buys} BUY
            </span>
            <span className="text-xs px-2 py-0.5 rounded bg-bg-hover text-red">
              ▼ {data.stats.total_sells} SELL
            </span>
            <span className="text-xs px-2 py-0.5 rounded bg-bg-hover text-text-secondary">
              Avg conf: {((data.stats.avg_conf || 0) * 100).toFixed(1)}%
            </span>
          </div>
        )}

        <button onClick={load} className="ml-auto text-text-secondary hover:text-text-primary">
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* Sub tabs */}
      <div className="flex gap-1 px-4 py-2 border-b border-border bg-bg-card flex-shrink-0">
        {(['signals', 'sectors', 'movers'] as const).map(t => (
          <button key={t} onClick={() => setTab(t)}
            className={`text-xs px-3 py-1 rounded capitalize ${tab === t ? 'bg-accent text-white' : 'text-text-secondary hover:text-text-primary'}`}>
            {t}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-auto p-4">
        {loading && !data && (
          <div className="flex items-center justify-center h-32 text-text-secondary text-xs animate-pulse">
            Loading market data...
          </div>
        )}

        {!loading && !data && (
          <div className="flex flex-col items-center justify-center h-32 gap-2 text-text-secondary">
            <p className="text-sm">No data available</p>
            <p className="text-xs">Run the pipeline from the Intelligence tab to generate signals.</p>
          </div>
        )}

        {/* Signals tab */}
        {tab === 'signals' && data && (
          <div className="grid grid-cols-2 gap-4">
            <SignalTable title="Top BUY Signals" rows={data.top_buy_signals} type="BUY" />
            <SignalTable title="Top SELL Signals" rows={data.top_sell_signals} type="SELL" />
          </div>
        )}

        {/* Sectors tab */}
        {tab === 'sectors' && data && (
          <div className="grid grid-cols-1 gap-2 max-w-2xl">
            {data.sector_summary.length === 0 ? (
              <p className="text-xs text-text-secondary">No sector data yet. Run the pipeline first.</p>
            ) : data.sector_summary.map(s => (
              <div key={s.sector} className="bg-bg-card border border-border rounded p-3 flex items-center gap-4">
                <span className="text-sm text-white font-medium w-48 truncate">{s.sector}</span>
                <div className="flex gap-3 flex-1">
                  <span className="text-xs text-green">▲ {s.buys} BUY</span>
                  <span className="text-xs text-red">▼ {s.sells} SELL</span>
                  <span className="text-xs text-text-secondary">{s.total} total</span>
                </div>
                {/* Heatmap bar */}
                <div className="w-32 h-2 rounded overflow-hidden bg-bg-hover">
                  <div className="h-full bg-green rounded"
                    style={{ width: `${s.total > 0 ? (s.buys / s.total) * 100 : 50}%` }} />
                </div>
                <span className="text-xs text-text-secondary w-16 text-right">
                  conf {((s.avg_conf || 0) * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Movers tab */}
        {tab === 'movers' && data && (
          <div className="grid grid-cols-2 gap-4 max-w-2xl">
            <div>
              <h3 className="text-xs font-medium text-text-secondary mb-2 flex items-center gap-1">
                <TrendingUp size={12} className="text-green" /> Top Gainers
              </h3>
              {gainers.length === 0 ? (
                <p className="text-xs text-text-secondary">Calculating...</p>
              ) : gainers.map(m => (
                <div key={m.ticker} className="flex items-center justify-between py-2 border-b border-border">
                  <span className="text-sm text-white font-medium">{m.ticker}</span>
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-text-primary">₹{m.close.toLocaleString('en-IN')}</span>
                    <span className="text-xs text-green font-medium">+{m.chg_pct.toFixed(2)}%</span>
                  </div>
                </div>
              ))}
            </div>
            <div>
              <h3 className="text-xs font-medium text-text-secondary mb-2 flex items-center gap-1">
                <TrendingDown size={12} className="text-red" /> Top Losers
              </h3>
              {losers.length === 0 ? (
                <p className="text-xs text-text-secondary">Calculating...</p>
              ) : losers.map(m => (
                <div key={m.ticker} className="flex items-center justify-between py-2 border-b border-border">
                  <span className="text-sm text-white font-medium">{m.ticker}</span>
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-text-primary">₹{m.close.toLocaleString('en-IN')}</span>
                    <span className="text-xs text-red font-medium">{m.chg_pct.toFixed(2)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function SignalTable({ title, rows, type }: { title: string; rows: SignalRow[]; type: string }) {
  const color = type === 'BUY' ? 'text-green' : 'text-red'
  return (
    <div className="bg-bg-card border border-border rounded overflow-hidden">
      <div className="px-4 py-2 border-b border-border">
        <span className={`text-xs font-medium ${color}`}>{title}</span>
      </div>
      {rows.length === 0 ? (
        <p className="text-xs text-text-secondary px-4 py-4">No {type} signals in last 7 days.</p>
      ) : (
        <table className="w-full text-xs">
          <thead>
            <tr className="text-text-secondary border-b border-border">
              <th className="text-left px-4 py-2">Ticker</th>
              <th className="text-right px-2 py-2">Price</th>
              <th className="text-right px-2 py-2">Target</th>
              <th className="text-right px-4 py-2">Conf</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((s, i) => (
              <tr key={i} className="border-b border-border hover:bg-bg-hover">
                <td className="px-4 py-2">
                  <div className="text-white font-medium">{s.ticker}</div>
                  {s.sector && <div className="text-text-secondary text-[10px]">{s.sector}</div>}
                </td>
                <td className="px-2 py-2 text-right text-text-primary">
                  ₹{s.price_at_signal?.toFixed(1)}
                </td>
                <td className={`px-2 py-2 text-right ${type === 'BUY' ? 'text-green' : 'text-red'}`}>
                  {s.target_price ? `₹${s.target_price.toFixed(1)}` : '—'}
                </td>
                <td className="px-4 py-2 text-right text-text-secondary">
                  {((s.final_confidence || 0) * 100).toFixed(0)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
