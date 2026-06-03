import { useState } from 'react'
import type { Signal } from '../../types'
import ReasoningDrawer from './ReasoningDrawer'

interface Props { signals: Signal[] }

export default function SignalFeed({ signals }: Props) {
  const [selected, setSelected] = useState<Signal | null>(null)

  const buys  = signals.filter(s => s.signal === 'BUY').length
  const sells = signals.filter(s => s.signal === 'SELL').length

  return (
    <>
      <div className="h-full flex flex-col bg-bg-card overflow-hidden">
        <div className="flex items-center gap-3 px-4 py-2 border-b border-border flex-shrink-0">
          <span className="text-xs font-medium text-text-secondary">LIVE SIGNAL FEED</span>
          <span className="text-xs px-2 py-0.5 rounded bg-bg-hover text-green">▲ BUY {buys}</span>
          <span className="text-xs px-2 py-0.5 rounded bg-bg-hover text-red">▼ SELL {sells}</span>
        </div>

        <div className="flex-1 overflow-y-auto">
          {signals.length === 0 ? (
            <p className="text-xs text-text-secondary px-4 py-3">No signals yet. Click Run Pipeline above.</p>
          ) : (
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-bg-card">
                <tr className="text-text-secondary border-b border-border">
                  <th className="text-left px-4 py-1.5">Signal</th>
                  <th className="text-left px-2 py-1.5">Ticker</th>
                  <th className="text-right px-2 py-1.5">Price</th>
                  <th className="text-right px-2 py-1.5">Target</th>
                  <th className="text-right px-2 py-1.5">Stop</th>
                  <th className="text-right px-4 py-1.5">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {signals.map((s, i) => (
                  <tr key={i} onClick={() => setSelected(s)}
                    className="border-b border-border cursor-pointer hover:bg-bg-hover transition-colors">
                    <td className={`px-4 py-2 font-medium ${s.signal === 'BUY' ? 'text-green' : 'text-red'}`}>
                      {s.signal === 'BUY' ? '▲' : '▼'} {s.signal}
                    </td>
                    <td className="px-2 py-2 text-white font-medium">{s.ticker}</td>
                    <td className="px-2 py-2 text-right text-text-primary">₹{s.price?.toFixed(1)}</td>
                    <td className="px-2 py-2 text-right text-green">₹{s.target?.toFixed(1) || '—'}</td>
                    <td className="px-2 py-2 text-right text-red">₹{s.stop_loss?.toFixed(1) || '—'}</td>
                    <td className="px-4 py-2 text-right">
                      <ConfBar pct={(s.confidence || 0) * 100} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {selected && <ReasoningDrawer signal={selected} onClose={() => setSelected(null)} />}
    </>
  )
}

function ConfBar({ pct }: { pct: number }) {
  const filled = Math.round(pct / 12.5)
  return (
    <span className="terminal-text text-text-secondary">
      {'█'.repeat(filled)}{'░'.repeat(8 - filled)} {pct.toFixed(0)}%
    </span>
  )
}
