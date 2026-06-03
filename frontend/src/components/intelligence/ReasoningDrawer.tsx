import type { Signal } from '../../types'
import { X } from 'lucide-react'

interface Props { signal: Signal; onClose: () => void }

export default function ReasoningDrawer({ signal, onClose }: Props) {
  const blocks = [
    { label: 'ML MODEL (LightGBM)', text: signal.ml_reasoning, color: '#00d4aa' },
    { label: 'KRONOS FOUNDATION',   text: signal.kronos_reasoning, color: '#3d85c8' },
    { label: 'SLM — NSE COMMANDER', text: signal.slm_reasoning, color: '#ffa502' },
    { label: 'CLAUDE SONNET',       text: signal.claude_reasoning, color: '#5c6bc0' },
  ].filter(b => b.text)

  return (
    <div className="fixed inset-y-0 right-0 w-[480px] bg-bg-card border-l border-border flex flex-col z-50 shadow-2xl">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <div>
          <span className={`text-lg font-bold ${signal.signal === 'BUY' ? 'text-green' : 'text-red'}`}>
            {signal.signal === 'BUY' ? '▲' : '▼'} {signal.signal} {signal.ticker}
          </span>
          <span className="ml-3 text-text-secondary text-sm">₹{signal.price?.toFixed(1)}</span>
        </div>
        <button onClick={onClose} className="text-text-secondary hover:text-white">
          <X size={18} />
        </button>
      </div>

      {/* Summary row */}
      <div className="flex gap-4 px-4 py-3 border-b border-border text-xs">
        <div><span className="text-text-secondary">Target </span>
          <span className="text-green font-medium">₹{signal.target?.toFixed(1) || '—'}</span></div>
        <div><span className="text-text-secondary">Stop </span>
          <span className="text-red font-medium">₹{signal.stop_loss?.toFixed(1) || '—'}</span></div>
        <div><span className="text-text-secondary">Confidence </span>
          <span className="text-white font-medium">{((signal.confidence || 0) * 100).toFixed(0)}%</span></div>
      </div>

      {/* Reasoning blocks */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4">
        {blocks.map((b, i) => (
          <div key={i} className="bg-bg-primary rounded p-3">
            <div className="text-xs font-medium mb-2 terminal-text" style={{ color: b.color }}>
              [{b.label}]
            </div>
            <pre className="text-xs text-text-primary terminal-text whitespace-pre-wrap leading-5">
              {b.text}
            </pre>
          </div>
        ))}
        {blocks.length === 0 && (
          <p className="text-xs text-text-secondary">No reasoning available for this signal.</p>
        )}
      </div>
    </div>
  )
}
