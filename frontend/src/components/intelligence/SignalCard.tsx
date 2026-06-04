import { useState } from 'react'
import type { Signal } from '../../types'
import ReasoningDrawer from './ReasoningDrawer'

interface Props { signal: Signal }

export default function SignalCard({ signal: s }: Props) {
  const [open, setOpen] = useState(false)
  const isBuy = s.signal === 'BUY'

  return (
    <>
      <div
        onClick={() => setOpen(true)}
        className="flex items-center gap-4 px-4 py-2.5 border-b border-border cursor-pointer hover:bg-bg-hover transition-colors text-xs"
      >
        <span className={`font-medium w-12 ${isBuy ? 'text-green' : 'text-red'}`}>
          {isBuy ? '▲ BUY' : '▼ SELL'}
        </span>
        <span className="font-medium text-white w-24">{s.ticker}</span>
        <span className="text-text-primary">₹{s.price?.toFixed(1)}</span>
        <span className="text-green">→ ₹{s.target?.toFixed(1) || '—'}</span>
        <span className="text-red">Stop ₹{s.stop_loss?.toFixed(1) || '—'}</span>
        <span className="ml-auto text-text-secondary terminal-text">
          {'█'.repeat(Math.round((s.confidence || 0) * 8))}{'░'.repeat(8 - Math.round((s.confidence || 0) * 8))} {((s.confidence || 0) * 100).toFixed(0)}%
        </span>
      </div>
      {open && <ReasoningDrawer signal={s} onClose={() => setOpen(false)} />}
    </>
  )
}
