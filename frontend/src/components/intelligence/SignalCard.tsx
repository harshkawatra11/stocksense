import { useState } from 'react'
import type { Signal, ComponentStatus } from '../../types'
import ReasoningDrawer from './ReasoningDrawer'

interface Props { signal: Signal }

const DOT_COLOR: Record<ComponentStatus['status'], string> = {
  ok: 'bg-green', degraded: 'bg-yellow', unavailable: 'bg-red',
}
const LABEL: Record<string, string> = {
  kronos: 'Kronos', lightgbm: 'LGBM', llm_synthesis: 'Synthesis', macro: 'Macro',
}

/** Minimal honest provenance dots — hover for what actually produced this signal. */
function ComponentDots({ components }: { components: Signal['components_json'] }) {
  if (!components) return null
  return (
    <span className="flex items-center gap-1">
      {(Object.entries(components) as [string, ComponentStatus][]).map(([name, c]) => (
        <span
          key={name}
          title={`${LABEL[name] || name}: ${c.status}${c.source ? ` (${c.source})` : ''} — ${c.detail}`}
          className={`w-1.5 h-1.5 rounded-full flex-shrink-0 cursor-default ${DOT_COLOR[c.status]}`}
        />
      ))}
    </span>
  )
}

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
        <ComponentDots components={s.components_json} />
        <span className="ml-auto text-text-secondary terminal-text">
          {'█'.repeat(Math.round((s.confidence || 0) * 8))}{'░'.repeat(8 - Math.round((s.confidence || 0) * 8))} {((s.confidence || 0) * 100).toFixed(0)}%
        </span>
      </div>
      {open && <ReasoningDrawer signal={s} onClose={() => setOpen(false)} />}
    </>
  )
}
