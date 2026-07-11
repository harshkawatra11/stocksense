import { useSystemHealth } from '../hooks/useSystemHealth'
import type { ComponentStatus } from '../types'

const DOT_COLOR: Record<ComponentStatus['status'], string> = {
  ok: 'bg-green',
  degraded: 'bg-yellow',
  unavailable: 'bg-red',
}

const LABEL: Record<string, string> = {
  kronos: 'Kronos',
  lightgbm: 'LightGBM',
  llm_synthesis: 'Synthesis',
  macro: 'Macro',
}

function sourceSuffix(c: ComponentStatus): string {
  if (c.source && c.source !== 'skipped') return ` (${c.source})`
  if (c.source === 'skipped') return ' (skipped)'
  return ''
}

function Chip({ name, c }: { name: string; c: ComponentStatus }) {
  const title = `${LABEL[name] || name}: ${c.status}${c.source ? ` [${c.source}]` : ''}${c.stale ? ' · stale' : ''}\n${c.detail}`
  return (
    <div
      title={title}
      className="flex items-center gap-1.5 text-[11px] text-text-secondary cursor-default"
    >
      <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${DOT_COLOR[c.status]}`} />
      <span>
        {LABEL[name] || name}
        <span className="opacity-60">{sourceSuffix(c)}</span>
        {c.stale && <span className="text-yellow ml-1">stale</span>}
      </span>
    </div>
  )
}

/**
 * Persistent small strip surfacing the Stage 0 truth layer: what's actually
 * running right now for each pipeline component, honestly. Green = ok,
 * amber = degraded (running but not ideal — e.g. pretrained Kronos, stale
 * LightGBM), red = unavailable. Hover any chip for the detail text.
 */
export default function SystemHealthBar() {
  const health = useSystemHealth(30_000)

  if (!health) {
    return (
      <div className="flex items-center gap-3 text-[11px] text-text-secondary">
        <span className="w-1.5 h-1.5 rounded-full bg-text-secondary/40 animate-pulse" />
        checking system health…
      </div>
    )
  }

  const { components, angel_one } = health

  return (
    <div className="flex items-center gap-3">
      {(Object.entries(components) as [string, ComponentStatus][]).map(([name, c]) => (
        <Chip key={name} name={name} c={c} />
      ))}
      {angel_one?.tripped && (
        <div
          title={`Angel One circuit breaker tripped: ${angel_one.last_error || 'unknown error'}\nRetrying in ${angel_one.retry_in_seconds}s`}
          className="flex items-center gap-1.5 text-[11px] text-red cursor-default"
        >
          <span className="w-1.5 h-1.5 rounded-full bg-red flex-shrink-0" />
          <span>Angel One offline</span>
        </div>
      )}
    </div>
  )
}
