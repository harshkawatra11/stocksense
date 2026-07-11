interface Props {
  /** Unix seconds timestamp of the last price tick, or null/undefined if unknown. */
  ts: number | null | undefined
  /** Whether the websocket itself is connected. */
  connected?: boolean
  className?: string
}

/**
 * Small freshness indicator for a live price:
 * green  = updated within last 2s
 * amber  = updated within last 30s
 * red/gray = stale beyond 30s or disconnected
 */
export default function PriceAgeDot({ ts, connected = true, className = '' }: Props) {
  let color = 'bg-text-secondary/40' // gray — unknown / disconnected
  let title = 'No live price'

  if (connected && ts != null) {
    const ageSec = Date.now() / 1000 - ts
    if (ageSec < 2) {
      color = 'bg-green animate-pulse'
      title = 'Live'
    } else if (ageSec < 30) {
      color = 'bg-yellow'
      title = `Updated ${Math.round(ageSec)}s ago`
    } else {
      color = 'bg-red'
      title = `Stale (${Math.round(ageSec)}s ago)`
    }
  } else if (!connected) {
    color = 'bg-red'
    title = 'Disconnected'
  }

  return (
    <span
      title={title}
      className={`inline-block w-1.5 h-1.5 rounded-full flex-shrink-0 ${color} ${className}`}
    />
  )
}
