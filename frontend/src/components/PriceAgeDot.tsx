interface Props {
  /** Unix seconds timestamp of the last price tick, or null/undefined if unknown. */
  ts: number | null | undefined
  /** Whether the websocket itself is connected. */
  connected?: boolean
  /**
   * Whether the market is currently open. When closed, staleness is expected —
   * show a calm neutral gray ("as of close") instead of an alarming red.
   */
  marketOpen?: boolean
  className?: string
}

/**
 * Small freshness indicator for a live price:
 * green  = updated within last 2s
 * amber  = updated within last 30s
 * red/gray = stale beyond 30s or disconnected
 * neutral gray = market closed (stale is normal, not an error)
 */
export default function PriceAgeDot({ ts, connected = true, marketOpen = true, className = '' }: Props) {
  let color = 'bg-text-secondary/40' // gray — unknown / disconnected
  let title = 'No live price'

  if (!marketOpen) {
    // Market closed: a stale price is the honest, expected state.
    color = 'bg-text-secondary/50'
    title = 'Market closed — as of 15:30 close'
  } else if (connected && ts != null) {
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
