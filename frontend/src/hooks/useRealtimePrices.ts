import { useCallback, useEffect, useRef, useState } from 'react'

export interface PriceTick {
  ltp: number
  close: number | null
  ts: number
}

export type PriceMap = Record<string, PriceTick>

/** Recent LTPs per symbol (oldest first) — ring buffer for sparklines. */
export type HistoryMap = Record<string, number[]>

interface WsMessage {
  type?: 'snapshot' | 'delta'
  quotes?: PriceMap
  history?: HistoryMap
}

const MAX_BACKOFF_MS = 30_000
const BASE_BACKOFF_MS = 1_000
const HISTORY_LEN = 30

/**
 * WebSocket client for GET /api/ws/prices.
 * Contract: on connect, server sends a full snapshot `{symbol: {ltp, close, ts}}`;
 * afterwards it sends periodic (~1s) partial updates in the same shape for symbols
 * whose price changed. This hook merges snapshot + deltas into one running map,
 * with auto-reconnect (exponential backoff) on disconnect.
 */
export function useRealtimePrices() {
  const [prices, setPrices] = useState<PriceMap>({})
  const [history, setHistory] = useState<HistoryMap>({})
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const backoffRef = useRef(BASE_BACKOFF_MS)
  const closedRef = useRef(false)

  useEffect(() => {
    closedRef.current = false

    const connect = () => {
      if (closedRef.current) return

      // Follow the same-origin/proxy pattern used elsewhere (EventSource('/api/...'))
      // but resolve to a ws:// or wss:// URL based on the current page protocol.
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const url = `${proto}//${window.location.host}/api/ws/prices`

      let ws: WebSocket
      try {
        ws = new WebSocket(url)
      } catch {
        scheduleReconnect()
        return
      }
      wsRef.current = ws

      ws.onopen = () => {
        backoffRef.current = BASE_BACKOFF_MS
        setConnected(true)
      }

      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data) as WsMessage | PriceMap
          // Server envelope: {type: "snapshot"|"delta", quotes: {...}, history?: {...}}.
          // Also tolerate a bare {symbol: tick} map (older server shape).
          const envelope = msg as WsMessage
          const quotes: PriceMap =
            envelope.quotes ?? ((msg as PriceMap) || {})
          if (envelope.quotes === undefined && envelope.type !== undefined) return

          setPrices((prev) => ({ ...prev, ...quotes }))

          setHistory((prev) => {
            const next = { ...prev }
            // Snapshot seeds full server-side history; deltas append client-side.
            if (envelope.history) {
              for (const [sym, pts] of Object.entries(envelope.history)) {
                next[sym] = pts.slice(-HISTORY_LEN)
              }
            }
            for (const [sym, tick] of Object.entries(quotes)) {
              if (tick.ltp == null) continue
              const buf = next[sym] ? [...next[sym]] : []
              if (buf[buf.length - 1] !== tick.ltp) {
                buf.push(tick.ltp)
                next[sym] = buf.slice(-HISTORY_LEN)
              }
            }
            return next
          })
        } catch {
          // ignore malformed frames
        }
      }

      ws.onerror = () => {
        ws.close()
      }

      ws.onclose = () => {
        setConnected(false)
        wsRef.current = null
        if (!closedRef.current) scheduleReconnect()
      }
    }

    const scheduleReconnect = () => {
      if (closedRef.current) return
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
      const delay = backoffRef.current
      backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS)
      reconnectTimerRef.current = setTimeout(connect, delay)
    }

    connect()

    return () => {
      closedRef.current = true
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
      wsRef.current?.close()
      wsRef.current = null
    }
  }, [])

  const isStale = useCallback(
    (symbol: string, thresholdSeconds = 30) => {
      const tick = prices[symbol]
      if (!tick) return true
      return Date.now() / 1000 - tick.ts > thresholdSeconds
    },
    [prices]
  )

  return { prices, history, isStale, connected }
}
