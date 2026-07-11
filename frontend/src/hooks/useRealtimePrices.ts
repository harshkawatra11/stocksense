import { useCallback, useEffect, useRef, useState } from 'react'

export interface PriceTick {
  ltp: number
  close: number | null
  ts: number
}

export type PriceMap = Record<string, PriceTick>

const MAX_BACKOFF_MS = 30_000
const BASE_BACKOFF_MS = 1_000

/**
 * WebSocket client for GET /api/ws/prices.
 * Contract: on connect, server sends a full snapshot `{symbol: {ltp, close, ts}}`;
 * afterwards it sends periodic (~1s) partial updates in the same shape for symbols
 * whose price changed. This hook merges snapshot + deltas into one running map,
 * with auto-reconnect (exponential backoff) on disconnect.
 */
export function useRealtimePrices() {
  const [prices, setPrices] = useState<PriceMap>({})
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
          const msg = JSON.parse(e.data) as PriceMap
          setPrices((prev) => ({ ...prev, ...msg }))
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

  return { prices, isStale, connected }
}
