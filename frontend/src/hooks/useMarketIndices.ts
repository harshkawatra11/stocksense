import { useState, useEffect, useRef } from 'react'
import type { PriceMap } from './useRealtimePrices'

export interface IndexQuote {
  symbol: string
  ltp: number | null
  change: number | null      // absolute change (not yet available from LTP-only feed)
  changePct: number | null
  available: boolean
  live?: boolean             // true if this tick came from the WS feed, not REST poll
  ts?: number | null         // WS tick timestamp, if live
}

const POLL_MS_OPEN   = 30_000   // 30s during market hours
const POLL_MS_CLOSED = 120_000  // 2 min outside hours

// Previous close reference (hardcoded as seed — updated once we have history)
const PREV_CLOSE: Record<string, number> = {
  'Nifty 50': 0,
  'Sensex': 0,
}

/**
 * Falls back to REST polling for indices, but if the live WS price feed (from
 * useRealtimePrices) has a tick for an index symbol, that takes priority —
 * indices may use special instrument keys not covered by the WS feed, so we
 * degrade gracefully per-symbol rather than assuming full coverage.
 */
export function useMarketIndices(isMarketOpen: boolean, livePrices?: PriceMap): IndexQuote[] {
  const [quotes, setQuotes] = useState<IndexQuote[]>([
    { symbol: 'Nifty 50', ltp: null, change: null, changePct: null, available: false },
    { symbol: 'Sensex',   ltp: null, change: null, changePct: null, available: false },
  ])
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  async function fetchIndices() {
    try {
      const res = await fetch('/api/market/indices', { signal: AbortSignal.timeout(8000) })
      if (!res.ok) return
      const json = await res.json()

      setQuotes((prev) =>
        (json.indices as Array<{ symbol: string; ltp: number | null; available: boolean }>).map(
          (item) => {
            const prevLtp = prev.find((q) => q.symbol === item.symbol)?.ltp ?? null
            const prevClose = PREV_CLOSE[item.symbol] || prevLtp

            let change: number | null = null
            let changePct: number | null = null
            if (item.ltp !== null && prevClose) {
              change = item.ltp - prevClose
              changePct = (change / prevClose) * 100
            }

            // Update seed with latest known price
            if (item.ltp !== null) PREV_CLOSE[item.symbol] = item.ltp

            return { symbol: item.symbol, ltp: item.ltp, change, changePct, available: item.available }
          }
        )
      )
    } catch {
      // silent — keep stale data
    } finally {
      timerRef.current = setTimeout(fetchIndices, isMarketOpen ? POLL_MS_OPEN : POLL_MS_CLOSED)
    }
  }

  useEffect(() => {
    fetchIndices()
    return () => { if (timerRef.current) clearTimeout(timerRef.current) }
  }, [isMarketOpen])

  // Overlay live WS ticks where available, per-symbol — REST poll remains the
  // fallback source of truth for symbols the WS feed doesn't cover.
  if (!livePrices) return quotes

  return quotes.map((q) => {
    const tick = livePrices[q.symbol]
    if (!tick) return q

    const prevClose = PREV_CLOSE[q.symbol] || tick.close || q.ltp
    let change: number | null = null
    let changePct: number | null = null
    if (tick.ltp !== null && prevClose) {
      change = tick.ltp - prevClose
      changePct = (change / prevClose) * 100
    }
    if (tick.ltp !== null) PREV_CLOSE[q.symbol] = tick.close ?? PREV_CLOSE[q.symbol] ?? tick.ltp

    return { symbol: q.symbol, ltp: tick.ltp, change, changePct, available: true, live: true, ts: tick.ts }
  })
}
