import { useState, useEffect, useRef } from 'react'

export interface IndexQuote {
  symbol: string
  ltp: number | null
  change: number | null      // absolute change (not yet available from LTP-only feed)
  changePct: number | null
  available: boolean
}

const POLL_MS_OPEN   = 30_000   // 30s during market hours
const POLL_MS_CLOSED = 120_000  // 2 min outside hours

// Previous close reference (hardcoded as seed — updated once we have history)
const PREV_CLOSE: Record<string, number> = {
  'Nifty 50': 0,
  'Sensex': 0,
}

export function useMarketIndices(isMarketOpen: boolean): IndexQuote[] {
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

  return quotes
}
