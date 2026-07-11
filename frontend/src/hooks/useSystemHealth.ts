import { useEffect, useState, useRef } from 'react'
import type { SystemHealth } from '../types'

/**
 * Polls GET /api/system/health — the Stage 0 truth layer rollup — every
 * ~30s. Mirrors the polling pattern of useBackendHealth but carries the
 * full per-component status payload instead of a single up/down flag.
 */
export function useSystemHealth(intervalMs = 30_000): SystemHealth | null {
  const [health, setHealth] = useState<SystemHealth | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  async function check() {
    try {
      const res = await fetch('/api/system/health', { signal: AbortSignal.timeout(6000) })
      if (res.ok) setHealth(await res.json())
    } catch {
      // keep the last known-good snapshot rather than flashing to null
    }
  }

  useEffect(() => {
    check()
    timerRef.current = setInterval(check, intervalMs)
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [intervalMs])

  return health
}
