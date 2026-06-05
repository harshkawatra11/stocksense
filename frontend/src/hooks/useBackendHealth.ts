import { useState, useEffect, useRef } from 'react'

export type BackendState = 'connected' | 'disconnected' | 'checking'

export function useBackendHealth(intervalMs = 10_000): BackendState {
  const [state, setState] = useState<BackendState>('checking')
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  async function check() {
    try {
      const res = await fetch('/api/health', { signal: AbortSignal.timeout(4000) })
      setState(res.ok ? 'connected' : 'disconnected')
    } catch {
      setState('disconnected')
    }
  }

  useEffect(() => {
    check()
    timerRef.current = setInterval(check, intervalMs)
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [intervalMs])

  return state
}
