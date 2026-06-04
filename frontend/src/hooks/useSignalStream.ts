import { useCallback, useRef } from 'react'
import { createSignalStream } from '../api/stream'
import type { StreamEvent } from '../api/stream'

export function useSignalStream(onEvent: (e: StreamEvent) => void, onClose: () => void) {
  const stopRef = useRef<(() => void) | null>(null)

  const start = useCallback(() => {
    stopRef.current?.()
    stopRef.current = createSignalStream(onEvent, onClose)
  }, [onEvent, onClose])

  const stop = useCallback(() => {
    stopRef.current?.()
    stopRef.current = null
  }, [])

  return { start, stop }
}
