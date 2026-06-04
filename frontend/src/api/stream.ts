export type StreamEvent = Record<string, unknown>
export type StreamHandler = (event: StreamEvent) => void

export function createSignalStream(onEvent: StreamHandler, onClose: () => void): () => void {
  let es: EventSource | null = null
  let closed = false

  const connect = () => {
    if (closed) return
    es = new EventSource('/api/stream/signals')

    es.onmessage = (e) => {
      try {
        onEvent(JSON.parse(e.data))
      } catch {}
    }

    es.onerror = () => {
      es?.close()
      if (!closed) onClose()
    }
  }

  connect()

  return () => {
    closed = true
    es?.close()
  }
}
