import { useCallback, useEffect, useState } from 'react'
import { ShieldCheck, Check, X, FlaskConical } from 'lucide-react'
import type { PendingConfirmation } from '../../types'

const POLL_MS = 20_000

/**
 * The human-confirmed trade queue. Every row here is a specific proposed
 * trade waiting for an explicit yes/no — approving places a real order on
 * Upstox SANDBOX only (never live money). See backend/routers/confirmations.py
 * and intelligence/live_confirmation.py for the full picture. Empty/hidden
 * in the common case: nothing queues here until LIVE_CONFIRMATION_ENABLED is
 * turned on in .env and someone triggers a scan.
 */
export default function PendingConfirmations() {
  const [items, setItems] = useState<PendingConfirmation[]>([])
  const [busyId, setBusyId] = useState<number | null>(null)
  const [queuing, setQueuing] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const res = await fetch('/api/confirmations/pending')
      if (!res.ok) return
      setItems(await res.json())
    } catch {
      // silent — keep last known list, next poll retries
    }
  }, [])

  useEffect(() => {
    refresh()
    const t = setInterval(refresh, POLL_MS)
    return () => clearInterval(t)
  }, [refresh])

  async function act(id: number, action: 'approve' | 'reject') {
    setBusyId(id)
    try {
      const res = await fetch(`/api/confirmations/${id}/${action}`, { method: 'POST' })
      if (res.ok) await refresh()
    } finally {
      setBusyId(null)
    }
  }

  async function scanNow() {
    setQueuing(true)
    try {
      await fetch('/api/confirmations/queue-fresh', { method: 'POST' })
      await refresh()
    } finally {
      setQueuing(false)
    }
  }

  if (items.length === 0) {
    return (
      <div className="flex items-center justify-between rounded-lg border border-neutral-800 bg-neutral-900/40 px-4 py-3 text-sm text-neutral-500">
        <div className="flex items-center gap-2">
          <ShieldCheck size={16} className="text-neutral-600" />
          No trades awaiting your approval right now.
        </div>
        <button
          onClick={scanNow}
          disabled={queuing}
          className="rounded-md border border-neutral-700 px-3 py-1 text-xs text-neutral-300 hover:bg-neutral-800 disabled:opacity-50"
        >
          {queuing ? 'Scanning…' : 'Scan for new trades'}
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-medium text-neutral-200">
          <ShieldCheck size={16} className="text-emerald-400" />
          Awaiting your approval ({items.length})
          <span className="flex items-center gap-1 rounded-full bg-amber-950/60 px-2 py-0.5 text-[10px] font-normal text-amber-400">
            <FlaskConical size={10} /> sandbox only — no real money
          </span>
        </div>
        <button
          onClick={scanNow}
          disabled={queuing}
          className="rounded-md border border-neutral-700 px-3 py-1 text-xs text-neutral-300 hover:bg-neutral-800 disabled:opacity-50"
        >
          {queuing ? 'Scanning…' : 'Scan again'}
        </button>
      </div>
      {items.map((c) => (
        <div
          key={c.id}
          className="flex items-center justify-between gap-3 rounded-lg border border-neutral-800 bg-neutral-900/60 px-4 py-3"
        >
          <div className="min-w-0">
            <div className="flex items-baseline gap-2">
              <span className={`text-sm font-semibold ${c.action === 'BUY' ? 'text-emerald-400' : 'text-rose-400'}`}>
                {c.action}
              </span>
              <span className="font-medium text-neutral-100">{c.ticker}</span>
              <span className="text-sm text-neutral-400">
                {c.quantity} @ ₹{c.price.toFixed(2)}
              </span>
            </div>
            {c.reasoning && <div className="mt-0.5 truncate text-xs text-neutral-500">{c.reasoning}</div>}
            {c.is_stale && (
              <div className="mt-1 text-[10px] font-medium text-amber-500">
                Quoted price may be stale — approving may be rejected as expired
              </div>
            )}
          </div>
          <div className="flex shrink-0 gap-2">
            <button
              onClick={() => act(c.id, 'reject')}
              disabled={busyId === c.id}
              className="flex items-center gap-1 rounded-md border border-neutral-700 px-3 py-1.5 text-xs text-neutral-300 hover:bg-neutral-800 disabled:opacity-50"
            >
              <X size={14} /> Reject
            </button>
            <button
              onClick={() => act(c.id, 'approve')}
              disabled={busyId === c.id}
              className="flex items-center gap-1 rounded-md bg-emerald-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
            >
              <Check size={14} /> Approve
            </button>
          </div>
        </div>
      ))}
    </div>
  )
}
