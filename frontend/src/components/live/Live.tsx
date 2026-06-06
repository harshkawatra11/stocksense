import { useCallback, useEffect, useState } from 'react'
import { ThumbsUp, ThumbsDown, ShoppingCart, X, RefreshCw, Play, Wallet, Clock, Target } from 'lucide-react'
import type { LiveSignal, Account, ActivityEvent, PositionReview } from '../../types'

const api = {
  get: (p: string) => fetch(`/api${p}`).then(r => r.json()),
  post: (p: string, body?: unknown) =>
    fetch(`/api${p}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
    }).then(async r => {
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`)
      return r.json()
    }),
}

const confColor = (c: number) =>
  c >= 0.75 ? 'text-green' : c >= 0.6 ? 'text-yellow' : 'text-text-secondary'

const eventStyle: Record<string, { color: string; label: string }> = {
  SUGGESTED:  { color: 'text-accent',  label: 'Suggested' },
  RATED:      { color: 'text-yellow',  label: 'Rated' },
  BOUGHT:     { color: 'text-green',   label: 'Bought' },
  SOLD:       { color: 'text-red',     label: 'Sold' },
  REANALYZED: { color: 'text-blue-400', label: 'Re-analyzed' },
  NOTE:       { color: 'text-text-secondary', label: 'Note' },
}

const statusStyle: Record<string, string> = {
  on_track: 'text-green', ahead: 'text-green', target_hit: 'text-green',
  behind: 'text-yellow', expired: 'text-red', stopped: 'text-red',
}

export default function Live() {
  const [signals, setSignals] = useState<LiveSignal[]>([])
  const [account, setAccount] = useState<Account | null>(null)
  const [activity, setActivity] = useState<ActivityEvent[]>([])
  const [reviews, setReviews] = useState<PositionReview[]>([])
  const [onlyAffordable, setOnlyAffordable] = useState(false)
  const [running, setRunning] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    const [sig, acc, act, rev] = await Promise.all([
      api.get(`/live/signals?limit=40${onlyAffordable ? '&only_affordable=true' : ''}`),
      api.get('/live/account'),
      api.get('/live/activity?limit=60'),
      api.get('/live/positions/reviews'),
    ])
    setSignals(Array.isArray(sig) ? sig : [])
    setAccount(acc)
    setActivity(Array.isArray(act) ? act : [])
    setReviews(Array.isArray(rev) ? rev : [])
  }, [onlyAffordable])

  useEffect(() => { refresh() }, [refresh])

  const act = async (fn: () => Promise<unknown>, key: string) => {
    setBusy(key); setError(null)
    try { await fn(); await refresh() }
    catch (e) { setError(e instanceof Error ? e.message : 'Action failed') }
    finally { setBusy(null) }
  }

  const rate = (s: LiveSignal, like: boolean) =>
    act(() => api.post('/live/rate', { signal_id: s.id, ticker: s.ticker, like, reason: like ? 'liked' : 'confidence too low' }), `rate-${s.id}-${like}`)

  const buy = (s: LiveSignal) => {
    const shares = s.shares_affordable && s.shares_affordable > 0 ? s.shares_affordable : 1
    return act(() => api.post('/live/buy', { signal_id: s.id, ticker: s.ticker, quantity: shares, price: s.price_at_signal, rationale: `Bought ${shares} @ ${s.price_at_signal}` }), `buy-${s.id}`)
  }

  const pass = (s: LiveSignal) =>
    act(() => api.post('/live/pass', { signal_id: s.id, ticker: s.ticker, reason: 'passed' }), `pass-${s.id}`)

  const runPipeline = async () => {
    setRunning(true); setError(null)
    try { await api.post('/live/run?limit=50'); await refresh() }
    catch (e) { setError(e instanceof Error ? e.message : 'Run failed') }
    finally { setRunning(false) }
  }

  const reviewPositions = () => act(() => api.post('/live/positions/review'), 'review')

  return (
    <div className="h-full flex flex-col overflow-hidden bg-bg-primary">
      {/* Header */}
      <div className="px-4 py-2.5 border-b border-border bg-bg-card flex items-center gap-4 flex-shrink-0">
        <span className="text-sm font-semibold text-white">Live Signals</span>
        {account && (
          <div className="flex items-center gap-2 text-xs">
            <Wallet size={13} className="text-green" />
            <span className="text-text-primary">₹{account.cash_available.toFixed(0)}</span>
            <span className="text-text-secondary">deployable</span>
            <span className="text-text-secondary">· ₹{account.cash_reserve.toFixed(0)} reserve</span>
          </div>
        )}
        <label className="flex items-center gap-1.5 text-xs text-text-secondary ml-2 cursor-pointer">
          <input type="checkbox" checked={onlyAffordable} onChange={e => setOnlyAffordable(e.target.checked)} />
          Affordable only
        </label>
        <div className="ml-auto flex items-center gap-2">
          <button onClick={refresh} className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 bg-bg-hover text-text-secondary rounded hover:text-text-primary">
            <RefreshCw size={12} /> Refresh
          </button>
          <button onClick={runPipeline} disabled={running}
            className="flex items-center gap-1.5 text-xs px-3 py-1.5 bg-accent text-white rounded disabled:opacity-50">
            <Play size={12} /> {running ? 'Running…' : 'Run Pipeline'}
          </button>
        </div>
      </div>

      {error && <div className="px-4 py-1.5 text-xs text-red bg-red/10 border-b border-border flex-shrink-0">{error}</div>}

      <div className="flex-1 flex overflow-hidden">
        {/* Signals */}
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {signals.length === 0 && (
            <div className="text-xs text-text-secondary text-center mt-8">
              No BUY signals yet. Click <span className="text-accent">Run Pipeline</span> to generate them.
            </div>
          )}
          {signals.map(s => (
            <div key={s.id} className="bg-bg-card border border-border rounded-lg p-3">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-white">{s.ticker}</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-bg-hover text-text-secondary">{s.timeframe}</span>
                {s.sector && <span className="text-[10px] text-text-secondary">{s.sector}</span>}
                <span className={`ml-auto text-xs font-medium ${confColor(s.final_confidence)}`}>
                  {(s.final_confidence * 100).toFixed(0)}% conf
                </span>
              </div>

              {/* The intelligent headline: buy @ X → Y (+Z%) in ~ETAd */}
              <div className="mt-2 flex items-center gap-2 text-sm">
                <span className="text-text-primary">₹{s.price_at_signal.toFixed(1)}</span>
                {s.target_price && (
                  <>
                    <Target size={12} className="text-green" />
                    <span className="text-green">₹{s.target_price.toFixed(1)}</span>
                    {s.expected_move_pct != null && (
                      <span className="text-green text-xs">(+{s.expected_move_pct.toFixed(1)}%)</span>
                    )}
                  </>
                )}
                {s.target_eta_days != null && (
                  <span className="flex items-center gap-1 text-xs text-text-secondary ml-1">
                    <Clock size={11} /> ~{s.target_eta_days}d
                  </span>
                )}
              </div>

              <div className="mt-1.5 flex items-center gap-3 text-[11px] text-text-secondary">
                {s.stop_loss && <span>SL ₹{s.stop_loss.toFixed(1)}</span>}
                {s.macro_sector_score != null && (
                  <span className={s.macro_sector_score >= 0 ? 'text-green' : 'text-red'}>
                    macro {s.macro_sector_score >= 0 ? '+' : ''}{s.macro_sector_score.toFixed(2)}
                  </span>
                )}
                <span className={s.affordable ? 'text-green' : 'text-text-secondary'}>
                  {s.affordable ? `${s.shares_affordable} share${s.shares_affordable === 1 ? '' : 's'} w/ ₹500` : 'not affordable w/ ₹500'}
                </span>
              </div>

              {/* Actions */}
              <div className="mt-2.5 flex items-center gap-2">
                <button onClick={() => rate(s, true)} disabled={!!busy}
                  className="flex items-center gap-1 text-[11px] px-2 py-1 rounded bg-bg-hover text-text-secondary hover:text-green disabled:opacity-40">
                  <ThumbsUp size={12} /> Like
                </button>
                <button onClick={() => rate(s, false)} disabled={!!busy}
                  className="flex items-center gap-1 text-[11px] px-2 py-1 rounded bg-bg-hover text-text-secondary hover:text-red disabled:opacity-40">
                  <ThumbsDown size={12} /> Dislike
                </button>
                <button onClick={() => buy(s)} disabled={!!busy || !s.affordable}
                  title={s.affordable ? 'Buy' : 'Not affordable with current capital'}
                  className="flex items-center gap-1 text-[11px] px-2.5 py-1 rounded bg-green/20 text-green hover:bg-green/30 disabled:opacity-40 disabled:cursor-not-allowed">
                  <ShoppingCart size={12} /> Buy
                </button>
                <button onClick={() => pass(s)} disabled={!!busy}
                  className="flex items-center gap-1 text-[11px] px-2 py-1 rounded bg-bg-hover text-text-secondary hover:text-text-primary disabled:opacity-40 ml-auto">
                  <X size={12} /> Pass
                </button>
              </div>
            </div>
          ))}
        </div>

        {/* Right column: positions + activity */}
        <div className="w-80 flex-shrink-0 border-l border-border flex flex-col overflow-hidden">
          {/* Position re-analysis */}
          <div className="border-b border-border">
            <div className="px-3 py-2 flex items-center gap-2">
              <span className="text-xs font-semibold text-white">Position Re-analysis</span>
              <button onClick={reviewPositions} disabled={!!busy}
                className="ml-auto flex items-center gap-1 text-[10px] px-2 py-1 rounded bg-bg-hover text-text-secondary hover:text-text-primary disabled:opacity-40">
                <RefreshCw size={10} /> Re-check
              </button>
            </div>
            <div className="max-h-52 overflow-y-auto px-3 pb-2 space-y-1.5">
              {reviews.length === 0 && <div className="text-[11px] text-text-secondary pb-2">No held positions yet.</div>}
              {reviews.map((r, i) => (
                <div key={i} className="text-[11px] bg-bg-primary rounded p-2">
                  <div className="flex items-center gap-2">
                    <span className="text-text-primary font-medium">{r.ticker}</span>
                    <span className={`${statusStyle[r.status] || 'text-text-secondary'}`}>{r.status}</span>
                    <span className="ml-auto text-text-secondary">{r.verdict}</span>
                  </div>
                  <div className="text-text-secondary mt-1">
                    ₹{r.entry_price?.toFixed(1)} → ₹{r.target_price?.toFixed(1)} · {r.progress_pct?.toFixed(0)}% done · {r.days_elapsed?.toFixed(1)}/{r.eta_days}d
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Activity feed */}
          <div className="flex-1 overflow-hidden flex flex-col">
            <div className="px-3 py-2 text-xs font-semibold text-white">Activity Log</div>
            <div className="flex-1 overflow-y-auto px-3 pb-3 space-y-1">
              {activity.length === 0 && <div className="text-[11px] text-text-secondary">No activity yet.</div>}
              {activity.map(e => {
                const st = eventStyle[e.event_type] || eventStyle.NOTE
                return (
                  <div key={e.id} className="text-[11px] flex gap-2 py-1 border-b border-border/40">
                    <span className={`${st.color} font-medium w-20 flex-shrink-0`}>{st.label}</span>
                    <div className="min-w-0">
                      <span className="text-text-primary">{e.ticker}</span>
                      {e.rating && <span className={e.rating === 'LIKE' ? 'text-green ml-1' : 'text-red ml-1'}>{e.rating}</span>}
                      {e.note && <span className="text-text-secondary ml-1">{e.note}</span>}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
