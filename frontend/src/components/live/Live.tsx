import { useCallback, useEffect, useState } from 'react'
import { ThumbsUp, ThumbsDown, RefreshCw, Wallet, Clock, Target, AlertTriangle, FlaskConical, Bot, ExternalLink } from 'lucide-react'
import type { LiveSignal, Account, ActivityEvent, PositionReview, DataStatus, TradingMode, ComponentStatus } from '../../types'
import { useRealtimePrices } from '../../hooks/useRealtimePrices'
import { useMarketStatus } from '../../hooks/useMarketStatus'
import { useFlash, flashBgClass } from '../FlashPrice'
import PriceAgeDot from '../PriceAgeDot'

const COMPONENT_DOT: Record<ComponentStatus['status'], string> = {
  ok: 'bg-green', degraded: 'bg-yellow', unavailable: 'bg-red',
}
const COMPONENT_LABEL: Record<string, string> = {
  kronos: 'Kronos', lightgbm: 'LGBM', llm_synthesis: 'Synth', macro: 'Macro',
}

/** Small honest per-signal provenance badges — e.g. "Kronos: pretrained (not NSE)". */
function ComponentBadges({ components }: { components: LiveSignal['components_json'] }) {
  if (!components) return null
  return (
    <div className="mt-1.5 flex items-center gap-2 flex-wrap">
      {(Object.entries(components) as [string, ComponentStatus][]).map(([name, c]) => (
        <span
          key={name}
          title={`${COMPONENT_LABEL[name] || name}: ${c.status}${c.source ? ` (${c.source})` : ''} — ${c.detail}`}
          className="flex items-center gap-1 text-[10px] text-text-secondary/80 cursor-default"
        >
          <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${COMPONENT_DOT[c.status]}`} />
          {COMPONENT_LABEL[name] || name}
          {c.source && c.source !== 'skipped' && <span className="opacity-60">·{c.source}</span>}
          {c.source === 'skipped' && <span className="opacity-60">·skipped</span>}
        </span>
      ))}
    </div>
  )
}

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
  SUGGESTED:    { color: 'text-accent',  label: 'Suggested' },
  RATED:        { color: 'text-yellow',  label: 'Rated' },
  BOUGHT:       { color: 'text-green',   label: 'Bought' },
  SOLD:         { color: 'text-red',     label: 'Sold' },
  REANALYZED:   { color: 'text-blue-400', label: 'Re-analyzed' },
  NOTE:         { color: 'text-text-secondary', label: 'Note' },
  AUTO_BUY:     { color: 'text-green',   label: '🤖 Auto-buy' },
  AUTO_SELL:    { color: 'text-red',     label: '🤖 Auto-sell' },
  AUTO_PASS:    { color: 'text-text-secondary', label: '🤖 Passed' },
  PARAM_CHANGE: { color: 'text-accent',  label: '🤖 Tuned' },
  RETRAIN:      { color: 'text-yellow',  label: '🤖 Retrain' },
  JOB_RUN:      { color: 'text-text-secondary', label: 'Job' },
}

// Angel One stock pages live at angelone.in/stocks/<company-name-slug> ("Limited" → "ltd").
// With a real company name we deep-link straight to the page; otherwise a plain
// Google search (NOT "I'm Feeling Lucky", which mis-redirected) that reliably
// surfaces the Angel One page as the top result.
const angelOneUrl = (ticker: string, name?: string | null) => {
  if (name && name.toUpperCase() !== ticker.toUpperCase()) {
    const slug = name.toLowerCase()
      .replace(/\blimited\b/g, 'ltd')
      .replace(/[&.,()'/]/g, ' ')
      .trim()
      .replace(/\s+/g, '-')
    return `https://www.angelone.in/stocks/${slug}`
  }
  return `https://www.google.com/search?q=${encodeURIComponent(`angelone.in ${ticker} share price`)}`
}

const statusStyle: Record<string, string> = {
  on_track: 'text-green', ahead: 'text-green', target_hit: 'text-green',
  behind: 'text-yellow', expired: 'text-red', stopped: 'text-red',
}

/** Live LTP next to the frozen price_at_signal, with % move and a freshness dot. */
function LivePriceBadge({
  ticker, priceAtSignal, ltp, ts, connected, marketOpen,
}: { ticker: string; priceAtSignal: number; ltp: number | null | undefined; ts: number | null | undefined; connected: boolean; marketOpen: boolean }) {
  const flash = useFlash(ltp)

  if (ltp == null) return null

  const movePct = ((ltp - priceAtSignal) / priceAtSignal) * 100
  const up = movePct >= 0

  return (
    <span
      key={ticker}
      className={`flex items-center gap-1 text-xs transition-colors duration-500 rounded px-1 ${flashBgClass(flash)}`}
    >
      <PriceAgeDot ts={ts} connected={connected} marketOpen={marketOpen} />
      <span className="text-text-primary">₹{ltp.toFixed(1)}</span>
      <span className={up ? 'text-green' : 'text-red'}>
        {up ? '▲' : '▼'}{Math.abs(movePct).toFixed(2)}%
      </span>
      {!marketOpen && <span className="text-[10px] text-text-secondary opacity-70">as of 15:30 close</span>}
    </span>
  )
}

export default function Live() {
  const [signals, setSignals] = useState<LiveSignal[]>([])
  const [account, setAccount] = useState<Account | null>(null)
  const [activity, setActivity] = useState<ActivityEvent[]>([])
  const [reviews, setReviews] = useState<PositionReview[]>([])
  const [dataStatus, setDataStatus] = useState<DataStatus | null>(null)
  const [mode, setMode] = useState<TradingMode | null>(null)
  const [onlyAffordable, setOnlyAffordable] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const { prices: livePrices, connected: wsConnected } = useRealtimePrices()
  const marketStatus = useMarketStatus()
  const marketOpen = marketStatus.state === 'open'
  const [activityStreaming, setActivityStreaming] = useState(false)

  const isPaper = mode?.mode !== 'LIVE'  // default to paper until proven otherwise

  const refresh = useCallback(async () => {
    const [sig, acc, act, rev, ds, md] = await Promise.all([
      api.get(`/live/signals?limit=40${onlyAffordable ? '&only_affordable=true' : ''}`),
      api.get('/live/account'),
      api.get('/live/activity?limit=60'),
      api.get('/live/positions/reviews'),
      api.get('/live/data-status'),
      api.get('/live/mode'),
    ])
    setSignals(Array.isArray(sig) ? sig : [])
    setAccount(acc)
    setActivity(Array.isArray(act) ? act : [])
    setReviews(Array.isArray(rev) ? rev : [])
    setDataStatus(ds && ds.source ? ds : null)
    setMode(md && md.mode ? md : null)
  }, [onlyAffordable])

  // Poll fallback: 60s normally, relaxed to 120s once the SSE activity stream
  // is live (incremental events arrive in real time; poll just reconciles).
  useEffect(() => {
    refresh()
    const t = setInterval(refresh, activityStreaming ? 120_000 : 60_000)
    return () => clearInterval(t)
  }, [refresh, activityStreaming])

  // Streaming activity feed — new activity_log events pushed via SSE, prepended
  // incrementally without a full-page refetch. Poll above remains the fallback.
  useEffect(() => {
    const es = new EventSource('/api/stream/activity')
    es.onopen = () => setActivityStreaming(true)
    es.onerror = () => setActivityStreaming(false) // EventSource auto-reconnects
    es.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)
        if (msg.type === 'connected') { setActivityStreaming(true); return }
        if (msg.type !== 'activity') return
        setActivity(prev => {
          if (prev.some(a => a.id === msg.id)) return prev
          const ev: ActivityEvent = {
            id: msg.id, event_type: msg.event_type, ticker: msg.ticker,
            rating: msg.rating, note: msg.note, payload: msg.payload,
            created_at: msg.created_at,
          }
          return [ev, ...prev].slice(0, 60)
        })
      } catch { /* ignore malformed frames */ }
    }
    return () => es.close()
  }, [])

  const act = async (fn: () => Promise<unknown>, key: string) => {
    setBusy(key); setError(null)
    try { await fn(); await refresh() }
    catch (e) { setError(e instanceof Error ? e.message : 'Action failed') }
    finally { setBusy(null) }
  }

  // Human feedback only — buying, passing, and pipeline runs are the brain's job now.
  const rate = (s: LiveSignal, like: boolean) =>
    act(() => api.post('/live/rate', { signal_id: s.id, ticker: s.ticker, like, reason: like ? 'liked' : 'confidence too low' }), `rate-${s.id}-${like}`)

  return (
    <div className="h-full flex flex-col overflow-hidden bg-bg-primary">
      {/* Header */}
      <div className="px-4 py-2.5 border-b border-border bg-bg-card flex items-center gap-4 flex-shrink-0">
        <span className="text-sm font-semibold text-white">Live Signals</span>
        {/* Paper vs Live mode chip */}
        {isPaper ? (
          <span
            title={mode?.reason || 'Tracking only until a track record exists'}
            className="flex items-center gap-1.5 text-[11px] px-2 py-0.5 rounded bg-yellow/15 text-yellow border border-yellow/30">
            <FlaskConical size={12} /> PAPER MODE
            {mode && <span className="text-yellow/70 font-normal">· {mode.reason}</span>}
          </span>
        ) : (
          <span className="flex items-center gap-1.5 text-[11px] px-2 py-0.5 rounded bg-green/15 text-green border border-green/30">
            🟢 LIVE
          </span>
        )}
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
          <span className="flex items-center gap-1.5 text-[11px] text-text-secondary">
            <Bot size={13} className="text-accent" /> brain trades autonomously every 30 min
          </span>
          <button onClick={refresh} className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 bg-bg-hover text-text-secondary rounded hover:text-text-primary">
            <RefreshCw size={12} /> Refresh
          </button>
        </div>
      </div>

      {/* Data freshness banner — is this LIVE intraday or PAST end-of-day? */}
      {dataStatus && (
        <div className={`px-4 py-1.5 text-xs border-b border-border flex items-center gap-2 flex-shrink-0 ${
          dataStatus.is_live ? 'text-green bg-green/10' : 'text-yellow bg-yellow/10'
        }`}>
          {dataStatus.is_live
            ? <>🟢 {dataStatus.label}{dataStatus.age_hours != null && <span className="opacity-70">· updated {dataStatus.age_hours}h ago</span>}</>
            : <><AlertTriangle size={13} className="flex-shrink-0" /> {dataStatus.label}</>}
        </div>
      )}

      {error && <div className="px-4 py-1.5 text-xs text-red bg-red/10 border-b border-border flex-shrink-0">{error}</div>}

      <div className="flex-1 flex overflow-hidden">
        {/* Signals */}
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {signals.length === 0 && (
            <div className="text-xs text-text-secondary text-center mt-8">
              No BUY signals yet — the brain generates them automatically every 30 minutes during market hours.
            </div>
          )}
          {signals.map(s => (
            <div key={s.id} className="bg-bg-card border border-border rounded-lg p-3">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-white">{s.ticker}</span>
                {s.name && s.name.toUpperCase() !== s.ticker.toUpperCase() && (
                  <span className="text-[11px] text-text-secondary truncate max-w-[180px]" title={s.name}>{s.name}</span>
                )}
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-bg-hover text-text-secondary">{s.timeframe}</span>
                {s.sector && <span className="text-[10px] text-text-secondary">{s.sector}</span>}
                <span className={`ml-auto text-xs font-medium ${confColor(s.final_confidence)}`}>
                  {(s.final_confidence * 100).toFixed(0)}% conf
                </span>
              </div>

              {/* The intelligent headline: buy @ X → Y (+Z%) in ~ETAd */}
              <div className="mt-2 flex items-center gap-2 text-sm">
                <span className="text-text-secondary">signal ₹{s.price_at_signal.toFixed(1)}</span>
                <LivePriceBadge
                  ticker={s.ticker}
                  priceAtSignal={s.price_at_signal}
                  ltp={livePrices[s.ticker]?.ltp}
                  ts={livePrices[s.ticker]?.ts}
                  connected={wsConnected}
                  marketOpen={marketOpen}
                />
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
                {isPaper && <span className="text-yellow/70 italic ml-auto">tracked, not executed</span>}
              </div>

              {/* Stage 0 truth layer — what actually produced this signal */}
              <ComponentBadges components={s.components_json} />

              {/* Human feedback — the only human input; trades are autonomous */}
              <div className="mt-2.5 flex items-center gap-2">
                <button onClick={() => rate(s, true)} disabled={!!busy}
                  className="flex items-center gap-1 text-[11px] px-2 py-1 rounded bg-bg-hover text-text-secondary hover:text-green disabled:opacity-40">
                  <ThumbsUp size={12} /> Like
                </button>
                <button onClick={() => rate(s, false)} disabled={!!busy}
                  className="flex items-center gap-1 text-[11px] px-2 py-1 rounded bg-bg-hover text-text-secondary hover:text-red disabled:opacity-40">
                  <ThumbsDown size={12} /> Dislike
                </button>
                <a href={angelOneUrl(s.ticker, s.name)} target="_blank" rel="noopener noreferrer"
                  className="flex items-center gap-1 text-[11px] px-2 py-1 rounded bg-accent/15 text-accent border border-accent/30 hover:bg-accent hover:text-white transition-colors">
                  <ExternalLink size={12} /> Buy on Angel One
                </a>
                <span className="ml-auto flex items-center gap-1 text-[10px] text-text-secondary italic">
                  <Bot size={11} /> brain decides
                </span>
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
              <span className="ml-auto text-[10px] text-text-secondary">auto, every 30 min</span>
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
            <div className="px-3 py-2 text-xs font-semibold text-white flex items-center gap-2">
              Activity Log
              {activityStreaming && (
                <span className="flex items-center gap-1 text-[10px] font-normal text-green">
                  <span className="w-1.5 h-1.5 rounded-full bg-green animate-pulse" /> streaming
                </span>
              )}
            </div>
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
