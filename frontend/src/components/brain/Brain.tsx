import { useCallback, useEffect, useRef, useState } from 'react'
import { createChart, ColorType, AreaSeries } from 'lightweight-charts'
import {
  BrainCircuit, Activity, Wallet, FlaskConical, Wifi, WifiOff,
  ArrowUpRight, ArrowDownRight, MinusCircle, SlidersHorizontal, History,
} from 'lucide-react'
import type { BrainStatus, BrainParamChange, JobRun, EquityPoint, Decision } from '../../types'

const api = { get: (p: string) => fetch(`/api${p}`).then(r => r.json()) }

const PARAM_LABELS: Record<string, string> = {
  confidence_threshold: 'Confidence gate',
  max_position_pct: 'Max position size',
  max_open_positions: 'Max open positions',
  macro_nudge_cap: 'Macro nudge cap',
  agreement_boost: 'Agreement boost',
}

const fmtTime = (iso: string | null) => {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleString('en-IN', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })
}

const ago = (iso: string | null) => {
  if (!iso) return '—'
  const mins = Math.floor((Date.now() - new Date(iso).getTime()) / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const h = Math.floor(mins / 60)
  if (h < 24) return `${h}h ago`
  return `${Math.floor(h / 24)}d ago`
}

function EquityCurve({ points }: { points: EquityPoint[] }) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!ref.current || points.length === 0) return
    const el = ref.current
    const chart = createChart(el, {
      layout: { background: { type: ColorType.Solid, color: '#1a1d24' }, textColor: '#9aa0ab' },
      grid: { vertLines: { color: '#2a2d3766' }, horzLines: { color: '#2a2d3766' } },
      rightPriceScale: { borderColor: '#2a2d37' },
      timeScale: { borderColor: '#2a2d37' },
      width: el.clientWidth,
      height: el.clientHeight,
    })
    const series = chart.addSeries(AreaSeries, {
      lineColor: '#00d4aa',
      topColor: '#00d4aa33',
      bottomColor: '#00d4aa05',
      lineWidth: 2,
    })
    series.setData(points.map(p => ({
      time: p.date as unknown as import('lightweight-charts').Time,
      value: p.equity,
    })))
    chart.timeScale().fitContent()
    const ro = new ResizeObserver(() => chart.applyOptions({ width: el.clientWidth }))
    ro.observe(el)
    return () => { ro.disconnect(); chart.remove() }
  }, [points])

  return <div ref={ref} className="w-full h-full" />
}

export default function Brain() {
  const [status, setStatus] = useState<BrainStatus | null>(null)
  const [equity, setEquity] = useState<EquityPoint[]>([])
  const [history, setHistory] = useState<BrainParamChange[]>([])
  const [decisions, setDecisions] = useState<Decision[]>([])
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      const [st, eq, hist, dec] = await Promise.all([
        api.get('/brain/status'),
        api.get('/brain/equity?days=180'),
        api.get('/brain/params/history?limit=50'),
        api.get('/live/decisions?limit=40'),
      ])
      setStatus(st)
      setEquity(Array.isArray(eq) ? eq : [])
      setHistory(Array.isArray(hist) ? hist : [])
      setDecisions(Array.isArray(dec) ? dec : [])
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load brain status')
    }
  }, [])

  useEffect(() => {
    refresh()
    const t = setInterval(refresh, 30_000)
    return () => clearInterval(t)
  }, [refresh])

  const autoDecisions = decisions.filter(d => d.rationale?.includes('[AUTO]'))
  const isPaper = status?.mode?.mode !== 'LIVE'
  const equityNow = status?.account?.equity ?? 0
  const equityStart = equity.length > 0 ? equity[0].equity : equityNow
  const equityPnl = equityNow - equityStart
  const equityUp = equityPnl >= 0

  return (
    <div className="h-full flex flex-col overflow-hidden bg-bg-primary">
      {/* Header — display only, the brain has no switches */}
      <div className="px-4 py-2.5 border-b border-border bg-bg-card flex items-center gap-4 flex-shrink-0">
        <span className="flex items-center gap-2 text-sm font-semibold text-white">
          <BrainCircuit size={16} className="text-accent" /> Autonomous Brain
        </span>
        <span className="flex items-center gap-1.5 text-[11px] px-2 py-0.5 rounded bg-green/15 text-green border border-green/30">
          <span className="w-1.5 h-1.5 rounded-full bg-green animate-pulse" /> ALWAYS ON
        </span>
        {isPaper ? (
          <span title={status?.mode?.reason}
            className="flex items-center gap-1.5 text-[11px] px-2 py-0.5 rounded bg-yellow/15 text-yellow border border-yellow/30">
            <FlaskConical size={12} /> PAPER {status?.mode?.rolling_accuracy != null &&
              <span className="text-yellow/70">· {(status.mode.rolling_accuracy * 100).toFixed(0)}% acc</span>}
          </span>
        ) : (
          <span className="text-[11px] px-2 py-0.5 rounded bg-green/15 text-green border border-green/30">LIVE</span>
        )}
        {status?.groww_feed && (
          <span className={`flex items-center gap-1 text-[11px] ${status.groww_feed.configured ? 'text-green' : 'text-text-secondary'}`}>
            {status.groww_feed.configured ? <Wifi size={12} /> : <WifiOff size={12} />}
            Groww feed {status.groww_feed.configured ? 'on' : 'not configured'}
          </span>
        )}
        {status?.data && (
          <span className={`text-[11px] ${status.data.is_live ? 'text-green' : 'text-text-secondary'}`}>
            data: {status.data.label}
          </span>
        )}
        <span className="ml-auto text-[11px] text-text-secondary">auto-refreshes · no controls by design</span>
      </div>

      {error && <div className="px-4 py-1.5 text-xs text-red bg-red/10 border-b border-border flex-shrink-0">{error}</div>}

      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {/* Top row: equity + account cards */}
        <div className="grid grid-cols-4 gap-3">
          <div className="col-span-3 bg-bg-card border border-border rounded-lg overflow-hidden">
            <div className="px-3 pt-2.5 pb-1 flex items-center gap-3">
              <span className="text-xs font-semibold text-white">Paper Equity</span>
              <span className="text-sm text-text-primary">₹{equityNow.toFixed(2)}</span>
              <span className={`flex items-center text-xs ${equityUp ? 'text-green' : 'text-red'}`}>
                {equityUp ? <ArrowUpRight size={13} /> : <ArrowDownRight size={13} />}
                {equityUp ? '+' : ''}{equityPnl.toFixed(2)} since start
              </span>
            </div>
            <div className="h-48">
              {equity.length > 0
                ? <EquityCurve points={equity} />
                : <div className="h-full flex items-center justify-center text-xs text-text-secondary">No equity history yet — the brain builds it as it trades.</div>}
            </div>
          </div>

          <div className="space-y-3">
            <div className="bg-bg-card border border-border rounded-lg p-3">
              <div className="flex items-center gap-2 text-xs text-text-secondary"><Wallet size={13} /> Account</div>
              <div className="mt-2 space-y-1 text-xs">
                <div className="flex justify-between"><span className="text-text-secondary">Cash</span><span className="text-text-primary">₹{status?.account.cash_available.toFixed(2) ?? '—'}</span></div>
                <div className="flex justify-between"><span className="text-text-secondary">Positions ({status?.account.positions ?? 0})</span><span className="text-text-primary">₹{status?.account.market_value.toFixed(2) ?? '—'}</span></div>
                <div className="flex justify-between border-t border-border pt-1 mt-1"><span className="text-text-secondary">Equity</span><span className="text-white font-medium">₹{status?.account.equity.toFixed(2) ?? '—'}</span></div>
              </div>
            </div>
            <div className="bg-bg-card border border-border rounded-lg p-3">
              <div className="flex items-center gap-2 text-xs text-text-secondary"><Activity size={13} /> Today (auto)</div>
              <div className="mt-2 grid grid-cols-2 gap-1 text-xs">
                <span className="text-text-secondary">Buys</span><span className="text-green text-right">{status?.today?.buys ?? 0}</span>
                <span className="text-text-secondary">Sells</span><span className="text-red text-right">{status?.today?.sells ?? 0}</span>
                <span className="text-text-secondary">Passes</span><span className="text-text-primary text-right">{status?.today?.passes ?? 0}</span>
                <span className="text-text-secondary">Realized P&L</span>
                <span className={`text-right ${(status?.today?.realized_pnl ?? 0) >= 0 ? 'text-green' : 'text-red'}`}>
                  ₹{(status?.today?.realized_pnl ?? 0).toFixed(2)}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Middle row: params + heartbeat */}
        <div className="grid grid-cols-2 gap-3">
          {/* Adaptive parameters */}
          <div className="bg-bg-card border border-border rounded-lg p-3">
            <div className="flex items-center gap-2 text-xs font-semibold text-white">
              <SlidersHorizontal size={13} className="text-accent" /> Adaptive Parameters
              <span className="text-[10px] text-text-secondary font-normal ml-auto">tuned nightly by Claude within hard bounds</span>
            </div>
            <div className="mt-2 space-y-2">
              {(status?.params ?? []).map(p => {
                const range = p.max_value - p.min_value
                const pct = range > 0 ? ((p.value - p.min_value) / range) * 100 : 0
                return (
                  <div key={p.param_name} className="text-xs">
                    <div className="flex items-center gap-2">
                      <span className="text-text-primary">{PARAM_LABELS[p.param_name] || p.param_name}</span>
                      <span className="text-white font-mono ml-auto">{p.value}</span>
                      <span className="text-[10px] text-text-secondary">[{p.min_value}–{p.max_value}]</span>
                    </div>
                    <div className="mt-1 h-1 bg-bg-hover rounded overflow-hidden">
                      <div className="h-full bg-accent" style={{ width: `${pct}%` }} />
                    </div>
                    {p.updated_by !== 'seed' && (
                      <div className="mt-0.5 text-[10px] text-text-secondary truncate" title={p.reason}>
                        {p.updated_by} · {ago(p.updated_at)}{p.reason && ` — ${p.reason}`}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>

          {/* Job heartbeat */}
          <div className="bg-bg-card border border-border rounded-lg p-3">
            <div className="text-xs font-semibold text-white">Scheduler Heartbeat</div>
            <div className="mt-2 space-y-1 max-h-72 overflow-y-auto">
              {(status?.jobs ?? []).length === 0 && (
                <div className="text-[11px] text-text-secondary">No job runs recorded yet — start the scheduler (python -m scheduler.market_runner).</div>
              )}
              {(status?.jobs ?? []).map(j => (
                <div key={j.job_id} className="flex items-center gap-2 text-[11px] py-1 border-b border-border/40">
                  <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                    j.status === 'ok' ? 'bg-green' : j.status === 'running' ? 'bg-yellow animate-pulse' : 'bg-red'
                  }`} />
                  <span className="text-text-primary w-36 truncate">{j.job_id}</span>
                  <span className="text-text-secondary truncate flex-1" title={j.error || j.summary}>
                    {j.status === 'error' ? (j.error || 'error') : (j.summary || j.status)}
                  </span>
                  <span className="text-text-secondary flex-shrink-0">{ago(j.started_at)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom row: auto decisions + param history */}
        <div className="grid grid-cols-2 gap-3">
          {/* Auto decision feed */}
          <div className="bg-bg-card border border-border rounded-lg p-3">
            <div className="text-xs font-semibold text-white">Autonomous Decisions</div>
            <div className="mt-2 space-y-1 max-h-80 overflow-y-auto">
              {autoDecisions.length === 0 && (
                <div className="text-[11px] text-text-secondary">No auto decisions yet — they appear as the pipeline runs during market hours.</div>
              )}
              {autoDecisions.map(d => (
                <div key={d.id} className="flex items-center gap-2 text-[11px] py-1 border-b border-border/40">
                  {d.action === 'BUY' ? <ArrowUpRight size={12} className="text-green flex-shrink-0" />
                    : d.action === 'SELL' ? <ArrowDownRight size={12} className="text-red flex-shrink-0" />
                    : <MinusCircle size={12} className="text-text-secondary flex-shrink-0" />}
                  <span className={`w-10 font-medium ${
                    d.action === 'BUY' ? 'text-green' : d.action === 'SELL' ? 'text-red' : 'text-text-secondary'
                  }`}>{d.action}</span>
                  <span className="text-text-primary w-24 truncate">{d.ticker}</span>
                  {d.quantity > 0 && d.price != null && (
                    <span className="text-text-secondary">{d.quantity} @ ₹{Number(d.price).toFixed(1)}</span>
                  )}
                  {d.pnl != null && (
                    <span className={Number(d.pnl) >= 0 ? 'text-green' : 'text-red'}>
                      {Number(d.pnl) >= 0 ? '+' : ''}₹{Number(d.pnl).toFixed(2)}
                    </span>
                  )}
                  <span className="text-text-secondary truncate flex-1" title={d.rationale}>
                    {d.rationale?.replace('[AUTO]', '').replace('[PAPER]', '').trim()}
                  </span>
                  <span className="text-text-secondary flex-shrink-0">{ago(d.decided_at)}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Param change history */}
          <div className="bg-bg-card border border-border rounded-lg p-3">
            <div className="flex items-center gap-2 text-xs font-semibold text-white">
              <History size={13} className="text-accent" /> Self-Calibration History
            </div>
            <div className="mt-2 space-y-1 max-h-80 overflow-y-auto">
              {history.length === 0 && (
                <div className="text-[11px] text-text-secondary">
                  No parameter changes yet. The brain calibrates itself nightly at 16:15 IST once it has trade outcomes to learn from.
                </div>
              )}
              {history.map((h, i) => (
                <div key={i} className="text-[11px] py-1 border-b border-border/40">
                  <div className="flex items-center gap-2">
                    <span className="text-text-primary">{PARAM_LABELS[h.param_name] || h.param_name}</span>
                    <span className="font-mono text-text-secondary">{h.old_value}</span>
                    <span className="text-text-secondary">→</span>
                    <span className="font-mono text-white">{h.new_value}</span>
                    <span className="ml-auto text-text-secondary">{fmtTime(h.changed_at)}</span>
                  </div>
                  {h.reason && <div className="text-text-secondary mt-0.5 truncate" title={h.reason}>{h.changed_by}: {h.reason}</div>}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
