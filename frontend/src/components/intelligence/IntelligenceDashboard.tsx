import { useState, useEffect, useCallback } from 'react'
import type { Signal, TerminalLine } from '../../types'
import TerminalBase from './terminals/TerminalBase'
import SignalFeed from './SignalFeed'
import LearningsView from './learnings/LearningsView'
import { Bot } from 'lucide-react'

type SubTab = 'live' | 'learnings'

interface ReasoningRow {
  id: number
  model_name: string
  reasoning: string
  created_at: string
  ticker: string
  signal_type: string
  timeframe: string
  final_confidence: number | null
}

const istTime = (iso: string) =>
  new Date(iso).toLocaleTimeString('en-IN', { hour12: false, timeZone: 'Asia/Kolkata' })

// First informative line of a model's reasoning blob.
const gist = (reasoning: string) => {
  const lines = reasoning.split('\n').map(l => l.trim()).filter(Boolean)
  return (lines[1] || lines[0] || '').replace(/^[•\-*]\s*/, '').slice(0, 160)
}

const toLine = (r: ReasoningRow): TerminalLine => {
  const conf = r.final_confidence != null ? ` ${(r.final_confidence * 100).toFixed(0)}%` : ''
  const color = r.signal_type === 'BUY' ? 'green' : r.signal_type === 'SELL' ? 'red' : 'yellow'
  return {
    ts: istTime(r.created_at),
    ticker: r.ticker,
    color,
    text: `${r.ticker} ${r.timeframe}  ${r.signal_type}${conf}  ${gist(r.reasoning)}`,
  }
}

export default function IntelligenceDashboard() {
  const [subTab, setSubTab] = useState<SubTab>('live')
  const [signals, setSignals] = useState<Signal[]>([])
  const [mlLines, setMlLines] = useState<TerminalLine[]>([])
  const [kronosLines, setKronosLines] = useState<TerminalLine[]>([])
  const [slmLines, setSlmLines] = useState<TerminalLine[]>([])
  const [claudeLines, setClaudeLines] = useState<TerminalLine[]>([])
  const [lastUpdate, setLastUpdate] = useState<string | null>(null)

  // Passive observatory: tail what the brain already reasoned, oldest → newest.
  const refresh = useCallback(async () => {
    try {
      const [rows, sigs] = await Promise.all([
        fetch('/api/live/reasoning?limit=400&hours=48').then(r => r.json()) as Promise<ReasoningRow[]>,
        fetch('/api/live/signals?limit=40').then(r => r.json()),
      ])
      if (Array.isArray(rows)) {
        const ordered = [...rows].reverse()
        setMlLines(ordered.filter(r => r.model_name === 'lgbm').map(toLine).slice(-300))
        setKronosLines(ordered.filter(r => r.model_name === 'kronos').map(toLine).slice(-300))
        setSlmLines(ordered.filter(r => r.model_name === 'macro' || r.model_name === 'slm').map(toLine).slice(-300))
        setClaudeLines(ordered.filter(r => r.model_name === 'claude' || r.model_name === 'combined').map(toLine).slice(-300))
        if (rows.length > 0) setLastUpdate(istTime(rows[0].created_at))
      }
      if (Array.isArray(sigs)) {
        // /api/live/signals field names → SignalFeed's Signal shape
        setSignals(sigs.map((s: Record<string, unknown>) => ({
          ...s,
          signal: s.signal_type,
          price: s.price_at_signal,
          target: s.target_price,
          confidence: s.final_confidence,
        })) as unknown as Signal[])
      }
    } catch { /* backend briefly down — next poll recovers */ }
  }, [])

  useEffect(() => {
    refresh()
    const t = setInterval(refresh, 30_000) // pipeline runs every 30 min; tail it continuously
    return () => clearInterval(t)
  }, [refresh])

  return (
    <div className="h-full flex flex-col bg-bg-primary overflow-hidden">
      {/* Sub-tab bar */}
      <div className="flex items-center gap-4 px-4 py-2 border-b border-border bg-bg-card flex-shrink-0">
        {(['live', 'learnings'] as SubTab[]).map(t => (
          <button key={t} onClick={() => setSubTab(t)}
            className={`text-sm px-3 py-1 rounded ${subTab === t ? 'bg-accent text-white' : 'text-text-secondary hover:text-text-primary'}`}>
            {t === 'live' ? '⚡ Live Intelligence' : '📚 Learnings'}
          </button>
        ))}
        <div className="ml-auto flex items-center gap-2 text-[11px] text-text-secondary">
          <Bot size={13} className="text-accent" />
          auto — terminals tail the brain's pipeline runs
          {lastUpdate && <span>· last reasoning {lastUpdate} IST</span>}
        </div>
      </div>

      {subTab === 'live' ? (
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* 4 Terminals */}
          <div className="flex-1 grid grid-cols-2 grid-rows-2 gap-px bg-border overflow-hidden">
            <TerminalBase title="TERMINAL 1 — ML MODEL (LightGBM)" lines={mlLines} />
            <TerminalBase title="TERMINAL 2 — KRONOS FOUNDATION" lines={kronosLines} />
            <TerminalBase title="TERMINAL 3 — MACRO / SLM" lines={slmLines} />
            <TerminalBase title="TERMINAL 4 — CLAUDE SONNET" lines={claudeLines} />
          </div>

          {/* Signal Feed */}
          <div className="h-44 flex-shrink-0 border-t border-border">
            <SignalFeed signals={signals} />
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-auto">
          <LearningsView />
        </div>
      )}
    </div>
  )
}
