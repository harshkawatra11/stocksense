import { useState, useEffect, useRef } from 'react'
import type { Signal, TerminalLine } from '../../types'
import TerminalBase from './terminals/TerminalBase'
import SignalFeed from './SignalFeed'
import LearningsView from './learnings/LearningsView'
import { Play, Square } from 'lucide-react'

type SubTab = 'live' | 'learnings'

export default function IntelligenceDashboard() {
  const [subTab, setSubTab] = useState<SubTab>('live')
  const [running, setRunning] = useState(false)
  const [signals, setSignals] = useState<Signal[]>([])

  const [mlLines, setMlLines]         = useState<TerminalLine[]>([])
  const [kronosLines, setKronosLines] = useState<TerminalLine[]>([])
  const [slmLines, setSlmLines]       = useState<TerminalLine[]>([])
  const [claudeLines, setClaudeLines] = useState<TerminalLine[]>([])

  const esRef = useRef<EventSource | null>(null)

  const ts = () => new Date().toLocaleTimeString('en-IN', { hour12: false, timeZone: 'Asia/Kolkata' })

  const appendML     = (l: TerminalLine) => setMlLines(p => [...p.slice(-300), l])
  const appendKronos = (l: TerminalLine) => setKronosLines(p => [...p.slice(-300), l])
  const appendSLM    = (l: TerminalLine) => setSlmLines(p => [...p.slice(-300), l])
  const appendClaude = (l: TerminalLine) => setClaudeLines(p => [...p.slice(-300), l])

  const startStream = () => {
    if (esRef.current) esRef.current.close()
    setRunning(true)
    appendML({ ts: ts(), text: '--- Starting pipeline batch ---', color: 'grey' })
    appendKronos({ ts: ts(), text: '--- Waiting for ML output ---', color: 'grey' })
    appendSLM({ ts: ts(), text: '--- Portfolio guard active ---', color: 'grey' })
    appendClaude({ ts: ts(), text: '--- Awaiting top signals ---', color: 'grey' })

    const es = new EventSource('/api/stream/signals')
    esRef.current = es

    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data)
        handleEvent(data)
      } catch {}
    }

    es.onerror = () => {
      setRunning(false)
      appendClaude({ ts: ts(), text: '--- Stream ended ---', color: 'grey' })
      es.close()
    }
  }

  const stopStream = () => {
    esRef.current?.close()
    setRunning(false)
  }

  const handleEvent = (data: Record<string, unknown>) => {
    const t = ts()
    const ticker = (data.ticker as string) || ''
    const signal = (data.signal as string) || 'HOLD'
    const conf   = ((data.confidence as number) || 0) * 100

    const sigColor = signal === 'BUY' ? 'green' : signal === 'SELL' ? 'red' : 'yellow'

    if (data.type === 'ml_result' || data.ml_reasoning) {
      appendML({ ts: t, ticker, color: sigColor,
        text: `${ticker}  ${signal}  ${conf.toFixed(0)}%  ${(data.ml_reasoning as string || '').split('\n')[1]?.trim() || ''}` })
    }

    if (data.type === 'kronos_result' || data.kronos_reasoning) {
      appendKronos({ ts: t, ticker, color: sigColor,
        text: `${ticker}  ${(data.kronos_reasoning as string || '').split('\n')[1]?.trim() || ''}` })
    }

    if (data.type === 'slm_result' || data.slm_reasoning) {
      const slmText = data.slm_reasoning as string || ''
      const skipFlag = signal === 'SKIP'
      appendSLM({ ts: t, ticker, color: skipFlag ? 'grey' : sigColor,
        text: `${ticker}  ${skipFlag ? '✗ SKIP (not in portfolio)' : slmText.split('\n')[1]?.trim() || signal}` })
    }

    if (data.type === 'claude_checking') {
      appendClaude({ ts: t, color: 'grey',
        text: `--- Reviewing ${data.count} signals ---` })
    }

    if (data.type === 'claude_enriched' && data.claude_reasoning) {
      const action = (data.claude_action as string) || 'CONFIRM'
      appendClaude({ ts: t, ticker, color: action === 'CONFIRM' ? 'green' : 'red',
        text: `${ticker}  ${action === 'CONFIRM' ? '✅' : '❌'} ${action}  ${(data.claude_reasoning as string).split('\n')[1]?.trim() || ''}` })
    }

    if (data.type === 'batch_complete') {
      const all = [mlLines, kronosLines, slmLines, claudeLines]
      void all
      appendClaude({ ts: t, color: 'grey', text: `--- Batch complete: ${data.total} signals processed ---` })
      setRunning(false)
    }

    // Add to signal feed if it's a final confirmed signal
    if (ticker && signal !== 'SKIP' && signal !== 'HOLD' && (data.confidence as number) > 0.55) {
      setSignals(p => [data as unknown as Signal, ...p.slice(0, 99)])
    }
  }

  useEffect(() => () => { esRef.current?.close() }, [])

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
        <div className="ml-auto flex items-center gap-2">
          {subTab === 'live' && (
            <button onClick={running ? stopStream : startStream}
              className={`flex items-center gap-2 px-3 py-1 rounded text-xs font-medium ${
                running ? 'bg-red text-white' : 'bg-green text-bg-primary'}`}>
              {running ? <><Square size={12} /> Stop</> : <><Play size={12} /> Run Pipeline</>}
            </button>
          )}
        </div>
      </div>

      {subTab === 'live' ? (
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* 4 Terminals */}
          <div className="flex-1 grid grid-cols-2 grid-rows-2 gap-px bg-border overflow-hidden">
            <TerminalBase title="TERMINAL 1 — ML MODEL (LightGBM)" lines={mlLines} />
            <TerminalBase title="TERMINAL 2 — KRONOS FOUNDATION" lines={kronosLines} />
            <TerminalBase title="TERMINAL 3 — SLM NSE COMMANDER" lines={slmLines} />
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
