import { useCallback, useEffect, useState } from 'react'
import { RefreshCw, Database, FileText, Activity, Brain, AlertTriangle, FlaskConical } from 'lucide-react'
import type { DataStatus, TradingMode, Learning, ActivityEvent } from '../../types'

const api = {
  get: (p: string) => fetch(`/api${p}`).then(r => r.json()).catch(() => null),
}

interface Decision {
  id: number
  ticker: string
  action: string
  quantity: number
  price: number | null
  cash_after: number | null
  rationale: string | null
  outcome: string | null
  pnl: number | null
  decided_at: string
}

interface AccuracySummary {
  total_resolved: number
  correct: number
  accuracy_pct: number
  total_buys: number
  total_sells: number
  avg_confidence: number
  today_total: number
}

interface ModelAccuracy {
  model_name: string
  total_signals: number
  correct_signals: number
  accuracy: number | null
}

interface RecentSignal {
  id: number
  ticker: string
  signal_type?: string
  timeframe?: string
  final_confidence?: number
  status?: string
  fired_at?: string
}

interface LogFiles {
  day: string
  path: string
  available_days: string[]
  lines: string[]
}

// Where everything is logged — shown as a reference at the top of the panel.
const LOG_MAP: { what: string; where: string; when: string }[] = [
  { what: 'Signals + targets/SL', where: 'Postgres · signals', when: 'each pipeline run' },
  { what: 'Per-model reasoning', where: 'Postgres · signal_reasoning', when: 'each run, 1 row/model' },
  { what: 'Paper trades / passes', where: 'Postgres · decisions (+[PAPER])', when: 'on Buy/Pass action' },
  { what: 'Lifecycle feed', where: 'Postgres · activity_log', when: 'suggest/rate/buy/review' },
  { what: 'Position re-checks', where: 'Postgres · position_reviews', when: 'every 30 min mkt hrs' },
  { what: 'Daily accuracy', where: 'Postgres · model_accuracy', when: 'EOD 15:45 + 20:00' },
  { what: 'Lessons learned', where: 'Postgres · learnings', when: 'EOD 15:45, weekend' },
  { what: 'Raw run output', where: 'logs/<date>/app.log', when: 'continuously' },
]

function Section({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return (
    <div className="bg-bg-card border border-border rounded-lg overflow-hidden">
      <div className="px-3 py-2 border-b border-border flex items-center gap-2">
        {icon}
        <span className="text-xs font-semibold text-white">{title}</span>
      </div>
      <div className="p-3">{children}</div>
    </div>
  )
}

export default function LogsPanel() {
  const [dataStatus, setDataStatus] = useState<DataStatus | null>(null)
  const [mode, setMode] = useState<TradingMode | null>(null)
  const [signals, setSignals] = useState<RecentSignal[]>([])
  const [decisions, setDecisions] = useState<Decision[]>([])
  const [activity, setActivity] = useState<ActivityEvent[]>([])
  const [summary, setSummary] = useState<AccuracySummary | null>(null)
  const [byModel, setByModel] = useState<ModelAccuracy[]>([])
  const [learnings, setLearnings] = useState<Learning[]>([])
  const [logFiles, setLogFiles] = useState<LogFiles | null>(null)
  const [loading, setLoading] = useState(false)

  const refresh = useCallback(async () => {
    setLoading(true)
    const [ds, md, sig, dec, act, sum, bm, learn, files] = await Promise.all([
      api.get('/live/data-status'),
      api.get('/live/mode'),
      api.get('/signals/recent?limit=20'),
      api.get('/live/decisions?limit=20'),
      api.get('/live/activity?limit=30'),
      api.get('/accuracy/summary'),
      api.get('/accuracy/by-model'),
      api.get('/logs/learnings?limit=20'),
      api.get('/logs/files?tail=200'),
    ])
    setDataStatus(ds && ds.source ? ds : null)
    setMode(md && md.mode ? md : null)
    setSignals(Array.isArray(sig) ? sig : [])
    setDecisions(Array.isArray(dec) ? dec : [])
    setActivity(Array.isArray(act) ? act : [])
    setSummary(sum && typeof sum.accuracy_pct === 'number' ? sum : null)
    setByModel(Array.isArray(bm) ? bm : [])
    setLearnings(Array.isArray(learn) ? learn : [])
    setLogFiles(files && Array.isArray(files.lines) ? files : null)
    setLoading(false)
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const isPaper = mode?.mode !== 'LIVE'

  return (
    <div className="h-full overflow-y-auto bg-bg-primary p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <span className="text-sm font-semibold text-white">Logs & Audit</span>
        <span className="text-xs text-text-secondary">Everything the system records — keep a check here</span>
        <button onClick={refresh} disabled={loading}
          className="ml-auto flex items-center gap-1.5 text-xs px-2.5 py-1.5 bg-bg-hover text-text-secondary rounded hover:text-text-primary disabled:opacity-50">
          <RefreshCw size={12} className={loading ? 'animate-spin' : ''} /> Refresh
        </button>
      </div>

      {/* 1. Data & Mode status */}
      <Section icon={<Database size={13} className="text-accent" />} title="Data & Mode Status">
        <div className="space-y-2">
          {dataStatus && (
            <div className={`flex items-center gap-2 text-xs ${dataStatus.is_live ? 'text-green' : 'text-yellow'}`}>
              {dataStatus.is_live ? '🟢' : <AlertTriangle size={13} />}
              {dataStatus.label}
              {dataStatus.age_hours != null && <span className="text-text-secondary">· {dataStatus.age_hours}h old</span>}
            </div>
          )}
          <div className="flex items-center gap-2 text-xs">
            {isPaper
              ? <span className="flex items-center gap-1.5 text-yellow"><FlaskConical size={13} /> PAPER MODE</span>
              : <span className="text-green">🟢 LIVE MODE</span>}
            {mode && <span className="text-text-secondary">{mode.reason}</span>}
          </div>
          {mode && (
            <div className="text-[11px] text-text-secondary">
              LIVE unlocks at ≥{mode.gate.min_days}d history, ≥{mode.gate.min_resolved} resolved signals, ≥{(mode.gate.min_accuracy * 100).toFixed(0)}% accuracy.
            </div>
          )}
        </div>
      </Section>

      <div className="grid grid-cols-2 gap-4">
        {/* 2. Recent signals */}
        <Section icon={<Activity size={13} className="text-green" />} title="Recent Signals">
          {signals.length === 0 ? <p className="text-[11px] text-text-secondary">No signals yet.</p> : (
            <div className="space-y-1 max-h-64 overflow-y-auto">
              {signals.map(s => (
                <div key={s.id} className="flex items-center gap-2 text-[11px] py-0.5 border-b border-border/30">
                  <span className="text-text-primary font-medium w-16">{s.ticker}</span>
                  <span className="text-text-secondary w-10">{s.timeframe}</span>
                  <span className={s.signal_type === 'BUY' ? 'text-green' : 'text-text-secondary'}>{s.signal_type}</span>
                  {s.final_confidence != null && <span className="text-text-secondary">{(s.final_confidence * 100).toFixed(0)}%</span>}
                  <span className="ml-auto text-text-secondary">{s.status}</span>
                </div>
              ))}
            </div>
          )}
        </Section>

        {/* 3. Decisions ledger */}
        <Section icon={<FileText size={13} className="text-blue-400" />} title="Decisions (paper ledger)">
          {decisions.length === 0 ? <p className="text-[11px] text-text-secondary">No decisions yet.</p> : (
            <div className="space-y-1 max-h-64 overflow-y-auto">
              {decisions.map(d => (
                <div key={d.id} className="flex items-center gap-2 text-[11px] py-0.5 border-b border-border/30">
                  <span className={d.action === 'BUY' ? 'text-green w-10' : 'text-text-secondary w-10'}>{d.action}</span>
                  <span className="text-text-primary font-medium w-16">{d.ticker}</span>
                  {d.rationale?.startsWith('[PAPER]') && <span className="text-yellow text-[10px]">PAPER</span>}
                  {d.price != null && <span className="text-text-secondary">₹{d.price.toFixed(1)}×{d.quantity}</span>}
                  <span className="ml-auto text-text-secondary">{d.outcome || 'open'}</span>
                </div>
              ))}
            </div>
          )}
        </Section>

        {/* 4. Activity feed */}
        <Section icon={<Activity size={13} className="text-accent" />} title="Activity Lifecycle">
          {activity.length === 0 ? <p className="text-[11px] text-text-secondary">No activity yet.</p> : (
            <div className="space-y-1 max-h-64 overflow-y-auto">
              {activity.map(e => (
                <div key={e.id} className="flex items-center gap-2 text-[11px] py-0.5 border-b border-border/30">
                  <span className="text-text-secondary w-20">{e.event_type}</span>
                  <span className="text-text-primary">{e.ticker}</span>
                  {e.note && <span className="text-text-secondary truncate">{e.note}</span>}
                </div>
              ))}
            </div>
          )}
        </Section>

        {/* 5. Accuracy */}
        <Section icon={<Brain size={13} className="text-yellow" />} title="Accuracy">
          {summary ? (
            <div className="space-y-2">
              <div className="flex items-center gap-3 text-xs">
                <span className="text-text-primary">{summary.accuracy_pct}% accurate</span>
                <span className="text-text-secondary">({summary.correct}/{summary.total_resolved} resolved)</span>
              </div>
              <div className="text-[11px] text-text-secondary">avg conf {(summary.avg_confidence * 100).toFixed(0)}% · {summary.today_total} fired today</div>
              {byModel.length > 0 && (
                <div className="space-y-0.5 max-h-32 overflow-y-auto pt-1 border-t border-border/30">
                  {byModel.slice(0, 8).map((m, i) => (
                    <div key={i} className="flex items-center gap-2 text-[11px]">
                      <span className="text-text-primary w-16">{m.model_name}</span>
                      <span className="text-text-secondary">{m.accuracy != null ? `${(m.accuracy * 100).toFixed(0)}%` : 'n/a'}</span>
                      <span className="text-text-secondary">{m.correct_signals}/{m.total_signals}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : <p className="text-[11px] text-text-secondary">No accuracy data yet (needs resolved signals).</p>}
        </Section>
      </div>

      {/* 6. Learnings */}
      <Section icon={<Brain size={13} className="text-green" />} title="Lessons Learned (self-learning loop)">
        {learnings.length === 0 ? <p className="text-[11px] text-text-secondary">No learnings yet — generated at EOD review (15:45 IST).</p> : (
          <div className="space-y-2 max-h-56 overflow-y-auto">
            {learnings.map(l => (
              <div key={l.id} className="text-[11px] border-b border-border/30 pb-1.5">
                <div className="flex items-center gap-2">
                  <span className="text-text-primary font-medium">{l.title}</span>
                  {l.ticker && <span className="text-text-secondary">{l.ticker}</span>}
                  <span className="ml-auto text-text-secondary">{l.learning_date}</span>
                </div>
                <p className="text-text-secondary mt-0.5">{l.body}</p>
              </div>
            ))}
          </div>
        )}
      </Section>

      {/* 7. Raw app.log tail */}
      <Section icon={<FileText size={13} className="text-text-secondary" />} title={`Raw app.log${logFiles ? ` · ${logFiles.day}` : ''}`}>
        {logFiles ? (
          <pre className="text-[10px] text-text-secondary font-mono max-h-64 overflow-y-auto whitespace-pre-wrap leading-4">
            {logFiles.lines.join('\n')}
          </pre>
        ) : <p className="text-[11px] text-text-secondary">No log file found.</p>}
      </Section>

      {/* Reference: where everything is logged */}
      <Section icon={<Database size={13} className="text-text-secondary" />} title="Where everything is logged">
        <table className="w-full text-[11px]">
          <thead>
            <tr className="text-text-secondary text-left">
              <th className="font-normal pb-1">What</th>
              <th className="font-normal pb-1">Where</th>
              <th className="font-normal pb-1">When</th>
            </tr>
          </thead>
          <tbody>
            {LOG_MAP.map((row, i) => (
              <tr key={i} className="border-t border-border/30">
                <td className="text-text-primary py-1">{row.what}</td>
                <td className="text-accent py-1 font-mono">{row.where}</td>
                <td className="text-text-secondary py-1">{row.when}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>
    </div>
  )
}
