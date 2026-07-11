import type { ReactNode } from 'react'
import type { Tab } from '../types'
import { Briefcase, List, TrendingUp, Globe, Cpu, Zap, ScrollText, BrainCircuit } from 'lucide-react'
import { useMarketStatus } from '../hooks/useMarketStatus'
import { useBackendHealth } from '../hooks/useBackendHealth'
import { useMarketIndices } from '../hooks/useMarketIndices'
import { useRealtimePrices } from '../hooks/useRealtimePrices'
import PriceAgeDot from './PriceAgeDot'
import SystemHealthBar from './SystemHealthBar'

interface Props {
  activeTab: Tab
  onTabChange: (t: Tab) => void
  children: ReactNode
}

const tabs: { id: Tab; label: string; icon: ReactNode }[] = [
  { id: 'brain',        label: 'Brain',        icon: <BrainCircuit size={18} /> },
  { id: 'live',         label: 'Live Signals', icon: <Zap size={18} /> },
  { id: 'intelligence', label: 'Intelligence', icon: <Cpu size={18} /> },
  { id: 'portfolio',    label: 'Portfolio',    icon: <Briefcase size={18} /> },
  { id: 'charts',       label: 'Charts',       icon: <TrendingUp size={18} /> },
  { id: 'market',       label: 'Market',       icon: <Globe size={18} /> },
  { id: 'watchlist',    label: 'Watchlist',    icon: <List size={18} /> },
  { id: 'logs',         label: 'Logs & Audit', icon: <ScrollText size={18} /> },
]

function IndexTicker({ isMarketOpen }: { isMarketOpen: boolean }) {
  const { prices: livePrices, connected: wsConnected } = useRealtimePrices()
  const indices = useMarketIndices(isMarketOpen, livePrices)

  return (
    <div className="flex gap-5 text-xs">
      {indices.map((q) => {
        const ltpStr = q.ltp !== null ? q.ltp.toLocaleString('en-IN') : '—'
        const up = q.changePct !== null && q.changePct >= 0
        const changeStr =
          q.changePct !== null
            ? `${up ? '▲' : '▼'}${Math.abs(q.changePct).toFixed(2)}%`
            : null
        const changeColor = !q.available
          ? 'text-text-secondary'
          : up ? 'text-green' : 'text-red'

        return (
          <span key={q.symbol} className="text-text-secondary flex items-center gap-1">
            {q.live && <PriceAgeDot ts={q.ts} connected={wsConnected} />}
            {q.symbol}{' '}
            <span className={`font-medium ${changeColor}`}>
              {ltpStr}
              {changeStr && <span className="ml-1">{changeStr}</span>}
              {!q.available && <span className="ml-1 text-[10px] opacity-50">(offline)</span>}
            </span>
          </span>
        )
      })}
    </div>
  )
}

function MarketBadge() {
  const ms = useMarketStatus()

  const dotColor =
    ms.state === 'open'   ? 'bg-green animate-pulse' :
    ms.state === 'pre'    ? 'bg-yellow animate-pulse' :
    ms.state === 'post'   ? 'bg-yellow' :
    'bg-red'

  const labelColor =
    ms.state === 'open'  ? 'text-green' :
    ms.state === 'pre'   ? 'text-yellow' :
    ms.state === 'post'  ? 'text-yellow' :
    'text-red'

  return (
    <div className="flex items-center gap-4">
      {/* Market status pill */}
      <div className="flex items-center gap-1.5">
        <span className={`w-2 h-2 rounded-full flex-shrink-0 ${dotColor}`} />
        <span className={`text-xs font-semibold tracking-wide ${labelColor}`}>{ms.label}</span>
        {ms.state === 'open' && ms.minutesToClose !== null && (
          <span className="text-xs text-text-secondary ml-1">
            ({ms.minutesToClose}m left)
          </span>
        )}
        {ms.state === 'pre' && ms.minutesToOpen !== null && (
          <span className="text-xs text-text-secondary ml-1">
            (opens in {ms.minutesToOpen}m)
          </span>
        )}
      </div>

      {/* Live clock */}
      <div className="flex flex-col items-end leading-tight">
        <span className="text-xs font-mono text-text-primary tabular-nums">
          {ms.timeStr} <span className="text-text-secondary">IST</span>
        </span>
        <span className="text-[10px] text-text-secondary">{ms.dateStr}</span>
      </div>
    </div>
  )
}

function BackendIndicator() {
  const state = useBackendHealth(10_000)

  const dotColor =
    state === 'connected'    ? 'bg-green' :
    state === 'checking'     ? 'bg-yellow animate-pulse' :
    'bg-red animate-pulse'

  const label =
    state === 'connected'    ? 'Backend' :
    state === 'checking'     ? 'Connecting' :
    'Offline'

  const labelColor =
    state === 'connected'    ? 'text-text-secondary' :
    state === 'checking'     ? 'text-yellow' :
    'text-red'

  return (
    <div className="flex items-center gap-1.5">
      <span className={`w-2 h-2 rounded-full flex-shrink-0 ${dotColor}`} />
      <span className={`text-xs ${labelColor}`}>{label}</span>
    </div>
  )
}

export default function Layout({ activeTab, onTabChange, children }: Props) {
  const ms = useMarketStatus()
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-bg-primary">
      {/* Sidebar */}
      <aside className="w-52 flex-shrink-0 bg-bg-card border-r border-border flex flex-col">
        <div className="px-4 py-4 border-b border-border">
          <span className="text-lg font-bold text-white tracking-wide">
            Stock<span className="text-green">Sense</span>
          </span>
        </div>

        <nav className="flex-1 py-3">
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => onTabChange(t.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 text-sm transition-colors ${
                activeTab === t.id
                  ? 'bg-bg-hover text-white border-l-2 border-accent'
                  : 'text-text-secondary hover:bg-bg-hover hover:text-text-primary'
              }`}
            >
              {t.icon}
              {t.label}
              {t.id === 'intelligence' && (
                <span className="ml-auto w-2 h-2 rounded-full bg-green animate-pulse" />
              )}
            </button>
          ))}
        </nav>

        <div className="px-4 py-3 border-t border-border">
          <p className="text-xs text-text-secondary">NSE Trading Intelligence</p>
          <p className="text-xs text-text-secondary mt-1">v0.1 · Local</p>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Top bar */}
        <header className="h-12 border-b border-border bg-bg-card flex items-center px-4 gap-6 flex-shrink-0">
          {/* Left: live index prices via Angel One */}
          <IndexTicker isMarketOpen={ms.state === 'open'} />

          {/* Divider */}
          <div className="w-px h-5 bg-border" />

          {/* Backend health */}
          <BackendIndicator />

          {/* Divider */}
          <div className="w-px h-5 bg-border" />

          {/* Stage 0 truth layer — per-component pipeline health */}
          <SystemHealthBar />

          {/* Right: market status + clock */}
          <div className="ml-auto">
            <MarketBadge />
          </div>
        </header>

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          {children}
        </div>
      </main>
    </div>
  )
}
