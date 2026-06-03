import type { ReactNode } from 'react'
import type { Tab } from '../types'
import { BarChart2, Briefcase, List, TrendingUp, Globe, Cpu } from 'lucide-react'

interface Props {
  activeTab: Tab
  onTabChange: (t: Tab) => void
  children: ReactNode
}

const tabs: { id: Tab; label: string; icon: ReactNode }[] = [
  { id: 'watchlist',     label: 'Watchlist',     icon: <List size={18} /> },
  { id: 'portfolio',     label: 'Portfolio',     icon: <Briefcase size={18} /> },
  { id: 'orders',        label: 'Orders',        icon: <BarChart2 size={18} /> },
  { id: 'charts',        label: 'Charts',        icon: <TrendingUp size={18} /> },
  { id: 'market',        label: 'Market',        icon: <Globe size={18} /> },
  { id: 'intelligence',  label: 'Intelligence',  icon: <Cpu size={18} /> },
]

export default function Layout({ activeTab, onTabChange, children }: Props) {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-bg-primary">
      {/* Sidebar */}
      <aside className="w-52 flex-shrink-0 bg-bg-card border-r border-border flex flex-col">
        {/* Logo */}
        <div className="px-4 py-4 border-b border-border">
          <span className="text-lg font-bold text-white tracking-wide">Stock<span className="text-green">Sense</span></span>
        </div>

        {/* Nav */}
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

        {/* Footer */}
        <div className="px-4 py-3 border-t border-border">
          <p className="text-xs text-text-secondary">NSE Trading Intelligence</p>
          <p className="text-xs text-text-secondary mt-1">v0.1 · Local</p>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Top bar */}
        <header className="h-11 border-b border-border bg-bg-card flex items-center px-4 gap-6 flex-shrink-0">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-green" />
            <span className="text-xs text-text-secondary">NSE OPEN</span>
          </div>
          <div className="flex gap-4 text-xs">
            <span className="text-text-secondary">Nifty 50 <span className="text-green">24,521 ▲0.4%</span></span>
            <span className="text-text-secondary">Sensex <span className="text-green">80,431 ▲0.3%</span></span>
          </div>
          <div className="ml-auto text-xs text-text-secondary">
            {new Date().toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata' })} IST
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
