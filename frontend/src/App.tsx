import { useEffect, useState } from 'react'
import Layout from './components/Layout'
import IntelligenceDashboard from './components/intelligence/IntelligenceDashboard'
import Live from './components/live/Live'
import Portfolio from './components/portfolio/Portfolio'
import MarketOverview from './components/market/MarketOverview'
import Watchlist from './components/watchlist/Watchlist'
import Charts from './components/charts/Charts'
import LogsPanel from './components/logs/LogsPanel'
import Brain from './components/brain/Brain'
import ProviderSetup from './components/setup/ProviderSetup'
import type { Tab } from './types'

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('brain')
  // null = still checking; true/false = whether engines are configured
  const [configured, setConfigured] = useState<boolean | null>(null)

  const checkConfig = () =>
    fetch('/api/providers/status')
      .then(r => r.json())
      .then(s => setConfigured(!!s.is_configured))
      .catch(() => setConfigured(true)) // don't lock the app out if the check fails

  useEffect(() => { checkConfig() }, [])

  if (configured === null) {
    return <div className="h-screen bg-bg-primary" />
  }

  if (!configured) {
    return <ProviderSetup onConfigured={() => setConfigured(true)} />
  }

  return (
    <Layout activeTab={activeTab} onTabChange={setActiveTab}>
      {activeTab === 'intelligence' && <IntelligenceDashboard />}
      {activeTab === 'live' && <Live />}
      {activeTab === 'portfolio' && <Portfolio />}
      {activeTab === 'market' && <MarketOverview />}
      {activeTab === 'watchlist' && <Watchlist />}
      {activeTab === 'charts' && <Charts />}
      {activeTab === 'logs' && <LogsPanel />}
      {activeTab === 'brain' && <Brain />}
    </Layout>
  )
}
