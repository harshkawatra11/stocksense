import { useState } from 'react'
import Layout from './components/Layout'
import IntelligenceDashboard from './components/intelligence/IntelligenceDashboard'
import Live from './components/live/Live'
import Portfolio from './components/portfolio/Portfolio'
import MarketOverview from './components/market/MarketOverview'
import Watchlist from './components/watchlist/Watchlist'
import Charts from './components/charts/Charts'
import LogsPanel from './components/logs/LogsPanel'
import Brain from './components/brain/Brain'
import type { Tab } from './types'

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('brain')

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
