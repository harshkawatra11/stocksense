import { useState } from 'react'
import Layout from './components/Layout'
import IntelligenceDashboard from './components/intelligence/IntelligenceDashboard'
import Portfolio from './components/portfolio/Portfolio'
import MarketOverview from './components/market/MarketOverview'
import type { Tab } from './types'

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('intelligence')

  return (
    <Layout activeTab={activeTab} onTabChange={setActiveTab}>
      {activeTab === 'intelligence' && <IntelligenceDashboard />}
      {activeTab === 'portfolio' && <Portfolio />}
      {activeTab === 'market' && <MarketOverview />}
      {activeTab === 'watchlist' && (
        <div className="flex items-center justify-center h-full text-text-secondary">
          Watchlist — connect Angel One SmartAPI in Phase 2
        </div>
      )}
      {activeTab === 'orders' && (
        <div className="flex items-center justify-center h-full text-text-secondary">
          Orders — connect Angel One SmartAPI in Phase 2
        </div>
      )}
      {activeTab === 'charts' && (
        <div className="flex items-center justify-center h-full text-text-secondary">
          Charts — TradingView widget in Phase 2
        </div>
      )}
    </Layout>
  )
}
