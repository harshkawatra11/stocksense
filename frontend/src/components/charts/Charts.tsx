import { useState } from 'react'

const POPULAR = ['RELIANCE', 'HDFCBANK', 'INFY', 'TCS', 'ICICIBANK', 'WIPRO', 'AXISBANK', 'SBIN']

export default function Charts() {
  const [ticker, setTicker] = useState('RELIANCE')
  const [input, setInput] = useState('')

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div className="px-4 py-3 border-b border-border bg-bg-card flex items-center gap-3 flex-shrink-0">
        <span className="text-sm font-medium text-white">Charts</span>
        <div className="flex gap-1 ml-4">
          {POPULAR.map(t => (
            <button key={t} onClick={() => setTicker(t)}
              className={`text-xs px-2 py-1 rounded ${ticker === t ? 'bg-accent text-white' : 'bg-bg-hover text-text-secondary hover:text-text-primary'}`}>
              {t}
            </button>
          ))}
        </div>
        <form onSubmit={e => { e.preventDefault(); if (input.trim()) { setTicker(input.trim().toUpperCase()); setInput('') } }}
          className="ml-auto flex gap-2">
          <input value={input} onChange={e => setInput(e.target.value)} placeholder="Type ticker..."
            className="bg-bg-hover text-text-primary text-xs px-3 py-1 rounded border border-border outline-none w-32" />
          <button type="submit" className="text-xs px-3 py-1 bg-accent text-white rounded">Go</button>
        </form>
      </div>

      <div className="flex-1 flex flex-col items-center justify-center text-text-secondary gap-3">
        <div className="text-4xl">📈</div>
        <p className="text-sm font-medium text-text-primary">{ticker}</p>
        <p className="text-xs text-center max-w-sm">
          TradingView Lightweight Charts integration — Phase 2.<br />
          Chart will render here using historical OHLCV data from TimescaleDB.
        </p>
        <p className="text-xs text-text-secondary mt-2">
          Run: <code className="bg-bg-hover px-2 py-0.5 rounded">npm install lightweight-charts</code> and wire chart component
        </p>
      </div>
    </div>
  )
}
