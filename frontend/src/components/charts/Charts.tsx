import { useEffect, useRef, useState, useCallback } from 'react'
import { createChart, ColorType, CandlestickSeries, HistogramSeries } from 'lightweight-charts'
import { Search, TrendingUp, TrendingDown } from 'lucide-react'

const POPULAR = ['RELIANCE', 'HDFCBANK', 'INFY', 'TCS', 'ICICIBANK', 'WIPRO', 'AXISBANK', 'SBIN', 'TATAMOTORS', 'ADANIENT']

const PERIODS = [
  { label: '1M', days: 30 },
  { label: '3M', days: 90 },
  { label: '6M', days: 180 },
  { label: '1Y', days: 365 },
  { label: '2Y', days: 730 },
  { label: '5Y', days: 1825 },
]

interface Candle { time: string; open: number; high: number; low: number; close: number; volume: number }
interface SignalMark { date: string; signal_type: string; price_at_signal: number; final_confidence: number }

export default function Charts() {
  const [ticker, setTicker] = useState('RELIANCE')
  const [inputVal, setInputVal] = useState('')
  const [period, setPeriod] = useState(365)
  const [candles, setCandles] = useState<Candle[]>([])
  const [signals, setSignals] = useState<SignalMark[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [latestClose, setLatestClose] = useState<number | null>(null)
  const [priceChange, setPriceChange] = useState<{ abs: number; pct: number } | null>(null)

  const chartRef = useRef<HTMLDivElement>(null)
  const volRef = useRef<HTMLDivElement>(null)

  const loadData = useCallback(async (t: string, days: number) => {
    setLoading(true)
    setError(null)
    try {
      const [candleRes, sigRes] = await Promise.all([
        fetch(`/api/ohlcv/${t}?days=${days}`),
        fetch(`/api/ohlcv/${t}/signals`),
      ])
      if (!candleRes.ok) {
        setError(`No data found for ${t}`)
        setCandles([])
        setLoading(false)
        return
      }
      const candleData = await candleRes.json()
      const sigData = sigRes.ok ? await sigRes.json() : []
      setCandles(candleData.candles || [])
      setSignals(Array.isArray(sigData) ? sigData : [])

      const cs: Candle[] = candleData.candles || []
      if (cs.length >= 2) {
        const last = cs[cs.length - 1]
        const prev = cs[cs.length - 2]
        setLatestClose(last.close)
        setPriceChange({ abs: last.close - prev.close, pct: (last.close - prev.close) / prev.close * 100 })
      }
    } catch {
      setError('Failed to load chart data')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadData(ticker, period) }, [ticker, period, loadData])

  // Build chart
  useEffect(() => {
    if (!chartRef.current || !volRef.current || candles.length === 0) return

    const chartEl = chartRef.current
    const volEl = volRef.current

    // Price chart
    const chart = createChart(chartEl, {
      layout: {
        background: { type: ColorType.Solid, color: '#0d0f14' },
        textColor: '#9aa0ab',
      },
      grid: {
        vertLines: { color: '#2a2d37' },
        horzLines: { color: '#2a2d37' },
      },
      crosshair: { mode: 1 },
      rightPriceScale: { borderColor: '#2a2d37' },
      timeScale: { borderColor: '#2a2d37', timeVisible: true },
      width: chartEl.clientWidth,
      height: chartEl.clientHeight,
    })

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#00d4aa',
      downColor: '#ff4757',
      borderUpColor: '#00d4aa',
      borderDownColor: '#ff4757',
      wickUpColor: '#00d4aa',
      wickDownColor: '#ff4757',
    })

    candleSeries.setData(
      candles.map(c => ({ time: c.time as unknown as import('lightweight-charts').Time, open: c.open, high: c.high, low: c.low, close: c.close }))
    )

    // Add price lines for signals (v5 compatible — no setMarkers)
    signals
      .filter(s => s.date && s.signal_type !== 'HOLD' && s.price_at_signal)
      .forEach(s => {
        candleSeries.createPriceLine({
          price: s.price_at_signal,
          color: s.signal_type === 'BUY' ? '#00d4aa' : '#ff4757',
          lineWidth: 1,
          lineStyle: 2,
          axisLabelVisible: true,
          title: `${s.signal_type} ${((s.final_confidence || 0) * 100).toFixed(0)}%`,
        })
      })

    chart.timeScale().fitContent()

    // Volume chart
    const volChart = createChart(volEl, {
      layout: {
        background: { type: ColorType.Solid, color: '#0d0f14' },
        textColor: '#9aa0ab',
      },
      grid: {
        vertLines: { color: '#2a2d37' },
        horzLines: { color: 'transparent' },
      },
      rightPriceScale: { borderColor: '#2a2d37', scaleMargins: { top: 0.1, bottom: 0 } },
      timeScale: { borderColor: '#2a2d37', visible: false },
      width: volEl.clientWidth,
      height: volEl.clientHeight,
    })

    const volSeries = volChart.addSeries(HistogramSeries, {
      color: '#3d85c8',
      priceFormat: { type: 'volume' },
      priceScaleId: '',
    })

    volSeries.setData(
      candles.map(c => ({
        time: c.time as unknown as import('lightweight-charts').Time,
        value: c.volume,
        color: c.close >= c.open ? '#00d4aa33' : '#ff475733',
      }))
    )

    volChart.timeScale().fitContent()

    // Sync time scales
    chart.timeScale().subscribeVisibleTimeRangeChange(() => {
      const range = chart.timeScale().getVisibleRange()
      if (range) volChart.timeScale().setVisibleRange(range)
    })

    // Resize
    const ro = new ResizeObserver(() => {
      chart.applyOptions({ width: chartEl.clientWidth })
      volChart.applyOptions({ width: volEl.clientWidth })
    })
    ro.observe(chartEl)

    return () => {
      ro.disconnect()
      chart.remove()
      volChart.remove()
    }
  }, [candles, signals])

  const up = priceChange ? priceChange.pct >= 0 : true

  return (
    <div className="h-full flex flex-col overflow-hidden bg-bg-primary">
      {/* Header */}
      <div className="px-4 py-2 border-b border-border bg-bg-card flex-shrink-0 flex items-center gap-3 flex-wrap">
        {/* Ticker info */}
        <div className="flex items-center gap-3">
          <span className="text-sm font-semibold text-white">{ticker}</span>
          {latestClose !== null && (
            <>
              <span className="text-sm text-text-primary">
                ₹{latestClose.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
              {priceChange && (
                <span className={`text-xs flex items-center gap-1 ${up ? 'text-green' : 'text-red'}`}>
                  {up ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                  {up ? '+' : ''}{priceChange.abs.toFixed(2)} ({up ? '+' : ''}{priceChange.pct.toFixed(2)}%)
                </span>
              )}
            </>
          )}
        </div>

        {/* Quick tickers */}
        <div className="flex gap-1 flex-wrap">
          {POPULAR.map(t => (
            <button key={t} onClick={() => setTicker(t)}
              className={`text-xs px-2 py-0.5 rounded ${ticker === t ? 'bg-accent text-white' : 'bg-bg-hover text-text-secondary hover:text-text-primary'}`}>
              {t}
            </button>
          ))}
        </div>

        {/* Period selector */}
        <div className="flex gap-1">
          {PERIODS.map(p => (
            <button key={p.label} onClick={() => setPeriod(p.days)}
              className={`text-xs px-2 py-0.5 rounded ${period === p.days ? 'bg-accent text-white' : 'bg-bg-hover text-text-secondary hover:text-text-primary'}`}>
              {p.label}
            </button>
          ))}
        </div>

        {/* Custom ticker search */}
        <form onSubmit={e => { e.preventDefault(); const v = inputVal.trim().toUpperCase(); if (v) { setTicker(v); setInputVal('') } }}
          className="ml-auto flex gap-2">
          <div className="relative">
            <Search size={12} className="absolute left-2 top-1/2 -translate-y-1/2 text-text-secondary" />
            <input value={inputVal} onChange={e => setInputVal(e.target.value)} placeholder="Search ticker..."
              className="bg-bg-hover text-text-primary text-xs pl-6 pr-3 py-1.5 rounded border border-border outline-none w-36 focus:border-accent" />
          </div>
          <button type="submit" className="text-xs px-3 py-1.5 bg-accent text-white rounded">Go</button>
        </form>
      </div>

      {/* Chart area */}
      <div className="flex-1 flex flex-col overflow-hidden relative">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-bg-primary/80 z-10">
            <div className="text-xs text-text-secondary animate-pulse">Loading chart data...</div>
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center z-10">
            <div className="text-xs text-red">{error}</div>
          </div>
        )}

        {/* Price candles — 80% height */}
        <div ref={chartRef} className="flex-1" />

        {/* Volume — 20% height */}
        <div ref={volRef} className="h-24 border-t border-border flex-shrink-0" />
      </div>

      {/* Signal overlay legend */}
      {signals.length > 0 && (
        <div className="px-4 py-1.5 border-t border-border bg-bg-card flex-shrink-0 flex items-center gap-4">
          <span className="text-xs text-text-secondary">Signals on chart:</span>
          <span className="text-xs text-green">▲ BUY ({signals.filter(s => s.signal_type === 'BUY').length})</span>
          <span className="text-xs text-red">▼ SELL ({signals.filter(s => s.signal_type === 'SELL').length})</span>
        </div>
      )}
    </div>
  )
}
