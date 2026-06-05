import { useState, useEffect } from 'react'

// NSE trading hours (IST)
const MARKET_OPEN_H = 9
const MARKET_OPEN_M = 15
const MARKET_CLOSE_H = 15
const MARKET_CLOSE_M = 30

// NSE holidays 2025–2026 (official NSE holiday calendar)
// Format: 'YYYY-MM-DD'
const NSE_HOLIDAYS = new Set([
  // 2025
  '2025-01-26', // Republic Day
  '2025-02-26', // Mahashivratri
  '2025-03-14', // Holi
  '2025-04-10', // Shri Ram Navami
  '2025-04-14', // Dr. Baba Saheb Ambedkar Jayanti
  '2025-04-18', // Good Friday
  '2025-05-01', // Maharashtra Day
  '2025-08-15', // Independence Day
  '2025-08-27', // Ganesh Chaturthi
  '2025-10-02', // Gandhi Jayanti / Dussehra
  '2025-10-20', // Diwali Laxmi Pujan (Muhurat Trading)
  '2025-10-21', // Diwali Balipratipada
  '2025-11-05', // Prakash Gurpurb Sri Guru Nanak Dev Ji
  '2025-11-26', // Constitution Day (exchange holiday)
  '2025-12-25', // Christmas
  // 2026
  '2026-01-26', // Republic Day
  '2026-03-19', // Holi
  '2026-04-02', // Shri Ram Navami (tentative)
  '2026-04-03', // Good Friday
  '2026-04-14', // Dr. Baba Saheb Ambedkar Jayanti
  '2026-05-01', // Maharashtra Day
  '2026-08-15', // Independence Day
  '2026-10-02', // Gandhi Jayanti
  '2026-11-14', // Diwali Laxmi Pujan (tentative)
  '2026-12-25', // Christmas
])

export type MarketState = 'open' | 'closed' | 'pre' | 'post'

export interface MarketStatus {
  now: Date           // current IST time (as Date object, updated every second)
  dateStr: string     // 'Thu, 05 Jun 2026'
  timeStr: string     // '15:27:43'
  state: MarketState  // open | closed | pre | post
  label: string       // human-readable status
  isHoliday: boolean
  holidayName: string | null
  minutesToOpen: number | null   // if pre-market
  minutesToClose: number | null  // if open
}

function getISTDate(date: Date): { y: number; m: number; d: number; h: number; min: number; sec: number; day: number } {
  const ist = new Date(date.toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }))
  return {
    y: ist.getFullYear(),
    m: ist.getMonth() + 1,
    d: ist.getDate(),
    h: ist.getHours(),
    min: ist.getMinutes(),
    sec: ist.getSeconds(),
    day: ist.getDay(), // 0=Sun, 6=Sat
  }
}

function toISODate(y: number, m: number, d: number): string {
  return `${y}-${String(m).padStart(2, '0')}-${String(d).padStart(2, '0')}`
}

function computeStatus(now: Date): MarketStatus {
  const { y, m, d, h, min, sec, day } = getISTDate(now)
  const dateKey = toISODate(y, m, d)
  const isWeekend = day === 0 || day === 6
  const isHoliday = NSE_HOLIDAYS.has(dateKey)

  const dateStr = now.toLocaleDateString('en-IN', {
    timeZone: 'Asia/Kolkata',
    weekday: 'short',
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  })

  const timeStr = `${String(h).padStart(2, '0')}:${String(min).padStart(2, '0')}:${String(sec).padStart(2, '0')}`

  const totalMin = h * 60 + min
  const openMin = MARKET_OPEN_H * 60 + MARKET_OPEN_M   // 555
  const closeMin = MARKET_CLOSE_H * 60 + MARKET_CLOSE_M // 930

  let state: MarketState
  let label: string
  let minutesToOpen: number | null = null
  let minutesToClose: number | null = null

  if (isWeekend || isHoliday) {
    state = 'closed'
    label = isHoliday ? 'NSE HOLIDAY' : 'WEEKEND'
  } else if (totalMin >= 9 * 60 && totalMin < openMin) {
    // 9:00–9:15 pre-open session
    state = 'pre'
    label = 'PRE-OPEN'
    minutesToOpen = openMin - totalMin
  } else if (totalMin >= openMin && totalMin < closeMin) {
    state = 'open'
    label = 'NSE OPEN'
    minutesToClose = closeMin - totalMin
  } else if (totalMin >= closeMin && totalMin < closeMin + 30) {
    state = 'post'
    label = 'POST-CLOSE'
  } else {
    state = 'closed'
    label = 'NSE CLOSED'
    // minutes to next open (tomorrow or next trading day — approximate)
    minutesToOpen = null
  }

  return {
    now,
    dateStr,
    timeStr,
    state,
    label,
    isHoliday,
    holidayName: null,
    minutesToOpen,
    minutesToClose,
  }
}

export function useMarketStatus(): MarketStatus {
  const [status, setStatus] = useState<MarketStatus>(() => computeStatus(new Date()))

  useEffect(() => {
    const id = setInterval(() => setStatus(computeStatus(new Date())), 1000)
    return () => clearInterval(id)
  }, [])

  return status
}
