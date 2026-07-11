const BASE = '/api'

export interface AccuracySummary {
  total_resolved: number
  correct: number
  accuracy_pct: number
  total_buys: number
  total_sells: number
  avg_confidence: number
  today_total: number
}

export interface ModelAccuracyRow {
  model_name: string
  period_start: string
  period_end: string
  total_signals: number
  correct_signals: number
  accuracy: number
  avg_confidence: number
}

export interface NetExpectancy {
  n: number
  win_rate: number | null
  net_win_rate: number | null
  net_expectancy_bps: number | null
  gross_expectancy_bps: number | null
}

export async function fetchAccuracySummary(): Promise<AccuracySummary | null> {
  const r = await fetch(`${BASE}/accuracy/summary`)
  if (!r.ok) return null
  return r.json()
}

export async function fetchModelAccuracy(): Promise<ModelAccuracyRow[]> {
  const r = await fetch(`${BASE}/accuracy/by-model`)
  if (!r.ok) return []
  const d = await r.json()
  return Array.isArray(d) ? d : []
}

export async function fetchDailyAccuracy(): Promise<{ date: string; total: number; correct: number; accuracy_pct: number }[]> {
  const r = await fetch(`${BASE}/accuracy/daily`)
  if (!r.ok) return []
  const d = await r.json()
  return Array.isArray(d) ? d : []
}

export async function fetchNetExpectancy(n = 60): Promise<NetExpectancy | null> {
  const r = await fetch(`${BASE}/accuracy/net-expectancy?n=${n}`)
  if (!r.ok) return null
  return r.json()
}
