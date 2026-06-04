export interface AccuracySummary {
  model_name: string
  signal_type: string
  timeframe: string
  total_signals: number
  correct_signals: number
  accuracy: number
  avg_confidence: number
  period_start: string
  period_end: string
}

const BASE = '/api'

export async function fetchAccuracySummary(): Promise<AccuracySummary[]> {
  const r = await fetch(`${BASE}/accuracy/summary`)
  if (!r.ok) return []
  const d = await r.json()
  return Array.isArray(d) ? d : []
}

export async function fetchModelAccuracy(model: string): Promise<AccuracySummary[]> {
  const r = await fetch(`${BASE}/accuracy/${model}`)
  if (!r.ok) return []
  const d = await r.json()
  return Array.isArray(d) ? d : []
}
