import type { Signal } from '../types'

const BASE = '/api'

export async function fetchTodaySignals(): Promise<Signal[]> {
  const r = await fetch(`${BASE}/signals/today`)
  if (!r.ok) throw new Error('Failed to fetch signals')
  const d = await r.json()
  return Array.isArray(d) ? d : []
}

export async function fetchRecentSignals(limit = 50): Promise<Signal[]> {
  const r = await fetch(`${BASE}/signals/recent?limit=${limit}`)
  if (!r.ok) throw new Error('Failed to fetch recent signals')
  const d = await r.json()
  return Array.isArray(d) ? d : []
}

export async function fetchSignalReasoning(id: number): Promise<Record<string, string>> {
  const r = await fetch(`${BASE}/signals/${id}/reasoning`)
  if (!r.ok) throw new Error('Failed to fetch reasoning')
  return r.json()
}
