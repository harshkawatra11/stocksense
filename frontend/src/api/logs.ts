import type { Learning } from '../types'

const BASE = '/api'

export async function fetchLearningsToday(): Promise<Learning[]> {
  const r = await fetch(`${BASE}/logs/learnings/today`)
  if (!r.ok) return []
  const d = await r.json()
  return Array.isArray(d) ? d : []
}

export async function fetchLearnings(limit = 100): Promise<Learning[]> {
  const r = await fetch(`${BASE}/logs/learnings?limit=${limit}`)
  if (!r.ok) return []
  const d = await r.json()
  return Array.isArray(d) ? d : []
}
