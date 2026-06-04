import { useEffect, useState } from 'react'
import { fetchAccuracySummary, fetchModelAccuracy } from '../api/accuracy'
import type { AccuracySummary, ModelAccuracyRow } from '../api/accuracy'

export function useAccuracy() {
  const [summary, setSummary] = useState<AccuracySummary | null>(null)
  const [rows, setRows] = useState<ModelAccuracyRow[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    Promise.all([fetchAccuracySummary(), fetchModelAccuracy()])
      .then(([s, r]) => { setSummary(s); setRows(r) })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  return { summary, rows, loading }
}
