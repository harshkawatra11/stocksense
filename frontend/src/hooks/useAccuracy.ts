import { useEffect, useState } from 'react'
import { fetchAccuracySummary, fetchModelAccuracy, fetchNetExpectancy } from '../api/accuracy'
import type { AccuracySummary, ModelAccuracyRow, NetExpectancy } from '../api/accuracy'

export function useAccuracy() {
  const [summary, setSummary] = useState<AccuracySummary | null>(null)
  const [rows, setRows] = useState<ModelAccuracyRow[]>([])
  const [netExpectancy, setNetExpectancy] = useState<NetExpectancy | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    Promise.all([fetchAccuracySummary(), fetchModelAccuracy(), fetchNetExpectancy()])
      .then(([s, r, e]) => { setSummary(s); setRows(r); setNetExpectancy(e) })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  return { summary, rows, netExpectancy, loading }
}
