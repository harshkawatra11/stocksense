import { useEffect, useState } from 'react'
import { fetchAccuracySummary } from '../api/accuracy'
import type { AccuracySummary } from '../api/accuracy'

export function useAccuracy() {
  const [rows, setRows] = useState<AccuracySummary[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    fetchAccuracySummary()
      .then(setRows)
      .catch(() => setRows([]))
      .finally(() => setLoading(false))
  }, [])

  return { rows, loading }
}
