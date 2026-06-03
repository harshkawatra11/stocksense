import { useEffect, useState } from 'react'
import type { Learning } from '../../../types'

type Period = 'today' | 'all'

export default function LearningsView() {
  const [period, setPeriod] = useState<Period>('today')
  const [learnings, setLearnings] = useState<Learning[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    const url = period === 'today' ? '/api/logs/learnings/today' : '/api/logs/learnings?limit=100'
    fetch(url)
      .then(r => r.json())
      .then(d => setLearnings(Array.isArray(d) ? d : []))
      .catch(() => setLearnings([]))
      .finally(() => setLoading(false))
  }, [period])

  return (
    <div className="p-4">
      <div className="flex gap-2 mb-4">
        {(['today', 'all'] as Period[]).map(p => (
          <button key={p} onClick={() => setPeriod(p)}
            className={`text-xs px-3 py-1 rounded ${period === p ? 'bg-accent text-white' : 'bg-bg-card text-text-secondary hover:text-text-primary'}`}>
            {p === 'today' ? 'Today' : 'All Time'}
          </button>
        ))}
      </div>

      {loading && <p className="text-xs text-text-secondary">Loading...</p>}
      {!loading && learnings.length === 0 && (
        <p className="text-xs text-text-secondary">No learnings yet. EOD review will populate this after market close.</p>
      )}

      <div className="space-y-3">
        {learnings.map(l => (
          <div key={l.id} className="bg-bg-card rounded p-4 border border-border">
            <div className="flex items-start justify-between mb-2">
              <span className="text-sm font-medium text-white">{l.title}</span>
              {l.ticker && <span className="text-xs text-accent ml-2 flex-shrink-0">{l.ticker}</span>}
            </div>
            <p className="text-xs text-text-secondary leading-5">{l.body}</p>
            <div className="flex gap-2 mt-2 flex-wrap">
              {l.tags?.map(tag => (
                <span key={tag} className="text-xs px-2 py-0.5 rounded bg-bg-hover text-text-secondary">{tag}</span>
              ))}
            </div>
            <p className="text-xs text-text-secondary mt-2">{new Date(l.learning_date).toLocaleDateString('en-IN')}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
