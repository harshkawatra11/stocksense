import type { Learning } from '../../../types'

interface Props { learning: Learning }

const TYPE_COLOR: Record<string, string> = {
  eod_review: 'text-accent',
  weekly_review: 'text-yellow',
  signal_outcome: 'text-green',
}

export default function LearningEntry({ learning: l }: Props) {
  const typeColor = TYPE_COLOR[l.learning_type] || 'text-text-secondary'

  return (
    <div className="bg-bg-card rounded p-4 border border-border">
      <div className="flex items-start justify-between mb-1.5">
        <span className="text-sm font-medium text-white leading-tight">{l.title}</span>
        <div className="flex items-center gap-2 ml-3 flex-shrink-0">
          {l.ticker && <span className="text-xs text-accent">{l.ticker}</span>}
          <span className={`text-xs ${typeColor}`}>{l.learning_type}</span>
        </div>
      </div>
      <p className="text-xs text-text-secondary leading-5">{l.body}</p>
      <div className="flex items-center gap-2 mt-2 flex-wrap">
        {l.tags?.map(tag => (
          <span key={tag} className="text-xs px-2 py-0.5 rounded bg-bg-hover text-text-secondary">{tag}</span>
        ))}
        <span className="ml-auto text-xs text-text-secondary">
          {new Date(l.learning_date).toLocaleDateString('en-IN')}
        </span>
      </div>
    </div>
  )
}
