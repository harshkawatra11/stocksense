interface Props {
  totalPnl: number
  totalPct: number
  dayPnl: number
}

export default function PnLCard({ totalPnl, totalPct, dayPnl }: Props) {
  const totalColor = totalPnl >= 0 ? 'text-green' : 'text-red'
  const dayColor   = dayPnl   >= 0 ? 'text-green' : 'text-red'

  return (
    <div className="flex gap-4">
      <div className="bg-bg-card rounded p-4 border border-border flex-1">
        <p className="text-xs text-text-secondary mb-1">Total P&L</p>
        <p className={`text-xl font-bold ${totalColor}`}>
          {totalPnl >= 0 ? '+' : ''}₹{totalPnl.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
        </p>
        <p className={`text-xs ${totalColor} mt-1`}>{totalPct >= 0 ? '+' : ''}{totalPct.toFixed(2)}%</p>
      </div>
      <div className="bg-bg-card rounded p-4 border border-border flex-1">
        <p className="text-xs text-text-secondary mb-1">Today's P&L</p>
        <p className={`text-xl font-bold ${dayColor}`}>
          {dayPnl >= 0 ? '+' : ''}₹{dayPnl.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
        </p>
        <p className="text-xs text-text-secondary mt-1">Mark-to-market</p>
      </div>
    </div>
  )
}
