import FlashPrice from '../FlashPrice'

interface Props {
  totalPnl: number
  totalPct: number
  dayPnl: number
  /** When false, values are labeled "as of 15:30 close" instead of implying freshness. */
  live?: boolean
}

export default function PnLCard({ totalPnl, totalPct, dayPnl, live = true }: Props) {
  const totalColor = totalPnl >= 0 ? 'text-green' : 'text-red'
  const dayColor   = dayPnl   >= 0 ? 'text-green' : 'text-red'

  return (
    <div className="flex gap-4">
      <div className="bg-bg-card rounded p-4 border border-border flex-1">
        <p className="text-xs text-text-secondary mb-1">
          Total P&L
          {!live && <span className="ml-1.5 text-[10px] opacity-70">as of 15:30 close</span>}
        </p>
        <p className={`text-xl font-bold ${totalColor}`}>
          <FlashPrice value={Math.round(totalPnl)}>
            {totalPnl >= 0 ? '+' : ''}₹{totalPnl.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
          </FlashPrice>
        </p>
        <p className={`text-xs ${totalColor} mt-1`}>{totalPct >= 0 ? '+' : ''}{totalPct.toFixed(2)}%</p>
      </div>
      <div className="bg-bg-card rounded p-4 border border-border flex-1">
        <p className="text-xs text-text-secondary mb-1">Today's P&L</p>
        <p className={`text-xl font-bold ${dayColor}`}>
          <FlashPrice value={Math.round(dayPnl)}>
            {dayPnl >= 0 ? '+' : ''}₹{dayPnl.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
          </FlashPrice>
        </p>
        <p className="text-xs text-text-secondary mt-1">Mark-to-market</p>
      </div>
    </div>
  )
}
