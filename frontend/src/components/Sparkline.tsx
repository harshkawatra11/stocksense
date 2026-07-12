interface Props {
  /** Recent values, oldest first. Renders nothing with fewer than 2 points. */
  points: number[]
  width?: number
  height?: number
  className?: string
}

/**
 * Minimal hand-rolled inline SVG sparkline (no charting library).
 * Colored green/red by net direction over the window.
 */
export default function Sparkline({ points, width = 48, height = 14, className = '' }: Props) {
  if (!points || points.length < 2) return null

  const min = Math.min(...points)
  const max = Math.max(...points)
  const span = max - min || 1
  const pad = 1

  const step = (width - pad * 2) / (points.length - 1)
  const path = points
    .map((v, i) => {
      const x = pad + i * step
      const y = pad + (1 - (v - min) / span) * (height - pad * 2)
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')

  const up = points[points.length - 1] >= points[0]

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className={`inline-block flex-shrink-0 ${className}`}
      aria-hidden="true"
    >
      <polyline
        points={path}
        fill="none"
        stroke={up ? '#22c55e' : '#ef4444'}
        strokeWidth="1.2"
        strokeLinejoin="round"
        strokeLinecap="round"
        opacity="0.85"
      />
    </svg>
  )
}
