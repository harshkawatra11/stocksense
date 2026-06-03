import { useEffect, useRef, useState } from 'react'
import type { TerminalLine } from '../../../types'

interface Props {
  title: string
  lines: TerminalLine[]
}

const COLOR_MAP = {
  green:  '#00d4aa',
  red:    '#ff4757',
  yellow: '#ffa502',
  grey:   '#9aa0ab',
  white:  '#e8eaed',
}

export default function TerminalBase({ title, lines }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const [paused, setPaused] = useState(false)

  useEffect(() => {
    if (!paused) bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [lines, paused])

  return (
    <div className="bg-bg-primary flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-bg-card border-b border-border flex-shrink-0">
        <span className="text-xs font-medium text-text-secondary terminal-text">{title}</span>
        <span className="text-xs text-text-secondary terminal-text">{lines.length} lines</span>
      </div>

      {/* Log area */}
      <div
        className="flex-1 overflow-y-auto px-3 py-2"
        onMouseEnter={() => setPaused(true)}
        onMouseLeave={() => setPaused(false)}
      >
        {lines.length === 0 ? (
          <p className="text-xs text-text-secondary terminal-text mt-2">Waiting for pipeline run...</p>
        ) : (
          lines.map((line, i) => (
            <div key={i} className="flex gap-2 text-xs terminal-text leading-5">
              <span className="text-text-secondary flex-shrink-0 select-none">{line.ts}</span>
              <span style={{ color: COLOR_MAP[line.color] }}>{line.text}</span>
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
