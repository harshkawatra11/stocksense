import { useEffect, useRef, useState, type ReactNode } from 'react'

/**
 * Shared green/red flash-on-change primitive for live prices.
 * Extracted from Live.tsx's LivePriceBadge so every price surface
 * (indices, signal cards, portfolio) flashes consistently.
 */

/** Returns 'up' | 'down' | null for ~500ms after `value` changes. */
export function useFlash(value: number | null | undefined): 'up' | 'down' | null {
  const [flash, setFlash] = useState<'up' | 'down' | null>(null)
  const prevRef = useRef<number | null | undefined>(value)

  useEffect(() => {
    const prev = prevRef.current
    if (value != null && prev != null && value !== prev) {
      setFlash(value > prev ? 'up' : 'down')
      prevRef.current = value
      const t = setTimeout(() => setFlash(null), 500)
      return () => clearTimeout(t)
    }
    prevRef.current = value
  }, [value])

  return flash
}

/** Tailwind background class for a flash state (or '' when idle). */
export function flashBgClass(flash: 'up' | 'down' | null): string {
  return flash === 'up' ? 'bg-green/20' : flash === 'down' ? 'bg-red/20' : ''
}

interface FlashPriceProps {
  /** The numeric value to watch for changes. */
  value: number | null | undefined
  /** Rendered content (defaults to the formatted value itself). */
  children?: ReactNode
  className?: string
}

/**
 * Wraps its children in a span that flashes green/red when `value` changes.
 * Direction comes from comparing consecutive values.
 */
export default function FlashPrice({ value, children, className = '' }: FlashPriceProps) {
  const flash = useFlash(value)
  return (
    <span className={`transition-colors duration-500 rounded ${flashBgClass(flash)} ${className}`}>
      {children ?? (value != null ? value.toLocaleString('en-IN') : '—')}
    </span>
  )
}
