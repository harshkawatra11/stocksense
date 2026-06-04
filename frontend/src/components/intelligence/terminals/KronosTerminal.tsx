import type { TerminalLine } from '../../../types'
import TerminalBase from './TerminalBase'

interface Props { lines: TerminalLine[] }

export default function KronosTerminal({ lines }: Props) {
  return <TerminalBase title="TERMINAL 2 — KRONOS FOUNDATION MODEL" lines={lines} />
}
