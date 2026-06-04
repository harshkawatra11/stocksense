import type { TerminalLine } from '../../../types'
import TerminalBase from './TerminalBase'

interface Props { lines: TerminalLine[] }

export default function ClaudeTerminal({ lines }: Props) {
  return <TerminalBase title="TERMINAL 4 — CLAUDE SONNET (Final Review)" lines={lines} />
}
