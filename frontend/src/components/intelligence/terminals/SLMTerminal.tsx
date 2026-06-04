import type { TerminalLine } from '../../../types'
import TerminalBase from './TerminalBase'

interface Props { lines: TerminalLine[] }

export default function SLMTerminal({ lines }: Props) {
  return <TerminalBase title="TERMINAL 3 — SLM NSE COMMANDER (Qwen2.5-7B)" lines={lines} />
}
