import type { TerminalLine } from '../../types'
import TerminalBase from './terminals/TerminalBase'

interface Props {
  mlLines: TerminalLine[]
  kronosLines: TerminalLine[]
  slmLines: TerminalLine[]
  claudeLines: TerminalLine[]
}

export default function MultiTerminalGrid({ mlLines, kronosLines, slmLines, claudeLines }: Props) {
  return (
    <div className="grid grid-cols-2 grid-rows-2 gap-px bg-border h-full overflow-hidden">
      <TerminalBase title="TERMINAL 1 — ML MODEL (LightGBM)" lines={mlLines} />
      <TerminalBase title="TERMINAL 2 — KRONOS FOUNDATION" lines={kronosLines} />
      <TerminalBase title="TERMINAL 3 — SLM NSE COMMANDER (Qwen2.5-7B)" lines={slmLines} />
      <TerminalBase title="TERMINAL 4 — CLAUDE SONNET" lines={claudeLines} />
    </div>
  )
}
