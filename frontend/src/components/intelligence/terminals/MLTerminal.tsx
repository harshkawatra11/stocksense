import type { TerminalLine } from '../../../types'
import TerminalBase from './TerminalBase'

interface Props { lines: TerminalLine[] }

export default function MLTerminal({ lines }: Props) {
  return <TerminalBase title="TERMINAL 1 — ML MODEL (LightGBM + SHAP)" lines={lines} />
}
