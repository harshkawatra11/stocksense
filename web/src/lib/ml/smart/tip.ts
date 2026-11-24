// Port of smart.html tip() (lines 457-464), with the flow mini-card appended when the row has one.
import { esc, pc } from "../format";
import { flowMiniHTML, px2 } from "./tiles";
import type { SmartStock } from "./types";

export function tipHTML(s: SmartStock): string {
  const st = (s.read || {}).struct;
  return `<b>${esc(s.sym)}</b> <span class="muted">· ${esc(s.sector)}</span><br>₹${px2(s.ltp)} · <span class="${s.pct >= 0 ? "up" : "down"}">${pc(s.pct)}</span><br>
    DISPLACEMENT ${s.disp > 0 ? "+" : ""}${s.disp.toFixed(2)} · ${s.side === "bull" ? "BULLISH" : "BEARISH"} #${s.rank}<br>FROM THE OPEN ${pc(s.from_open)}<br>STRUCTURE ${st ? `<b class="${st.dir === "bull" ? "up" : "down"}" style="font:700 10.5px JetBrains Mono">${esc(st.type)} ${st.dir === "bull" ? "↑" : "↓"} ${esc(st.time)}</b>` : "–"}${flowMiniHTML(s.flow)}`;
}

/** Position of the tooltip next to the pointer, kept inside the window (smart.html line 460). */
export function tipPos(clientX: number, clientY: number, innerW: number, innerH: number): { left: number; top: number } {
  return { left: Math.min(innerW - 240, clientX + 14), top: Math.min(innerH - 120, clientY + 14) };
}
