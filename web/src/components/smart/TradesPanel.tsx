"use client";
import { useEffect, useMemo, useRef, type MouseEvent } from "react";
import { candidateLabel, tpanelHTML } from "@/lib/ml/smart/trades";
import type { Trade, TradesResponse } from "@/lib/ml/smart/types";
import type { TradesCtl } from "./useTrades";

/** The strategy and exit selects of the toolbar (#tstrat, #texit), grouped by source. Hidden while Trades is off. */
export function TradeControls({ ctl, on }: { ctl: TradesCtl; on: boolean }) {
  if (!on) return null;
  return (
    <span className="tctl">
      <select id="tstrat" title="Whose trades are drawn" value={ctl.strat} onChange={(e) => ctl.setStrat(e.target.value)}>
        <option value="candidate">{candidateLabel(ctl.idx)}</option>
        {ctl.groups.map((g) => (
          <optgroup key={g.label} label={g.label}>
            {g.items.map((r) => (
              <option key={r.key} value={r.key}>
                {r.name}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
      <select id="texit" title="Exit rule" value={ctl.exit} onChange={(e) => ctl.setExit(e.target.value)}>
        <option value="">Default exit</option>
        {ctl.exitOptions.map(([k, l]) => (
          <option key={k} value={k}>
            {l}
          </option>
        ))}
      </select>
    </span>
  );
}

export interface TradesPanelProps {
  TR: TradesResponse | null;
  sym: string;
  on: boolean;
  firstCandleTime: number | null;
  onShow: () => void;
  onHot: (t: Trade | null) => void;
  onSeek: (t: Trade) => void;
}

/** #tpanel: the header (with the EXPLORATORY badge and gate chip), setup state, summary and the trade list. */
export function TradesPanel({ TR, sym, on, firstCandleTime, onShow, onHot, onSeek }: TradesPanelProps) {
  const html = useMemo(() => tpanelHTML(TR, sym, on), [TR, sym, on]);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // trades older than the chart's own history are dimmed and cannot be scrolled to
    box.current?.querySelectorAll<HTMLElement>(".tr").forEach((r) => {
      const t = TR?.trades[Number(r.dataset.i)];
      if (t && firstCandleTime != null && t.t_in < firstCandleTime) {
        r.style.opacity = "0.55";
        r.title = "Older than this chart's history";
      }
    });
  }, [html, TR, firstCandleTime]);

  const row = (e: MouseEvent): { t: Trade; old: boolean } | null => {
    const r = (e.target as HTMLElement).closest<HTMLElement>(".tr");
    const t = r && TR?.trades[Number(r.dataset.i)];
    return t ? { t, old: firstCandleTime == null || t.t_in < firstCandleTime } : null;
  };

  return (
    <div
      id="tpanel"
      ref={box}
      dangerouslySetInnerHTML={{ __html: html }}
      onClick={(e) => {
        if ((e.target as HTMLElement).closest(".ton")) return onShow();
        const r = row(e);
        if (r && !r.old) onSeek(r.t);
      }}
      onMouseOver={(e) => onHot(row(e)?.t ?? null)}
      onMouseLeave={() => onHot(null)}
    />
  );
}
