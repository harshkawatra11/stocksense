"use client";
import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";
import { LabActionsProvider, type LabActions } from "@/lib/ml/actions";
import { navigate, viewUrl } from "@/lib/ml/sector/route";
import type { SmartLive, Tf } from "@/lib/ml/smart/types";

export interface ChartRequest {
  sym: string;
  /** Strategy preselected by the Swing Lab (labShow); also turns the Trades layer on. */
  strat?: string;
  tf?: Tf;
  trades?: boolean;
  seq: number;
}

interface HostValue {
  req: ChartRequest | null;
  close: () => void;
  live: SmartLive | null;
  setLive: (d: SmartLive | null) => void;
}

const HostCtx = createContext<HostValue>({ req: null, close: () => {}, live: null, setLive: () => {} });
export const useChartHost = (): HostValue => useContext(HostCtx);

/** Owns the single pop-up chart request and provides the LabActions context (docs/PLAN-M.md 5.6). */
export function ChartHost({ children }: { children: ReactNode }) {
  const [req, setReq] = useState<ChartRequest | null>(null);
  const [live, setLive] = useState<SmartLive | null>(null);
  const seq = useRef(0);
  const close = useCallback(() => setReq(null), []);

  const actions = useMemo<LabActions>(
    () => ({
      openChart: (sym) => setReq({ sym, seq: ++seq.current }),
      labShow: (sym, strat, tf) => setReq({ sym, strat: strat || "candidate", tf: tf as Tf | undefined, trades: true, seq: ++seq.current }),
      go: (view, arg) => navigate(viewUrl(view, arg)),
    }),
    [],
  );
  const value = useMemo(() => ({ req, close, live, setLive }), [req, close, live]);

  return (
    <HostCtx.Provider value={value}>
      <LabActionsProvider value={actions}>{children}</LabActionsProvider>
    </HostCtx.Provider>
  );
}
