"use client";
import { createContext, useContext, useMemo, type ReactNode } from "react";

// Replaces Market Lab's window.openChart / window.labShow / window.go (docs/PLAN-M.md 5.6).
export interface LabActions {
  openChart: (sym: string) => void;
  labShow: (sym: string, strat?: string, tf?: string) => void;
  go: (view: string, arg?: string) => void;
}

const noop: LabActions = { openChart: () => {}, labShow: () => {}, go: () => {} };
const Ctx = createContext<LabActions>(noop);

export function LabActionsProvider({ value, children }: { value?: Partial<LabActions>; children: ReactNode }) {
  const merged = useMemo(() => ({ ...noop, ...value }), [value]);
  return <Ctx.Provider value={merged}>{children}</Ctx.Provider>;
}

export const useLabActions = (): LabActions => useContext(Ctx);
