"use client";
import { useEffect, useState } from "react";
import { getJSON } from "@/lib/api";

export interface PanelState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

/** One GET per `path` (null = idle). The state belongs to the path it was fetched for, so a change shows "loading". */
export function usePanel<T>(path: string | null): PanelState<T> {
  const [st, setSt] = useState<{ path: string; data: T | null; error: string | null } | null>(null);
  useEffect(() => {
    if (path === null) return;
    const c = new AbortController();
    getJSON<T>(path, c.signal).then(
      (d) => !c.signal.aborted && setSt({ path, data: d, error: null }),
      (e: unknown) => !c.signal.aborted && setSt({ path, data: null, error: e instanceof Error ? e.message : String(e) }),
    );
    return () => c.abort();
  }, [path]);
  const mine = st && st.path === path ? st : null;
  return { data: mine?.data ?? null, error: mine?.error ?? null, loading: path !== null && !mine };
}
