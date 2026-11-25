"use client";
import { useEffect, useRef, useState } from "react";
import { getJSON } from "@/lib/api";
import { mergePaths, nowMin } from "@/lib/ml/smart/stage";
import type { PathPt, SmartLive } from "@/lib/ml/smart/types";

export type LiveState = "loading" | "warming" | "ok" | "down";

/**
 * /api/sm/live every 4000 ms (smart.html load()): the full response with paths every 30 s or more,
 * otherwise paths=0 and the cached paths are extended with the latest point.
 */
export function useSmartLive(ms = 4000): { data: SmartLive | null; state: LiveState } {
  const [data, setData] = useState<SmartLive | null>(null);
  const [state, setState] = useState<LiveState>("loading");
  const paths = useRef<Record<string, PathPt[]>>({});
  const lastFull = useRef(0);

  useEffect(() => {
    let alive = true;
    let ctl: AbortController | null = null;
    const load = async () => {
      if (document.visibilityState !== "visible") return;
      ctl?.abort();
      const c = new AbortController();
      ctl = c;
      try {
        const full = Date.now() - lastFull.current > 30000;
        const nd = await getJSON<SmartLive>(full ? "/api/sm/live?paths=1" : "/api/sm/live?paths=0", c.signal);
        if (!alive || c.signal.aborted) return;
        if (!nd.stocks || !nd.stocks.length) {
          setState("warming");
          return;
        }
        if (full) lastFull.current = Date.now();
        mergePaths(nd.stocks, paths.current, full, nowMin());
        setData(nd);
        setState("ok");
      } catch {
        if (alive && !c.signal.aborted) setState("down");
      }
    };
    void load();
    const id = setInterval(load, ms);
    return () => {
      alive = false;
      clearInterval(id);
      ctl?.abort();
    };
  }, [ms]);

  return { data, state };
}
