"use client";
import { useCallback, useEffect, useRef, useState } from "react";

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

export async function getJSON<T>(path: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(API_BASE + path, { cache: "no-store", signal });
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json() as Promise<T>;
}

/**
 * Polls `path` every `ms` while enabled and the tab is visible.
 * One AbortController per request; everything is cleaned up on unmount or path change.
 */
export function usePoll<T>(
  path: string | null,
  ms: number,
  opts?: { enabled?: boolean },
): { data: T | null; error: Error | null; refresh: () => void } {
  const enabled = opts?.enabled ?? true;
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [tick, setTick] = useState(0);
  const refresh = useCallback(() => setTick((n) => n + 1), []);
  const ctl = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!enabled || path === null) return;
    let alive = true;
    const run = () => {
      if (document.visibilityState !== "visible") return;
      ctl.current?.abort();
      const c = new AbortController();
      ctl.current = c;
      getJSON<T>(path, c.signal).then(
        (d) => {
          if (alive && !c.signal.aborted) {
            setData(d);
            setError(null);
          }
        },
        (e: unknown) => {
          if (alive && !c.signal.aborted) setError(e instanceof Error ? e : new Error(String(e)));
        },
      );
    };
    run();
    const id = setInterval(run, ms);
    const onVis = () => {
      if (document.visibilityState === "visible") run();
    };
    document.addEventListener("visibilitychange", onVis);
    return () => {
      alive = false;
      clearInterval(id);
      document.removeEventListener("visibilitychange", onVis);
      ctl.current?.abort();
    };
  }, [path, ms, enabled, tick]);

  return { data, error, refresh };
}
