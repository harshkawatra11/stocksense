"use client";
import { useEffect, useMemo, useRef, useState, type RefObject } from "react";
import { getJSON } from "@/lib/api";
import { isIdx } from "@/lib/ml/smart/chart";
import { groupStrategies, INDEX_EXITS, strategiesForTf, type StrategyGroup } from "@/lib/ml/smart/trades";
import type { StrategyRow, TradesResponse } from "@/lib/ml/smart/types";

interface Lists {
  rows: StrategyRow[] | null; // /api/lab/strategies (StockSense library); null when the endpoint is missing or down
  swing: string[]; // Market Lab's /api/lab/swing list (fallback)
  swingExits: Record<string, string>;
  intraday: string[]; // Market Lab's /api/lab/summary list (fallback)
}

async function tryJSON<T>(path: string): Promise<T | null> {
  try {
    return await getJSON<T>(path);
  } catch {
    return null; // 404 / 503 / network: the lab is optional
  }
}

export interface TradesCtl {
  TR: TradesResponse | null;
  t0: RefObject<number>;
  strat: string;
  exit: string;
  setStrat: (s: string) => void;
  setExit: (s: string) => void;
  groups: StrategyGroup[];
  exitOptions: [string, string][];
  idx: boolean;
}

/** Strategy lists and the trades of the open chart, refreshed every 60 s while the Trades layer is on. */
export function useTrades(sym: string, tf: string, on: boolean, initialStrat?: string): TradesCtl {
  const [strat, setStrat] = useState(initialStrat || "candidate");
  const [exit, setExit] = useState("");
  const [lists, setLists] = useState<Lists | null>(null);
  const [TR, setTR] = useState<TradesResponse | null>(null);
  const t0 = useRef(0);
  const seq = useRef(0);
  const last = useRef<TradesResponse | null>(null);
  const idx = isIdx(sym);

  useEffect(() => {
    if (!on || lists) return;
    let alive = true;
    void (async () => {
      const rows = await tryJSON<StrategyRow[]>("/api/lab/strategies");
      let swing: string[] = [];
      let swingExits: Record<string, string> = {};
      let intraday: string[] = [];
      if (!Array.isArray(rows)) {
        const s = await tryJSON<{ list?: string[]; exits?: Record<string, string> }>("/api/lab/swing");
        const i = await tryJSON<{ list?: string[] }>("/api/lab/summary");
        swing = s?.list ?? [];
        swingExits = s?.exits ?? {};
        intraday = i?.list ?? [];
      }
      if (alive) setLists({ rows: Array.isArray(rows) ? rows : null, swing, swingExits, intraday });
    })();
    return () => {
      alive = false;
    };
  }, [on, lists]);

  const rows = useMemo(() => (lists?.rows ? strategiesForTf(lists.rows, tf) : null), [lists, tf]);
  const groups = useMemo<StrategyGroup[]>(() => {
    if (rows) return groupStrategies(rows);
    const names = idx ? lists?.intraday : lists?.swing;
    if (!names || !names.length) return [];
    return [
      {
        label: idx ? "Market Lab intraday" : "Market Lab swing",
        items: names.map((n) => ({ key: n, name: n, source: idx ? "marketlab_intraday" : "marketlab_swing" })),
      },
    ];
  }, [rows, lists, idx]);

  // a strategy or exit that the list no longer has falls back to the default (smart.html tload)
  const known = useMemo(() => new Set(groups.flatMap((g) => g.items.flatMap((r) => [r.key, r.name]))), [groups]);
  const effStrat = strat === "candidate" || !lists || known.has(strat) ? strat : "candidate";
  const picked = rows?.find((r) => r.key === effStrat || r.name === effStrat);
  const exitOptions = useMemo<[string, string][]>(() => {
    if (picked?.exits && picked.exits.length) return picked.exits.map((e) => [e, e]);
    return idx ? INDEX_EXITS : Object.entries(lists?.swingExits ?? {});
  }, [picked, idx, lists]);
  const effExit = !exit || !lists || exitOptions.some(([k]) => k === exit) ? exit : "";

  useEffect(() => {
    if (!on) return;
    const sq = seq;
    const my = ++sq.current;
    const run = async () => {
      const q = new URLSearchParams({ sym, tf, strat: effStrat === "candidate" ? "" : effStrat, exit: effExit });
      try {
        const r = await getJSON<TradesResponse>("/api/lab/trades?" + q.toString());
        if (sq.current !== my) return; // a newer chart was opened meanwhile
        const p = last.current;
        if (!p || p.sym !== r.sym || p.strat !== r.strat || p.exit !== r.exit) t0.current = performance.now();
        last.current = r;
        setTR(r);
      } catch {
        if (sq.current === my) setTR({ trades: [], note: "Strategy lab is not answering." });
      }
    };
    void run();
    const id = setInterval(run, 60000);
    return () => {
      clearInterval(id);
      sq.current++;
    };
  }, [on, sym, tf, effStrat, effExit]);

  return { TR: on ? TR : null, t0, strat: effStrat, exit: effExit, setStrat, setExit, groups, exitOptions, idx };
}
