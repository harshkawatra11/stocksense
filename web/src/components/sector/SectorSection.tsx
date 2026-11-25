"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { getJSON } from "@/lib/api";
import { esc, pc, cls } from "@/lib/ml/format";
import { useLabActions } from "@/lib/ml/actions";
import { navigate, parseRoute, RANGES, RL, viewUrl, type Range } from "@/lib/ml/sector/route";
import type { SearchHit } from "@/lib/ml/sector/types";
import { useChartHost } from "@/components/smart/ChartHost";
import { SectorEngine, type Crumb } from "./sectorEngine";

export const SEARCH_DEBOUNCE_MS = 180;

function Crumbs({ c, go }: { c: Crumb | null; go: (view: string, arg?: string) => void }) {
  return (
    <div className="crumbs" id="crumbs">
      <a onClick={() => go("horizon")}>All sectors</a>
      {c?.view === "sector" && c.sector && (
        <>
          {" "}
          &nbsp;›&nbsp; <span className="cur">{c.sector}</span>
        </>
      )}
      {c?.view === "stock" && (
        <>
          {c.sector && (
            <>
              {" "}
              &nbsp;›&nbsp; <a onClick={() => go("sector", c.sector as string)}>{c.sector}</a>
            </>
          )}{" "}
          &nbsp;›&nbsp; <span className="cur">{c.sym}</span>
        </>
      )}
    </div>
  );
}

function Search({ go }: { go: (view: string, arg?: string) => void }) {
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (!(e.target as HTMLElement).closest(".search")) setHits(null);
    };
    document.addEventListener("click", close);
    return () => {
      document.removeEventListener("click", close);
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);

  const onInput = (v: string) => {
    setQ(v);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(async () => {
      const t = v.trim();
      if (!t) return setHits(null);
      try {
        setHits(await getJSON<SearchHit[]>("/api/search?q=" + encodeURIComponent(t)));
      } catch {
        setHits([]);
      }
    }, SEARCH_DEBOUNCE_MS);
  };
  const pick = (view: string, arg: string) => {
    setHits(null);
    setQ("");
    go(view, arg);
  };

  return (
    <div className="search" ref={box}>
      <input id="q" placeholder="Find a sector or stock" autoComplete="off" value={q} onChange={(e) => onInput(e.target.value)} />
      <div id="results" style={{ display: hits ? "block" : "none" }}>
        {hits &&
          (hits.length ? (
            hits.map((x) =>
              x.type === "sector" ? (
                <div key={"s:" + x.name} data-sec={x.name} onClick={() => pick("sector", x.name)}>
                  <b>{x.name}</b>
                  <span className="muted">sector</span>
                </div>
              ) : (
                <div key={"k:" + x.sym} data-sym={x.sym} onClick={() => pick("stock", x.sym)}>
                  <span>
                    <b>{x.sym}</b> <span className="muted">{(x.name || "").slice(0, 26)}</span>
                  </span>
                  <span className={`${cls(x.pct)} mono`}>{pc(x.pct)}</span>
                </div>
              ),
            )
          ) : (
            <div className="muted">Nothing found</div>
          ))}
      </div>
    </div>
  );
}

/** Port of sector.html: header, ranges, search, crumbs, and the three imperative panes filled by SectorEngine. */
export function SectorSection() {
  const params = useSearchParams();
  const key = params ? params.toString() : "";
  const route = useMemo(() => parseRoute(new URLSearchParams(key)), [key]);
  const { openChart } = useLabActions();
  const { req } = useChartHost();
  const [crumb, setCrumb] = useState<Crumb | null>(null);
  const [live, setLive] = useState<{ market: string; time: string } | null>(null);

  const view = useRef<HTMLDivElement>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const side = useRef<HTMLDivElement>(null);
  const tip = useRef<HTMLDivElement>(null);
  const engine = useRef<SectorEngine | null>(null);
  const routeRef = useRef(route);
  const openRef = useRef(openChart);
  useEffect(() => {
    routeRef.current = route;
    openRef.current = openChart;
  });

  const go = (view: string, arg?: string) => navigate(viewUrl(view, arg, routeRef.current.range));

  useEffect(() => {
    const v = view.current!;
    const eng = new SectorEngine(
      { root: v.closest<HTMLElement>(".ml-sector") ?? v.closest<HTMLElement>(".wrap") ?? v, view: v, bottom: bottom.current!, side: side.current!, tip: tip.current! },
      {
        go: (view, arg) => navigate(viewUrl(view, arg, routeRef.current.range)),
        openChart: (sym) => openRef.current(sym),
        crumbs: setCrumb,
        live: setLive,
      },
    );
    engine.current = eng;
    eng.start();
    return () => {
      eng.destroy();
      engine.current = null;
    };
  }, []);
  useEffect(() => {
    engine.current?.setRoute(route);
  }, [route]);
  useEffect(() => {
    if (engine.current) engine.current.modalOpen = !!req;
  }, [req]);

  const setRange = (r: Range) =>
    navigate(route.view === "sector" ? viewUrl("sector", route.name ?? "", r) : route.view === "stock" ? viewUrl("stock", route.sym ?? "", r) : viewUrl("horizon", undefined, r));

  return (
    <>
      <header>
        <div>
          <h1>Sector Map</h1>
          <div className="sub">Which sectors are running, what is moving them, and the stocks inside</div>
        </div>
        <div className="sp" />
        <span className="pill" id="livep">
          {live && live.market !== "open" ? (
            <>
              <span className="dot" style={{ background: "#9aa3b2", animation: "none" }} />
              Market {esc(live.market)}
            </>
          ) : (
            <>
              <span className="dot" />
              Live · NSE indices{live ? ` · ${new Date(live.time).toLocaleTimeString("en-GB", { timeZone: "Asia/Kolkata" })}` : ""}
            </>
          )}
        </span>
        <div className="seg" id="ranges">
          {RANGES.map((r) => (
            <button key={r} data-r={r} className={route.range === r ? "on" : ""} onClick={() => setRange(r)}>
              {RL[r]}
            </button>
          ))}
        </div>
        <Search go={go} />
        <a
          className="pill"
          href="#smart"
          onClick={(e) => {
            e.preventDefault();
            window.scrollTo({ top: 0, behavior: "smooth" });
          }}
        >
          Smart money ↑
        </a>
      </header>
      <Crumbs c={crumb} go={go} />
      <div className="wrap">
        <div className="main">
          <div id="view" ref={view} />
          <div id="bottom" ref={bottom} />
        </div>
        <div className="side" id="side" ref={side} />
      </div>
      <div className="foot">
        A technical screen built from NSE and Yahoo data · prices 2-3 s behind NSE for what is on screen, about a minute for the rest · not investment advice
      </div>
      <div className="tipc" id="stip" ref={tip} />
    </>
  );
}
