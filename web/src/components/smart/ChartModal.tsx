"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  createChart,
  type AutoscaleInfo,
  type IChartApi,
  type IPriceLine,
  type ISeriesApi,
  type UTCTimestamp,
} from "lightweight-charts";
import { getJSON } from "@/lib/api";
import { esc } from "@/lib/ml/format";
import {
  applyQuote,
  DEFAULT_LAYERS,
  drawZones,
  eventsHTML,
  isIdx,
  LAYER_BUTTONS,
  levelsHTML,
  liquidityLines,
  priceHTML,
  structureMarkers,
  TF_BUTTONS,
  visibleRange,
  zoneSpecs,
  type LayerKey,
  type Layers,
} from "@/lib/ml/smart/chart";
import { drawTrades, tlevels, tradeMarkers } from "@/lib/ml/smart/trades";
import type { ChartData, QuoteData, SmartLive, Tf, Trade } from "@/lib/ml/smart/types";
import { useChartHost, type ChartRequest } from "./ChartHost";
import { ResearchNote } from "./ResearchNote";
import { TradeControls, TradesPanel } from "./TradesPanel";
import { useTrades } from "./useTrades";

const toTime = (t: number) => t as UTCTimestamp;

interface Api {
  chart: IChartApi;
  candle: ISeriesApi<"Candlestick">;
  vwap: ISeriesApi<"Line">;
  lines: IPriceLine[];
}

function ModalBody({ req, live, onClose }: { req: ChartRequest; live: SmartLive | null; onClose: () => void }) {
  const [sym, setSym] = useState(req.sym);
  const [tf, setTf] = useState<Tf>(req.tf ?? (isIdx(req.sym) ? "5m" : "1d"));
  const [lay, setLay] = useState<Layers>({ ...DEFAULT_LAYERS, trades: !!req.trades });
  const [md, setMd] = useState<{ d: ChartData | null; err: string | null; sym: string; tf: Tf } | null>(null);
  const [quote, setQuote] = useState<{ key: string; ltp: number; prev: number | null } | null>(null);
  const ctl = useTrades(sym, tf, lay.trades, req.strat);

  const mc = useRef<HTMLDivElement>(null);
  const mcw = useRef<HTMLDivElement>(null);
  const mo = useRef<HTMLCanvasElement>(null);
  const api = useRef<Api | null>(null);
  const hot = useRef<Trade | null>(null);

  const cur = md && md.sym === sym && md.tf === tf && md.d ? md.d : null;
  const err = md && md.sym === sym && md.tf === tf ? md.err : null;

  // what the animation loop and the price-scale callback read each frame
  const latest = useRef({ cur, lay, TR: ctl.TR, sym });
  useEffect(() => {
    latest.current = { cur, lay, TR: ctl.TR, sym };
  });

  // ---- the chart, created once per open with Market Lab's options ----
  useEffect(() => {
    const el = mc.current;
    if (!el) return;
    const chart = createChart(el, {
      layout: { background: { color: "#ffffff" }, textColor: "#6b7385", fontFamily: "JetBrains Mono", fontSize: 11 },
      grid: { vertLines: { color: "#f0f2f5" }, horzLines: { color: "#f0f2f5" } },
      timeScale: { timeVisible: true, borderColor: "#d9dee6", rightOffset: 6 },
      rightPriceScale: { borderColor: "#d9dee6" },
      crosshair: { mode: 0 },
      localization: {
        timeFormatter: (t: number) =>
          new Date(t * 1000).toLocaleString("en-IN", { timeZone: "Asia/Kolkata", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }),
      },
    });
    const candle = chart.addCandlestickSeries({
      upColor: "#2f9e5f",
      downColor: "#d6455d",
      borderVisible: false,
      wickUpColor: "#2f9e5f",
      wickDownColor: "#d6455d",
    });
    const vwap = chart.addLineSeries({ color: "#3b5bdb", lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false });
    // keep a live trade's stop and target inside the price scale (smart.html tscale)
    candle.applyOptions({
      autoscaleInfoProvider: (original: () => AutoscaleInfo | null) => {
        const r = original();
        const { TR, sym: s, lay: l } = latest.current;
        const lv = tlevels(TR, s, l.trades);
        if (r && r.priceRange && lv.length) {
          r.priceRange.minValue = Math.min(r.priceRange.minValue, ...lv);
          r.priceRange.maxValue = Math.max(r.priceRange.maxValue, ...lv);
        }
        return r;
      },
    });
    api.current = { chart, candle, vwap, lines: [] };
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(() => chart.applyOptions({ width: el.clientWidth, height: el.clientHeight })) : null;
    ro?.observe(el);
    return () => {
      ro?.disconnect();
      api.current = null;
      chart.remove();
    };
  }, []);

  // ---- candles and the smart-money read ----
  useEffect(() => {
    const c = new AbortController();
    getJSON<ChartData>(`/api/sm/chart?sym=${encodeURIComponent(sym)}&tf=${tf}`, c.signal).then(
      (d) => {
        if (!c.signal.aborted) setMd({ d: d.error ? null : d, err: d.error ?? null, sym, tf });
      },
      (e: unknown) => {
        if (!c.signal.aborted) setMd({ d: null, err: e instanceof Error ? e.message : String(e), sym, tf });
      },
    );
    return () => c.abort();
  }, [sym, tf]);

  useEffect(() => {
    const a = api.current;
    if (!a || !cur) return;
    a.candle.setMarkers([]);
    a.candle.setData(cur.candles.map((c) => ({ time: toTime(c.time), open: c.open, high: c.high, low: c.low, close: c.close })));
    a.candle.priceScale().applyOptions({ autoScale: true });
    a.chart.priceScale("right").applyOptions({ autoScale: true });
    a.chart.timeScale().setVisibleLogicalRange(visibleRange(cur.candles.length, cur.tf));
  }, [cur]);

  useEffect(() => {
    const a = api.current;
    if (!a || !cur) return;
    a.vwap.setData(lay.vwap ? (cur.read?.chart?.vwap ?? []).map((p) => ({ time: toTime(p.time), value: p.value })) : []);
    a.lines.forEach((l) => a.candle.removePriceLine(l));
    a.lines = liquidityLines(cur.read, lay.liquidity).map((s) =>
      a.candle.createPriceLine({ price: s.price, color: s.color, lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: s.title }),
    );
  }, [cur, lay.vwap, lay.liquidity]);

  useEffect(() => {
    const a = api.current;
    if (!a || !cur) return;
    const have = new Set(cur.candles.map((c) => c.time));
    const m = [...structureMarkers(cur.read, lay.structure).filter((k) => have.has(k.time)), ...tradeMarkers(ctl.TR, cur.candles, sym, lay.trades)];
    m.sort((x, y) => x.time - y.time);
    a.candle.setMarkers(m.map((k) => ({ ...k, time: toTime(k.time) })));
  }, [cur, lay.structure, lay.trades, ctl.TR, sym]);

  useEffect(() => {
    // room on the right so a live trade's risk:reward box is not squeezed against the price axis
    const open = lay.trades && !!ctl.TR && (ctl.TR.trades || []).some((t) => t.status !== "CLOSED");
    api.current?.chart.timeScale().applyOptions({ rightOffset: open ? 18 : 6 });
  }, [lay.trades, ctl.TR]);

  // ---- the last candle moves every 3 s ----
  useEffect(() => {
    if (!cur) return;
    let alive = true;
    const tick = async () => {
      try {
        const q = await getJSON<QuoteData>("/api/sm/quote?sym=" + encodeURIComponent(sym));
        if (!alive || !q.ltp) return;
        const c = applyQuote(cur.candles, q, cur.tf);
        if (!c) return;
        api.current?.candle.update({ time: toTime(c.time), open: c.open, high: c.high, low: c.low, close: c.close });
        setQuote({ key: `${sym}|${cur.tf}`, ltp: q.ltp, prev: q.prev ?? null });
      } catch {
        /* the next tick tries again */
      }
    };
    const id = setInterval(tick, 3000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [cur, sym]);

  const ctlT0 = ctl.t0;
  // ---- overlay canvas (#mo), redrawn every animation frame: zones and the trade boxes ----
  useEffect(() => {
    if (typeof requestAnimationFrame === "undefined") return;
    let raf = 0;
    const loop = () => {
      raf = requestAnimationFrame(loop);
      const cv = mo.current;
      const w = mcw.current;
      const a = api.current;
      if (!cv || !w || !a) return;
      const dpr = window.devicePixelRatio || 1;
      const W = w.clientWidth;
      const Hh = w.clientHeight;
      if (cv.width !== W * dpr || cv.height !== Hh * dpr) {
        cv.width = W * dpr;
        cv.height = Hh * dpr;
        cv.style.width = W + "px";
        cv.style.height = Hh + "px";
      }
      const x = cv.getContext("2d");
      if (!x) return;
      x.setTransform(dpr, 0, 0, dpr, 0, 0);
      x.clearRect(0, 0, W, Hh);
      const { cur: d, lay: l, TR, sym: s } = latest.current;
      if (!d) return;
      const ts = a.chart.timeScale();
      const pw = ts.width();
      const timeToX = (t: number) => ts.timeToCoordinate(toTime(t));
      const priceToY = (p: number) => a.candle.priceToCoordinate(p);
      x.save();
      x.beginPath();
      x.rect(0, 0, pw, Hh - 26);
      x.clip();
      drawZones(x, zoneSpecs(d.read, l), pw, timeToX, priceToY);
      if (TR)
        drawTrades(x, {
          TR,
          sym: s,
          tradesOn: l.trades,
          hot: hot.current,
          t0: ctlT0.current,
          now: performance.now(),
          pw,
          lastCandleTime: d.candles.length ? d.candles[d.candles.length - 1].time : null,
          timeToX,
          priceToY,
        });
      x.restore();
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [ctlT0]);

  const s = live?.stocks.find((q) => q.sym === sym);
  const ix = live?.indices[sym as keyof SmartLive["indices"]];
  const sub = s
    ? `${s.sector} · ${s.side === "bull" ? "BULLISH" : "BEARISH"} #${s.rank} OF ${s.side === "bull" ? live!.bull.length : live!.bear.length}`
    : ix
      ? "INDEX"
      : "";
  const chg = s ? s.pct : ix ? ix.pct : 0;
  const q = quote && quote.key === `${sym}|${tf}` ? quote : null;
  const price = priceHTML(q ? q.ltp : cur?.ltp, q ? (q.prev ?? cur?.prev) : cur?.prev);
  const levels = useMemo(() => (err ? `<div class="down">${esc(err)}</div>` : cur ? levelsHTML(cur.read) : '<div class="muted">Loading…</div>'), [cur, err]);
  const events = useMemo(() => (cur ? eventsHTML(cur.read) : ""), [cur]);

  const toggle = (k: LayerKey) => setLay((l) => ({ ...l, [k]: !l[k] }));

  return (
    <div className="box" id="mbox" style={{ borderLeftColor: chg >= 0 ? "var(--up)" : "var(--down)" }}>
      <div className="mh">
        <div>
          <h2 id="mname">{sym}</h2>
          <div className="muted" id="msub">
            {sub}
          </div>
          <div style={{ fontSize: 15, fontWeight: 700, marginTop: 3 }} id="mprice" dangerouslySetInnerHTML={{ __html: price }} />
        </div>
        <button id="mx" onClick={onClose}>
          ✕
        </button>
      </div>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 8 }}>
        <span id="mtfs">
          {TF_BUTTONS.map(([k, l]) => (
            <span key={k}>
              <button data-tf={k} className={k === tf ? "on" : ""} onClick={() => setTf(k)}>
                {l}
              </button>{" "}
            </span>
          ))}
        </span>
        <span style={{ width: 10 }} />
        <span id="mlay">
          {LAYER_BUTTONS.map(([k, l]) => (
            <span key={k}>
              <button data-l={k} className={lay[k] ? "on" : ""} onClick={() => toggle(k)}>
                {l}
              </button>{" "}
            </span>
          ))}
        </span>
        <TradeControls ctl={ctl} on={lay.trades} />
      </div>
      <div className="mg">
        <div>
          <div id="mcw" ref={mcw}>
            <div id="mc" ref={mc} />
            <canvas id="mo" ref={mo} />
          </div>
          <div className="evs" id="mev" dangerouslySetInnerHTML={{ __html: events }} />
          <ResearchNote sym={sym} onPeer={setSym} />
        </div>
        <div className="lv">
          <TradesPanel
            TR={ctl.TR}
            sym={sym}
            on={lay.trades}
            firstCandleTime={cur && cur.candles.length ? cur.candles[0].time : null}
            onShow={() => setLay((l) => ({ ...l, trades: true }))}
            onHot={(t) => {
              hot.current = t;
            }}
            onSeek={(t) => {
              const k = cur ? cur.candles.findIndex((c) => c.time >= t.t_in) : -1;
              if (k >= 0) api.current?.chart.timeScale().setVisibleLogicalRange({ from: k - 60, to: k + 60 });
            }}
          />
          <div id="mlv" dangerouslySetInnerHTML={{ __html: levels }} />
        </div>
      </div>
      <div className="note">Scroll or pinch to zoom · drag to move · switch each layer above · the last candle updates live every 3 seconds.</div>
    </div>
  );
}

/** The single pop-up chart, mounted once at page level inside .ml-smart styling (docs/PLAN-M.md M6.1). */
export function ChartModal() {
  const { req, close, live } = useChartHost();
  useEffect(() => {
    if (!req) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [req, close]);
  return (
    <div
      id="modal"
      style={{ display: req ? "flex" : "none" }}
      onClick={(e) => {
        if ((e.target as HTMLElement).id === "modal") close();
      }}
    >
      {req && <ModalBody key={req.seq} req={req} live={live} onClose={close} />}
    </div>
  );
}
