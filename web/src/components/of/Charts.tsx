"use client";
import { useEffect, useRef } from "react";
import { createChart, type SeriesMarker, type Time, type UTCTimestamp } from "lightweight-charts";
import type { DeltaReply, OfBars, Series } from "@/lib/of/types";

const tt = (t: number) => t as UTCTimestamp;
const OPTS = {
  layout: { background: { color: "#ffffff" }, textColor: "#6b7385", fontFamily: "JetBrains Mono", fontSize: 11 },
  grid: { vertLines: { color: "#f0f2f5" }, horzLines: { color: "#f0f2f5" } },
  timeScale: { timeVisible: true, borderColor: "#d9dee6" },
  rightPriceScale: { borderColor: "#d9dee6" },
  localization: {
    timeFormatter: (t: number) =>
      new Date(t * 1000).toLocaleTimeString("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit" }),
  },
};
const toLine = (ts: number[], v: Series | undefined) =>
  (v ?? []).flatMap((x, i) => (x == null || !isFinite(x) ? [] : [{ time: tt(ts[i]), value: x }]));

/** Candles with VWAP, +-1/2 sigma bands and an anchored VWAP; clicking a candle calls onAnchor(time). */
export function CandleChart({ d, onAnchor }: { d: OfBars; onAnchor: (ts: number) => void }) {
  const el = useRef<HTMLDivElement>(null);
  const cb = useRef(onAnchor);
  useEffect(() => {
    cb.current = onAnchor;
  });
  useEffect(() => {
    if (!el.current) return;
    const chart = createChart(el.current, { ...OPTS, autoSize: true });
    const candle = chart.addCandlestickSeries({
      upColor: "#2f9e5f",
      downColor: "#d6455d",
      borderVisible: false,
      wickUpColor: "#2f9e5f",
      wickDownColor: "#d6455d",
    });
    const ts = d.bars.map((b) => b.time);
    candle.setData(d.bars.map((b) => ({ time: tt(b.time), open: b.open, high: b.high, low: b.low, close: b.close })));
    const add = (v: Series | undefined, color: string, lineStyle: number, lineWidth: 1 | 2) => {
      if (!v) return;
      chart
        .addLineSeries({ color, lineWidth, lineStyle, priceLineVisible: false, lastValueVisible: false })
        .setData(toLine(ts, v));
    };
    add(d.vwap, "#3b5bdb", 0, 2);
    add(d.bands?.u1, "#98a1b3", 2, 1);
    add(d.bands?.l1, "#98a1b3", 2, 1);
    add(d.bands?.u2, "#c5cad4", 2, 1);
    add(d.bands?.l2, "#c5cad4", 2, 1);
    add(d.anchored, "#f5a623", 0, 2);
    chart.timeScale().fitContent();
    chart.subscribeClick((p) => {
      if (typeof p.time === "number") cb.current(p.time);
    });
    return () => chart.remove();
  }, [d]);
  return <div ref={el} className="of-chart" aria-label="Candles with VWAP" />;
}

/** CVD line with divergence markers (up arrow = bullish, down arrow = bearish). */
export function CvdChart({ d }: { d: DeltaReply }) {
  const el = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!el.current) return;
    const chart = createChart(el.current, { ...OPTS, autoSize: true });
    const s = chart.addLineSeries({ color: "#141a26", lineWidth: 2, priceLineVisible: false });
    s.setData(d.ts.map((t, i) => ({ time: tt(t), value: d.cvd[i] })));
    const m: SeriesMarker<Time>[] = d.ts.flatMap((t, i) =>
      d.divergence[i]
        ? [
            {
              time: tt(t),
              position: d.divergence[i] > 0 ? ("belowBar" as const) : ("aboveBar" as const),
              shape: d.divergence[i] > 0 ? ("arrowUp" as const) : ("arrowDown" as const),
              color: d.divergence[i] > 0 ? "#2f9e5f" : "#d6455d",
              text: d.divergence[i] > 0 ? "bull div" : "bear div",
            },
          ]
        : [],
    );
    s.setMarkers(m);
    chart.timeScale().fitContent();
    return () => chart.remove();
  }, [d]);
  return <div ref={el} className="of-chart sm" aria-label="Cumulative delta" />;
}
