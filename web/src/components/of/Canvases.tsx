"use client";
// Imperative canvas islands (docs/PLAN-M.md 5.6): React owns the element, pure helpers in lib/of/draw.ts do the maths.
import { useEffect, useRef } from "react";
import {
  cellText,
  footprintCellColor,
  heatColor,
  heatMax,
  inStack,
  maxAbsDelta,
  priceY,
  profileWidths,
  tpoGrid,
} from "@/lib/of/draw";
import type { FootprintReply, HeatReply, TpoProfile, VolumeProfile } from "@/lib/of/types";

const MONO = '11px "JetBrains Mono", monospace';

function setup(cv: HTMLCanvasElement, h: number): { ctx: CanvasRenderingContext2D; w: number } | null {
  const w = cv.clientWidth || 600;
  const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  cv.style.height = `${h}px`;
  const ctx = cv.getContext("2d");
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w };
}

function hline(ctx: CanvasRenderingContext2D, y: number, w: number, color: string, label: string) {
  ctx.strokeStyle = color;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(0, y);
  ctx.lineTo(w, y);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = color;
  ctx.font = MONO;
  ctx.fillText(label, 4, y - 3);
}

export function VolumeProfileCanvas({ p }: { p: VolumeProfile }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    const s = cv && setup(cv, 360);
    if (!s) return;
    const { ctx, w } = s;
    const lo = p.bins[0],
      hi = p.bins[p.bins.length - 1] + p.bin;
    const widths = profileWidths(p.vol, w - 70);
    p.bins.forEach((b, i) => {
      const y1 = priceY(b + p.bin, lo, hi, 360, 6),
        y0 = priceY(b, lo, hi, 360, 6);
      const inVa = b >= p.val - 1e-9 && b + p.bin <= p.vah + 1e-9;
      ctx.fillStyle = inVa ? "rgba(59,91,219,0.55)" : "rgba(152,161,179,0.5)";
      ctx.fillRect(w - widths[i], y1, widths[i], Math.max(1, y0 - y1 - 1));
    });
    hline(ctx, priceY(p.poc, lo, hi, 360, 6), w, "#d6455d", `POC ${p.poc}`);
    hline(ctx, priceY(p.vah, lo, hi, 360, 6), w, "#3b5bdb", `VAH ${p.vah}`);
    hline(ctx, priceY(p.val, lo, hi, 360, 6), w, "#3b5bdb", `VAL ${p.val}`);
  }, [p]);
  return <canvas ref={ref} aria-label="Volume profile" />;
}

export function TpoCanvas({ t }: { t: TpoProfile }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    const rows = tpoGrid(t);
    const rh = 14;
    const s = cv && setup(cv, rows.length * rh + 4);
    if (!s) return;
    const { ctx, w } = s;
    ctx.font = MONO;
    rows.forEach((r, i) => {
      const y = 2 + i * rh;
      if (r.va) {
        ctx.fillStyle = "rgba(59,91,219,0.10)";
        ctx.fillRect(60, y, w - 60, rh);
      }
      if (r.single) {
        ctx.fillStyle = "rgba(245,166,35,0.35)";
        ctx.fillRect(60, y, w - 60, rh);
      }
      ctx.fillStyle = r.poc ? "#d6455d" : "#141a26";
      ctx.fillText(r.letters, 64, y + 11);
      ctx.fillStyle = "#6b7487";
      ctx.fillText(r.price.toFixed(1), 2, y + 11);
    });
    const ib = rows.map((r, i) => (r.ib ? i : -1)).filter((i) => i >= 0);
    if (ib.length) {
      ctx.strokeStyle = "#3b5bdb";
      ctx.strokeRect(w - 6, 2 + ib[0] * rh, 4, (ib[ib.length - 1] - ib[0] + 1) * rh);
    }
  }, [t]);
  return <canvas ref={ref} aria-label="TPO letters by price" />;
}

export function FootprintCanvas({ fp }: { fp: FootprintReply }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv || !fp.bars.length) return;
    const lo = Math.min(...fp.bars.map((b) => b.low)),
      hi = Math.max(...fp.bars.map((b) => b.high));
    const nRows = Math.max(1, Math.round((hi - lo) / fp.bin) + 1);
    const rh = 15,
      colW = 92,
      x0 = 52,
      h = nRows * rh + 20;
    cv.style.width = `${x0 + fp.bars.length * colW}px`;
    const s = setup(cv, h);
    if (!s) return;
    const { ctx } = s;
    const mx = maxAbsDelta(fp.bars);
    ctx.font = "10px 'JetBrains Mono', monospace";
    fp.bars.forEach((b, ci) => {
      const x = x0 + ci * colW;
      b.rows.forEach((r) => {
        const y = 4 + Math.round((hi - r.price) / fp.bin) * rh;
        ctx.fillStyle = footprintCellColor(r.delta, mx);
        ctx.fillRect(x, y, colW - 2, rh - 1);
        ctx.fillStyle = "#141a26";
        ctx.fillText(cellText(r), x + 3, y + 11);
        const st = inStack(b, r.price);
        if (st) {
          ctx.strokeStyle = st === "buy" ? "#1b7a45" : "#a3263d";
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x, y, colW - 2, rh - 1);
          ctx.lineWidth = 1;
        }
        if (r.price === b.poc) {
          ctx.strokeStyle = "#141a26";
          ctx.strokeRect(x + 1, y + 1, colW - 4, rh - 3);
        }
      });
      ctx.fillStyle = b.delta >= 0 ? "#2f9e5f" : "#d6455d";
      ctx.fillText(`delta ${Math.round(b.delta)}`, x + 3, h - 4);
    });
    ctx.fillStyle = "#6b7487";
    for (let k = 0; k < nRows; k += 2) ctx.fillText((hi - k * fp.bin).toFixed(0), 2, 4 + k * rh + 11);
  }, [fp]);
  return (
    <div style={{ overflowX: "auto" }}>
      <canvas ref={ref} aria-label="Footprint" style={{ width: "100%" }} />
    </div>
  );
}

export function HeatCanvas({ h }: { h: HeatReply }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv || !h.qty || !h.times || !h.prices) return;
    const s = setup(cv, 240);
    if (!s) return;
    const { ctx, w } = s;
    const mx = heatMax(h) || 1;
    const cw = w / h.times.length,
      ch = 240 / h.prices.length;
    h.qty.forEach((col, ti) =>
      col.forEach((q, pi) => {
        ctx.fillStyle = heatColor(q / mx);
        ctx.fillRect(ti * cw, 240 - (pi + 1) * ch, Math.ceil(cw), Math.ceil(ch));
      }),
    );
  }, [h]);
  return <canvas ref={ref} aria-label="Depth heat map" />;
}
