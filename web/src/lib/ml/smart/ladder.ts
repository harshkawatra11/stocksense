// Port of smart.html ladder() (lines 188-206): each bar a column of short dashes up to its close.
import type { Bar } from "./types";

export const C = {
  up: "#2f9e5f",
  down: "#d6455d",
  upL: "#9fd8b5",
  downL: "#f0a9b5",
  ink: "#141a26",
  blue: "#3b5bdb",
  grid: "#dfe3ea",
};

export interface LadderRect {
  x: number;
  y: number;
  w: number;
  h: number;
  color: string;
}

/** The pure part of ladder(): every dash rectangle and the y of the dashed last-price line. */
export function ladderPlan(
  bars: Bar[] | null | undefined,
  W: number,
  Hh: number,
  step = 3,
): { rects: LadderRect[]; ly: number } | null {
  if (!bars || bars.length < 2) return null;
  const lo = Math.min(...bars.map((b) => b[2]));
  const hi = Math.max(...bars.map((b) => b[1]));
  const pad = (hi - lo) * 0.08 || 1;
  const Y = (v: number) => Hh - 3 - ((v - (lo - pad)) / (hi + pad - (lo - pad))) * (Hh - 6);
  const n = bars.length;
  const w = W / n;
  const base = bars[0][0];
  const rects: LadderRect[] = [];
  bars.forEach((b, i) => {
    const up = b[3] >= base;
    const top = Y(b[3]);
    const color = up ? (b[3] >= b[0] ? C.up : "#5cbf86") : b[3] < b[0] ? C.down : "#e88a9a";
    for (let y = Hh - 2; y > top; y -= step) rects.push({ x: i * w + 0.5, y, w: Math.max(1, w - 1.2), h: 1.1, color });
  });
  return { rects, ly: Y(bars[bars.length - 1][3]) };
}

export function ladder(cv: HTMLCanvasElement, bars: Bar[] | null | undefined, opt: { step?: number } = {}): number | undefined {
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth;
  const Hh = cv.clientHeight;
  if (!W || !Hh) return;
  cv.width = W * dpr;
  cv.height = Hh * dpr;
  const x = cv.getContext("2d");
  if (!x) return;
  x.setTransform(dpr, 0, 0, dpr, 0, 0);
  x.clearRect(0, 0, W, Hh);
  const plan = ladderPlan(bars, W, Hh, opt.step || 3);
  if (!plan) return;
  for (const r of plan.rects) {
    x.fillStyle = r.color;
    x.fillRect(r.x, r.y, r.w, r.h);
  }
  x.strokeStyle = C.ink;
  x.setLineDash([2, 2]);
  x.lineWidth = 1;
  x.beginPath();
  x.moveTo(0, plan.ly);
  x.lineTo(W, plan.ly);
  x.stroke();
  x.setLineDash([]);
  return plan.ly;
}
