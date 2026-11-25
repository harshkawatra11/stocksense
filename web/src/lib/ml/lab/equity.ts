// Port of lab.html draw() (equity canvas with CASH bands and hover) and its onmousemove index maths.
// Ported from market-lab web/lab.html (MIT, ba3f771); changes: typed, geometry split from drawing.
import type { CurvePoint } from "./types";

export interface Band {
  /** First index of the cash run. */
  from: number;
  /** Index where the run ends: the first filter-on bar after it, or the last bar. */
  to: number;
}

/** Cash bands from the market-filter flags (1 = on, 0 = cash): runs of 0, each ending at the next 1 or the last index.
 *  [1,1,0,0,1] gives one band from 2 to 4. */
export function cashBands(flags: number[]): Band[] {
  const n = flags.length;
  const out: Band[] = [];
  let s0: number | null = null;
  flags.forEach((f, i) => {
    if (!f && s0 == null) s0 = i;
    if ((f || i === n - 1) && s0 != null) {
      out.push({ from: s0, to: f ? i : n - 1 });
      s0 = null;
    }
  });
  return out;
}

export interface EquityData {
  lines: [CurvePoint[], CurvePoint[], CurvePoint[]];
  regime: [string, number][];
  t0: number;
}

export const PAD = { L: 40, R: 8, T: 8, B: 20 };

/** Index under the pointer, clamped to the series (lab.html onmousemove). */
export function hoverIndex(mx: number, W: number, n: number): number {
  return Math.max(0, Math.min(n - 1, Math.round(((mx - PAD.L) / (W - PAD.L - PAD.R)) * (n - 1))));
}

/** Number of points revealed `elapsed` ms after the start of the 1.5 s draw-in animation. */
export const revealed = (n: number, elapsed: number): number => Math.max(2, Math.floor(n * Math.min(1, elapsed / 1500)));

export function tooltipText(d: EquityData, i: number): string {
  const v = (j: number) => d.lines[j][Math.min(i, d.lines[j].length - 1)][1];
  const cash = d.regime[i] && !d.regime[i][1];
  return `${d.lines[0][i][0]} · system ${v(0).toFixed(0)} · universe ${v(1).toFixed(0)} · Nifty ${v(2).toFixed(0)}${cash ? " · filter off" : ""}`;
}

/** Draws one frame; returns the [Lp, R, W, n] geometry the hover maths needs, or null when there is nothing to draw. */
export function drawEquity(
  cv: HTMLCanvasElement,
  W: number,
  H: number,
  dpr: number,
  d: EquityData | null,
  hover: number | null,
  now: number,
): [number, number, number, number] | null {
  if (W && (cv.width !== W * dpr || cv.height !== H * dpr)) {
    cv.width = W * dpr;
    cv.height = H * dpr;
  }
  const x = cv.getContext("2d");
  if (!x) return null;
  x.setTransform(dpr, 0, 0, dpr, 0, 0);
  x.clearRect(0, 0, W, H);
  if (!d || d.lines[0].length <= 1) return null;
  const { L: Lp, R, T, B } = PAD;
  const all = d.lines.flat().map((p) => p[1]);
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const n = d.lines[0].length;
  const X = (i: number) => Lp + ((W - Lp - R) * i) / (n - 1);
  const Y = (v: number) => T + (H - T - B) * (1 - (v - lo) / (hi - lo || 1));
  for (const b of cashBands(d.regime.map((r) => r[1]))) {
    const w0 = X(b.to) - X(b.from);
    x.fillStyle = "#f1f2f6";
    x.fillRect(X(b.from), T, w0, H - T - B);
    if (w0 > 40) {
      x.fillStyle = "#a1a8b6";
      x.font = "600 9.5px Inter";
      x.fillText("CASH", X(b.from) + 4, T + 11);
    }
  }
  x.font = "10px JetBrains Mono";
  x.lineWidth = 1;
  for (let k = 0; k <= 4; k++) {
    const v = lo + ((hi - lo) * k) / 4;
    const y = Y(v);
    x.strokeStyle = "rgba(200,206,216,.5)";
    x.beginPath();
    x.moveTo(Lp, y);
    x.lineTo(W - R, y);
    x.stroke();
    x.fillStyle = "#8a93a6";
    x.fillText(v.toFixed(0), 4, y + 3);
  }
  let ly = "";
  d.lines[0].forEach((p, i) => {
    const y = p[0].slice(0, 4);
    if (y !== ly && i) {
      x.fillStyle = "#8a93a6";
      x.fillText(y, X(i) - 12, H - 5);
    }
    ly = y;
  });
  const k = revealed(n, now - d.t0);
  const series: [CurvePoint[], string, number][] = [
    [d.lines[2], "#d97706", 1.3],
    [d.lines[1], "#9aa3b2", 1.3],
    [d.lines[0], "#2563eb", 2.2],
  ];
  for (const [c, col, lw] of series) {
    x.beginPath();
    for (let i = 0; i < Math.min(k, c.length); i++) {
      if (i) x.lineTo(X(i), Y(c[i][1]));
      else x.moveTo(X(i), Y(c[i][1]));
    }
    x.strokeStyle = col;
    x.lineWidth = lw;
    x.stroke();
  }
  const j = Math.min(k, n) - 1;
  const e = d.lines[0][j];
  x.fillStyle = "#2563eb";
  x.beginPath();
  x.arc(X(j), Y(e[1]), 3.5, 0, 7);
  x.fill();
  if (hover != null && hover < k) {
    x.strokeStyle = "#141a26";
    x.lineWidth = 1;
    x.beginPath();
    x.moveTo(X(hover), T);
    x.lineTo(X(hover), H - B);
    x.stroke();
  }
  return [Lp, R, W, n];
}
