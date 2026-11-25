// Port of smart.html's stage drawing (lines 222-236, 285-445, 509-539): geometry, the big canvas
// (lines, waterfall, brackets, connectors, zero line), the bead strip and the travelling lights.
// All functions are pure over explicit arguments; the DOM glue lives in components/smart/stageEngine.ts.
import { esc, pc } from "../format";
import { C } from "./ladder";
import type { Hover, PathPt, SmartIndex, SmartLive, SmartSector, SmartStock } from "./types";

type Ctx = CanvasRenderingContext2D;

/** Minutes since 09:15 IST, clamped to 1..375 (smart.html line 186). */
export const nowMin = (nowMs: number = Date.now()): number =>
  Math.max(1, Math.min(375, Math.floor(((nowMs / 1000 + 19800) % 86400) / 60) - 555));

export const yScale = (v: number): number => Math.sign(v) * Math.log1p(Math.abs(v) / 0.12);

export interface Geometry {
  W: number;
  Hs: number;
  L: number;
  CH: number;
  T: number;
  B: number;
  chartR: number;
  wfL: number;
  wfR: number;
  tilesL: number;
  tilesW: number;
  beadsH: number;
  Y: (v: number) => number;
  X: (m: number) => number;
  lim: number;
  hiV: number;
  loV: number;
  nm: number;
}

export function geometry(D: SmartLive, W: number, Hs: number, nm: number): Geometry {
  const tilesW = Math.round(W * 0.37);
  const wfW = Math.round(W * 0.065);
  const gap = Math.round(W * 0.07);
  const chartR = W - tilesW - gap - wfW;
  const beadsH = 84;
  const CH = 118;
  const L = CH + 34;
  const T = beadsH + 16;
  const B = Hs - 18;
  const vals = D.stocks.map((s) => s.from_open).sort((a, b) => a - b);
  const q = (f: number) => vals[Math.floor(f * (vals.length - 1))];
  const hiV = Math.max(0.4, Math.min(8, q(0.985)));
  const loV = Math.min(-0.4, Math.max(-8, q(0.015)));
  const tTop = yScale(hiV);
  const tBot = yScale(loV);
  const lim = Math.max(hiV, -loV);
  const Y = (v: number) => T + ((tTop - yScale(Math.max(loV, Math.min(hiV, v)))) / (tTop - tBot)) * (B - T);
  const X = (m: number) => L + (m / nm) * (chartR - L);
  return { W, Hs, L, CH, T, B, chartR, wfL: chartR, wfR: chartR + wfW, tilesL: W - tilesW, tilesW, beadsH, Y, X, lim, hiV, loV, nm };
}

// ---- smooth moves between refreshes ----
export const ease = (t: number): number => 1 - Math.pow(1 - t, 3);
export interface Anim {
  t0: number;
  from: Record<string, [number, number]>;
}
export function tween(anim: Anim, now: number): { FO: (s: SmartStock) => number; DP: (s: SmartStock) => number } {
  const k = () => Math.min(1, (now - anim.t0) / 650);
  return {
    FO: (s) => {
      const f = anim.from[s.sym];
      return f ? f[0] + (s.from_open - f[0]) * ease(k()) : s.from_open;
    },
    DP: (s) => {
      const f = anim.from[s.sym];
      return f ? f[1] + (s.disp - f[1]) * ease(k()) : s.disp;
    },
  };
}
export const animDone = (anim: Anim, now: number): boolean => (now - anim.t0) / 650 >= 1;

export function valAt(p: PathPt[], m: number): number {
  let v = 0;
  for (const q of p) {
    if (q[0] <= m) v = q[1];
    else break;
  }
  return v;
}

export function smoothPath(x: Ctx, pts: [number, number][]): void {
  if (pts.length < 2) return;
  x.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length - 1; i++) {
    const mx = (pts[i][0] + pts[i + 1][0]) / 2;
    const my = (pts[i][1] + pts[i + 1][1]) / 2;
    x.quadraticCurveTo(pts[i][0], pts[i][1], mx, my);
  }
  const last = pts[pts.length - 1];
  x.lineTo(last[0], last[1]);
}

export interface Row {
  s: SmartStock;
  y: number;
  h: number;
}
export interface Conn {
  sym: string;
  side: "bull" | "bear";
  rank: number;
  sx: number;
  sy: number;
  cx: number;
  tx: number;
  ty: number;
}
export interface DrawEnv {
  D: SmartLive;
  BY: Record<string, SmartStock>;
  g: Geometry;
  H: Hover;
  FO: (s: SmartStock) => number;
  DP: (s: SmartStock) => number;
  /** Vertical centre of a stock's tile inside the stage, or null when it has no tile. */
  tileMid: (sym: string) => number | null;
  now: Date;
}

export const matchesH =
  (H: Hover) =>
  (s: SmartStock): boolean =>
    !H || (H.type === "stock" ? s.sym === H.key : s.sector === H.key);

export interface DrawResult {
  PTS: Record<string, [number, number][]>;
  ROWS: Row[];
  CONN: Conn[];
}

/** The big canvas: lines, waterfall, connectors, zero line (smart.html draw()). */
export function drawStage(x: Ctx, env: DrawEnv): DrawResult {
  const { D, BY, g, H, FO, DP } = env;
  const matches = matchesH(H);
  const y0 = g.Y(0);
  x.font = "10px JetBrains Mono";
  x.fillStyle = "#6b7385";
  x.strokeStyle = C.grid;
  x.lineWidth = 1;
  [5, 3, 2, 1, 0.5, 0.2, 0, -0.2, -0.5, -0.75, -1, -1.5, -2, -3, -5]
    .filter((v) => v <= g.hiV + 1e-9 && v >= g.loV - 1e-9)
    .forEach((v) => {
      const y = g.Y(v);
      x.beginPath();
      x.moveTo(g.L, y);
      x.lineTo(g.chartR, y);
      x.stroke();
      x.fillText((v > 0 ? "+" : "") + v, g.CH + 6, y + 3);
    });
  for (let m = 45; m < g.nm; m += 60) {
    const xx = g.X(m);
    x.beginPath();
    x.moveTo(xx, g.T);
    x.lineTo(xx, g.B);
    x.stroke();
  }
  (
    [
      ["09:15", 0],
      ["10:00", 45],
      ["11:00", 105],
      ["12:00", 165],
      ["13:00", 225],
      ["14:00", 285],
      ["15:00", 345],
    ] as [string, number][]
  ).forEach(([t, m]) => {
    if (m < g.nm - 20) x.fillText(t, g.X(m) - 14, g.Hs - 4);
  });
  const tnow = env.now.toLocaleTimeString("en-GB", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit" });
  x.fillStyle = C.ink;
  x.fillText(tnow, g.chartR - 16, g.Hs - 4);
  // side bar: green above zero, red below
  x.fillStyle = C.up;
  x.fillRect(g.L - 6, g.T, 2, y0 - g.T);
  x.fillStyle = C.down;
  x.fillRect(g.L - 6, y0, 2, g.B - y0);
  // stock lines
  const PTS: Record<string, [number, number][]> = {};
  const ordered = D.stocks.slice().sort((a, b) => (matches(a) ? 1 : 0) - (matches(b) ? 1 : 0));
  for (const s of ordered) {
    const p = s.path || [];
    if (p.length < 2) continue;
    const pts: [number, number][] = [];
    for (let i = 0; i < p.length; i += 3) pts.push([g.X(p[i][0]), g.Y(p[i][1])]);
    pts.push([g.X(p[p.length - 1][0]), g.Y(FO(s))]);
    PTS[s.sym] = pts;
    const strong = s.rank <= 20;
    const up = FO(s) >= 0;
    const m = matches(s);
    if (H && !m) {
      x.strokeStyle = up ? "rgba(47,158,95,.08)" : "rgba(214,69,93,.08)";
      x.lineWidth = 1;
    } else if (H && H.type === "stock") {
      x.strokeStyle = C.ink;
      x.lineWidth = 2.6;
    } else if (H) {
      x.strokeStyle = up ? C.up : C.down;
      x.lineWidth = 1.8;
    } else {
      x.strokeStyle = up ? (strong ? "rgba(47,158,95,.9)" : "rgba(47,158,95,.28)") : strong ? "rgba(214,69,93,.9)" : "rgba(214,69,93,.28)";
      x.lineWidth = strong ? 1.4 : 1;
    }
    x.beginPath();
    smoothPath(x, pts);
    x.stroke();
    if ((strong && !H) || (H && m)) {
      x.fillStyle = H && H.type === "stock" ? C.ink : up ? C.up : C.down;
      ((s.read || {}).ev || []).forEach(([mm]) => {
        if (mm <= g.nm) {
          const yy = g.Y(valAt(p, mm));
          x.beginPath();
          x.arc(g.X(mm), yy, 2.4, 0, 7);
          x.fill();
        }
      });
    }
  }
  // indices, dashed
  x.setLineDash([5, 4]);
  x.lineWidth = 1.5;
  for (const k of ["SENSEX", "NIFTY", "BANKNIFTY"] as const) {
    const i: SmartIndex | undefined = D.indices[k];
    if (!i || !i.path || i.path.length < 2) continue;
    const pts: [number, number][] = [];
    for (let j = 0; j < i.path.length; j += 3) pts.push([g.X(i.path[j][0]), g.Y(i.path[j][1])]);
    x.strokeStyle = H ? "rgba(59,91,219,.25)" : C.blue;
    x.beginPath();
    smoothPath(x, pts);
    x.stroke();
    const lp = pts[pts.length - 1];
    x.setLineDash([]);
    x.font = "700 9.5px JetBrains Mono";
    const tw = x.measureText(k).width;
    x.fillStyle = "#fff";
    x.fillRect(lp[0] - tw - 10, lp[1] - 7, tw + 6, 13);
    x.fillStyle = C.blue;
    x.fillText(k, lp[0] - tw - 7, lp[1] + 3);
    x.setLineDash([5, 4]);
  }
  x.setLineDash([]);
  // waterfall: every stock now, strongest at the top; positives fill down to the zero line
  const bull = D.bull.map((s) => BY[s]).filter(Boolean);
  const bear = D.bear.map((s) => BY[s]).filter(Boolean).reverse();
  const rows: Row[] = [];
  const wfW = g.wfR - g.wfL - 4;
  const mxd = Math.max(1, ...D.stocks.map((s) => Math.abs(DP(s))));
  bull.forEach((s, i) => rows.push({ s, y: g.T + ((i + 0.5) * (y0 - g.T)) / bull.length, h: (y0 - g.T) / bull.length }));
  bear.forEach((s, i) => rows.push({ s, y: y0 + ((i + 0.5) * (g.B - y0)) / bear.length, h: (g.B - y0) / bear.length }));
  for (const r of rows) {
    const s = r.s;
    const up = s.side === "bull";
    const strong = s.rank <= 20;
    const m = matches(s);
    const len = Math.max(3, (Math.abs(DP(s)) / mxd) * wfW);
    x.fillStyle =
      H && H.type === "stock" && m
        ? C.ink
        : H && !m
          ? up
            ? "rgba(47,158,95,.12)"
            : "rgba(214,69,93,.12)"
          : up
            ? strong
              ? C.up
              : C.upL
            : strong
              ? C.down
              : C.downL;
    x.fillRect(g.wfL + 2, r.y - Math.max(0.6, r.h * 0.35), len, Math.max(1.2, r.h * 0.7));
  }
  // brackets round the 20 strongest and the 20 weakest
  const brk = (a: Row, b: Row) => {
    const yA = a.y - a.h / 2 - 2;
    const yB = b.y + b.h / 2 + 2;
    x.strokeStyle = C.ink;
    x.lineWidth = 1.5;
    (
      [
        [g.wfL, 1],
        [g.wfR, -1],
      ] as [number, number][]
    ).forEach(([xx, d]) => {
      x.beginPath();
      x.moveTo(xx + d * 6, yA);
      x.lineTo(xx, yA);
      x.lineTo(xx, yB);
      x.lineTo(xx + d * 6, yB);
      x.stroke();
    });
  };
  if (bull.length) brk(rows[0], rows[Math.min(19, bull.length - 1)]);
  if (bear.length) brk(rows[rows.length - Math.min(20, bear.length)], rows[rows.length - 1]);
  x.font = "9.5px JetBrains Mono";
  x.fillStyle = "#6b7385";
  x.fillText(String(D.stocks.length), g.wfL - 24, y0 - 3);
  x.fillText(String(bear.length), g.wfR + 3, y0 - 3);
  // connectors from each of the 40 bars to its tile
  const CONN: Conn[] = [];
  for (const r of rows) {
    if (r.s.rank > 20) continue;
    const ty = env.tileMid(r.s.sym);
    if (ty == null) continue;
    const tx = g.tilesL - 2;
    const m = matches(r.s);
    const sel = !!H && H.type === "stock" && m;
    if (H && !m) continue;
    x.strokeStyle = sel ? C.ink : r.s.side === "bull" ? C.up : C.down;
    x.lineWidth = sel ? 2 : 1;
    x.setLineDash(sel ? [] : [4, 3]);
    const sx = g.wfR;
    const cx = (sx + tx) / 2;
    x.beginPath();
    x.moveTo(sx, r.y);
    x.bezierCurveTo(cx, r.y, cx, ty, tx, ty);
    x.stroke();
    CONN.push({ sym: r.s.sym, side: r.s.side, rank: r.s.rank, sx, sy: r.y, cx, tx, ty });
  }
  x.setLineDash([]);
  // the zero line: across the chart, then an S-curve to the gap between the bullish and bearish tiles
  const divY = g.Hs / 2;
  x.strokeStyle = "#2c3e8f";
  x.lineWidth = 1.6;
  x.beginPath();
  x.moveTo(g.L, y0);
  x.lineTo(g.wfR, y0);
  const cx = (g.wfR + g.tilesL) / 2;
  x.bezierCurveTo(cx, y0, cx, divY, g.tilesL, divY);
  x.lineTo(g.W, divY);
  x.stroke();
  (
    [
      [g.X(Math.min(g.nm * 0.55, 200)), y0],
      [g.wfL, y0],
      [g.wfR, y0],
      [g.tilesL, divY],
    ] as [number, number][]
  ).forEach(([a, b]) => {
    x.fillStyle = "#2c3e8f";
    x.beginPath();
    x.arc(a, b, 3, 0, 7);
    x.fill();
  });
  // the hovered stock's own connector, even when it is not in the top 20
  if (H && H.type === "stock") {
    const r = rows.find((q) => q.s.sym === H.key);
    if (r) {
      x.fillStyle = C.ink;
      x.fillRect(g.wfL + 2, r.y - 1.5, Math.max(4, (Math.abs(DP(r.s)) / mxd) * wfW), 3);
    }
  }
  return { PTS, ROWS: rows, CONN };
}

// ---- beads ----
export interface Bead {
  s: SmartStock;
  x: number;
  y: number;
}
export const BEAD_LIM = 4;

/** x of a % from the open on the bead strip of width W (smart.html beads() X): +-4% map to the two edges. */
export function beadScale(W: number): (v: number) => number {
  const L = 20;
  const R = W - 22;
  return (v) => {
    const c = Math.max(-BEAD_LIM, Math.min(BEAD_LIM, v));
    return L + ((c + BEAD_LIM) / (2 * BEAD_LIM)) * (R - L);
  };
}

export interface BeadsResult {
  BEADS: Bead[];
  lead: SmartSector | undefined;
  lag: SmartSector | undefined;
  leadX: number;
  lagX: number;
}

export function drawBeads(x: Ctx, env: Omit<DrawEnv, "tileMid" | "now" | "DP">): BeadsResult {
  const { D, BY, g, H, FO } = env;
  const matches = matchesH(H);
  const W = g.wfR;
  const Hb = g.beadsH;
  const R = W - 22;
  const ax = Hb - 26;
  const X = beadScale(W);
  x.strokeStyle = "#c5ccd6";
  x.lineWidth = 1;
  x.beginPath();
  x.moveTo(20, ax);
  x.lineTo(R, ax);
  x.stroke();
  x.beginPath();
  x.moveTo(X(0), 4);
  x.lineTo(X(0), Hb);
  x.stroke();
  x.fillStyle = "#6b7385";
  x.font = "9px JetBrains Mono";
  x.fillText("-4%", 0, ax + 3);
  x.fillText("+4%", R + 3, ax + 3);
  for (let v = -3; v <= 3; v++)
    if (v) {
      x.beginPath();
      x.moveTo(X(v), ax - 3);
      x.lineTo(X(v), ax + 3);
      x.stroke();
    }
  const BEADS: Bead[] = [];
  const stack: Record<number, number> = {};
  const list = D.stocks.slice().sort((a, b) => Math.abs(FO(a)) - Math.abs(FO(b)));
  for (const s of list) {
    const bx = Math.round(X(FO(s)) / 6) * 6;
    const k = (stack[bx] = (stack[bx] || 0) + 1);
    const up = FO(s) >= 0;
    const by = up ? ax - 6 - (k - 1) * 6 : ax + 6 + (k - 1) * 6;
    if (by < 2 || by > Hb - 2) continue;
    const m = matches(s);
    if (s.trail != null && Math.abs(s.trail - FO(s)) > 0.15 && (!H || m)) {
      const tx = X(s.trail);
      const gr = x.createLinearGradient(tx, 0, bx, 0);
      gr.addColorStop(0, up ? "rgba(47,158,95,0)" : "rgba(214,69,93,0)");
      gr.addColorStop(1, up ? "rgba(47,158,95,.35)" : "rgba(214,69,93,.35)");
      x.strokeStyle = gr;
      x.lineWidth = 3;
      x.beginPath();
      x.moveTo(tx, by);
      x.lineTo(bx, by);
      x.stroke();
    }
    x.fillStyle = H && !m ? (up ? "rgba(47,158,95,.15)" : "rgba(214,69,93,.15)") : H && H.type === "stock" ? C.ink : up ? C.up : C.down;
    x.beginPath();
    x.arc(bx, by, H && H.type === "stock" && m ? 4.5 : 2.6, 0, 7);
    x.fill();
    BEADS.push({ s, x: bx, y: by });
    if (H && H.type === "stock" && m) {
      x.font = "700 10px JetBrains Mono";
      x.fillStyle = C.ink;
      x.fillText(`${s.sym} ${pc(FO(s))}`, Math.min(R - 90, bx - 30), up ? by - 8 : by + 16);
    }
  }
  // extremes circled, indices as dots, leading / lagging sector as diamonds
  const ext = [BY[D.bull[0]], BY[D.bear[0]]].filter(Boolean);
  ext.forEach((s) => {
    const b = BEADS.find((q) => q.s === s);
    if (b) {
      x.strokeStyle = s.side === "bull" ? C.up : C.down;
      x.lineWidth = 2;
      x.beginPath();
      x.arc(b.x, b.y, 6, 0, 7);
      x.stroke();
    }
  });
  (["NIFTY", "BANKNIFTY", "SENSEX"] as const).forEach((k) => {
    const i = D.indices[k];
    if (i) {
      x.fillStyle = C.ink;
      x.beginPath();
      x.arc(X(i.from_open), ax, 3, 0, 7);
      x.fill();
    }
  });
  const real = D.sectors.filter((s) => s.name !== "OTHERS" && s.n >= 3);
  const lead = real[0];
  const lag = real[real.length - 1];
  const dia = (s: SmartSector | undefined) => {
    if (!s) return 0;
    const xx = X(s.pct);
    x.fillStyle = C.ink;
    x.beginPath();
    x.moveTo(xx, ax - 5);
    x.lineTo(xx + 5, ax);
    x.lineTo(xx, ax + 5);
    x.lineTo(xx - 5, ax);
    x.fill();
    x.setLineDash([2, 2]);
    x.strokeStyle = "#9aa2b1";
    x.beginPath();
    x.moveTo(xx, ax - 6);
    x.lineTo(xx, 14);
    x.stroke();
    x.setLineDash([]);
    return xx;
  };
  const lagX = dia(lag);
  const leadX = dia(lead);
  return { BEADS, lead, lag, leadX, lagX };
}

/** Text of the leading / lagging sector labels above the bead strip. */
export function sectorLabelHTML(kind: "lead" | "lag", s: SmartSector): string {
  const cls = s.pct >= 0 ? "up" : "down";
  return kind === "lag"
    ? `LAGGING SECTOR <b>${esc(s.name)}</b> <span class="${cls}">${pc(s.pct)}</span> <span class="down">↘</span>`
    : `<span class="up">↗</span> LEADING SECTOR <b>${esc(s.name)}</b> <span class="${cls}">${pc(s.pct)}</span> `;
}
export const labelLeft = (xx: number, W: number): number => Math.max(0, Math.min(W - 250, xx - 120));

// ---- hover hit-testing (smart.html lines 465-478) ----
export function nearestLine(PTS: Record<string, [number, number][]>, mx: number, my: number): string | null {
  let best: string | null = null;
  let bd = 9;
  for (const [sym, pts] of Object.entries(PTS))
    for (const p of pts) {
      if (Math.abs(p[0] - mx) > 7) continue;
      const d = Math.abs(p[1] - my);
      if (d < bd) {
        bd = d;
        best = sym;
      }
    }
  return best;
}

export function hitSym(
  g: Geometry,
  mx: number,
  my: number,
  BEADS: Bead[],
  ROWS: Row[],
  PTS: Record<string, [number, number][]>,
): string | null {
  if (my < g.beadsH && mx < g.wfR) {
    let bd = 8;
    let sym: string | null = null;
    BEADS.forEach((b) => {
      const d = Math.hypot(b.x - mx, b.y - my);
      if (d < bd) {
        bd = d;
        sym = b.s.sym;
      }
    });
    return sym;
  }
  if (mx >= g.wfL && mx <= g.wfR) {
    const r = ROWS.find((q) => Math.abs(q.y - my) <= Math.max(2, q.h / 2));
    return r ? r.s.sym : null;
  }
  if (mx < g.chartR && mx > g.L) return nearestLine(PTS, mx, my);
  return null;
}

// ---- the travelling lights (fxLoop) ----
export function fxPoint(pts: [number, number][], f: number): [number, number] {
  const i = Math.min(pts.length - 2, Math.max(0, Math.floor(f)));
  const t = Math.min(1, Math.max(0, f - i));
  return [pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t];
}
export function bez(c: Conn, t: number): [number, number] {
  const u = 1 - t;
  return [
    u * u * u * c.sx + 3 * u * u * t * c.cx + 3 * u * t * t * c.cx + t * t * t * c.tx,
    u * u * u * c.sy + 3 * u * u * t * c.sy + 3 * u * t * t * c.ty + t * t * t * c.ty,
  ];
}
/** Each light spends 4 time units on the thread, then 1 on the connector; ranks are staggered. */
export const fxPhase = (T: number, rank: number): number => (T / 4 + rank * 0.05) % 1.25;

export function drawFx(
  x: Ctx,
  env: { CONN: Conn[]; PTS: Record<string, [number, number][]>; BY: Record<string, SmartStock>; H: Hover; t0: number; now: number },
): void {
  const matches = matchesH(env.H);
  const T = ((env.now - env.t0) / 1000) * 0.18;
  const dot = (p: [number, number], rgb: string) => {
    x.fillStyle = `rgba(${rgb},.18)`;
    x.beginPath();
    x.arc(p[0], p[1], 5, 0, 7);
    x.fill();
    x.fillStyle = `rgb(${rgb})`;
    x.beginPath();
    x.arc(p[0], p[1], 2.2, 0, 7);
    x.fill();
  };
  for (const c of env.CONN) {
    const s = env.BY[c.sym];
    const pts = env.PTS[c.sym];
    if (!s || !pts || pts.length < 2 || (env.H && !matches(s))) continue;
    const rgb = c.side === "bull" ? "22,163,74" : "220,38,38";
    const f = fxPhase(T, c.rank);
    dot(f < 1 ? fxPoint(pts, f * (pts.length - 1)) : bez(c, (f - 1) / 0.25), rgb);
  }
}

/** Merge a fresh /api/sm/live row set into the path cache (smart.html lines 496-508). */
export function mergePaths(
  stocks: SmartStock[],
  PATHS: Record<string, PathPt[]>,
  full: boolean,
  m: number,
): void {
  if (full) for (const s of stocks) PATHS[s.sym] = s.path || [];
  stocks.forEach((s) => {
    const p = (PATHS[s.sym] = PATHS[s.sym] || []);
    const last = p[p.length - 1];
    if (p.length && last[0] >= m) p[p.length - 1] = [last[0], s.from_open];
    else p.push([m, s.from_open]);
    s.path = p;
  });
}
