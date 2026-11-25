// Port of sector.html horizon code (lines 266-361): the bubble colour ramp, replay values and the SVG layout.
import { esc, pc } from "../format";
import type { HorizonData, HSector } from "./types";

/** Bubble colour: a ramp from pale to deep green (v >= 0) or red (v < 0), saturating at +-3% (sector.html line 267). */
export const COLORS = (v: number | null | undefined): string => {
  const x = v || 0;
  const a = Math.min(1, Math.abs(x) / 3);
  return x >= 0
    ? `rgb(${Math.round(160 - 120 * a)},${Math.round(215 - 55 * a)},${Math.round(175 - 85 * a)})`
    : `rgb(${Math.round(240 - 20 * a)},${Math.round(170 - 100 * a)},${Math.round(180 - 90 * a)})`;
};

/** Minute offset from 09:15 as "HH:MM" (sector.html hm). */
export function hm(m: number): string {
  const t = 555 + m;
  return String(Math.floor(t / 60)).padStart(2, "0") + ":" + String(t % 60).padStart(2, "0");
}

export const nowMinOf = (d: HorizonData): number => (d.snaps.length ? d.snaps[d.snaps.length - 1].m : 0);

/** Sector values now, or as of replay minute `at` (sector.html hValues). */
export function hValues(d: HorizonData, at: number | null): { secs: HSector[]; nv: number | null } {
  const secs = d.sectors.map((s) => ({ ...s }));
  let nv = d.nifty.v;
  if (at != null) {
    const pick = (p: [number, number | null][], m: number): number | null => {
      let v: number | null = null;
      for (const q of p) if (q[0] <= m && q[1] != null) v = q[1];
      return v;
    };
    nv = pick(d.snaps.map((s) => [s.m, s.nifty] as [number, number | null]), at);
    secs.forEach((s) => {
      s.v = pick(s.path, at);
      s.trail = pick(s.path, at - 30);
    });
  }
  return { secs: secs.filter((s) => s.v != null), nv };
}

interface Placed extends HSector {
  x: number;
  y: number;
  r: number;
}

export interface HorizonSvg {
  g: string;
  tails: string;
  bubs: string;
}

export function horizonSvg(d: HorizonData, at: number | null, W: number, H: number): HorizonSvg {
  const { secs: base, nv } = hValues(d, at);
  const secs = base as Placed[];
  const L = 70;
  const R = W - 24;
  const T = 70;
  const B = H - 70;
  const vals = secs.map((s) => s.v as number).concat([nv || 0, 0]);
  const lo = Math.min(...vals) - 0.6;
  const hi = Math.max(...vals) + 0.8;
  const Y = (x: number) => T + ((hi - x) / (hi - lo)) * (B - T);
  const groups = d.groups;
  const gw = (R - L) / groups.length;
  const mx = Math.max(1, ...secs.map((s) => s.size || 0));
  const rad = (s: HSector) => 17 + 24 * Math.sqrt((s.size || mx * 0.05) / mx);
  const split = Math.max(0, Math.min(1, Y(nv || 0) / H));
  let g = `<defs><linearGradient id="bgud" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#dff3e7"/><stop offset="${split}" stop-color="#f4fbf6"/><stop offset="${split}" stop-color="#fdf3f4"/><stop offset="1" stop-color="#fbe2e6"/></linearGradient>
    <filter id="glow"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
    <rect x="0" y="0" width="${W}" height="${H}" fill="url(#bgud)"/>`;
  const step = hi - lo > 8 ? 2 : hi - lo > 4 ? 1 : 0.5;
  for (let x = Math.ceil(lo / step) * step; x <= hi; x += step)
    g += `<line x1="${L}" x2="${R}" y1="${Y(x)}" y2="${Y(x)}" stroke="#00000010"/><text x="${L - 8}" y="${Y(x) + 4}" font-size="10.5" fill="#6b7487" text-anchor="end">${x > 0 ? "+" : ""}${+x.toFixed(1)}%</text>`;
  if (nv != null) {
    g += `<line x1="${L}" x2="${R}" y1="${Y(nv + 1)}" y2="${Y(nv + 1)}" stroke="#22a05a" stroke-dasharray="5 5" opacity=".55"/><text x="${L + 6}" y="${Y(nv + 1) - 5}" font-size="10" font-weight="700" fill="#22a05a" letter-spacing=".8">RUNNING ZONE · 1%+ AHEAD OF NIFTY</text>`;
    g += `<line x1="${L - 40}" x2="${R}" y1="${Y(nv)}" y2="${Y(nv)}" stroke="#2563eb" stroke-width="2.2" filter="url(#glow)"/>
      <g><rect x="6" y="${Y(nv) - 14}" width="92" height="28" rx="7" fill="#2563eb"/><text x="52" y="${Y(nv) - 2}" fill="#fff" font-size="10" font-weight="800" text-anchor="middle" letter-spacing=".5">NIFTY 50</text><text x="52" y="${Y(nv) + 10}" fill="#fff" font-size="11" font-weight="700" text-anchor="middle">${pc(nv)}</text></g>
      <text x="8" y="${Y(nv) - 20}" font-size="9.5" font-weight="700" fill="#22a05a">▲ BEATING</text><text x="8" y="${Y(nv) + 26}" font-size="9.5" font-weight="700" fill="#dc3f54">▼ TRAILING</text>`;
  }
  groups.forEach((grp, i) => {
    const gs = secs.filter((s) => s.group === grp);
    const avg = gs.length ? gs.reduce((a, s) => a + (s.v as number), 0) / gs.length : null;
    g += `<text x="${L + gw * i + gw / 2}" y="${H - 34}" font-size="10.5" font-weight="800" fill="#4b5565" text-anchor="middle" letter-spacing="1">${esc(grp.toUpperCase())}</text>
      <text x="${L + gw * i + gw / 2}" y="${H - 20}" font-size="11" font-weight="600" text-anchor="middle" fill="${(avg ?? 0) >= 0 ? "#22a05a" : "#dc3f54"}">${avg == null ? "" : pc(avg)}</text>`;
    gs.sort((a, b) => (b.v as number) - (a.v as number)).forEach((s, j) => {
      s.x = L + gw * i + (gw * (j + 1)) / (gs.length + 1) + (j % 2 ? 8 : -8);
      s.y = Y(s.v as number);
      s.r = rad(s);
    });
  });
  // keep bubbles from sitting on each other: nudge sideways inside their column, height stays the truth
  for (let it = 0; it < 80; it++) {
    let moved = false;
    for (let i = 0; i < secs.length; i++)
      for (let j = i + 1; j < secs.length; j++) {
        const A = secs[i];
        const Bb = secs[j];
        const dx = Bb.x - A.x;
        const dy = Bb.y - A.y;
        const dist = Math.hypot(dx, dy);
        const need = A.r + Bb.r + 4;
        if (dist < need) {
          const push = ((need - dist) / 2) * (dx >= 0 ? 1 : -1) || 1;
          A.x -= push;
          Bb.x += push;
          moved = true;
        }
      }
    secs.forEach((s) => {
      const i = groups.indexOf(s.group);
      const lo2 = L + gw * i + s.r + 2;
      const hi2 = L + gw * (i + 1) - s.r - 2;
      s.x = Math.max(lo2, Math.min(hi2, s.x));
    });
    if (!moved) break;
  }
  let tails = "";
  let bubs = "";
  secs.forEach((s) => {
    const v = s.v as number;
    const run = nv != null && v - nv >= 1;
    if (s.trail != null && Math.abs(Y(s.trail) - s.y) > 2)
      tails += `<line x1="${s.x}" x2="${s.x}" y1="${Y(s.trail)}" y2="${s.y}" stroke="${v >= s.trail ? "#22a05a" : "#dc3f54"}" stroke-opacity=".28" stroke-width="${Math.max(3, s.r * 0.5)}" stroke-linecap="round"/>`;
    bubs += `<g class="bub" data-s="${esc(s.name)}" style="transform:translate(${s.x}px,${s.y}px)">
      ${run ? `<circle class="run" r="${s.r + 6}" fill="none" stroke="#22a05a" stroke-width="2"/><circle r="${s.r + 10}" fill="none" stroke="#22a05a" stroke-opacity=".25"/>` : ""}
      <circle class="b" r="${s.r}" fill="${COLORS(v)}" stroke="${v >= 0 ? "#1f8a4d" : "#c8344a"}" stroke-width="1.2" fill-opacity=".92"/>
      ${run ? `<g transform="translate(0,${-s.r - 16})"><rect x="-30" y="-8" width="60" height="15" rx="4" fill="#22a05a"/><text y="3" font-size="9" font-weight="800" fill="#fff" text-anchor="middle" letter-spacing=".6">RUNNING</text></g>` : ""}
      <text y="-1" font-size="${s.r > 28 ? 12 : 10.5}" font-weight="800" fill="#0f1a12" text-anchor="middle">${esc(s.name)}</text>
      <text y="13" font-size="${s.r > 28 ? 11.5 : 10}" font-weight="600" fill="#0f1a12" text-anchor="middle" data-k="hs:${esc(s.name)}" data-v="${v}">${pc(v)}</text></g>`;
  });
  return { g, tails, bubs };
}
