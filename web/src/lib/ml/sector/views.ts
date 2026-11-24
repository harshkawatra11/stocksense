// Port of sector.html's HTML-string views (lines 362-560): horizon side panel, the sector hub view, the stock view,
// the bottom line chart and the tooltips. Inline onclick="go(...)" became data-gv / data-ga attributes read by one
// delegated handler in components/sector/SectorSection.tsx. esc() stays on every interpolated string.
import { esc, pc, cls, rs } from "../format";
import { COLORS } from "./horizon";
import { RL, type Range } from "./route";
import type { HorizonData, NewsItem, RelChart, SectorData, SGroup, SRow, StockData } from "./types";

export const goAttr = (view: string, arg?: string): string => `data-gv="${esc(view)}" data-ga="${esc(arg ?? "")}"`;

const IST = (d: string) => new Date(d).toLocaleDateString("en-IN", { timeZone: "Asia/Kolkata", day: "numeric" });
const dmy = (d: string, o: Intl.DateTimeFormatOptions) => new Date(d).toLocaleDateString("en-IN", { timeZone: "Asia/Kolkata", ...o });

export function newsItem(h: NewsItem, sec?: string): string {
  const d = new Date(h.time);
  return `<div class="newsi"><div class="d"><b>${IST(h.time)}</b>${d.toLocaleDateString("en-IN", { timeZone: "Asia/Kolkata", month: "short" })}</div><div><span class="tg ${esc(h.tag[0])}">${esc(h.tag)}</span>
    <div style="font-weight:600;margin-top:3px"><a href="${esc(h.link)}" target="_blank" rel="noopener">${esc(h.title)}</a></div><div class="muted" style="font-size:11px">${esc(h.source)}</div>
    ${sec && h.since != null ? `<span class="chipx">${esc(sec)} since <b class="${cls(h.since)}">${pc(h.since)}</b></span>` : ""}</div></div>`;
}

// ---- bottom: a line with headline pins (sector.html lineChart) ----
export function lineChartHTML(title: string, sub: string, ch: RelChart | null, W0: number): string {
  if (!ch)
    return `<div class="btitle">${title}</div><div class="muted" style="margin-top:20px">The price history loads in the background on the first start (a few minutes)…</div>`;
  const W = W0 || 900;
  const H = 118;
  const L = 6;
  const R = W - 46;
  const T = 26;
  const B = H - 16;
  const all = [...ch.a, ...(ch.b || [])].filter((x): x is number => x != null);
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const X = (i: number) => L + (i / (ch.dates.length - 1)) * (R - L);
  const Y = (v: number) => T + ((hi - v) / (hi - lo || 1)) * (B - T);
  const pts = (arr: (number | null)[]) => arr.map((v, i) => (v == null ? null : `${X(i)},${Y(v)}`)).filter(Boolean).join(" ");
  const aLast = ch.a[ch.a.length - 1] as number;
  let g = `<line x1="${L}" x2="${R}" y1="${Y(100)}" y2="${Y(100)}" stroke="#00000018" stroke-dasharray="3 4"/>`;
  if (ch.b) {
    const bLast = ch.b[ch.b.length - 1] as number;
    const poly = ch.a
      .map((v, i) => `${X(i)},${Y(v as number)}`)
      .concat(ch.b.map((v, i) => `${X(i)},${Y(v as number)}`).reverse())
      .join(" ");
    g += `<polygon points="${poly}" fill="${aLast >= bLast ? "rgba(34,160,90,.13)" : "rgba(220,63,84,.13)"}"/>`;
    g += `<polyline fill="none" stroke="#6b7487" stroke-width="1.5" points="${pts(ch.b)}"/>`;
  }
  g += `<polyline fill="none" stroke="${ch.b ? "#2563eb" : "#2b3445"}" stroke-width="1.8" points="${pts(ch.a)}" stroke-dasharray="2000" stroke-dashoffset="2000"><animate attributeName="stroke-dashoffset" to="0" dur="1.2s" fill="freeze"/></polyline>`;
  g += `<text x="${R + 4}" y="${Y(aLast) + 4}" font-size="10" font-weight="700" fill="${ch.b ? "#2563eb" : "#2b3445"}">${aLast.toFixed(1)}</text>`;
  if (ch.b) {
    const bLast = ch.b[ch.b.length - 1] as number;
    g += `<text x="${R + 4}" y="${Y(bLast) + (Math.abs(Y(bLast) - Y(aLast)) < 10 ? 14 : 4)}" font-size="10" fill="#6b7487">${bLast.toFixed(1)}</text>`;
  }
  const months: Record<string, number> = {};
  ch.dates.forEach((d, i) => {
    const m = d.slice(0, 7);
    if (!(m in months)) months[m] = i;
  });
  Object.entries(months).forEach(([m, i]) => {
    if (i > 0) g += `<line x1="${X(i)}" x2="${X(i)}" y1="${T}" y2="${B}" stroke="#00000010"/>`;
    g += `<text x="${X(i) + 2}" y="${H - 2}" font-size="10" fill="#6b7487">${new Date(m + "-01").toLocaleDateString("en-IN", { month: "short" })}</text>`;
  });
  Object.entries(ch.pins || {}).forEach(([d, ps]) => {
    const i = ch.dates.indexOf(d);
    if (i < 0) return;
    const y0 = Y(ch.a[i] as number);
    g += `<line x1="${X(i)}" x2="${X(i)}" y1="${8}" y2="${y0}" stroke="#00000022" stroke-dasharray="2 2"/><circle cx="${X(i)}" cy="${y0}" r="2.5" fill="#2b3445"/>`;
    ps.slice(0, 3).forEach((p, j) => {
      const c = p.tag === "R" ? "#1d4ed8" : p.tag === "P" ? "#d97706" : "#4b5565";
      g += `<g class="pin" data-t="${esc(p.title)}" data-s="${esc(p.source)}" data-l="${esc(p.link)}"><circle cx="${X(i)}" cy="${8 + j * 12}" r="6" fill="#fff" stroke="${c}" stroke-width="1.3"/><text x="${X(i)}" y="${11 + j * 12}" font-size="7.5" font-weight="800" fill="${c}" text-anchor="middle">${esc(p.tag)}</text></g>`;
    });
    if (ps.length > 3) g += `<text x="${X(i)}" y="${8 + 3 * 12 + 3}" font-size="8" fill="#6b7487" text-anchor="middle">+${ps.length - 3}</text>`;
  });
  return `<div class="btitle">${title}<small>${sub}</small></div><svg>${g}</svg>`;
}

export const pinTipHTML = (title: string, source: string): string =>
  `<div style="max-width:300px"><b style="font-size:12.5px">${esc(title)}</b><div class="muted">${esc(source)}</div></div>`;

// ================= HORIZON =================
export function horizonViewHTML(range: Range): string {
  return `<div class="vt"><b>Sector horizon · ${RL[range]}</b><div>Higher = stronger · above the glowing NIFTY line = beating the market · click a sector to open it</div></div>
      ${range === "today" ? `<div class="replay"><button id="play">▶ Replay today</button><span class="muted" id="at">Now · live</span><input type="range" id="slider" min="0" max="375" value="375"></div>` : ""}
      <svg id="hsvg" class="fade-in" width="100%" height="100%"></svg>
      <div class="legend"><i style="background:var(--up)"></i>Up <i style="background:var(--down)"></i>Down &nbsp; Bigger bubble = bigger sector &nbsp; Light trail = where it was 30 minutes ago &nbsp; <span class="badge">RUNNING</span> = 1%+ ahead of NIFTY ${range === "today" ? "today" : RL[range]}</div>`;
}

export function bubbleTipHTML(d: HorizonData, name: string, range: Range): string | null {
  const s = d.sectors.find((x) => x.name === name);
  if (!s) return null;
  return `<b>${esc(s.name)}</b> <span class="muted">${esc(s.index)}</span><table><tr><td>${RL[range]}</td><td class="${cls(s.v)}">${pc(s.v)}</td></tr><tr><td>Rank</td><td>#${s.rank} of 20</td></tr><tr><td>Stocks inside</td><td>${s.n}</td></tr>${s.rank_open ? `<tr><td>At the open</td><td>#${s.rank_open}</td></tr>` : ""}</table><div class="muted" style="margin-top:4px">Click to open</div>`;
}

export function hSideHTML(d: HorizonData, range: Range): string {
  const secs = d.sectors.filter((s) => s.v != null).sort((a, b) => (b.v as number) - (a.v as number));
  const mx = Math.max(1, ...secs.map((s) => Math.abs(s.v as number)));
  const cl = d.climb.map((c) => ({ ...c, d: c.from - c.to }));
  let placed = false;
  let list = "";
  secs.forEach((s, i) => {
    const v = s.v as number;
    if (!placed && d.nifty.v != null && v < d.nifty.v) {
      list += `<div class="niftysep"><span>NIFTY 50</span><span>${pc(d.nifty.v)}</span></div>`;
      placed = true;
    }
    list += `<div class="row" ${goAttr("sector", s.name)}><span class="muted">${i + 1}</span><span>${esc(s.name)}</span><div class="bar" style="width:${(Math.abs(v) / mx) * 100}%;background:${v >= 0 ? "var(--up)" : "var(--down)"}"></div><span class="mono ${cls(v)}" style="text-align:right;font-size:12px" data-k="hl:${esc(s.name)}" data-v="${v}">${pc(v)}</span></div>`;
  });
  const since = range === "today" ? "Today · since the open" : `${RL[range]} · vs last week`;
  return (
    (d.themes.length
      ? `<h3>Themes beating NIFTY <small>Today · middle stock</small></h3>` +
        d.themes
          .map(
            (t) =>
              `<div class="kv"><span><b>${esc(t.name)}</b><div class="muted" style="font-size:11px">${esc(t.parent || "")}</div></span><span class="mono" style="text-align:right">${t.beat} of ${t.n}<br><span class="up">${pc(t.pct)}</span></span></div>`,
          )
          .join("") +
        `<div class="muted" style="font-size:11px;margin-top:4px">NSE has no index for these groups — each shows its middle stock (half its stocks did better, half worse), and how many of its stocks beat NIFTY.</div>`
      : "") +
    `<h3>Climbing the ranks <small>${since}</small></h3>` +
    (cl
      .filter((c) => c.d > 0)
      .sort((a, b) => b.d - a.d)
      .slice(0, 4)
      .map((c) => `<div class="kv"><span>${esc(c.name)}</span><span class="mono">#${c.from} → #${c.to} <b class="up">▲${c.d}</b></span></div>`)
      .join("") || '<div class="muted">Builds through the day.</div>') +
    `<h3>Slipping</h3>` +
    (cl
      .filter((c) => c.d < 0)
      .sort((a, b) => a.d - b.d)
      .slice(0, 4)
      .map((c) => `<div class="kv"><span>${esc(c.name)}</span><span class="mono">#${c.from} → #${c.to} <b class="down">▼${-c.d}</b></span></div>`)
      .join("") || '<div class="muted">–</div>') +
    `<h3>All 20 sectors · ${RL[range]} <small>above the line = beating NIFTY</small></h3>${list}` +
    `<h3>News &amp; policy radar <small>last 30 days · headlines as published</small></h3>` +
    d.news.map((h) => newsItem(h, h.sector)).join("") +
    `<h3>How to read the map</h3><div class="muted" style="font-size:12px">Every sector is a bubble. Its height is its move ${RL[range].toLowerCase()}, and the glowing line is NIFTY 50: above it = beating the market. <span class="up">Green</span> / <span class="down">red</span> = up / down. A bigger bubble = a bigger sector (the market value of its stocks). The light trail shows where it was 30 minutes ago.</div>
      <div class="muted" style="font-size:11px;margin-top:10px">Screener output · not investment advice</div>`
  );
}

// ================= SECTOR =================
export function sectorViewHTML(d: SectorData, W: number, H: number): string {
  const cx = W * 0.52;
  const cy = H * 0.5;
  const groups = d.groups;
  const n = groups.length;
  const angles = n === 1 ? [-90] : groups.map((_, i) => -125 + i * (360 / n));
  const RG = Math.min(W * 0.21, H * 0.25);
  const hasCards = d.news.length > 0;
  const minX = hasCards ? 285 : 30;
  let svg = `<defs><radialGradient id="halo"><stop offset="0" stop-color="${d.ix.pct >= 0 ? "#22a05a" : "#dc3f54"}" stop-opacity=".16"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient></defs>
    <circle cx="${cx}" cy="${cy}" r="${RG * 1.5}" fill="url(#halo)"/>`;
  [0.55, 0.9, 1.25].forEach((k) => (svg += `<circle cx="${cx}" cy="${cy}" r="${RG * k}" fill="none" stroke="#00000009"/>`));
  let nodes = "";
  const up = d.ix.pct >= 0;
  const hubC = up ? "#22a05a" : "#dc3f54";
  // first place every group and stock, then push apart anything that overlaps, then draw
  const G = groups.map((gp, i) => {
    const a = (angles[i] * Math.PI) / 180;
    return { gp, a, gx: cx + Math.cos(a) * RG, gy: cy + Math.sin(a) * RG * 0.92 };
  });
  interface P {
    s: SRow;
    gx: number;
    gy: number;
    a: number;
    t: number;
    x: number;
    y: number;
    r: number;
    gi: number;
  }
  const P: P[] = [];
  G.forEach(({ gp, a, gx, gy }, gi) => {
    const shown = gp.stocks.slice(0, 14);
    const spread = Math.min(2.6, 0.4 + shown.length * 0.19);
    shown.forEach((s, j) => {
      const t = shown.length === 1 ? 0 : (j / (shown.length - 1) - 0.5) * spread;
      const dist = 58 + (j % 3) * 22;
      P.push({ s, gx, gy, a, t, gi, x: gx + Math.cos(a + t) * dist, y: gy + Math.sin(a + t) * dist * 0.9, r: 6 + Math.min(10, Math.abs(s.pct || 0) * 2.2) });
    });
  });
  const fixed = [{ x: cx, y: cy, r: 60 }, ...G.map((g) => ({ x: g.gx, y: g.gy, r: 30 }))];
  for (let it = 0; it < 120; it++) {
    for (let i = 0; i < P.length; i++) {
      const A = P[i];
      for (let j = i + 1; j < P.length; j++) {
        const B2 = P[j];
        const dx = B2.x - A.x;
        const dy = B2.y - A.y;
        const dd = Math.hypot(dx, dy) || 0.1;
        const need = A.r + B2.r + 26;
        if (dd < need) {
          const k = (need - dd) / dd / 2;
          A.x -= dx * k;
          A.y -= dy * k * 0.6;
          B2.x += dx * k;
          B2.y += dy * k * 0.6;
        }
      }
      for (const F of fixed) {
        const dx = A.x - F.x;
        const dy = A.y - F.y;
        const dd = Math.hypot(dx, dy) || 0.1;
        const need = A.r + F.r;
        if (dd < need) {
          A.x += (dx / dd) * (need - dd);
          A.y += (dy / dd) * (need - dd);
        }
      }
      A.x = Math.max(minX + A.r, Math.min(W - 70, A.x));
      A.y = Math.max(58 + A.r, Math.min(H - 40, A.y));
    }
  }
  G.forEach(({ gp, a, gx, gy }, gi) => {
    const gc = (gp.mid ?? 0) >= 0 ? "#22a05a" : "#dc3f54";
    const busy = Math.max(0.6, Math.min(4, 4 - gp.stocks.reduce((s, r) => s + (r.vol_ratio || 1), 0) / gp.stocks.length));
    svg += `<line x1="${cx}" y1="${cy}" x2="${gx}" y2="${gy}" stroke="${gc}" stroke-width="${3 + Math.min(5, gp.n / 8)}" stroke-opacity=".75" stroke-linecap="round"/>
      <line x1="${cx}" y1="${cy}" x2="${gx}" y2="${gy}" stroke="#fff" stroke-width="2.4" class="flow" style="animation-duration:${busy}s" opacity=".9"/>`;
    const shown = gp.stocks.slice(0, 14);
    const extra = gp.stocks.length - shown.length;
    const out = a;
    P.filter((q) => q.gi === gi).forEach((q) => {
      const s = q.s;
      const sx = q.x;
      const sy = q.y;
      const r = q.r;
      const t = Math.atan2(sy - gy, sx - gx) - out;
      const pct = s.pct || 0;
      const col = pct >= 0 ? COLORS(Math.max(0.4, pct)) : COLORS(Math.min(-0.4, pct));
      svg += `<line x1="${gx}" y1="${gy}" x2="${sx}" y2="${sy}" stroke="${pct >= 0 ? "#22a05a" : "#dc3f54"}" stroke-opacity=".35" stroke-width="1"/>`;
      if ((s.vol_ratio || 0) > 1.2)
        svg += `<line x1="${gx}" y1="${gy}" x2="${sx}" y2="${sy}" stroke="${pct >= 0 ? "#22a05a" : "#dc3f54"}" stroke-width="2" class="flow" style="animation-duration:${Math.max(0.5, 3 - (s.vol_ratio as number))}s"/>`;
      const lx = sx + (Math.cos(out + t) >= 0 ? r + 4 : -r - 4);
      const anc = Math.cos(out + t) >= 0 ? "start" : "end";
      nodes += `<g class="node" data-sym="${esc(s.sym)}"><circle class="b" cx="${sx}" cy="${sy}" r="${r}" fill="${col}" stroke="${pct >= 0 ? "#1f8a4d" : "#c8344a"}" stroke-width="1.2" ${s.above50 ? "" : 'stroke-dasharray="3 2"'}/>
        <text x="${lx}" y="${sy - 1}" font-size="10" font-weight="800" fill="#121826" text-anchor="${anc}">${esc(s.sym)}</text><text x="${lx}" y="${sy + 10}" font-size="9.5" font-weight="600" fill="${pct >= 0 ? "#22a05a" : "#dc3f54"}" text-anchor="${anc}" data-k="ss:${esc(s.sym)}" data-v="${s.pct}">${pc(s.pct)}</text></g>`;
    });
    if (extra > 0) {
      const ex = gx + Math.cos(out) * 52;
      const ey = gy + Math.sin(out) * 46;
      nodes += `<g><circle cx="${ex}" cy="${ey}" r="10" fill="#fff" stroke="#c9ced8"/><text x="${ex}" y="${ey + 3}" font-size="8.5" font-weight="700" fill="#4b5565" text-anchor="middle">+${extra}</text></g>`;
    }
    const share = gp.n ? gp.above50 / gp.n : 0;
    const rr = 17;
    const circ = 2 * Math.PI * rr;
    nodes += `<g class="node grp" data-g="${esc(gp.name)}"><circle class="b" cx="${gx}" cy="${gy}" r="${rr}" fill="#fff" stroke="${gc}" stroke-width="2"/>
      <circle cx="${gx}" cy="${gy}" r="${rr + 4}" fill="none" stroke="${gc}" stroke-width="3" stroke-dasharray="${circ * 1.24 * share} ${circ * 2}" transform="rotate(-90 ${gx} ${gy})" opacity=".8"/>
      <text x="${gx}" y="${gy + 4}" font-size="11" font-weight="800" fill="${gc}" text-anchor="middle">${(gp.mid ?? 0) > 0 ? "+" : ""}${gp.mid == null ? "–" : gp.mid.toFixed(1)}</text>
      <text x="${gx}" y="${gy + rr + 18}" font-size="11.5" font-weight="800" fill="#121826" text-anchor="middle" letter-spacing=".6">${esc(gp.name.toUpperCase())}</text>
      <text x="${gx}" y="${gy + rr + 31}" font-size="10" fill="#6b7487" text-anchor="middle">${gp.n} stocks · ${gp.above50} above 50-DMA</text></g>`;
  });
  nodes += `<g class="hub"><circle cx="${cx}" cy="${cy}" r="48" fill="${hubC}" stroke="#fff" stroke-width="4"/><circle cx="${cx}" cy="${cy}" r="54" fill="none" stroke="${hubC}" stroke-opacity=".3" stroke-dasharray="3 4"/>
    <text x="${cx}" y="${cy - 6}" font-size="${d.name.length > 10 ? 14 : 19}" font-weight="800" fill="#fff" text-anchor="middle">${esc(d.name)}</text>
    <text x="${cx}" y="${cy + 12}" font-size="13" font-weight="700" fill="#fff" text-anchor="middle" data-k="hub:${esc(d.name)}" data-v="${d.ix.pct}">${pc(d.ix.pct)}</text><text x="${cx}" y="${cy + 26}" font-size="8.5" font-weight="700" fill="#ffffffcc" text-anchor="middle" letter-spacing=".8">INDEX · TODAY</text></g>`;
  const news = d.news.filter((h) => h.tag !== "POLICY").slice(0, 3);
  const pol = d.news.filter((h) => h.tag === "POLICY").slice(0, 2);
  const card = (items: NewsItem[]) =>
    items
      .map(
        (h) => `<div class="fn"><span class="tg ${esc(h.tag[0])}">${esc(h.tag)}</span> <span class="muted" style="font-size:11px">${dmy(h.time, { weekday: "short", day: "numeric", month: "short" })}</span>
    <div class="t"><a href="${esc(h.link)}" target="_blank" rel="noopener">${esc(h.title)}</a></div><div class="muted" style="font-size:11px;display:flex;justify-content:space-between"><span>${esc(h.source)}</span>${h.since != null ? `<span>${esc(d.name)} since <b class="${cls(h.since)}">${pc(h.since)}</b></span>` : ""}</div></div>`,
      )
      .join("");
  return `<div class="vt"><b>Inside ${esc(d.name)} · ${d.n} stocks in ${groups.length} groups</b><div>Centre: ${esc(d.index)} index <b class="${cls(d.ix.pct)}">${pc(d.ix.pct)}</b> · the ${d.n} stocks here <b class="${cls(d.middle)}">${pc(d.middle)}</b> (middle stock) · click a stock</div></div>
    <svg width="100%" height="100%" class="fade-in" id="ssvg">${svg}${nodes}</svg>
    ${news.length ? `<div class="fcard" style="left:14px;top:60px;width:250px"><h4>✦ News &amp; results</h4>${card(news)}</div>` : ""}
    ${pol.length ? `<div class="fcard" style="left:14px;bottom:44px;width:250px"><h4 style="color:var(--gold)">⚖ Government &amp; policy</h4>${card(pol)}</div>` : ""}
    <div class="legend"><i style="background:var(--up)"></i>Up <i style="background:var(--down)"></i>Down &nbsp; Bigger dot = bigger move today &nbsp; ◯ Solid ring = above its 50-day average &nbsp; ◌ Dashed = below &nbsp; Moving dots = more shares traded than normal</div>`;
}

export function stockTipHTML(s: SRow): string {
  return `<b>${esc(s.sym)}</b> <span class="muted">${esc(s.basic || s.group)}</span><table><tr><td>Price</td><td>${rs(s.ltp)}</td></tr><tr><td>Today</td><td class="${cls(s.pct)}">${pc(s.pct)}</td></tr><tr><td>Week</td><td class="${cls(s.w1)}">${pc(s.w1)}</td></tr><tr><td>Month</td><td class="${cls(s.m1)}">${pc(s.m1)}</td></tr><tr><td>Shares traded</td><td>${s.vol_ratio != null ? s.vol_ratio.toFixed(2) + "× normal" : "–"}</td></tr><tr><td>50-day average</td><td>${s.above50 == null ? "–" : s.above50 ? "above" : "below"}</td></tr></table>`;
}
export function groupTipHTML(gp: SGroup): string {
  return `<b>${esc(gp.name)}</b><table><tr><td>Stocks</td><td>${gp.n}</td></tr><tr><td>Middle stock today</td><td class="${cls(gp.mid)}">${pc(gp.mid)}</td></tr><tr><td>Above 50-day average</td><td>${gp.above50} of ${gp.n}</td></tr></table><div class="muted" style="margin-top:4px">${esc(gp.stocks.slice(0, 6).map((s) => s.sym).join(", "))}${gp.n > 6 ? "…" : ""}</div>`;
}

export function sSideHTML(d: SectorData): string {
  const ix = d.ix;
  const nf = d.nifty;
  const ins = d.inside;
  const moved = d.rank_open && d.rank ? d.rank_open - d.rank : null;
  const rh = d.rank_hist || [];
  let spark = "";
  if (rh.length > 1) {
    const W = 320;
    const H = 54;
    const X = (i: number) => (i / (rh.length - 1)) * W;
    const Y = (r: number) => 4 + ((r - 1) / 19) * (H - 8);
    const line = rh.map((p, i) => `${X(i)},${Y(p.rank)}`).join(" ");
    const day = (s: string) => new Date(s).toLocaleDateString("en-IN", { day: "numeric", month: "short" });
    spark = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:${H}px;background:#f6f8fb;border-radius:8px"><polygon points="0,${H} ${line} ${W},${H}" fill="rgba(37,99,235,.12)"/><polyline fill="none" stroke="#2563eb" stroke-width="2" points="${line}"/></svg>
      <div class="muted" style="font-size:11.5px">From <b>#${rh[0].rank}</b> on ${day(rh[0].date)} to <b>#${rh[rh.length - 1].rank}</b> on ${day(rh[rh.length - 1].date)}</div>`;
  }
  const bar = (lab: string, a: number, b: number, note?: string) =>
    `<div class="kv" style="margin-top:6px"><span>${lab}</span><b>${a} of ${b}</b></div><div class="prog"><i style="width:${b ? (a / b) * 100 : 0}%"></i></div>${note ? `<div class="muted" style="font-size:11px;margin-top:-6px">${note}</div>` : ""}`;
  const peOk = ix.pe && nf.pe;
  const mv = (rows: SRow[]) =>
    rows.map((m) => `<div class="kv" style="cursor:pointer" ${goAttr("stock", m.sym)}><span><b>${esc(m.sym)}</b></span><span class="mono ${cls(m.w1)}">${pc(m.w1)}</span></div>`).join("");
  return `<div class="scard"><div style="display:flex;justify-content:space-between;align-items:center"><b style="font-size:18px">${esc(d.name)}</b><span class="badge">#${d.rank} OF 20${moved ? ` · ${moved > 0 ? "▲" : "▼"}${Math.abs(moved)} SINCE THE OPEN` : ""}</span></div>
      <div style="margin-top:4px"><span class="muted">${esc(d.index)}</span> <b class="mono" data-k="ixl" data-v="${ix.last}">${Number(ix.last || 0).toLocaleString("en-IN", { minimumFractionDigits: 2 })}</b> <b class="${cls(ix.pct)}">${pc(ix.pct)}</b> <span class="muted">today${ix.live ? " · live" : ""}</span></div>
      <div class="box3"><div>WEEK<b class="${cls(ix.w1)}">${pc(ix.w1)}</b>NIFTY ${pc(nf.w1)}</div><div>1 MONTH<b class="${cls(ix.m1)}">${pc(ix.m1)}</b>NIFTY ${pc(nf.m1)}</div><div>3 MONTHS<b class="${cls(ix.m3)}">${pc(ix.m3)}</b>NIFTY ${pc(nf.m3)}</div></div></div>
    <h3>Index vs the stocks here <small>Today</small></h3><div class="box2"><div>${esc(d.index)} index<b class="${cls(ix.pct)}">${pc(ix.pct)}</b></div><div>The ${d.n} stocks here, middle stock<b class="${cls(d.middle)}">${pc(d.middle)}</b></div></div>
    ${d.others.length ? `<div class="muted" style="font-size:12px;margin:8px 0 4px">The index also holds ${d.others.length} stocks this map shows under other sectors:</div>` + d.others.map((o) => `<div class="kv" style="cursor:pointer" ${goAttr("stock", o.sym)}><span><b>${esc(o.sym)}</b><div class="muted" style="font-size:10.5px">${esc(o.sector)}</div></span><span class="mono">${rs(o.ltp)} <b class="${cls(o.pct)}">${pc(o.pct)}</b></span></div>`).join("") : ""}
    <h3>Rank among 20 sectors <small>1 month · last 10 weeks</small></h3>${spark || '<div class="muted">Builds once the price history has loaded.</div>'}
    <h3>Inside the sector <small>${ins.n} stocks</small></h3>${bar("Above their 50-day average", ins.above50, ins.n, `A week ago: ${ins.above50_wk} of ${ins.n}`)}${bar("Rose this week", ins.rose_week, ins.n)}${bar("Within 5% of a 52-week high", ins.near_high, ins.n)}${bar("Trading 1.5× their normal volume today", ins.busy, ins.n)}
    ${peOk ? `<h3>Valuation <small>price ÷ earnings</small></h3><div class="kv"><span>${esc(d.index)}</span><b>${ix.pe}</b></div><div class="kv"><span>NIFTY 50</span><b>${nf.pe}</b></div><div class="muted" style="font-size:11.5px">${(ix.pe as number) > (nf.pe as number) ? "Priced above" : "Priced below"} the market by ${Math.abs(((ix.pe as number) / (nf.pe as number) - 1) * 100).toFixed(0)}%</div>` : ""}
    <h3>Moved most this week</h3>${mv(d.movers)}
    <h3>Slowed lately</h3>${mv(d.laggards)}
    <div class="muted" style="font-size:11px;margin-top:12px">Screener output · not investment advice</div>`;
}

// ================= STOCK =================
export function stockViewHTML(d: StockData, W: number, H: number): string {
  const r = d.row;
  const cx = W * 0.5;
  const cy = H * 0.3;
  const up = (r.pct || 0) >= 0;
  const c = up ? "#22a05a" : "#dc3f54";
  const below = r.high52 && r.ltp ? (1 - r.ltp / r.high52) * 100 : null;
  const peers = d.peers.filter((p) => p.sym !== d.sym);
  const lines = d.lines;
  let price = "";
  if (lines) {
    const all = [...lines.stock, ...(lines.sector || []), ...(lines.nifty || [])].filter((x) => x != null);
    const lo = Math.min(...all);
    const hi = Math.max(...all);
    const PW = 220;
    const PH = 70;
    const P = (arr: number[]) => arr.map((x, i) => `${(i / (arr.length - 1)) * PW},${PH - 4 - ((x - lo) / (hi - lo || 1)) * (PH - 8)}`).join(" ");
    price = `<svg viewBox="0 0 ${PW} ${PH}" style="width:100%;height:${PH}px">${lines.nifty ? `<polyline fill="none" stroke="#9aa3b2" stroke-width="1.3" points="${P(lines.nifty)}"/>` : ""}${lines.sector ? `<polyline fill="none" stroke="#22a05a" stroke-width="1.3" points="${P(lines.sector)}"/>` : ""}<polyline fill="none" stroke="#2563eb" stroke-width="1.8" points="${P(lines.stock)}"/></svg>
      <div style="font-size:11px"><span style="color:#2563eb">— ${esc(d.sym)} <b>${pc(lines.m3.stock)}</b></span> &nbsp; ${lines.sector && d.sector ? `<span style="color:#22a05a">— ${esc(d.sector.name)} <b>${pc(lines.m3.sector)}</b></span>` : ""}<br><span class="muted">— NIFTY 50 <b>${pc(lines.m3.nifty)}</b></span></div>`;
  }
  const back = d.sector ? goAttr("sector", d.sector.name) : goAttr("horizon");
  const news = d.news
    .slice(0, 2)
    .map(
      (h) =>
        `<div class="fn"><span class="tg ${esc(h.tag[0])}">${esc(h.tag)}</span> <span class="muted" style="font-size:11px">${dmy(h.time, { weekday: "short", day: "numeric", month: "short" })}</span><div class="t"><a href="${esc(h.link)}" target="_blank" rel="noopener">${esc(h.title)}</a></div><div class="muted" style="font-size:11px">${esc(h.source)}</div></div>`,
    )
    .join("");
  const flowPaths = ([
    [260, 150],
    [260, 330],
    [260, 470],
    [W - 260, 140],
    [W - 260, 330],
    [W - 260, 450],
  ] as [number, number][])
    .map(
      ([x, y], i) =>
        `<path d="M${cx} ${cy} C ${(cx + x) / 2} ${cy}, ${(cx + x) / 2} ${y}, ${x} ${y}" fill="none" stroke="#9aa3b2" stroke-width="1"/><path d="M${cx} ${cy} C ${(cx + x) / 2} ${cy}, ${(cx + x) / 2} ${y}, ${x} ${y}" fill="none" stroke="#2563eb" stroke-width="2" class="flow" style="animation-duration:${2 + i * 0.3}s" opacity=".7"/>`,
    )
    .join("");
  return `<div class="vt"><b>${esc(d.sym)} · up close</b><div>${esc(d.cls.basic || "")} · ${esc(d.sector ? d.sector.name : d.cls.sector || "")} · everything about this one stock</div></div>
    <button class="close" ${back}>✕</button>
    <svg width="100%" height="100%" class="fade-in" id="stsvg">
      ${[90, 140, 200, 270].map((k) => `<circle cx="${cx}" cy="${cy}" r="${k}" fill="none" stroke="#00000008"/>`).join("")}
      ${flowPaths}
      <g class="hub"><circle cx="${cx}" cy="${cy}" r="58" fill="${c}" stroke="#fff" stroke-width="5"/>
      ${below != null ? `<circle cx="${cx}" cy="${cy}" r="66" fill="none" stroke="#2563eb" stroke-width="3" stroke-dasharray="${2 * Math.PI * 66 * Math.max(0.02, 1 - below / 100)} 999" transform="rotate(-90 ${cx} ${cy})"/>` : ""}
      <text x="${cx}" y="${cy - 10}" font-size="${d.sym.length > 9 ? 13 : 18}" font-weight="800" fill="#fff" text-anchor="middle">${esc(d.sym)}</text>
      <text x="${cx}" y="${cy + 9}" font-size="13" font-weight="700" fill="#fff" text-anchor="middle" data-k="sp:${esc(d.sym)}" data-v="${r.ltp}">${rs(r.ltp)}</text>
      <text x="${cx}" y="${cy + 26}" font-size="11.5" font-weight="700" fill="#fff" text-anchor="middle">${pc(r.pct)} · Today</text></g>
      ${below != null ? `<text x="${cx}" y="${cy + 86}" font-size="10.5" fill="#2563eb" text-anchor="middle">blue ring: ${below.toFixed(1)}% below its 52-week high</text>` : ""}
      <text x="${cx}" y="${cy + 112}" font-size="10.5" font-weight="800" fill="#4b5565" text-anchor="middle" letter-spacing="1">PEERS · ${esc((r.group || "").toUpperCase())}</text>
      <text x="${cx}" y="${cy + 125}" font-size="10" fill="#6b7487" text-anchor="middle">click one to switch</text></svg>
    <div class="peers" style="left:${cx - 320}px;width:640px;top:${cy + 134}px">${peers.slice(0, 11).map((p) => `<div class="peer" ${goAttr("stock", p.sym)}>${esc(p.sym)}<span class="${cls(p.pct)}" data-k="pp:${esc(p.sym)}" data-v="${p.pct}">${pc(p.pct)}</span></div>`).join("")}${peers.length > 11 ? `<div class="peer" style="color:var(--blue)">+${peers.length - 11} more<span class="muted">in Peers</span></div>` : ""}</div>
    <div class="fcard" style="left:14px;top:60px;width:240px"><h4>✦ News &amp; policy <small>naming ${esc(d.sym)}</small></h4>${news || '<div class="muted">No headlines in 60 days.</div>'}</div>
    <div class="fcard" style="left:14px;top:270px;width:240px"><h4>Price · last 3 months <small>all start at 100</small></h4>${price || '<div class="muted">History loading…</div>'}</div>
    <div class="fcard" style="left:14px;top:430px;width:240px"><h4>Where the price sits</h4><div class="kv"><span>52-week high</span><b>${below != null ? below.toFixed(1) + "% below" : "–"}</b></div><div class="mini"><i style="display:block;height:100%;width:${below != null ? 100 - below : 0}%;background:#2563eb"></i></div>
      <div class="kv" style="margin-top:6px"><span>50-day average</span><b class="${r.above50 ? "up" : r.above50 === false ? "down" : ""}">${r.above50 == null ? "–" : r.above50 ? "Price above it" : "Price below it"}</b></div><div class="kv"><span>Today</span><b class="${cls(r.pct)}">${pc(r.pct)}</b></div></div>
    <div class="fcard" style="right:14px;top:60px;width:250px"><h4>Against its sector <small>vs ${esc(d.sector ? d.sector.index : "")}</small></h4><table class="ag"><tr><th></th><th>${esc(d.sym.slice(0, 9))}</th><th>SECTOR</th><th>GAP</th></tr>${d.against.map((a) => `<tr><td>${esc(a.label)}</td><td class="${cls(a.stock)}">${pc(a.stock)}</td><td class="${cls(a.sector)}">${pc(a.sector)}</td><td class="${cls(a.gap)}">${pc(a.gap)}</td></tr>`).join("")}</table>
      ${d.rank && d.rank.pos ? `<div class="muted" style="font-size:11px;margin-top:4px">#${d.rank.pos} of ${d.rank.of} ${esc(d.sector ? d.sector.name : "")} stocks this week</div>` : ""}</div>
    <div class="fcard" style="right:14px;top:250px;width:250px"><h4>Delivery % <small>taken home · today</small></h4><b style="font-size:22px">${d.delivery != null ? d.delivery.toFixed(1) + "%" : "–"}</b><div class="muted" style="font-size:11px">of today's traded shares were taken delivery of</div><div class="mini"><i style="display:block;height:100%;width:${d.delivery || 0}%;background:#2563eb"></i></div></div>
    <div class="fcard" style="right:14px;top:380px;width:250px"><h4>Shares traded <small>today</small></h4><b style="font-size:22px">${d.volume.ratio != null ? d.volume.ratio.toFixed(2) + "×" : "–"}</b><div class="muted" style="font-size:11px">its 60-day normal</div>
      <div class="mini" style="height:8px"><i style="display:block;height:100%;width:${Math.min(100, ((d.volume.ratio || 0) / 3) * 100)}%;background:linear-gradient(90deg,#22a05a,#2563eb)"></i></div><div class="muted" style="font-size:10px;display:flex;justify-content:space-around"><span>1×</span><span>2×</span><span>3×</span></div></div>`;
}

export function stockSideHTML(d: StockData, starred: boolean): string {
  const r = d.row;
  const below = r.high52 && r.ltp ? (1 - r.ltp / r.high52) * 100 : null;
  const peers = d.peers.filter((p) => p.sym !== d.sym);
  return `<div class="scard"><b style="font-size:19px">${esc(d.sym)}</b><div class="muted" style="font-size:12px">${esc(d.cls.basic || "")} · ${esc(d.sector ? d.sector.name : "")}${d.sector ? " · in its NSE sector index" : ""}</div>
      <div style="margin-top:6px"><b style="font-size:18px" class="mono" data-k="sr:${esc(d.sym)}" data-v="${r.ltp}">${rs(r.ltp)}</b> <b class="${cls(r.pct)}">${pc(r.pct)} today</b></div>
      <div class="box2"><div>THIS WEEK<b class="${cls(r.w1)}">${pc(r.w1)}</b></div><div>1 MONTH<b class="${cls(r.m1)}">${pc(r.m1)}</b></div><div>3 MONTHS<b class="${cls(r.m3)}">${pc(r.m3)}</b></div><div>52-WEEK HIGH<b>${below != null ? below.toFixed(1) + "%" : "–"}</b>below</div>
      <div>50-DAY AVERAGE<b>${r.above50 == null ? "–" : r.above50 ? "Above" : "Below"}</b></div><div>DELIVERY % (TODAY)<b>${d.delivery != null ? d.delivery.toFixed(1) + "%" : "–"}</b></div><div>SHARES TRADED<b>${d.volume.ratio != null ? d.volume.ratio.toFixed(2) + "×" : "–"}</b>its 60-day normal</div><div>IN THE SECTOR<b>${d.rank && d.rank.pos ? "#" + d.rank.pos + " of " + d.rank.of : "–"}</b>this week</div>
      ${d.cls.pe ? `<div>PE<b>${d.cls.pe}</b></div>` : ""}<div>MARKET VALUE<b>${d.cls.mcap ? "₹" + Math.round(d.cls.mcap / 1e7).toLocaleString("en-IN") + " cr" : "–"}</b></div></div>
      <div style="display:flex;gap:6px;align-items:center"><button class="btn" id="openc">Open chart</button><button class="mbtn" id="star" title="Add to watchlist" style="margin-top:8px;font-size:15px">${starred ? "★" : "☆"}</button></div></div>
    <h3>Peers · ${peers.length} <small>${esc(r.group || "")} · Today · highest % first</small></h3>
    ${peers.map((p) => `<div class="kv" style="cursor:pointer" ${goAttr("stock", p.sym)}><b>${esc(p.sym)}</b><span class="mono">${rs(p.ltp)} <b class="${cls(p.pct)}" data-k="pl:${esc(p.sym)}" data-v="${p.pct}">${pc(p.pct)}</b></span></div>`).join("") || '<div class="muted">No peers in this group.</div>'}
    <div class="muted" style="font-size:11px;margin-top:12px;text-align:center">Screener output · not investment advice</div>`;
}
