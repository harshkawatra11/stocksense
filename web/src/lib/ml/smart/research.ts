// Port of smart.html's research note renderer (lines 837-923): ring, qchart, rdraw. Every interpolated string goes through esc().
import { esc, rf, rsg } from "../format";
import type { ResearchNote } from "./types";

export function ringColor(score: number | null | undefined): string {
  const s = score ?? 0;
  return s >= 75 ? "#16a34a" : s >= 62 ? "#65a30d" : s >= 48 ? "#d97706" : s >= 35 ? "#ea580c" : "#dc2626";
}

export function ring(score: number | null | undefined, grade: string | undefined): string {
  const c = 2 * Math.PI * 34;
  const col = ringColor(score);
  return `<svg viewBox="0 0 84 84" class="rp-ring"><circle cx="42" cy="42" r="34" fill="none" stroke="#eef1f5" stroke-width="8"/>
    <circle cx="42" cy="42" r="34" fill="none" stroke="${col}" stroke-width="8" stroke-linecap="round" stroke-dasharray="${c}" stroke-dashoffset="${c}" transform="rotate(-90 42 42)" style="--to:${c * (1 - (score || 0) / 100)}" class="rp-arc"/>
    <text x="42" y="44" text-anchor="middle" font-size="22" font-weight="800" fill="${col}" font-family="Inter">${score ?? "–"}</text>
    <text x="42" y="58" text-anchor="middle" font-size="8.5" font-weight="700" fill="#6b7385" font-family="Inter">OUT OF 100</text></svg>
    <div class="rp-grade" style="color:${col}">${esc(grade || "")}</div>`;
}

export function qchart(rs: ResearchNote["results"]): string {
  if (!rs.length) return '<div class="muted">No quarterly results filed with NSE.</div>';
  const W = 320;
  const H = 120;
  const n = rs.length;
  const bw = W / n;
  const mx = Math.max(...rs.map((x) => x.income || 0), 1);
  const pm = Math.max(...rs.map((x) => Math.abs(x.profit || 0)), 1);
  const bars = rs
    .map((x, i) => {
      const h = ((x.income || 0) / mx) * (H - 38);
      return `<rect x="${i * bw + bw * 0.16}" y="${H - 18 - h}" width="${bw * 0.68}" height="${h}" rx="2" fill="#dbe5f7" class="rp-bar" style="animation-delay:${i * 0.05}s"><title>${esc(x.label)}: income ₹${rf(x.income)} cr</title></rect>`;
    })
    .join("");
  const pts = rs
    .map((x, i) => (x.profit == null ? null : ([i * bw + bw / 2, H - 18 - (x.profit > 0 ? (x.profit / pm) * (H - 46) : 0) - 4, x] as const)))
    .filter((p): p is NonNullable<typeof p> => p !== null);
  const line = pts.length > 1 ? `<polyline points="${pts.map((p) => p[0] + "," + p[1]).join(" ")}" fill="none" stroke="#16a34a" stroke-width="2" class="rp-line"/>` : "";
  const dots = pts
    .map(([x, y, r]) => `<circle cx="${x}" cy="${y}" r="3.2" fill="${(r.profit ?? 0) >= 0 ? "#16a34a" : "#dc2626"}"><title>${esc(r.label)}: profit ₹${rf(r.profit)} cr</title></circle>`)
    .join("");
  const labs = rs.map((x, i) => `<text x="${i * bw + bw / 2}" y="${H - 5}" text-anchor="middle" font-size="9" fill="#8a93a6">${esc(x.label)}</text>`).join("");
  return `<svg viewBox="0 0 ${W} ${H}" class="rp-q">${bars}${line}${dots}${labs}</svg><div class="rp-leg"><i style="background:#dbe5f7"></i>Income (bars) <i style="background:#16a34a;height:3px"></i>Profit (line) · ₹ crore, from NSE filings</div>`;
}

type Eco = ResearchNote["economy"][number];
export const pushHTML = (e: Eco): string =>
  e.push === 1
    ? '<b class="up">▲ helping now</b>'
    : e.push === -1
      ? '<b class="down">▼ hurting now</b>'
      : e.push === 0
        ? '<b style="color:#d97706">⚠ big swing</b>'
        : '<span class="muted">steady</span>';
const eff = (e: Eco) => (e.effect > 0 ? "helps when it rises" : e.effect < 0 ? "hurts when it rises" : "matters both ways");
export const effInr = (e: Eco): string =>
  e.factor === "inr"
    ? e.effect > 0
      ? "helps when the rupee weakens"
      : e.effect < 0
        ? "hurts when the rupee weakens"
        : "matters both ways"
    : e.factor === "rates"
      ? e.effect > 0
        ? "helps when rates fall"
        : "matters both ways"
      : eff(e);

const cu = (v: number | null | undefined) => ((v || 0) >= 0 ? "up" : "down");

export function researchHTML(r: ResearchNote): string {
  const t = r.ttm || {};
  const sec = r.sector_strength || {};
  const tech = r.tech || {};
  const sc = Object.entries(r.scores || {}).filter((kv): kv is [string, number] => kv[1] != null);
  const bar = (v: number) => `<div class="rp-sb"><i style="width:${v * 10}%;background:${v >= 7 ? "#16a34a" : v >= 5 ? "#d97706" : "#dc2626"}"></i></div>`;
  const lo = r.low52;
  const hi = r.high52;
  const pos = lo && hi && r.ltp ? Math.max(0, Math.min(100, ((r.ltp - lo) / (hi - lo)) * 100)) : null;
  const peVs = r.pe && r.sector_pe ? r.pe / r.sector_pe : null;
  const sw = r.swot || {};
  const w = r.weights;
  return `<div class="rp">
    <div class="rp-top">
      <div class="rp-id"><div class="h">RESEARCH NOTE · ${esc(new Date(r.built * 1000).toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" }))}</div>
        <div class="rp-name">${esc(r.name || r.sym)}</div>
        <div class="rp-tags">${[r.macro, r.sector, r.industry, r.basic].filter(Boolean).map((x) => `<span>${esc(x)}</span>`).join("<em>›</em>")}${r.index ? `<span class="b">${esc(r.index)}</span>` : ""}</div></div>
      <div class="rp-score">${ring(r.total, r.grade)}</div>
      <div class="rp-sc">${sc.map(([k, v]) => `<div class="rp-sr"><span>${esc(k)}</span>${bar(v)}<b>${v.toFixed(1)}</b></div>`).join("")}</div>
    </div>

    <div class="rp-sec"><div class="h">IN SIMPLE WORDS</div>
      ${r.summary.map((l) => `<p>${esc(l)}</p>`).join("")}
      ${r.wiki ? `<p class="rp-wiki">${esc(r.wiki.text)} ${r.wiki.url ? `<a href="${esc(r.wiki.url)}" target="_blank" rel="noopener">Wikipedia ↗</a>` : ""}</p>` : ""}</div>

    <div class="rp-grid">
      <div class="rp-card"><div class="h">RESULTS · LAST ${r.results.length} QUARTERS</div>${qchart(r.results)}
        <div class="rp-kv"><div>12-month income<b>₹${rf(t.income)} cr</b></div><div>12-month profit<b class="${cu(t.profit)}">₹${rf(t.profit)} cr</b></div>
          <div>Profit on ₹100 sales<b>₹${rf(t.margin, 1)}</b><span>typical ~₹${esc(t.norm)}</span></div><div>Income growth<b class="${cu(t.g_inc)}">${rsg(t.g_inc)}%</b><span>vs a year ago</span></div>
          <div>Profit growth<b class="${cu(t.g_pat)}">${rsg(t.g_pat)}%</b><span>vs a year ago</span></div><div>Profitable quarters<b>${t.profit_q} of ${t.n_q}</b></div></div></div>

      <div class="rp-card"><div class="h">FINANCES & VALUATION</div>
        <div class="rp-kv"><div>Market value<b>₹${rf(r.mcap_cr)} cr</b></div><div>Price ÷ yearly profit (P/E)<b>${rf(r.pe, 1)}×</b><span>sector ~${rf(r.sector_pe, 1)}×</span></div>
          ${r.financial ? `<div>Debt check<b>n/a</b><span>banks & lenders borrow by design</span></div>` : `<div>Debt ÷ own capital<b class="${(r.balance.de ?? 0) < 1 ? "up" : "down"}">${rf(r.balance.de, 2)}×</b><span>last detailed filing ${esc(r.balance.as_of || "–")} · own business</span></div>`}
          <div>Delivery today<b>${rf(r.delivery, 1)}%</b><span>shares actually taken home</span></div></div>
        ${peVs ? `<div class="rp-gl"><span>Cheaper than sector</span><span>Same</span><span>Costlier</span></div><div class="rp-g"><i style="left:${Math.max(2, Math.min(98, ((peVs - 0.4) / 1.6) * 100))}%"></i></div>` : ""}
        ${pos != null ? `<div class="h" style="margin-top:10px">52-WEEK RANGE</div><div class="rp-rg"><span>₹${rf(lo, 1)}</span><div><i style="left:${pos}%"></i></div><span>₹${rf(hi, 1)}</span></div>` : ""}
        <div class="rp-kv" style="margin-top:6px"><div>1-year return<b class="${cu(tech.ret1y)}">${rsg(tech.ret1y, 1)}%</b></div><div>vs Nifty (3 months)<b class="${cu(tech.rs63)}">${rsg(tech.rs63, 1)}</b></div></div></div>

      <div class="rp-card"><div class="h">INDUSTRY · ${esc(r.basic || r.industry || "")}</div>
        <p class="rp-p">${esc(r.what)}</p>
        <div class="rp-kv"><div>${esc(sec.sector || "Sector")} · 3 months<b class="${cu(sec.sector_3m)}">${rsg(sec.sector_3m, 1)}%</b></div><div>Nifty · 3 months<b class="${cu(sec.nifty_3m)}">${rsg(sec.nifty_3m, 1)}%</b></div></div>
        ${r.peers.length ? `<table class="rp-t"><tr><th>Peers</th><th>Value ₹ cr</th><th>P/E</th></tr>${r.peers.map((p) => `<tr data-s="${esc(p.sym)}"><td><b>${esc(p.sym)}</b> <span class="muted">${esc(p.name)}</span></td><td>${rf(p.mcap_cr)}</td><td>${rf(p.pe, 1)}</td></tr>`).join("")}</table>` : ""}</div>

      <div class="rp-card"><div class="h">ECONOMY · WHAT MOVES THIS BUSINESS</div>
        ${r.economy.map((e) => `<div class="rp-ec"><div><b>${esc(e.name)}</b> <span class="muted">· ${effInr(e)}</span><div class="muted">${esc(e.why)}</div></div>
          <div class="rp-ecr">${e.reading ? `<span class="mono">${rsg(e.reading.chg3m, 1)}% <span class="muted">3 mo</span></span><br>` : ""}${pushHTML(e)}</div></div>`).join("") || '<div class="muted">No specific drivers.</div>'}</div>
    </div>

    <div class="rp-swot">${(
      [
        ["Strengths", "s"],
        ["Weaknesses", "w"],
        ["Opportunities", "o"],
        ["Threats", "t"],
      ] as const
    )
      .map(([k, c]) => `<div class="rp-sw ${c}"><div class="h">${k.toUpperCase()}</div><ul>${(sw[k] || []).map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div>`)
      .join("")}</div>
    <div class="rp-foot">Rules-based note built from ${esc(r.sources.join(" · "))}. The score weighs business ${w.Business}, growth ${w.Growth}, profitability ${w.Profitability}, financial health ${w["Financial health"]}, valuation ${w.Valuation}, industry & economy ${w["Industry & economy"]}, momentum ${w.Momentum}. Debt figures come from NSE's older detailed filing (date shown). Not investment advice.</div>
  </div>`;
}
