"use client";
// Options positioning page (/flow/options?und=NIFTY): strike ladder, GEX, IV smile, history. Descriptive only.
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { createChart, type Time } from "lightweight-charts";
import { usePoll } from "@/lib/api";
import { cls, rf } from "@/lib/ml/format";
import {
  barLen,
  cumulativeGex,
  historySeries,
  ivPct,
  levelY,
  lin,
  oiMax,
  smileSegments,
  windowStrikes,
} from "@/lib/flow/options";
import type { OptionsReply } from "@/lib/flow/types";
import { Foot } from "./FlowPage";

export const UNDERLYINGS = ["NIFTY", "BANKNIFTY"];
const num = (v: number | null | undefined, d = 2): string => (v == null || !isFinite(v) ? "–" : v.toFixed(d));

const LEVELS: { key: "spot" | "max_pain" | "call_wall" | "put_wall" | "zero_gamma"; label: string; color: string; dash?: string }[] = [
  { key: "spot", label: "spot", color: "#2563eb" },
  { key: "max_pain", label: "max pain", color: "#7c3aed", dash: "4 3" },
  { key: "call_wall", label: "call wall", color: "#22a05a", dash: "2 3" },
  { key: "put_wall", label: "put wall", color: "#dc3f54", dash: "2 3" },
  { key: "zero_gamma", label: "zero-gamma", color: "#c27c0e", dash: "6 3" },
];

export function Ladder({ d }: { d: OptionsReply }) {
  const rowH = 14;
  const W = 640;
  const L = 58;
  const R = 70;
  const strikes = useMemo(() => windowStrikes(d.strikes, d.spot), [d.strikes, d.spot]);
  const max = oiMax(strikes);
  const half = (W - L - R) / 2;
  const cx = L + half;
  const H = strikes.length * rowH + 4;
  const lv: Record<string, number | null | undefined> = { spot: d.spot, ...d.totals };
  return (
    <div className="fl-card" id="c-ladder">
      <div className="fl-h">Strike ladder · put OI left, call OI right, ΔOI thin overlay</div>
      <svg viewBox={`0 0 ${W} ${H}`} className="fl-svg" role="img" aria-label="Strike ladder">
        <line x1={cx} x2={cx} y1={0} y2={H} stroke="#c8ced8" />
        {[...strikes].reverse().map((s, i) => {
          const y = i * rowH;
          const cl = barLen(s.call_oi, max, half);
          const pl = barLen(s.put_oi, max, half);
          const cd = barLen(s.call_doi, max, half);
          const pd = barLen(s.put_doi, max, half);
          return (
            <g key={s.strike} className="fl-lrow" data-strike={s.strike}>
              <title>{`${s.strike}: call OI ${rf(s.call_oi)} (Δ ${rf(s.call_doi)}), put OI ${rf(s.put_oi)} (Δ ${rf(s.put_doi)})`}</title>
              <text x={L - 6} y={y + rowH - 3} textAnchor="end" fontSize="9.5" fill="#6b7387">{s.strike}</text>
              <rect className="fl-pbar" x={cx - pl} y={y + 2} width={pl} height={rowH - 4} fill="#f2a9b4" />
              <rect className="fl-cbar" x={cx} y={y + 2} width={cl} height={rowH - 4} fill="#8fd3ac" />
              <rect className="fl-pd" x={cx - pd} y={y + 5} width={pd} height={rowH - 10} fill={(s.put_doi ?? 0) >= 0 ? "#b4233b" : "#7a8294"} />
              <rect className="fl-cd" x={cx} y={y + 5} width={cd} height={rowH - 10} fill={(s.call_doi ?? 0) >= 0 ? "#17803f" : "#7a8294"} />
            </g>
          );
        })}
        {LEVELS.map((l) => {
          const p = levelY(lv[l.key], strikes, rowH);
          if (!p) return null;
          const yy = p.y; // levelY counts from the top row, which is the highest strike
          return (
            <g key={l.key} className="fl-level" data-level={l.key} data-y={yy.toFixed(2)} data-clamped={p.clamped ? "1" : "0"}>
              <line x1={L} x2={W - R + 4} y1={yy} y2={yy} stroke={l.color} strokeWidth={l.key === "spot" ? 1.6 : 1.2} strokeDasharray={l.dash} />
              <text x={W - R + 6} y={yy + 3} fontSize="9" fontWeight="700" fill={l.color}>{l.label}{p.clamped ? " ↑↓" : ""}</text>
            </g>
          );
        })}
      </svg>
      <Foot />
    </div>
  );
}

export function GexChart({ d }: { d: OptionsReply }) {
  const W = 640;
  const H = 240;
  const P = { l: 44, r: 12, t: 10, b: 26 };
  const all = d.strikes;
  const cum = cumulativeGex(all);
  const from = all.length ? all.indexOf(windowStrikes(all, d.spot)[0]) : 0;
  const win = windowStrikes(all, d.spot);
  const cw = cum.slice(from, from + win.length);
  const vals = win.map((s) => s.gex_cr ?? 0);
  const vmax = Math.max(1e-9, ...vals.map(Math.abs));
  const cmax = Math.max(1e-9, ...cw.map(Math.abs));
  const kx = { lo: win[0]?.strike ?? 0, hi: win[win.length - 1]?.strike ?? 1 };
  const y0 = P.t + (H - P.t - P.b) / 2;
  const bw = Math.max(2, ((W - P.l - P.r) / Math.max(1, win.length)) * 0.7);
  const yBar = (v: number) => y0 - (v / vmax) * ((H - P.t - P.b) / 2);
  const yCum = (v: number) => y0 - (v / cmax) * ((H - P.t - P.b) / 2);
  const x = (k: number) => lin(k, kx, P.l + bw / 2, W - P.r - bw / 2);
  const line = win.map((s, i) => `${x(s.strike).toFixed(1)},${yCum(cw[i]).toFixed(1)}`).join(" ");
  const zg = d.totals.zero_gamma;
  return (
    <div className="fl-card" id="c-gex">
      <div className="fl-h">Gamma exposure by strike (bars) and cumulative (line), crore</div>
      <svg viewBox={`0 0 ${W} ${H}`} className="fl-svg" role="img" aria-label="GEX by strike">
        <line x1={P.l} x2={W - P.r} y1={y0} y2={y0} stroke="#c8ced8" />
        {win.map((s, i) => {
          const v = vals[i];
          return <rect key={s.strike} className="fl-gbar" x={x(s.strike) - bw / 2} y={Math.min(y0, yBar(v))} width={bw} height={Math.abs(yBar(v) - y0)} fill={v >= 0 ? "#8fd3ac" : "#f2a9b4"}><title>{`${s.strike}: ${num(v)} cr`}</title></rect>;
        })}
        <polyline className="fl-cum" fill="none" stroke="#141a26" strokeWidth="1.5" points={line} />
        {d.spot != null && d.spot >= kx.lo && d.spot <= kx.hi && <line className="fl-spot" x1={x(d.spot)} x2={x(d.spot)} y1={P.t} y2={H - P.b} stroke="#2563eb" />}
        {zg != null && zg >= kx.lo && zg <= kx.hi && <line className="fl-zg" x1={x(zg)} x2={x(zg)} y1={P.t} y2={H - P.b} stroke="#c27c0e" strokeDasharray="6 3" />}
        <text x={P.l} y={H - 8} fontSize="9.5" fill="#6b7387">{kx.lo}</text>
        <text x={W - P.r} y={H - 8} fontSize="9.5" fill="#6b7387" textAnchor="end">{kx.hi}</text>
        <text x={4} y={P.t + 8} fontSize="9" fill="#6b7387">+{num(vmax, 1)}</text>
        <text x={4} y={H - P.b} fontSize="9" fill="#6b7387">-{num(vmax, 1)}</text>
      </svg>
      <Foot />
    </div>
  );
}

export function Smile({ d }: { d: OptionsReply }) {
  const W = 640;
  const H = 220;
  const P = { l: 38, r: 12, t: 10, b: 24 };
  const win = windowStrikes(d.strikes, d.spot);
  const ivs = win.flatMap((s) => [s.call_iv, s.put_iv]).filter((v): v is number => v != null && isFinite(v) && v > 0).map(ivPct);
  const iv = { lo: Math.min(...ivs, Infinity), hi: Math.max(...ivs, -Infinity) };
  const kx = { lo: win[0]?.strike ?? 0, hi: win[win.length - 1]?.strike ?? 1 };
  const x = (k: number) => lin(k, kx, P.l, W - P.r);
  const y = (v: number) => lin(v, iv, H - P.b, P.t);
  const has = ivs.length > 0;
  return (
    <div className="fl-card" id="c-smile">
      <div className="fl-h">IV smile · call (green) and put (red) implied vol by strike</div>
      {has ? (
        <svg viewBox={`0 0 ${W} ${H}`} className="fl-svg" role="img" aria-label="IV smile">
          {smileSegments(win, "put_iv", x, y).map((p, i) => <polyline key={"p" + i} className="fl-ivp" fill="none" stroke="#dc3f54" strokeWidth="1.6" points={p} />)}
          {smileSegments(win, "call_iv", x, y).map((p, i) => <polyline key={"c" + i} className="fl-ivc" fill="none" stroke="#22a05a" strokeWidth="1.6" points={p} />)}
          {d.spot != null && d.spot >= kx.lo && d.spot <= kx.hi && <line x1={x(d.spot)} x2={x(d.spot)} y1={P.t} y2={H - P.b} stroke="#2563eb" strokeDasharray="3 3" />}
          <text x={4} y={P.t + 8} fontSize="9" fill="#6b7387">{iv.hi.toFixed(1)}%</text>
          <text x={4} y={H - P.b} fontSize="9" fill="#6b7387">{iv.lo.toFixed(1)}%</text>
          <text x={P.l} y={H - 8} fontSize="9.5" fill="#6b7387">{kx.lo}</text>
          <text x={W - P.r} y={H - 8} fontSize="9.5" fill="#6b7387" textAnchor="end">{kx.hi}</text>
        </svg>
      ) : (
        <div className="fl-empty">No implied vols stored for this expiry.</div>
      )}
      <Foot />
    </div>
  );
}

export function History({ d }: { d: OptionsReply }) {
  const el = useRef<HTMLDivElement>(null);
  const gex = useMemo(() => historySeries(d.history, "gex_total_cr"), [d.history]);
  const pcr = useMemo(() => historySeries(d.history, "pcr_oi"), [d.history]);
  useEffect(() => {
    const host = el.current;
    if (!host) return;
    const chart = createChart(host, {
      height: 220,
      layout: { background: { color: "#ffffff" }, textColor: "#6b7385", fontFamily: "JetBrains Mono", fontSize: 11 },
      grid: { vertLines: { color: "#f0f2f5" }, horzLines: { color: "#f0f2f5" } },
      rightPriceScale: { borderColor: "#d9dee6" },
      leftPriceScale: { visible: true, borderColor: "#d9dee6" },
      timeScale: { borderColor: "#d9dee6" },
    });
    const g = chart.addLineSeries({ color: "#2563eb", lineWidth: 2, priceScaleId: "right", title: "GEX cr" });
    const p = chart.addLineSeries({ color: "#c27c0e", lineWidth: 2, priceScaleId: "left", title: "PCR" });
    g.setData(gex.map((x) => ({ time: x.time as Time, value: x.value })));
    p.setData(pcr.map((x) => ({ time: x.time as Time, value: x.value })));
    chart.timeScale().fitContent();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(() => chart.applyOptions({ width: host.clientWidth })) : null;
    ro?.observe(host);
    return () => {
      ro?.disconnect();
      chart.remove();
    };
  }, [gex, pcr]);
  return (
    <div className="fl-card" id="c-hist">
      <div className="fl-h">History · total GEX (blue, right axis) and PCR (amber, left axis)</div>
      <div ref={el} className="fl-hist" data-points={gex.length} />
      <Foot />
    </div>
  );
}

export function Totals({ d }: { d: OptionsReply }) {
  const t = d.totals;
  const items: [string, string, string?][] = [
    ["Spot", rf(d.spot, 0)],
    ["Total GEX", t.gex_total_cr == null ? "–" : num(t.gex_total_cr, 1) + " cr", cls(t.gex_total_cr)],
    ["Zero-gamma", rf(t.zero_gamma, 0)],
    ["Call wall", rf(t.call_wall, 0)],
    ["Put wall", rf(t.put_wall, 0)],
    ["Max pain", rf(t.max_pain, 0)],
    ["PCR (OI)", num(t.pcr_oi)],
    ["PCR (ΔOI)", num(t.pcr_doi)],
  ];
  return (
    <div className="fl-strip" id="otot" style={{ gridTemplateColumns: "repeat(8,minmax(0,1fr))" }}>
      {items.map(([k, v, c]) => (
        <div className="fl-card" key={k}>
          <div className="fl-h">{k}</div>
          <div className={"fl-mid mono " + (c ?? "")}>{v}</div>
          <Foot />
        </div>
      ))}
    </div>
  );
}

function OptionsView({ und }: { und: string }) {
  const [expiry, setExpiry] = useState<string | null>(null);
  const path = `/api/flow/options?und=${encodeURIComponent(und)}` + (expiry ? `&expiry=${expiry}` : "");
  const { data: d, error } = usePoll<OptionsReply>(path, 60000);
  return (
    <div className="flow-page">
      <div className="fl-title">
        <h1>Options positioning · {und}</h1>
        <div className="fl-sub">
          {UNDERLYINGS.map((u, i) => (
            <span key={u}>
              {i ? " · " : ""}
              {u === und ? <b>{u}</b> : <Link href={`/flow/options/?und=${u}`}>{u}</Link>}
            </span>
          ))}
          {" · "}
          <Link href="/flow/">← Money Flow</Link>
        </div>
      </div>
      {!d ? (
        <div className="fl-empty" style={{ padding: "16px 22px" }}>{error ? `No option data for ${und}.` : "Loading…"}</div>
      ) : (
        <>
          <div className="fl-ctrl">
            <label>
              Expiry{" "}
              <select id="expiry" value={expiry ?? d.expiry ?? ""} onChange={(e) => setExpiry(e.target.value)}>
                {d.expiries.map((e) => <option key={e} value={e}>{e}</option>)}
              </select>
            </label>
            <span className="fl-sub">Totals are the whole chain; the ladder, GEX and smile are the chosen expiry.</span>
          </div>
          <Totals d={d} />
          <div className="fl-body">
            <Ladder d={d} />
            <GexChart d={d} />
            <div className="fl-two">
              <Smile d={d} />
              <History d={d} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export function OptionsPage() {
  const und = (useSearchParams().get("und") ?? "NIFTY").toUpperCase();
  return <OptionsView key={und} und={und} />;
}
