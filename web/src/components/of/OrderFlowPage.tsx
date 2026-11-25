"use client";
import { useState } from "react";
import { useSearchParams } from "next/navigation";
import type {
  BigTradesReply,
  DeltaReply,
  FootprintReply,
  HeatReply,
  OfBars,
  TimescalesReply,
  TpoProfile,
  VolumeProfile,
} from "@/lib/of/types";
import { CandleChart, CvdChart } from "./Charts";
import { FootprintCanvas, HeatCanvas, TpoCanvas, VolumeProfileCanvas } from "./Canvases";
import { Panel } from "./Panel";
import { usePanel } from "./usePanel";

const sgn = (x: number) => (x > 0 ? "up" : x < 0 ? "down" : "");
const arrow = (x: number) => (x > 0 ? "▲" : x < 0 ? "▼" : "•");
const pct = (x: number | null | undefined) => (x == null ? "–" : (x * 100).toFixed(2) + "%");
const hhmm = (t: number) =>
  new Date(t * 1000).toLocaleTimeString("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit" });

function Tog<T extends string>({ opts, value, onChange }: { opts: T[]; value: T; onChange: (v: T) => void }) {
  return (
    <span className="of-tog">
      {opts.map((o) => (
        <button key={o} className={o === value ? "on" : ""} onClick={() => onChange(o)}>
          {o}
        </button>
      ))}
    </span>
  );
}

export function OrderFlowPage() {
  const sp = useSearchParams();
  const sym = (sp.get("sym") ?? "RELIANCE").toUpperCase();
  const date = sp.get("date") ?? "";
  const [tf, setTf] = useState<"1" | "5">("5");
  const [method, setMethod] = useState<"bvc" | "clv">("bvc");
  const [anchor, setAnchor] = useState<number | null>(null);

  const q = `sym=${encodeURIComponent(sym)}${date ? `&date=${encodeURIComponent(date)}` : ""}`;
  const bars = usePanel<OfBars>(`/api/of/bars?${q}&tf=${tf}${anchor ? `&anchored=${anchor}` : ""}`);
  const vol = usePanel<VolumeProfile>(`/api/of/profile?${q}&kind=volume`);
  const tpo = usePanel<TpoProfile>(`/api/of/profile?${q}&kind=tpo`);
  const delta = usePanel<DeltaReply>(`/api/of/delta?${q}&method=${method}`);
  const fp = usePanel<FootprintReply>(`/api/of/footprint?${q}&tf=5`);
  const big = usePanel<BigTradesReply>(`/api/of/bigtrades?${q}`);
  const heat = usePanel<HeatReply>(`/api/of/heatmap?${q}`);
  const ts = usePanel<TimescalesReply>(`/api/of/timescales?sym=${encodeURIComponent(sym)}`);

  const meta = (d: { data_source?: string; estimated?: boolean } | null) => ({
    source: d?.data_source,
    estimated: d?.estimated,
  });

  return (
    <main className="of-page">
      <h1>Order-flow workbench</h1>
      <form className="of-bar" method="get" action="/of/">
        <input name="sym" defaultValue={sym} aria-label="Symbol" />
        <input name="date" type="date" defaultValue={date} aria-label="Date" />
        <button type="submit">Load</button>
        <span className="of-note">
          All order-flow figures are estimated from 1-minute and 5-minute bars (NSE publishes no trade-by-trade tape).
        </span>
      </form>
      {tpo.data && (
        <p>
          {tpo.data.day_type && <span className="of-badge">day type: {tpo.data.day_type.replace(/_/g, " ")}</span>}
          {tpo.data.open_type && <span className="of-badge">open: {tpo.data.open_type.replace(/_/g, " ")}</span>}
          {tpo.data.poor_high && <span className="of-badge">poor high</span>}
          {tpo.data.poor_low && <span className="of-badge">poor low</span>}
        </p>
      )}
      <div className="of-grid">
        <div>
          <Panel
            name={`${sym} candles, VWAP ±1/2σ${anchor ? ", anchored VWAP" : ""}`}
            method={bars.data?.method ?? "session VWAP and bands from the bars shown"}
            topic="06-vwap"
            state={bars}
            {...meta(bars.data)}
          >
            <Tog opts={["1", "5"] as ("1" | "5")[]} value={tf} onChange={setTf} />
            {anchor && (
              <button onClick={() => setAnchor(null)} style={{ marginLeft: 8 }}>
                Clear anchor
              </button>
            )}
            <span className="of-note"> Click a candle to anchor a VWAP there.</span>
            {bars.data && <CandleChart d={bars.data} onAnchor={setAnchor} />}
          </Panel>
          <Panel
            name="Cumulative delta"
            method={delta.data?.method ?? (method === "bvc" ? "BVC estimate from 1-minute bars" : "CLV estimate from bar close location")}
            topic="02-delta-and-cvd"
            state={delta}
            {...meta(delta.data)}
          >
            <Tog opts={["bvc", "clv"] as ("bvc" | "clv")[]} value={method} onChange={setMethod} />
            {delta.data && <CvdChart d={delta.data} />}
          </Panel>
          <Panel
            name="Footprint"
            method={fp.data?.method ?? "BVC estimate from 1-minute bars"}
            topic="03-footprint"
            state={fp}
            {...meta(fp.data)}
          >
            {fp.data && <FootprintCanvas fp={fp.data} />}
            <div className="of-note">Cells show sell×buy; outlined cells are stacked imbalances; boxed cell is the bar POC.</div>
          </Panel>
          <Panel
            name="Big trades and absorption"
            method={big.data?.method ?? "bars with volume above a multiple of their time-of-day median"}
            topic="04-big-trades-and-absorption"
            state={big}
            {...meta(big.data)}
          >
            {big.data && (
              <table>
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Side</th>
                    <th>Volume ×</th>
                    <th>+5m</th>
                    <th>+15m</th>
                    <th>+30m</th>
                  </tr>
                </thead>
                <tbody>
                  {big.data.big.map((b) => (
                    <tr key={b.ts}>
                      <td>{hhmm(b.ts)}</td>
                      <td className={sgn(b.side)}>{b.side > 0 ? "buy" : b.side < 0 ? "sell" : "–"}</td>
                      <td>{b.ratio.toFixed(1)}×</td>
                      <td>{pct(b.ret_5m)}</td>
                      <td>{pct(b.ret_15m)}</td>
                      <td>{pct(b.ret_30m)}</td>
                    </tr>
                  ))}
                  {big.data.absorption.map((a) => (
                    <tr key={`a${a.ts}`}>
                      <td>{hhmm(a.ts)}</td>
                      <td colSpan={5}>
                        absorption at {String(a.level)}: {a.side === "bid" ? "selling absorbed (bid)" : "buying absorbed (offer)"} ·
                        volume {a.volume_ratio.toFixed(1)}×, range {a.range_ratio.toFixed(1)}×
                      </td>
                    </tr>
                  ))}
                  {!big.data.big.length && !big.data.absorption.length && (
                    <tr>
                      <td colSpan={6}>None today.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            )}
          </Panel>
          <Panel
            name="Depth heat map"
            method={heat.data?.method ?? "resting depth captured forward-only"}
            topic="05-heatmap-and-liquidity"
            state={heat}
            {...meta(heat.data)}
          >
            {heat.data?.empty ? <div className="of-note">{heat.data.reason ?? "no depth for this date"}</div> : heat.data && <HeatCanvas h={heat.data} />}
          </Panel>
        </div>
        <div>
          <Panel
            name="Volume profile"
            method={vol.data?.method ?? "volume spread uniformly over each bar's range"}
            topic="07-volume-profile"
            state={vol}
            {...meta(vol.data)}
          >
            {vol.data && <VolumeProfileCanvas p={vol.data} />}
          </Panel>
          <Panel
            name="TPO (market profile)"
            method={tpo.data?.method ?? "30-minute letters A to M"}
            topic="08-market-profile-and-amt"
            state={tpo}
            {...meta(tpo.data)}
          >
            {tpo.data && <TpoCanvas t={tpo.data} />}
            <div className="of-note">Shaded amber rows are single prints; the bracket on the right is the initial balance.</div>
          </Panel>
          <Panel
            name="Timescales"
            method={ts.data?.method ?? "sign of structure, value and rate of change per timeframe"}
            topic="09-time-scales"
            state={ts}
            {...meta(ts.data)}
          >
            {ts.data && (
              <>
                <table>
                  <thead>
                    <tr>
                      <th>TF</th>
                      <th>Trend</th>
                      <th>vs value</th>
                      <th>vs VA</th>
                      <th>ROC</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ts.data.rows.map((r) => (
                      <tr key={r.tf}>
                        <td>{r.tf}</td>
                        {[r.trend, r.vs_value, r.vs_va, r.roc].map((v, i) => (
                          <td key={i} className={sgn(v)}>
                            {arrow(v)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="of-note">Alignment score {ts.data.score.toFixed(2)} (−1 to +1).</div>
              </>
            )}
          </Panel>
        </div>
      </div>
    </main>
  );
}
