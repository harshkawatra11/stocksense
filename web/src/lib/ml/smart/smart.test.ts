import { describe, expect, it } from "vitest";
import live from "@/test/fixtures/sm_live.json";
import research from "@/test/fixtures/research.json";
import { applyQuote, candleTime, levelsHTML, liquidityLines, priceHTML, structureMarkers, visibleRange, zoneSpecs, isIdx } from "./chart";
import { ladderPlan } from "./ladder";
import { researchHTML, qchart, ring, ringColor } from "./research";
import { beadScale, geometry, mergePaths, nowMin, yScale, fxPhase, fxPoint } from "./stage";
import { columnRanks, COLS, flowMiniHTML, halfHTML, stars, tileHTML } from "./tiles";
import { tipHTML } from "./tip";
import { exploratoryBadgeHTML, gateChipHTML, groupStrategies, rrbar, strategiesForTf, tlevels, tpanelHTML, tradeMarkers, tstate } from "./trades";
import type { Candle, ChartRead, ResearchNote, SmartLive, SmartStock, StrategyRow, TradesResponse } from "./types";

const D = live as unknown as SmartLive;
const BY = Object.fromEntries(D.stocks.map((s) => [s.sym, s]));

describe("ladder", () => {
  // Expected numbers: smart.html ladder() (lines 188-206) copied verbatim into Node and run on this fixture
  // (RELIANCE bars.slice(-36), canvas 100x40, step 2.5) with a recording fake 2d context.
  const bars = BY.RELIANCE.bars!.slice(-36);
  const plan = ladderPlan(bars, 100, 40, 2.5)!;
  it("matches the original JS output", () => {
    expect(bars).toHaveLength(25);
    expect(plan.rects).toHaveLength(211);
    expect(plan.ly).toBeCloseTo(5.349822248773378, 12);
    expect(plan.rects[0]).toEqual({ x: 0.5, y: 38, w: 2.8, h: 1.1, color: "#d6455d" });
    expect(plan.rects.at(-1)).toEqual({ x: 96.5, y: 5.5, w: 2.8, h: 1.1, color: "#2f9e5f" });
    const per: Record<number, number> = {};
    plan.rects.forEach((r) => (per[Math.round((r.x - 0.5) / 4)] = (per[Math.round((r.x - 0.5) / 4)] || 0) + 1));
    expect(Object.values(per)).toEqual([2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9, 10, 11, 11, 12, 12, 12, 13, 13, 14, 14]);
    expect([...new Set(plan.rects.map((r) => r.color))].sort()).toEqual(["#2f9e5f", "#5cbf86", "#d6455d"]);
  });
  it("needs two bars", () => {
    expect(ladderPlan([[1, 2, 0, 1]], 100, 40)).toBeNull();
    expect(ladderPlan(null, 100, 40)).toBeNull();
  });
});

describe("tiles", () => {
  it("assigns ranks 1..20 to the four columns, each exactly once", () => {
    expect(COLS).toEqual([[13, 20], [8, 12], [4, 7], [1, 3]]);
    const bull = COLS.flatMap((_, i) => columnRanks("bull", i));
    expect(bull).toEqual([13, 14, 15, 16, 17, 18, 19, 20, 8, 9, 10, 11, 12, 4, 5, 6, 7, 1, 2, 3]);
    expect([...bull].sort((a, b) => a - b)).toEqual(Array.from({ length: 20 }, (_, i) => i + 1));
    // the weakest sit at the bottom: bear columns run in reverse
    expect(columnRanks("bear", 0)).toEqual([20, 19, 18, 17, 16, 15, 14, 13]);
    expect(columnRanks("bear", 3)).toEqual([3, 2, 1]);
  });
  it("stars", () => {
    expect(stars(0.5)).toBe(1);
    expect(stars(0.9)).toBe(2);
    expect(stars(4.1)).toBe(5);
    expect(stars(-0.9)).toBe(2);
    expect(stars(0)).toBe(1);
    expect(stars(99)).toBe(5);
  });
  it("renders a hidden tile for a missing rank and escapes the symbol", () => {
    expect(tileHTML(undefined, false)).toContain("visibility:hidden");
    const evil: SmartStock = { ...BY.RELIANCE, sym: "<script>x</script>" };
    const html = tileHTML(evil, true);
    expect(html).not.toContain("<script");
    expect(html).toContain("&lt;script&gt;x&lt;/script&gt;");
  });
  it("shows the flow mini-card only when the row has one", () => {
    expect(flowMiniHTML(null)).toBe("");
    const f = flowMiniHTML({ oi_quad_label: "Long buildup", rvol: 1.8, deliv_z20_prev: 2.1, flow_score: 1.234 });
    expect(f).toContain("Long buildup");
    expect(f).toContain("tag bull");
    expect(f).toContain("1.8×");
    expect(f).toContain("2.1");
    expect(f).toContain("1.23");
    expect(flowMiniHTML({ oi_quad_label: "Short buildup" })).toContain("tag bear");
    const withFlow = tileHTML({ ...BY.RELIANCE, rank: 1, flow: { rvol: 2 } }, true);
    expect(withFlow).toContain("FLOW");
    expect(tileHTML({ ...BY.RELIANCE, rank: 1 }, true)).not.toContain("FLOW");
    expect(tipHTML({ ...BY.RELIANCE, flow: { rvol: 2 } })).toContain("FLOW");
  });
  it("lays out 20 slots per side", () => {
    expect((halfHTML("bull", D.bull, BY).match(/class="tile/g) || []).length).toBe(20);
    expect((halfHTML("bear", D.bear, BY).match(/data-s=/g) || []).length).toBe(D.bear.length);
  });
});

describe("stage maths", () => {
  it("bead x maps from_open +-4% to the two edges", () => {
    const X = beadScale(500);
    expect(X(-4)).toBe(20);
    expect(X(4)).toBe(478);
    expect(X(0)).toBe(249);
    expect(X(-9)).toBe(20); // clipped
    expect(X(12)).toBe(478);
  });
  it("yScale is odd and log shaped", () => {
    expect(yScale(0)).toBe(0);
    expect(yScale(1)).toBeCloseTo(Math.log1p(1 / 0.12), 12);
    expect(yScale(-1)).toBeCloseTo(-yScale(1), 12);
  });
  it("nowMin is minutes since 09:15 IST, clamped", () => {
    const at = (iso: string) => nowMin(Date.parse(iso));
    expect(at("2026-11-12T03:45:00Z")).toBe(1); // 09:15 IST clamps to 1
    expect(at("2026-11-12T05:45:00Z")).toBe(120); // 11:15 IST
    expect(at("2026-11-12T11:00:00Z")).toBe(375); // after the close
  });
  it("geometry keeps the axis around zero and maps minutes into the chart", () => {
    const g = geometry(D, 1200, 640, 120);
    expect(g.hiV).toBeGreaterThanOrEqual(0.4);
    expect(g.loV).toBeLessThanOrEqual(-0.4);
    expect(g.X(0)).toBe(g.L);
    expect(g.X(120)).toBeCloseTo(g.chartR, 9);
    expect(g.Y(g.hiV)).toBeCloseTo(g.T, 9);
    expect(g.Y(g.loV)).toBeCloseTo(g.B, 9);
    expect(g.tilesW).toBe(444);
  });
  it("light phases and points", () => {
    expect(fxPhase(0, 1)).toBeCloseTo(0.05, 12);
    expect(fxPhase(4, 0)).toBeCloseTo(1, 12);
    expect(fxPoint([[0, 0], [10, 20]], 0.5)).toEqual([5, 10]);
  });
  it("mergePaths extends the last point or appends", () => {
    const paths: Record<string, [number, number][]> = {};
    const mk = (fo: number, path?: [number, number][]) => [{ sym: "A", from_open: fo, path } as unknown as SmartStock];
    mergePaths(mk(1, [[0, 0], [5, 0.5]]), paths, true, 5);
    expect(paths.A).toEqual([[0, 0], [5, 1]]);
    mergePaths(mk(2), paths, false, 9);
    expect(paths.A.at(-1)).toEqual([9, 2]);
  });
});

describe("chart helpers", () => {
  const candles = (): Candle[] => [{ time: 1000, open: 10, high: 12, low: 9, close: 11 }];
  it("5m candle buckets are aligned to 09:15 IST", () => {
    // 09:15 IST = 03:45Z = 13500 s into the UTC day; a 5m bucket starts every 300 s from there
    const t0 = Date.parse("2026-11-12T03:45:00Z") / 1000;
    expect(candleTime("5m", t0 + 130, t0 - 300)).toBe(t0);
    expect(candleTime("5m", t0 + 299, t0)).toBe(t0);
    expect(candleTime("5m", t0 + 301, t0)).toBe(t0 + 300);
    // daily candles: the same IST day keeps the last candle, a new day opens one at 09:15 IST
    expect(candleTime("1d", t0 + 100, t0 - 86400)).toBe(t0);
    expect(candleTime("1d", t0 + 100, t0 - 3600)).toBe(t0 - 3600);
  });
  it("applyQuote widens the last candle or opens a new one", () => {
    const t0 = Date.parse("2026-11-12T03:45:00Z") / 1000;
    const c: Candle[] = [{ time: t0, open: 10, high: 12, low: 9, close: 11 }];
    expect(applyQuote(c, { ltp: 13, time: t0 + 60 }, "5m")).toMatchObject({ time: t0, high: 13, low: 9, close: 13 });
    expect(applyQuote(c, { ltp: 8, time: t0 + 70 }, "5m")).toMatchObject({ low: 8, close: 8 });
    expect(applyQuote(c, { ltp: 9, time: t0 + 301 }, "5m")).toEqual({ time: t0 + 300, open: 9, high: 9, low: 9, close: 9 });
    expect(c).toHaveLength(2);
    expect(applyQuote(candles(), { ltp: 0, time: 1 }, "5m")).toBeNull();
  });
  it("priceHTML and index detection", () => {
    expect(priceHTML(110, 100)).toBe('₹110.00 · <span class="up">+10.00%</span>');
    expect(priceHTML(90, 100)).toContain("down");
    expect(priceHTML(null, 1)).toBe("");
    expect(isIdx("NIFTY")).toBe(true);
    expect(isIdx("^NSEI")).toBe(true);
    expect(isIdx("RELIANCE")).toBe(false);
  });
  it("visible range", () => {
    expect(visibleRange(100, "5m")).toEqual({ from: 0, to: 106 });
    expect(visibleRange(500, "1d")).toEqual({ from: 300, to: 506 });
  });
  const read: ChartRead = {
    chart: { vwap: [], events: [{ time: 5, dir: "bull", type: "BOS" }, { time: 6, dir: "bear", type: "CHoCH" }], obs: [{ dir: 1, lo: 1, hi: 2, dist_pct: -1, time: 3 }], fvgs: [{ dir: -1, lo: 5, hi: 6, dist_pct: 2, status: "open", time: 4 }] },
    liquidity: { pdh: 12, pdl: 9, or_hi: 11, or_lo: null, swept: ["PDH"] },
    structure: { last: { type: "BOS", dir: "bull", time: "10:05", level: 101.5 }, breaks_today: 1 },
    order_blocks: [{ dir: 1, lo: 1, hi: 2, dist_pct: -1.5, tested: false }],
    fvgs: [],
    events: [{ time: "10:05", text: "BOS up <b>" }],
    buyers_pct: 61,
    vwap: 100,
    price: 101,
  };
  it("markers, price lines and zones follow the layer switches", () => {
    expect(structureMarkers(read, true).map((m) => [m.shape, m.position, m.text])).toEqual([["arrowUp", "belowBar", "BOS"], ["arrowDown", "aboveBar", "CHoCH"]]);
    expect(structureMarkers(read, false)).toEqual([]);
    expect(liquidityLines(read, true).map((l) => [l.title, l.price])).toEqual([["PDH", 12], ["PDL", 9], ["Opening range", 11]]);
    expect(liquidityLines(read, false)).toEqual([]);
    expect(zoneSpecs(read, { ob: true, fvg: false }).map((z) => z.label)).toEqual(["Order Block 1–2"]);
    expect(zoneSpecs(read, { ob: true, fvg: true }).map((z) => z.label)).toEqual(["Order Block 1–2", "FVG 5–6 · open"]);
  });
  it("levels panel text, with the event text escaped", () => {
    const h = levelsHTML(read);
    expect(h).toContain("BOS up · 10:05");
    expect(h).toContain("price hasn't come back yet");
    expect(h).toContain("Buyers 61%");
    expect(h).toContain("VWAP 100 · price above");
    expect(levelsHTML(null)).toContain("Not enough bars");
  });
});

describe("trades panel", () => {
  it("rrbar: entry 100, stop 98, target 106 is risk 2.00 (2.0%), reward 6.00 (6.0%), 1 : 3.0", () => {
    const h = rrbar(100, 98, 106);
    expect(h).toContain("Risk ₹2.00 (2.0%)");
    expect(h).toContain("Reward ₹6.00 (6.0%)");
    expect(h).toContain("1 : 3.0");
    expect(h).toContain("width:25%");
    expect(rrbar(100, 98, null)).toBe("");
  });
  it("setup state strings", () => {
    expect(tstate({ state: "FORMING", trigger: 100, stop: 95, target: 115, dist_pct: 1.2, mkt_up: true })).toContain("SETUP READY");
    expect(tstate({ state: "SIGNAL", p_in: 100, stop: 95, target: 115 })).toContain("TRIGGERED");
    expect(tstate({ state: "NONE", trend: true, near_high: false, tight: true, from_high: -4 })).toContain("✗ Near 52-week high (-4%)");
    expect(tstate(null)).toBe("");
  });
  it("gate chip and exploratory badge", () => {
    expect(gateChipHTML({ verdict: "PASS", lines_passed: 9 })).toContain("GATE PASS 9/9");
    expect(gateChipHTML({ verdict: "FAIL", lines_passed: 4 })).toContain("GATE FAIL 4/9");
    expect(gateChipHTML(null)).toContain("NOT YET TESTED");
    expect(gateChipHTML(undefined)).toContain("NOT YET TESTED");
    expect(gateChipHTML({ verdict: "PASS", lines_passed: 9 })).toContain("#16a34a");
    expect(gateChipHTML({ verdict: "FAIL", lines_passed: 4 })).toContain("#dc2626");
    expect(exploratoryBadgeHTML()).toContain("EXPLORATORY · net of StockSense costs");
  });
  const TR: TradesResponse = {
    sym: "RELIANCE",
    strat: "VCP tightening break",
    exit: "rr3_40",
    exit_name: "3R, 40d",
    swing: true,
    cost_pct: 0.4,
    exploratory: true,
    gated: { verdict: "FAIL", lines_passed: 4 },
    trades: [
      { t_in: 100, t_out: 200, p_in: 10, p_out: 12, stop: 9, target: 13, side: 1, status: "CLOSED", R: 2, pct: 20 },
      { t_in: 300, p_in: 11, stop: 10, target: 14, side: 1, status: "OPEN" },
    ],
    summary: { n: 2, win: 100, totR: 2, pct: 20, avgR: 1 },
    days_held: [3, 5],
  };
  it("the panel header carries the badge and the gate chip", () => {
    const h = tpanelHTML(TR, "RELIANCE", true);
    expect(h).toContain("SWING SYSTEM");
    expect(h).toContain("after 0.4% costs");
    expect(h).toContain("EXPLORATORY · net of StockSense costs");
    expect(h).toContain("GATE FAIL 4/9");
    expect(h).toContain("Avg hold<b>4d</b>");
    expect(tpanelHTML({ ...TR, gated: null }, "RELIANCE", true)).toContain("NOT YET TESTED");
    // a Market Lab-shaped reply (no exploratory/gated keys) shows neither
    const old = { ...TR } as Partial<TradesResponse>;
    delete old.exploratory;
    delete old.gated;
    expect(tpanelHTML(old as TradesResponse, "RELIANCE", true)).not.toContain("EXPLORATORY");
  });
  it("loading, off, and note states; strategy names are escaped", () => {
    expect(tpanelHTML(null, "RELIANCE", true)).toContain("Loading…");
    expect(tpanelHTML(null, "RELIANCE", false)).toContain("Show swing trades");
    expect(tpanelHTML(null, "NIFTY", false)).toContain("Show intraday trades");
    expect(tpanelHTML({ trades: [], note: "Swing rules are for stocks" }, "NIFTY", true)).toContain("Swing rules are for stocks");
    expect(tpanelHTML({ ...TR, strat: "<img src=x>" }, "RELIANCE", true)).not.toContain("<img");
  });
  it("levels and markers only for the open chart", () => {
    expect(tlevels(TR, "RELIANCE", true)).toEqual([10, 14, 11]);
    expect(tlevels(TR, "TCS", true)).toEqual([]);
    expect(tlevels(TR, "RELIANCE", false)).toEqual([]);
    const m = tradeMarkers(TR, [{ time: 100, open: 1, high: 1, low: 1, close: 1 }, { time: 200, open: 1, high: 1, low: 1, close: 1 }], "RELIANCE", true);
    expect(m.map((x) => x.text)).toEqual(["BUY", "+2R"]);
  });
  it("groups strategies by source, tiers separate, and filters by timeframe", () => {
    const rows: StrategyRow[] = [
      { key: "a", name: "A", source: "marketlab_swing", timeframe: "1d" },
      { key: "b", name: "B", source: "marketlab_intraday", timeframe: "5m" },
      { key: "c", name: "C", source: "tier", tier: "C", timeframe: "1d" },
      { key: "f", name: "F", source: "tier", tier: "F", timeframe: "1d" },
      { key: "k", name: "K", source: "k151", timeframe: "1d" },
      { key: "w", name: "W", source: "flow", timeframe: "5m" },
      { key: "z", name: "Z", source: "mystery" },
    ];
    expect(groupStrategies(rows).map((g) => g.label)).toEqual(["Market Lab swing", "Market Lab intraday", "Tier F", "Tier C", "151 Trading Strategies", "Flow", "Other"]);
    expect(strategiesForTf(rows, "1d").map((r) => r.key)).toEqual(["a", "c", "f", "k", "z"]);
    expect(strategiesForTf(rows, "5m").map((r) => r.key)).toEqual(["b", "w"]);
  });
});

describe("research note", () => {
  const note = research as unknown as ResearchNote;
  it("ring colours by score band", () => {
    expect([80, 70, 50, 40, 10].map(ringColor)).toEqual(["#16a34a", "#65a30d", "#d97706", "#ea580c", "#dc2626"]);
    expect(ring(80, "A")).toContain("OUT OF 100");
  });
  it("qchart", () => {
    expect(qchart([])).toContain("No quarterly results");
    expect(qchart([{ label: "Q1", income: 10, profit: 2 }, { label: "Q2", income: 20, profit: -1 }])).toContain("<polyline");
  });
  it("renders the stored note and escapes every string", () => {
    const html = researchHTML(note);
    expect(html).toContain("RESEARCH NOTE");
    expect(html).toContain("rp-swot");
    const evil = { ...note, name: "<script>alert(1)</script>", what: "<b onmouseover=x>", summary: ["<img src=x onerror=y>"] } as ResearchNote;
    const h = researchHTML(evil);
    expect(h).not.toContain("<script");
    expect(h).not.toContain("<img");
    expect(h).toContain("&lt;script&gt;alert(1)&lt;/script&gt;");
  });
});
