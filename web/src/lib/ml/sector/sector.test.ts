import { afterEach, describe, expect, it, vi } from "vitest";
import horizonFx from "@/test/fixtures/horizon.json";
import sectorFx from "@/test/fixtures/sector.json";
import stockFx from "@/test/fixtures/stock.json";
import { COLORS, hm, horizonSvg, hValues, nowMinOf } from "./horizon";
import { applyTickToHorizon, onScreen, watchList, watchToggle, WATCH_KEY } from "./live";
import { navigate, parseRoute, viewUrl } from "./route";
import type { HorizonData, SectorData, StockData, TickData } from "./types";
import { bubbleTipHTML, groupTipHTML, hSideHTML, lineChartHTML, pinTipHTML, sectorViewHTML, sSideHTML, stockSideHTML, stockTipHTML, stockViewHTML } from "./views";

const H = horizonFx as unknown as HorizonData;
const S = sectorFx as unknown as SectorData;
const K = stockFx as unknown as StockData;

describe("COLORS", () => {
  // Expected strings: sector.html line 267 (COLORS) copied verbatim into Node and run once.
  it("is the exact ramp, saturating at +-3%", () => {
    expect(COLORS(0)).toBe("rgb(160,215,175)");
    expect(COLORS(1.5)).toBe("rgb(100,188,133)");
    expect(COLORS(3)).toBe("rgb(40,160,90)");
    expect(COLORS(9)).toBe("rgb(40,160,90)");
    expect(COLORS(-0.4)).toBe("rgb(237,157,168)");
    expect(COLORS(-3)).toBe("rgb(220,70,90)");
    expect(COLORS(-9)).toBe("rgb(220,70,90)");
  });
  it("hm is the minute offset from 09:15", () => {
    expect([0, 1, 45, 375].map(hm)).toEqual(["09:15", "09:16", "10:00", "15:30"]);
  });
});

describe("routing", () => {
  it("builds query routes", () => {
    expect(viewUrl("horizon")).toBe("/");
    expect(viewUrl("horizon", undefined, "w1")).toBe("/?range=w1");
    expect(viewUrl("sector", "Energy")).toBe("/?view=sector&name=Energy&range=today");
    expect(viewUrl("sector", "Oil & Gas", "m1")).toBe("/?view=sector&name=Oil%20%26%20Gas&range=m1");
    expect(viewUrl("stock", "RELIANCE")).toBe("/?view=stock&sym=RELIANCE");
    expect(viewUrl("stock", "RELIANCE", "m3")).toBe("/?view=stock&sym=RELIANCE&range=m3");
  });
  it("parses them back, falling back to the horizon", () => {
    const p = (q: string) => parseRoute(new URLSearchParams(q));
    expect(p("")).toEqual({ view: "horizon", name: null, sym: null, range: "today" });
    expect(p("view=sector&name=Energy&range=today")).toEqual({ view: "sector", name: "Energy", sym: null, range: "today" });
    expect(p("view=stock&sym=RELIANCE")).toEqual({ view: "stock", name: null, sym: "RELIANCE", range: "today" });
    expect(p("view=sector").view).toBe("horizon"); // no name
    expect(p("view=stock&sym=X&range=bogus").range).toBe("today");
    expect(p("range=m3").range).toBe("m3");
    expect(parseRoute(null).view).toBe("horizon");
  });
  it("navigate pushes the URL", () => {
    navigate("/?view=stock&sym=TCS");
    expect(window.location.search).toBe("?view=stock&sym=TCS");
    navigate("/");
  });
});

describe("horizon", () => {
  it("draws one bubble per sector with the replay values", () => {
    const { secs, nv } = hValues(H, null);
    expect(secs).toHaveLength(20);
    expect(nv).toBe(-1.25);
    const svg = horizonSvg(H, null, 900, 690);
    expect((svg.bubs.match(/class="bub"/g) || []).length).toBe(20);
    // 13 sectors are 1% or more ahead of NIFTY (-1.25) in this fixture
    expect((svg.bubs.match(/>RUNNING</g) || []).length).toBe(13);
    expect(svg.g).toContain("NIFTY 50");
    expect(svg.g).toContain("RUNNING ZONE");
  });
  it("replays: sector values come from the path at minute AT, the trail 30 minutes earlier", () => {
    const r = hValues(H, 30);
    const s0 = H.sectors[0];
    const at = (m: number) => {
      let v: number | null = null;
      for (const q of s0.path) if (q[0] <= m && q[1] != null) v = q[1];
      return v;
    };
    expect(r.secs.find((s) => s.name === s0.name)!.v).toBe(at(30));
    expect(r.nv).toBe(H.snaps.find((s) => s.m === 30)!.nifty);
    expect(nowMinOf(H)).toBe(65);
  });
  it("tick folds sector values and NIFTY into the data", () => {
    const d = structuredClone(H);
    const t = { sectors: { [d.sectors[0].name]: { v: 9.9 }, nope: { v: 1 } }, nifty: { v: 0.5 }, stocks: {}, time: "", market: "open" } as TickData;
    applyTickToHorizon(d, t);
    expect(d.sectors[0].v).toBe(9.9);
    expect(d.nifty.v).toBe(0.5);
  });
  it("escapes sector names and has the tooltip", () => {
    const d = structuredClone(H);
    d.sectors[0].name = "<script>x</script>";
    const svg = horizonSvg(d, null, 900, 690);
    expect(svg.bubs).not.toContain("<script");
    expect(hSideHTML(d, "today")).not.toContain("<script");
    expect(bubbleTipHTML(H, H.sectors[0].name, "today")).toContain("Click to open");
    expect(bubbleTipHTML(H, "nope", "today")).toBeNull();
  });
});

describe("views", () => {
  it("sector view has a hub, groups and nodes; names are escaped", () => {
    const html = sectorViewHTML(S, 1000, 690);
    expect(html).toContain('class="hub"');
    expect((html.match(/class="node grp"/g) || []).length).toBe(S.groups.length);
    expect((html.match(/class="node" data-sym=/g) || []).length).toBe(S.groups.reduce((a, g) => a + Math.min(14, g.stocks.length), 0));
    const evil = structuredClone(S);
    evil.name = "<script>x</script>";
    evil.groups[0].stocks[0].sym = "<img src=x>";
    const e = sectorViewHTML(evil, 1000, 690) + sSideHTML(evil);
    expect(e).not.toContain("<script");
    expect(e).not.toContain("<img");
    expect(stockTipHTML(S.groups[0].stocks[0])).toContain("Shares traded");
    expect(groupTipHTML(S.groups[0])).toContain("Middle stock today");
  });
  it("stock view lists the peers (self excluded) and links with data attributes, not inline handlers", () => {
    const html = stockViewHTML(K, 1000, 690);
    const side = stockSideHTML(K, false);
    const peers = K.peers.filter((p) => p.sym !== K.sym);
    expect((html.match(/class="peer" data-gv="stock"/g) || []).length).toBe(Math.min(11, peers.length));
    expect((side.match(/data-gv="stock"/g) || []).length).toBe(peers.length);
    expect(side).toContain(`Peers · ${peers.length}`);
    expect(html + side).not.toMatch(/onclick=/i);
    expect(side).toContain("☆");
    expect(stockSideHTML(K, true)).toContain("★");
    const evil = structuredClone(K);
    evil.sym = "<script>x</script>";
    expect(stockViewHTML(evil, 1000, 690) + stockSideHTML(evil, false)).not.toContain("<script");
  });
  it("line chart: pins, escapes, empty state", () => {
    expect(lineChartHTML("T", "s", null, 900)).toContain("loads in the background");
    const html = lineChartHTML("T", "s", H.chart, 900);
    expect(html).toContain("<svg>");
    expect(html).toContain('class="pin"');
    const evil = structuredClone(H.chart!);
    const d0 = Object.keys(evil.pins!)[0];
    evil.pins![d0][0].title = "<script>x</script>";
    expect(lineChartHTML("T", "s", evil, 900)).not.toContain("<script");
    expect(pinTipHTML("<b>", "s")).not.toContain("<b>");
  });
});

describe("live and watchlist", () => {
  afterEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });
  it("onScreen picks the visible symbols for each view", () => {
    expect(onScreen("horizon", H)).toEqual([]);
    expect(onScreen("sector", S)).toEqual(S.groups.flatMap((g) => g.stocks.slice(0, 14).map((s) => s.sym)));
    expect(onScreen("stock", K)).toEqual([K.sym, ...K.peers.map((p) => p.sym)]);
    expect(onScreen("stock", null)).toEqual([]);
  });
  it("toggles sm_watch in localStorage", () => {
    expect(watchList()).toEqual([]);
    expect(watchToggle("TCS")).toBe(true);
    expect(JSON.parse(localStorage.getItem(WATCH_KEY)!)).toEqual(["TCS"]);
    expect(watchToggle("INFY")).toBe(true);
    expect(watchToggle("TCS")).toBe(false);
    expect(watchList()).toEqual(["INFY"]);
  });
  it("survives blocked or corrupt storage", () => {
    localStorage.setItem(WATCH_KEY, "{not json");
    expect(watchList()).toEqual([]);
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    expect(watchList()).toEqual([]);
    expect(() => watchToggle("TCS")).not.toThrow();
  });
});
