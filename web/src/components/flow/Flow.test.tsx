import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render } from "@testing-library/react";
import market from "@/test/fixtures/flow_market.json";
import stocks from "@/test/fixtures/flow_stocks.json";
import oi from "@/test/fixtures/flow_oi.json";
import sectors from "@/test/fixtures/flow_sectors_today.json";
import deals from "@/test/fixtures/flow_deals.json";
import participants from "@/test/fixtures/flow_participants.json";
import preopen from "@/test/fixtures/flow_preopen.json";
import { followTick, inPreopenWindow, istMinutes, preopenView, scatterAxes, scatterPoints, scatterXY, sparkPoints, treemap } from "@/lib/flow/charts";
import { QUADS, quadColor, quadStyle } from "@/lib/flow/quad";
import { filterStocks, gammaBadge, sortRows, zeroGammaDistancePct } from "@/lib/flow/table";
import { rowsOf, type FlowStock, type OiRow, type Preopen } from "@/lib/flow/types";

const h = vi.hoisted(() => ({ data: {} as Record<string, unknown>, openChart: vi.fn() }));
vi.mock("@/lib/api", () => ({
  API_BASE: "",
  getJSON: vi.fn(async () => ({})),
  usePoll: vi.fn((path: string | null) => {
    const k = Object.keys(h.data).find((p) => path?.startsWith(p));
    return { data: k ? h.data[k] : null, error: null, refresh: () => {} };
  }),
}));
vi.mock("@/lib/ml/actions", () => ({
  useLabActions: () => ({ openChart: h.openChart, labShow: vi.fn(), go: vi.fn() }),
}));

import { FlowPage } from "./FlowPage";

const ST = (stocks as { rows: FlowStock[] }).rows;
const base = (): Record<string, unknown> => ({
  "/api/flow/market": market,
  "/api/flow/participants": participants,
  "/api/flow/stocks": stocks,
  "/api/flow/preopen": preopen,
  "/api/flow/sectors": sectors,
  "/api/flow/oi": oi,
  "/api/flow/deals": deals,
});

describe("quad chips", () => {
  it("has the four colours: green, amber, orange, red", () => {
    expect(QUADS.map((q) => q.label)).toEqual(["long buildup", "short covering", "long unwinding", "short buildup"]);
    expect(quadStyle("long buildup")).toMatchObject({ fg: "#17603a", bg: "#e8f6ee" });
    expect(quadStyle("short covering")).toMatchObject({ fg: "#8a5a00", bg: "#fdf1d0" });
    expect(quadStyle("long unwinding")).toMatchObject({ fg: "#b4501a", bg: "#fde6d6" });
    expect(quadStyle("short buildup")).toMatchObject({ fg: "#9b1c2e", bg: "#fdecef" });
    expect(quadStyle("nope")).toBeNull();
    expect(quadColor(null)).toBe("#8a93a6");
  });
});

describe("chart geometry", () => {
  it("scatter axes are symmetric whole percents, at least 1", () => {
    expect(scatterAxes([{ x: -2.4, y: 5 }, { x: 1.2, y: -8 }])).toEqual({ xmax: 3, ymax: 8 });
    expect(scatterAxes([])).toEqual({ xmax: 1, ymax: 1 });
    expect(scatterAxes([{ x: 0.2, y: 0.1 }])).toEqual({ xmax: 1, ymax: 1 });
  });
  it("scatter maps zero to the middle, +x right, +y up", () => {
    const ax = { xmax: 3, ymax: 8 };
    const b = { W: 200, H: 200, pad: 0 };
    expect(scatterXY({ x: 0, y: 0 }, ax, b)).toEqual([100, 100]);
    expect(scatterXY({ x: 3, y: 8 }, ax, b)).toEqual([200, 0]);
    expect(scatterXY({ x: -3, y: -8 }, ax, b)).toEqual([0, 200]);
    expect(scatterXY({ x: 1.5, y: 0 }, ax, { W: 200, H: 200, pad: 20 })).toEqual([140, 100]);
  });
  it("scatter points from the oi fixture: one per stock with both values", () => {
    const rows = oi.rows as OiRow[];
    const pts = scatterPoints(rows);
    expect(pts.map((p) => [p.sym, p.x, p.y])).toEqual([["RELIANCE", 1, 5], ["TCS", 1, 3]]);
    expect(scatterAxes(pts)).toEqual({ xmax: 1, ymax: 5 });
    expect(scatterPoints([{ ...rows[0], oi_chg_pct: null }])).toEqual([]);
  });
  it("sparkline points scale into the box", () => {
    expect(sparkPoints([0, 1, 2], 10, 10)).toBe("2.0,8.0 5.0,5.0 8.0,2.0");
    expect(sparkPoints([1, null, 3], 10, 10)).toBe("2.0,8.0 8.0,2.0");
    expect(sparkPoints([1], 10, 10)).toBe("");
  });
  it("treemap area is proportional to the value and tiles the box", () => {
    expect(treemap([3, 1])).toEqual([
      { i: 0, x: 0, y: 0, w: 75, h: 100 },
      { i: 1, x: 75, y: 0, w: 25, h: 100 },
    ]);
    const t = treemap([2, 1, 1]);
    expect(t.find((r) => r.i === 0)).toEqual({ i: 0, x: 0, y: 0, w: 50, h: 100 });
    expect(t.reduce((s, r) => s + r.w * r.h, 0)).toBeCloseTo(10000, 6);
    const areas = treemap([5, 3, 1, 1]).map((r) => (r.w * r.h) / 100);
    expect(areas.map((a) => +a.toFixed(6))).toEqual([50, 30, 10, 10]);
  });
});

describe("pre-open phase", () => {
  const live: Preopen = {
    descriptive: true,
    phase: "live",
    nifty: null,
    breadth: { adv: 1, dec: 1 },
    rows: [
      { symbol: "A", iep: 1, iep_gap_pct: 1, imb: 0.1, total_qty: 5 },
      { symbol: "B", iep: 1, iep_gap_pct: -1, imb: -0.9, total_qty: 5 },
      { symbol: "C", iep: 1, iep_gap_pct: 2, imb: 0.5, total_qty: 5 },
    ],
  };
  it("switches on the fixture phase and sorts the live table by |imbalance|", () => {
    expect(preopenView(null).kind).toBe("before");
    expect(preopenView({ ...live, phase: "before" }).kind).toBe("before");
    expect(preopenView(preopen as unknown as Preopen).kind).toBe("after");
    const v = preopenView(live);
    expect(v.kind).toBe("live");
    expect(v.kind === "live" && v.rows.map((r) => r.symbol)).toEqual(["B", "C", "A"]);
    expect(preopenView({ ...live, phase: "after" }).kind).toBe("after");
  });
  it("follow-through ticks", () => {
    expect([followTick(true), followTick(false), followTick(null)]).toEqual(["✓", "✗", "–"]);
  });
  it("the poll window is 09:00 to 09:15 IST", () => {
    expect(istMinutes(new Date(Date.UTC(2026, 10, 25, 3, 30)))).toBe(540);
    expect(inPreopenWindow(new Date(Date.UTC(2026, 10, 25, 3, 30)))).toBe(true);
    expect(inPreopenWindow(new Date(Date.UTC(2026, 10, 25, 3, 45)))).toBe(true);
    expect(inPreopenWindow(new Date(Date.UTC(2026, 10, 25, 3, 46)))).toBe(false);
    expect(inPreopenWindow(new Date(Date.UTC(2026, 10, 25, 3, 29)))).toBe(false);
  });
});

describe("table helpers", () => {
  it("sorts with missing values last in both directions", () => {
    const r = (sym: string, flow_score: number | null) => ({ ...ST[0], sym, flow_score });
    const rows = [r("A", 1), r("B", null), r("C", 3), r("D", 2)];
    expect(sortRows(rows, "flow_score", "desc").map((x) => x.sym)).toEqual(["C", "D", "A", "B"]);
    expect(sortRows(rows, "flow_score", "asc").map((x) => x.sym)).toEqual(["A", "D", "C", "B"]);
    expect(sortRows(rows, "sym", "desc").map((x) => x.sym)).toEqual(["D", "C", "B", "A"]);
  });
  it("filters by quad and sector", () => {
    expect(filterStocks(ST, "long buildup", "all").map((x) => x.sym)).toEqual(["RELIANCE"]);
    expect(filterStocks(ST, "all", "REALTY").every((x) => x.sector === "REALTY")).toBe(true);
    expect(filterStocks(ST, "long buildup", "REALTY")).toEqual([]);
  });
  it("gamma badge and zero-gamma distance", () => {
    expect(gammaBadge({ regime: "positive" } as never).text).toBe("POSITIVE GAMMA · range / mean-reverting");
    expect(gammaBadge({ regime: "negative" } as never).text).toBe("NEGATIVE GAMMA · trending / volatile");
    expect(gammaBadge(null).kind).toBe("none");
    expect(zeroGammaDistancePct({ spot: 24000, zero_gamma: 24240 } as never)).toBeCloseTo(1, 9);
    expect(zeroGammaDistancePct({ spot: 24000, zero_gamma: null } as never)).toBeNull();
  });
  it("rowsOf unwraps {descriptive, rows} and tolerates bare lists", () => {
    expect(rowsOf({ descriptive: true, rows: [1, 2] })).toEqual([1, 2]);
    expect(rowsOf([3])).toEqual([3]);
    expect(rowsOf(null)).toEqual([]);
  });
});

describe("FlowPage", () => {
  beforeEach(() => {
    h.data = base();
    h.openChart.mockClear();
  });
  afterEach(cleanup);

  it("renders the market strip and a descriptive footer on every card", () => {
    const { container } = render(<FlowPage />);
    expect(container.querySelectorAll("#strip > .fl-card")).toHaveLength(6);
    expect(container.querySelector("#g-NIFTY .fl-badge")).toHaveTextContent("POSITIVE GAMMA · range / mean-reverting");
    expect(container.querySelector("#c-fii")).toHaveTextContent("+250 cr");
    expect(container.querySelector("#c-fii")).toHaveTextContent("+150 cr");
    const cards = container.querySelectorAll(".fl-card");
    const foots = container.querySelectorAll(".fl-card .fl-foot");
    expect(foots).toHaveLength(cards.length);
    expect(foots[0]).toHaveTextContent(
      "Descriptive: shows where money moved; whether it predicts returns is tested on the Strategies page (m-flow).",
    );
  });

  it("renders every F&O stock in the flow table; headers sort; a row opens the chart", () => {
    const { container } = render(<FlowPage />);
    const rows = () => Array.from(container.querySelectorAll("#flowtbl tbody tr"));
    expect(rows()).toHaveLength(ST.length);
    const scores = rows().map((r) => r.querySelector("td:last-child")!.textContent);
    expect(scores[0]).toBe(String(Math.max(...ST.map((s) => s.flow_score ?? -1)).toFixed(3)));
    fireEvent.click(container.querySelector('th[data-key="sym"]')!);
    expect(rows().map((r) => r.getAttribute("data-sym"))).toEqual(ST.map((s) => s.sym).sort());
    fireEvent.click(rows()[0]);
    expect(h.openChart).toHaveBeenCalledWith(rows()[0].getAttribute("data-sym"));
  });

  it("quad chips use the quad colours and the quad filter narrows the table", () => {
    const { container } = render(<FlowPage />);
    const chip = container.querySelector('#flowtbl [data-quad="long buildup"]') as HTMLElement;
    expect(chip).toHaveStyle({ color: "rgb(23, 96, 58)" });
    fireEvent.change(container.querySelector("#fquad")!, { target: { value: "short buildup" } });
    expect(container.querySelectorAll("#flowtbl tbody tr")).toHaveLength(1);
  });

  it("pre-open board follows the fixture phase", () => {
    const { container, rerender } = render(<FlowPage />);
    expect(container.querySelector("#c-preopen")).toHaveTextContent("No pre-open rows were captured today");
    h.data["/api/flow/preopen"] = {
      descriptive: true,
      phase: "live",
      nifty: null,
      breadth: { adv: 0, dec: 0 },
      rows: [
        { symbol: "A", iep: 1, iep_gap_pct: 1, imb: 0.1, total_qty: 5 },
        { symbol: "B", iep: 1, iep_gap_pct: -1, imb: -0.9, total_qty: 5 },
      ],
    };
    rerender(<FlowPage />);
    expect(container.querySelector("#pre-live")).not.toBeNull();
    expect(Array.from(container.querySelectorAll("#pre-live tbody .fl-sym")).map((t) => t.textContent)).toEqual(["B", "A"]);
    h.data["/api/flow/preopen"] = { ...(h.data["/api/flow/preopen"] as object), phase: "after", rows: [{ symbol: "A", iep: 1, iep_gap_pct: 1, imb: 0.1, total_qty: 5, follow_through: true }] };
    rerender(<FlowPage />);
    expect(container.querySelector("#pre-after .fl-tick")).toHaveTextContent("✓");
  });

  it("draws one tile per sector, the scatter dots and the deals", () => {
    const { container } = render(<FlowPage />);
    expect(container.querySelectorAll("#sectree .fl-tile")).toHaveLength(sectors.rows.length);
    expect(container.querySelectorAll("#c-scatter circle")).toHaveLength(2);
    expect(container.querySelectorAll("#deals tbody tr")).toHaveLength(1);
  });

  it("copes with empty replies", () => {
    h.data = {};
    const { container } = render(<FlowPage />);
    expect(container.querySelectorAll("#flowtbl tbody tr")).toHaveLength(1);
    expect(container.querySelector("#c-preopen")).toHaveTextContent("The board opens at 09:00");
  });
});
