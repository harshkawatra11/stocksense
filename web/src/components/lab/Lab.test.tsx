import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import swing from "@/test/fixtures/lab_swing.json";
import strategies from "@/test/fixtures/lab_strategies.json";
import { installFakeCanvas } from "@/test/canvas";
import {
  barMax,
  barWidth,
  boardRows,
  classify,
  expectedMaxSharpeNull,
  filterRows,
  gateKind,
  invNormCdf,
  multipleTestingNote,
} from "@/lib/ml/lab/board";
import { cashBands, hoverIndex, revealed } from "@/lib/ml/lab/equity";
import type { SwingData, SwingStrategy } from "@/lib/ml/lab/types";

const h = vi.hoisted(() => ({ swing: null as unknown, strategies: null as unknown, labShow: vi.fn() }));
vi.mock("@/lib/api", () => ({
  API_BASE: "",
  getJSON: vi.fn(async () => ({})),
  usePoll: vi.fn((path: string | null) => ({
    data: path === "/api/lab/swing" ? h.swing : path === "/api/lab/strategies" ? h.strategies : null,
    error: null,
    refresh: () => {},
  })),
}));
vi.mock("@/lib/ml/actions", () => ({
  useLabActions: () => ({ openChart: vi.fn(), labShow: h.labShow, go: vi.fn() }),
}));

import { LabSection } from "./LabSection";

const SW = swing as unknown as SwingData;
const strat = (key: string, excess: number, oos: number | null = null): SwingStrategy => ({
  key,
  exit: "trail3",
  exit_name: "x",
  n: 1,
  win: 1,
  net: 1,
  excess,
  days: 1,
  IS: null,
  OOS: null,
  OOS_exc: oos,
  years: {},
  gated: null,
});

describe("leaderboard maths", () => {
  it("sorts by excess, best first", () => {
    const rows = boardRows({ a: strat("ml_swing_a", -1), b: strat("ml_swing_b", 4), c: strat("tier_f_c", 1.5) });
    expect(rows.map((r) => r.k)).toEqual(["b", "c", "a"]);
    const fx = boardRows(SW.strategies).map((r) => r.excess);
    expect(fx).toEqual([...fx].sort((a, b) => b - a));
  });
  it("classifies by registry or key prefix, and filters by source and tier", () => {
    expect(classify("tier_c_orb_15")).toEqual({ source: "tier", tier: "C" });
    expect(classify("k151_x")).toEqual({ source: "k151", tier: "none" });
    expect(classify("flow_x")).toEqual({ source: "flow", tier: "none" });
    expect(classify("ml_swing_vcp", new Map([["ml_swing_vcp", { key: "ml_swing_vcp", name: "n", source: "z", tier: "B", timeframe: "1d", gated: null }]]))).toEqual({ source: "z", tier: "B" });
    const rows = boardRows({ a: strat("ml_swing_a", 1), b: strat("tier_f_b", 2), c: strat("tier_b_c", 3), d: strat("k151_d", 0) });
    expect(filterRows(rows, "tier", "all").map((r) => r.k)).toEqual(["c", "b"]);
    expect(filterRows(rows, "tier", "F").map((r) => r.k)).toEqual(["b"]);
    expect(filterRows(rows, "all", "all")).toHaveLength(4);
    expect(filterRows(rows, "marketlab_swing", "F")).toHaveLength(0);
  });
  it("scales bars: floor 0.3, 50 px at the maximum", () => {
    expect(barMax(boardRows({ a: strat("ml_swing_a", 0.1, -0.2) }))).toBe(0.3);
    expect(barMax(boardRows({ a: strat("ml_swing_a", 2, -4) }))).toBe(4);
    expect(barWidth(-4, 4)).toBe(50);
    expect(barWidth(1, 2)).toBe(25);
  });
  it("gate kind", () => {
    expect([gateKind("PASS"), gateKind("FAIL"), gateKind(null)]).toEqual(["pass", "fail", "none"]);
  });
  it("expected best-of-N Sharpe matches explore.expected_max_sharpe_null", () => {
    expect(invNormCdf(0.975)).toBeCloseTo(1.959964, 5);
    expect(invNormCdf(0.5)).toBeCloseTo(0, 9);
    expect(invNormCdf(0.001)).toBeCloseTo(-3.090232, 5);
    // values printed by the Python function
    expect(expectedMaxSharpeNull(174, 1260)).toBeCloseTo(1.2162790864267545, 6);
    expect(expectedMaxSharpeNull(2, 252)).toBeCloseTo(0.5197553442805939, 6);
    expect(expectedMaxSharpeNull(1, 252)).toBe(0);
    expect(multipleTestingNote(16)).toContain("16 strategy×exit combinations shown");
    expect(multipleTestingNote(16)).toContain("≈ 0.81 by luck");
  });
});

describe("equity geometry", () => {
  it("CASH band for regime [1,1,0,0,1] runs from index 2 to 4", () => {
    expect(cashBands([1, 1, 0, 0, 1])).toEqual([{ from: 2, to: 4 }]);
    expect(cashBands([1, 0, 0])).toEqual([{ from: 1, to: 2 }]);
    expect(cashBands([0, 1, 0, 1])).toEqual([{ from: 0, to: 1 }, { from: 2, to: 3 }]);
    expect(cashBands([1, 1])).toEqual([]);
  });
  it("hover index clamps and rounds", () => {
    expect(hoverIndex(40, 400, 11)).toBe(0);
    expect(hoverIndex(216, 400, 11)).toBe(5);
    expect(hoverIndex(999, 400, 11)).toBe(10);
    expect(hoverIndex(0, 400, 11)).toBe(0);
  });
  it("draw-in reveals points over 1.5 s", () => {
    expect(revealed(100, 0)).toBe(2);
    expect(revealed(100, 750)).toBe(50);
    expect(revealed(100, 5000)).toBe(100);
  });
});

describe("LabSection", () => {
  beforeEach(() => {
    installFakeCanvas();
    h.swing = swing;
    h.strategies = strategies;
    h.labShow.mockClear();
  });
  afterEach(cleanup);

  it("renders the radar lists from lab_swing.json", () => {
    const { container } = render(<LabSection />);
    const L = SW.live!;
    expect(container.querySelectorAll("#form .row")).toHaveLength(L.forming.length);
    expect(container.querySelectorAll("#trig .row")).toHaveLength(L.signals.length + L.open.length + (L.recent?.length ?? 0));
    expect(container.querySelector("#ntrig")).toHaveTextContent(`${L.signals.length} triggered · ${L.open.length} open`);
    expect(container.querySelector("#swmk")).toHaveTextContent(L.mkt_up ? "Market filter ON" : "Market filter OFF");
  });

  it("shows the leaderboard sorted by excess with a Gate column and the multiple-testing note", () => {
    const { container } = render(<LabSection />);
    const trs = container.querySelectorAll("#swlb tbody tr");
    expect(trs).toHaveLength(Object.keys(SW.strategies).length);
    const heads = Array.from(container.querySelectorAll("#swlb thead th")).map((t) => t.textContent);
    expect(heads[heads.length - 1]).toBe("Gate");
    expect(trs[0].querySelector("td:last-child")).toHaveTextContent("NOT YET TESTED");
    expect(container.querySelector("#mtnote")).toHaveTextContent(`${trs.length} strategy×exit combinations shown`);
  });

  it("Tier and Source filters narrow the rows; a row click calls labShow with the registry key", () => {
    const { container } = render(<LabSection />);
    fireEvent.change(container.querySelector("#ftier") as HTMLSelectElement, { target: { value: "F" } });
    const trs = container.querySelectorAll("#swlb tbody tr");
    expect(trs).toHaveLength(1);
    expect(trs[0]).toHaveTextContent("RSI(14) oversold cross");
    expect(container.querySelector("#mtnote")).toHaveTextContent("1 strategy×exit combinations shown");
    fireEvent.click(trs[0]);
    expect(h.labShow).toHaveBeenCalledWith("RELIANCE", "tier_f_rsi_oversold", "1d");
    fireEvent.change(container.querySelector("#ftier") as HTMLSelectElement, { target: { value: "all" } });
    fireEvent.change(container.querySelector("#fsrc") as HTMLSelectElement, { target: { value: "flow" } });
    expect(container.querySelectorAll("#swlb tbody tr")).toHaveLength(0);
  });

  it("clicking a radar row opens the chart on the candidate", () => {
    const { container } = render(<LabSection />);
    fireEvent.click(container.querySelector("#form .row") as HTMLElement);
    expect(h.labShow).toHaveBeenCalledWith(SW.live!.forming[0].sym, "candidate", "1d");
  });

  it("escapes symbols (JSX text, no script element)", () => {
    const bad = structuredClone(SW);
    bad.live!.forming[0].sym = "<script>x</script>";
    h.swing = bad;
    const { container } = render(<LabSection />);
    expect(container.querySelector("script")).toBeNull();
    expect(within(container.querySelector("#form") as HTMLElement).getByText(/<script>x<\/script>/)).toBeTruthy();
  });

  it("says so when the lab answers with an error, and loads before data", () => {
    h.swing = null;
    const { rerender } = render(<LabSection />);
    expect(screen.getByText("Loading…")).toBeTruthy();
    h.swing = { error: "run: stocksense lab scan" };
    rerender(<LabSection />);
    expect(screen.getByText("run: stocksense lab scan")).toBeTruthy();
  });
});
