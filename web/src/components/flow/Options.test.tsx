import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render } from "@testing-library/react";
import options from "@/test/fixtures/flow_options.json";
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
import type { OptionStrike, OptionsReply } from "@/lib/flow/types";

const h = vi.hoisted(() => ({
  data: null as unknown,
  paths: [] as string[],
  q: "und=NIFTY",
  setData: vi.fn(),
  create: vi.fn(),
}));
vi.mock("@/lib/api", () => ({
  API_BASE: "",
  getJSON: vi.fn(async () => ({})),
  usePoll: vi.fn((path: string | null) => {
    if (path) h.paths.push(path);
    return { data: h.data, error: null, refresh: () => {} };
  }),
}));
vi.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams(h.q) }));
vi.mock("lightweight-charts", () => {
  const chart = {
    addLineSeries: vi.fn(() => ({ setData: h.setData })),
    timeScale: () => ({ fitContent: vi.fn() }),
    applyOptions: vi.fn(),
    remove: vi.fn(),
  };
  return {
    createChart: vi.fn((...a: unknown[]) => {
      h.create(...a);
      return chart;
    }),
  };
});

import { OptionsPage } from "./OptionsPage";

const S = (strike: number, o: Partial<OptionStrike> = {}): OptionStrike => ({
  strike,
  call_oi: null,
  put_oi: null,
  call_doi: null,
  put_doi: null,
  call_iv: null,
  put_iv: null,
  gex_cr: null,
  ...o,
});

describe("options geometry", () => {
  it("windows 2k+1 strikes around the one nearest to spot", () => {
    const ks = Array.from({ length: 10 }, (_, i) => S(i * 10));
    expect(windowStrikes(ks, 55, 2).map((s) => s.strike)).toEqual([30, 40, 50, 60, 70]);
    expect(windowStrikes(ks, 2, 2).map((s) => s.strike)).toEqual([0, 10, 20, 30, 40]);
    expect(windowStrikes(ks, 500, 2).map((s) => s.strike)).toEqual([50, 60, 70, 80, 90]);
    expect(windowStrikes(ks, 55, 25)).toBe(ks);
  });
  it("scales bars: the largest OI (or delta) is `half` pixels", () => {
    const ks = [S(1, { call_oi: 100, put_oi: 40, call_doi: -20 }), S(2, { put_oi: 50, put_doi: 10 })];
    expect(oiMax(ks)).toBe(100);
    expect(barLen(100, 100, 200)).toBe(200);
    expect(barLen(50, 100, 200)).toBe(100);
    expect(barLen(-20, 100, 200)).toBe(40);
    expect(barLen(null, 100, 200)).toBe(0);
    expect(barLen(5, 0, 200)).toBe(0);
    expect(oiMax([])).toBe(0);
  });
  it("places a level between strike rows (highest strike on top) and pins outliers", () => {
    const ks = [S(23900), S(24000), S(24100)];
    expect(levelY(24100, ks, 20)).toEqual({ y: 10, clamped: false });
    expect(levelY(24000, ks, 20)).toEqual({ y: 30, clamped: false });
    expect(levelY(23900, ks, 20)).toEqual({ y: 50, clamped: false });
    expect(levelY(24050, ks, 20)).toEqual({ y: 20, clamped: false });
    expect(levelY(23950, ks, 20)).toEqual({ y: 40, clamped: false });
    expect(levelY(24200, ks, 20)).toEqual({ y: 10, clamped: true });
    expect(levelY(23800, ks, 20)).toEqual({ y: 50, clamped: true });
    expect(levelY(null, ks, 20)).toBeNull();
    expect(levelY(24000, [], 20)).toBeNull();
  });
  it("cumulative GEX runs from the lowest strike and treats gaps as 0", () => {
    expect(cumulativeGex([S(1, { gex_cr: 1 }), S(2, { gex_cr: 2 }), S(3, { gex_cr: -1 }), S(4)])).toEqual([1, 3, 2, 2]);
  });
  it("linear scale and iv units", () => {
    expect(lin(15, { lo: 10, hi: 20 }, 0, 100)).toBe(50);
    expect(ivPct(0.15)).toBeCloseTo(15, 9);
    expect(ivPct(15)).toBe(15);
  });
  it("smile polylines break at gaps", () => {
    const ks = [S(1, { call_iv: 0.1 }), S(2, { call_iv: 0.2 }), S(3), S(4, { call_iv: 0.3 })];
    const segs = smileSegments(ks, "call_iv", (k) => k, (v) => v);
    expect(segs).toEqual(["1.0,10.0 2.0,20.0", "4.0,30.0"]);
  });
  it("history series are ascending, unique and drop missing values", () => {
    const hs = [
      { date: "2026-11-11", gex_total_cr: 2, pcr_oi: null, spot: 1 },
      { date: "2026-11-10", gex_total_cr: 1, pcr_oi: 1.2, spot: 1 },
      { date: "2026-11-11", gex_total_cr: 3, pcr_oi: 1.3, spot: 1 },
    ];
    expect(historySeries(hs, "gex_total_cr")).toEqual([{ time: "2026-11-10", value: 1 }, { time: "2026-11-11", value: 3 }]);
    expect(historySeries(hs, "pcr_oi").map((p) => p.value)).toEqual([1.2, 1.3]);
    expect(historySeries((options as unknown as OptionsReply).history, "gex_total_cr")).toHaveLength(2);
  });
});

describe("OptionsPage", () => {
  beforeEach(() => {
    h.data = options;
    h.paths = [];
    h.q = "und=NIFTY";
    h.setData.mockClear();
    h.create.mockClear();
  });
  afterEach(cleanup);

  it("reads the underlying from the query and renders ladder, GEX, smile and history", () => {
    const { container } = render(<OptionsPage />);
    expect(h.paths[0]).toBe("/api/flow/options?und=NIFTY");
    expect(container.querySelector("h1")).toHaveTextContent("Options positioning · NIFTY");
    expect(container.querySelectorAll("#c-ladder .fl-lrow")).toHaveLength(2);
    expect(container.querySelectorAll("#c-gex .fl-gbar")).toHaveLength(2);
    expect(container.querySelectorAll("#c-smile polyline").length).toBe(2);
    expect(container.querySelector("#c-hist")).not.toBeNull();
    expect(h.create).toHaveBeenCalledTimes(1);
    expect(h.setData).toHaveBeenCalledTimes(2);
    expect(container.querySelectorAll(".fl-card .fl-foot")).toHaveLength(container.querySelectorAll(".fl-card").length);
  });

  it("mirrors the bars: puts left of the centre line, calls right, at the right scale", () => {
    const { container } = render(<OptionsPage />);
    const row = container.querySelector('.fl-lrow[data-strike="24000"]')!;
    const p = row.querySelector(".fl-pbar")!;
    const c = row.querySelector(".fl-cbar")!;
    // call_oi 100 and put_oi 200 on a common scale with max 200 and half = (640-58-70)/2 = 256
    expect(Number(c.getAttribute("width"))).toBeCloseTo(128, 6);
    expect(Number(p.getAttribute("width"))).toBeCloseTo(256, 6);
    expect(Number(c.getAttribute("x"))).toBeCloseTo(314, 6);
    expect(Number(p.getAttribute("x")) + Number(p.getAttribute("width"))).toBeCloseTo(314, 6);
  });

  it("places spot at its strike row and pins an out-of-range zero-gamma to the edge", () => {
    const { container } = render(<OptionsPage />);
    const spot = container.querySelector('[data-level="spot"]')!;
    expect(spot.getAttribute("data-y")).toBe("7.00"); // top row (24000), mid-row at rowH 14
    expect(spot.getAttribute("data-clamped")).toBe("0");
    const zg = container.querySelector('[data-level="zero_gamma"]')!;
    expect(zg.getAttribute("data-y")).toBe("21.00"); // fixture zero-gamma 1.0 is below the range: pinned to the last row
    expect(zg.getAttribute("data-clamped")).toBe("1");
  });

  it("puts an in-range zero-gamma between its two strikes", () => {
    const d = structuredClone(options) as unknown as OptionsReply;
    d.totals.zero_gamma = 23950;
    h.data = d;
    const { container } = render(<OptionsPage />);
    const zg = container.querySelector('[data-level="zero_gamma"]')!;
    expect(zg.getAttribute("data-y")).toBe("14.00");
    expect(zg.getAttribute("data-clamped")).toBe("0");
    expect(container.querySelector("#c-gex .fl-zg")).not.toBeNull();
  });

  it("the expiry selector lists the expiries and refetches with ?expiry=", () => {
    const { container } = render(<OptionsPage />);
    const sel = container.querySelector("#expiry") as HTMLSelectElement;
    expect(Array.from(sel.options).map((o) => o.value)).toEqual(["2026-11-17", "2026-11-24"]);
    fireEvent.change(sel, { target: { value: "2026-11-24" } });
    expect(h.paths[h.paths.length - 1]).toBe("/api/flow/options?und=NIFTY&expiry=2026-11-24");
  });

  it("takes BANKNIFTY from the query and shows a message without data", () => {
    h.q = "und=banknifty";
    h.data = null;
    const { container } = render(<OptionsPage />);
    expect(h.paths[0]).toBe("/api/flow/options?und=BANKNIFTY");
    expect(container.querySelector("h1")).toHaveTextContent("BANKNIFTY");
    expect(container).toHaveTextContent("Loading…");
  });
});
