import { describe, expect, it } from "vitest";
import { gateOf, median, pivot, rColor, tierSentence, tierSummary, type LibRow, type MatrixRow } from "./strategies";

const row = (key: string, tier: string, gated: LibRow["gated"]): LibRow => ({
  key,
  name: key,
  source: "s",
  tier,
  timeframe: "1d",
  kind: "trend",
  gated,
});

describe("gate normalising", () => {
  it("reads a verdict string and an object", () => {
    expect(gateOf("PASS")).toMatchObject({ tested: true, pass: true, lines: null });
    expect(gateOf("fail").pass).toBe(false);
    expect(gateOf({ verdict: "FAIL", lines_passed: 6, net_sharpe_5l: 0.4 })).toMatchObject({ tested: true, lines: 6, sharpe5: 0.4 });
    expect(gateOf(null)).toMatchObject({ tested: false, pass: false, verdict: null });
  });
});

describe("tier vs evidence", () => {
  const obj = (s: number, v = "FAIL") => ({ verdict: v, net_sharpe_5l: s });
  const rows = [
    row("f1", "F", obj(0.9)),
    row("f2", "F", obj(0.5)),
    row("c1", "C", obj(0.2, "PASS")),
    row("b1", "B", obj(-0.1)),
    row("b2", "B", obj(0.3)),
    row("b3", "B", null),
    row("n1", "none", obj(5, "PASS")),
  ];
  it("counts and takes the median per tier, ignoring the none tier", () => {
    const s = tierSummary(rows);
    expect(s.map((x) => x.tier)).toEqual(["F", "C", "B"]);
    expect(s[0]).toMatchObject({ n: 2, tested: 2, passing: 0, medianSharpe: 0.7 });
    expect(s[1]).toMatchObject({ n: 1, passing: 1, medianSharpe: 0.2 });
    expect(s[2]).toMatchObject({ n: 3, tested: 2 });
    expect(s[2].medianSharpe).toBeCloseTo(0.1, 12); // median of -0.1 and 0.3
  });
  it("states claimed vs measured order", () => {
    expect(tierSentence(tierSummary(rows))).toBe("Claimed order F < C < B; measured order B < C < F.");
  });
  it("says so when nothing is measured yet", () => {
    const s = tierSentence(tierSummary([row("f", "F", null), row("c", "C", "FAIL")]));
    expect(s).toContain("not available yet");
  });
  it("median of an even and an empty list", () => {
    expect(median([3, 1, 2, 4])).toBe(2.5);
    expect(median([])).toBeNull();
  });
});

describe("matrix helpers", () => {
  const mk = (strat: string, exit: string, avgR: number | null): MatrixRow => ({ strat, name: strat.toUpperCase(), tier: "none", exit, n: 3, avgR });
  it("pivots rows into strategies x exits", () => {
    const p = pivot([mk("a", "x", 1), mk("a", "y", 2), mk("b", "x", 3)]);
    expect(p.exits).toEqual(["x", "y"]);
    expect(p.strats.map((s) => s.strat)).toEqual(["a", "b"]);
    expect(p.strats[0].cells.y.avgR).toBe(2);
    expect(p.strats[1].cells.y).toBeUndefined();
  });
  it("colours by sign and saturates at 0.5 R", () => {
    expect(rColor(0.25)).toBe("rgba(47,158,95,0.33)");
    expect(rColor(-2)).toBe("rgba(214,69,93,0.65)");
    expect(rColor(null)).toBe("#f1f3f6");
    expect(rColor(1, 0)).toBe("#f1f3f6");
  });
});
