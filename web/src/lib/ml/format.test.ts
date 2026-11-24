import { describe, expect, it } from "vitest";
import { cls, esc, pc, rf, rs, rsg, sg } from "@/lib/ml/format";

describe("format", () => {
  it("pc", () => {
    expect(pc(1.234)).toBe("+1.23%");
    expect(pc(null)).toBe("–");
    expect(pc(-0.5)).toBe("-0.50%");
    expect(pc(0)).toBe("0.00%");
    expect(pc(NaN)).toBe("–");
  });
  it("rs", () => {
    expect(rs(1234.5)).toBe("₹1,234.50");
    expect(rs(null)).toBe("–");
    expect(rs(1234567.891)).toBe("₹12,34,567.89");
  });
  it("esc", () => {
    expect(esc('<a "b">')).toBe("&lt;a &quot;b&quot;&gt;");
    expect(esc(null)).toBe("");
    expect(esc("a&b")).toBe("a&amp;b");
  });
  it("cls, sg, rf, rsg", () => {
    expect(cls(1)).toBe("up");
    expect(cls(-1)).toBe("down");
    expect(cls(0)).toBe("");
    expect(sg(3)).toBe("+3");
    expect(sg(-3)).toBe("-3");
    expect(sg(null)).toBe("–");
    expect(rf(1234567, 0)).toBe("12,34,567");
    expect(rf(Infinity)).toBe("–");
    expect(rsg(5)).toBe("+5");
    expect(rsg(-5)).toBe("-5");
    expect(rsg(null)).toBe("–");
  });
});
