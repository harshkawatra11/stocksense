import { describe, expect, it } from "vitest";
import {
  binIndex,
  cellText,
  footprintCellColor,
  heatColor,
  HEAT_STOPS,
  inStack,
  panelTitle,
  priceY,
  profileWidths,
  tpoGrid,
} from "./draw";
import type { FpBar, TpoProfile, VolumeProfile } from "./types";
import fp from "@/test/fixtures/of_footprint.json";
import tpo from "@/test/fixtures/of_profile_tpo.json";
import vol from "@/test/fixtures/of_profile_volume.json";

describe("footprint cell colour", () => {
  it("is green for buying, red for selling, alpha scaled by |delta|/max", () => {
    expect(footprintCellColor(500, 1000)).toBe("rgba(47,158,95,0.400)"); // 0.1 + 0.6*0.5
    expect(footprintCellColor(-1000, 1000)).toBe("rgba(214,69,93,0.700)");
    expect(footprintCellColor(5000, 1000)).toBe("rgba(47,158,95,0.700)"); // clamped
  });
  it("is neutral for zero delta or no scale", () => {
    expect(footprintCellColor(0, 1000)).toContain("120,130,150");
    expect(footprintCellColor(10, 0)).toContain("120,130,150");
  });
  it("labels a cell sell x buy and finds stacked rows", () => {
    expect(cellText({ price: 1, buy: 1200.4, sell: 799.6, delta: 400 })).toBe("800×1200");
    const b = (fp.bars as FpBar[])[3];
    expect(inStack(b, b.stacked_imbalances[0].from_price)).toBe("buy");
    expect(inStack(b, 99999)).toBeNull();
  });
});

describe("profile scaling", () => {
  it("gives the largest bin the full width and scales the rest linearly", () => {
    expect(profileWidths([10, 20, 40, 20, 10], 100)).toEqual([25, 50, 100, 50, 25]);
    expect(profileWidths([0, 0], 100)).toEqual([0, 0]);
  });
  it("maps prices to y with the high at the top", () => {
    expect(priceY(110, 100, 110, 200)).toBe(0);
    expect(priceY(100, 100, 110, 200)).toBe(200);
    expect(priceY(105, 100, 110, 200, 10)).toBe(100);
  });
  it("finds the bin of the POC in the fixture", () => {
    const p = vol as unknown as VolumeProfile;
    expect(binIndex(p, p.poc)).toBe(14);
    expect(binIndex(p, 1000)).toBe(-1);
  });
});

describe("TPO grid", () => {
  const rows = tpoGrid(tpo as unknown as TpoProfile);
  it("lists the highest price first, one row per bin", () => {
    expect(rows.length).toBe(30);
    expect(rows[0].price).toBe(1523);
    expect(rows[29].price).toBe(1494);
  });
  it("flags single prints, the POC row and the initial balance", () => {
    expect(rows.filter((r) => r.single).map((r) => r.price)).toEqual([1521, 1520]);
    expect(rows.filter((r) => r.poc).map((r) => r.price)).toEqual([1508]);
    expect(rows.find((r) => r.price === 1508)!.letters.length).toBeGreaterThan(0);
    expect(rows.find((r) => r.price === 1504)!.ib).toBe(true);
    expect(rows.find((r) => r.price === 1513)!.ib).toBe(false);
    expect(rows.find((r) => r.price === 1510)!.va).toBe(true);
  });
});

describe("heat ramp and titles", () => {
  it("hits the first and last stop exactly and clamps", () => {
    expect(heatColor(0)).toBe(`rgb(${HEAT_STOPS[0].join(",")})`);
    expect(heatColor(1)).toBe(`rgb(${HEAT_STOPS[4].join(",")})`);
    expect(heatColor(9)).toBe(heatColor(1));
    expect(heatColor(0.5)).toBe(`rgb(${HEAT_STOPS[2].join(",")})`);
  });
  it("puts the method in the panel title", () => {
    expect(panelTitle("Delta", "BVC estimate from 1-minute bars")).toBe("Delta (BVC estimate from 1-minute bars)");
  });
});
