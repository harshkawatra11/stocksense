import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { installFakeCanvas } from "@/test/canvas";
import bars from "@/test/fixtures/of_bars.json";
import profVol from "@/test/fixtures/of_profile_volume.json";
import profTpo from "@/test/fixtures/of_profile_tpo.json";
import delta from "@/test/fixtures/of_delta.json";
import footprint from "@/test/fixtures/of_footprint.json";
import bigtrades from "@/test/fixtures/of_bigtrades.json";
import heatEmpty from "@/test/fixtures/of_heatmap_empty.json";
import timescales from "@/test/fixtures/of_timescales.json";

const h = vi.hoisted(() => ({ calls: [] as string[] }));
vi.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams("sym=reliance&date=2026-11-24") }));
vi.mock("@/lib/api", () => ({
  API_BASE: "",
  getJSON: vi.fn(async (p: string) => {
    h.calls.push(p);
    if (p.startsWith("/api/of/bars")) return bars;
    if (p.startsWith("/api/of/profile") && p.includes("kind=tpo")) return profTpo;
    if (p.startsWith("/api/of/profile")) return profVol;
    if (p.startsWith("/api/of/delta")) return delta;
    if (p.startsWith("/api/of/footprint")) return footprint;
    if (p.startsWith("/api/of/bigtrades")) return bigtrades;
    if (p.startsWith("/api/of/heatmap")) return heatEmpty;
    if (p.startsWith("/api/of/timescales")) return timescales;
    throw new Error("404");
  }),
}));
vi.mock("lightweight-charts", () => {
  const series = () => ({ setData: vi.fn(), setMarkers: vi.fn() });
  const chart = {
    addCandlestickSeries: vi.fn(series),
    addLineSeries: vi.fn(series),
    timeScale: () => ({ fitContent: vi.fn() }),
    subscribeClick: vi.fn(),
    remove: vi.fn(),
  };
  return { createChart: vi.fn(() => chart) };
});

import { OrderFlowPage } from "./OrderFlowPage";

beforeEach(() => {
  installFakeCanvas();
  h.calls.length = 0;
});
afterEach(cleanup);

describe("order-flow workbench", () => {
  it("fetches every /api/of route for the symbol and date", async () => {
    render(<OrderFlowPage />);
    await waitFor(() => expect(screen.getByText(/alignment score 0.40/i)).toBeInTheDocument());
    for (const r of ["bars", "profile", "delta", "footprint", "bigtrades", "heatmap", "timescales"])
      expect(h.calls.some((c) => c.startsWith(`/api/of/${r}?`) && c.includes("sym=RELIANCE"))).toBe(true);
    expect(h.calls).toContain("/api/of/bars?sym=RELIANCE&date=2026-11-24&tf=5");
  });

  it("puts the method label in every panel title and an info link to /learn", async () => {
    render(<OrderFlowPage />);
    await waitFor(() => expect(screen.getByText(/alignment score/i)).toBeInTheDocument());
    const panels = document.querySelectorAll("section.of-panel");
    expect(panels.length).toBe(8);
    panels.forEach((p) => {
      expect(p.querySelector("h2 span")!.textContent).toMatch(/\(.+\)$/);
      expect(p.querySelector("h2 a")!.getAttribute("href")).toMatch(/^\/learn\/\?topic=\d\d-/);
    });
    expect(screen.getByText("Cumulative delta (BVC estimate from 1-minute bars)")).toBeInTheDocument();
  });

  it("shows the heat map reason when depth is not captured, and the big-trade list", async () => {
    render(<OrderFlowPage />);
    await waitFor(() => expect(screen.getByText("depth not captured for this date")).toBeInTheDocument());
    expect(screen.getByText(/5\.4×/)).toBeInTheDocument();
    expect(screen.getByText(/selling absorbed \(bid\)/)).toBeInTheDocument();
    expect(screen.getByText("day type: normal variation")).toBeInTheDocument();
  });
});
