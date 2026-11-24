import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render } from "@testing-library/react";
import horizonFx from "@/test/fixtures/horizon.json";
import sectorFx from "@/test/fixtures/sector.json";
import stockFx from "@/test/fixtures/stock.json";
import searchFx from "@/test/fixtures/search.json";
import { installFakeCanvas } from "@/test/canvas";

const h = vi.hoisted(() => ({
  qs: "",
  calls: [] as string[],
  handler: ((): unknown => new Error("404")) as (p: string) => unknown,
}));
vi.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams(h.qs) }));
vi.mock("@/lib/api", () => ({
  API_BASE: "",
  getJSON: vi.fn(async (p: string) => {
    h.calls.push(p);
    const r = h.handler(p);
    if (r instanceof Error) throw r;
    return r;
  }),
  usePoll: vi.fn(() => ({ data: null, error: null, refresh: () => {} })),
}));

import { LabActionsProvider } from "@/lib/ml/actions";
import { WATCH_KEY } from "@/lib/ml/sector/live";
import type { StockData } from "@/lib/ml/sector/types";
import { SectorSection, SEARCH_DEBOUNCE_MS } from "./SectorSection";

const K = stockFx as unknown as StockData;

function route(over: Record<string, (p: string) => unknown> = {}) {
  h.handler = (p) => {
    for (const [prefix, fn] of Object.entries(over)) if (p.startsWith(prefix)) return fn(p);
    if (p.startsWith("/api/status")) return { classified: 20 };
    if (p.startsWith("/api/horizon")) return horizonFx;
    if (p.startsWith("/api/sector")) return sectorFx;
    if (p.startsWith("/api/stock")) return stockFx;
    if (p.startsWith("/api/search")) return searchFx;
    if (p.startsWith("/api/tick")) return { stocks: {}, sectors: {}, nifty: { v: null }, time: "2026-11-12T11:15:30+05:30", market: "open" };
    return new Error("404");
  };
}
// fake timers: flush microtasks and let timers of `ms` elapse
const flush = async (ms = 0) => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
};

beforeEach(() => {
  vi.useFakeTimers();
  installFakeCanvas();
  h.qs = "";
  h.calls = [];
  route();
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
  localStorage.clear();
});

describe("Horizon", () => {
  it("renders 20 bubbles from horizon.json, the side lists and the crumbs", async () => {
    const { container } = render(<SectorSection />);
    await flush();
    expect(container.querySelectorAll("#view .bub")).toHaveLength(20);
    expect(container.querySelector("#side")!.textContent).toContain("All 20 sectors · Today");
    expect(container.querySelector("#bottom svg")).not.toBeNull();
    expect(container.querySelector("#crumbs")!.textContent).toBe("All sectors");
    expect(h.calls.some((c) => c === "/api/horizon?range=today")).toBe(true);
  });

  it("a bubble click navigates to the sector query route; a range click keeps the view", async () => {
    const push = vi.spyOn(window.history, "pushState");
    const { container, getByText } = render(<SectorSection />);
    await flush();
    fireEvent.click(container.querySelector('.bub[data-s="Energy"]')!);
    expect(push).toHaveBeenLastCalledWith(null, "", "/?view=sector&name=Energy&range=today");
    fireEvent.click(getByText("1 week"));
    expect(push).toHaveBeenLastCalledWith(null, "", "/?range=w1");
  });

  it("the replay slider moves the bubbles to that minute and says when", async () => {
    const { container } = render(<SectorSection />);
    await flush();
    const slider = container.querySelector("#slider") as HTMLInputElement;
    slider.value = "30";
    fireEvent.input(slider);
    expect(container.querySelector("#at")!.textContent).toBe("At 09:45");
    slider.value = "375";
    fireEvent.input(slider);
    expect(container.querySelector("#at")!.textContent).toBe("Now · live");
  });

  it("shows 'Preparing the map' while the server classifies, and retries every 4 s", async () => {
    let n = 0;
    route({ "/api/status": () => (n++ < 1 ? { classified: 0, progress: "loading history" } : { classified: 20 }) });
    const { container } = render(<SectorSection />);
    await flush();
    expect(container.querySelector("#view")!.textContent).toContain("Preparing the map: loading history…");
    await flush(4000);
    expect(container.querySelectorAll("#view .bub")).toHaveLength(20);
  });

  it("an API without a `classified` field is taken as ready; a failure says it will retry", async () => {
    route({ "/api/status": () => ({}), "/api/horizon": () => new Error("/api/horizon: 503") });
    const { container } = render(<SectorSection />);
    await flush();
    expect(container.querySelector("#view")!.textContent).toContain("Could not load (/api/horizon: 503). Retrying…");
    route({ "/api/status": () => ({}) });
    await flush(30000); // the 30 s refresh
    expect(container.querySelectorAll("#view .bub")).toHaveLength(20);
  });

  it("ticks the live pill and the bubbles every second, only while the tab is visible", async () => {
    const { container } = render(<SectorSection />);
    await flush();
    const n0 = h.calls.filter((c) => c.startsWith("/api/tick")).length;
    await flush(1000);
    expect(h.calls.filter((c) => c.startsWith("/api/tick")).length).toBe(n0 + 1);
    expect(container.querySelector("#livep")!.textContent).toContain("Live · NSE indices · 11:15:30");
  });
});

describe("Sector view", () => {
  it("renders the hub, group nodes and the side card", async () => {
    h.qs = "view=sector&name=Energy&range=today";
    const { container } = render(<SectorSection />);
    await flush();
    expect(container.querySelector("#view .hub")).not.toBeNull();
    expect(container.querySelectorAll("#view .node.grp").length).toBeGreaterThan(0);
    expect(h.calls).toContain("/api/sector?name=Energy&range=today");
    expect(container.querySelector("#side")!.textContent).toContain("OF 20");
    expect(container.querySelector("#crumbs")!.textContent).toContain("Energy");
    // tooltips are read from the data
    const node = container.querySelector("#view .node[data-sym]")!;
    fireEvent.mouseMove(node, { clientX: 20, clientY: 20 });
    expect((container.querySelector("#stip") as HTMLElement).style.display).toBe("block");
    expect(container.querySelector("#stip")!.textContent).toContain("Shares traded");
    fireEvent.click(node);
  });
});

describe("Stock view", () => {
  const peerCount = K.peers.filter((p) => p.sym !== K.sym).length;

  it("renders the peers from stock.json, in the view and in the side list", async () => {
    h.qs = `view=stock&sym=${K.sym}`;
    const { container } = render(<SectorSection />);
    await flush();
    expect(container.querySelectorAll("#view .peers .peer[data-gv]")).toHaveLength(Math.min(11, peerCount));
    expect(container.querySelectorAll('#side .kv[data-gv="stock"]')).toHaveLength(peerCount);
    expect(container.querySelector("#side")!.textContent).toContain(`Peers · ${peerCount}`);
    expect(container.querySelector("#crumbs")!.textContent).toContain(K.sym);
    expect(container.querySelector("#crumbs a:nth-of-type(2)")!.textContent).toBe(K.sector!.name);
  });

  it("a peer click goes to that stock", async () => {
    h.qs = `view=stock&sym=${K.sym}`;
    const push = vi.spyOn(window.history, "pushState");
    const { container } = render(<SectorSection />);
    await flush();
    const p = container.querySelector("#view .peer[data-gv]") as HTMLElement;
    fireEvent.click(p);
    expect(push).toHaveBeenLastCalledWith(null, "", `/?view=stock&sym=${encodeURIComponent(p.dataset.ga!)}`);
  });

  it("Open chart uses the shared LabActions.openChart", async () => {
    h.qs = `view=stock&sym=${K.sym}`;
    const openChart = vi.fn();
    const { container } = render(
      <LabActionsProvider value={{ openChart }}>
        <SectorSection />
      </LabActionsProvider>,
    );
    await flush();
    fireEvent.click(container.querySelector("#openc")!);
    expect(openChart).toHaveBeenCalledWith(K.sym);
  });

  it("the star toggles the sm_watch watchlist and survives blocked storage", async () => {
    h.qs = `view=stock&sym=${K.sym}`;
    const { container } = render(<SectorSection />);
    await flush();
    const star = container.querySelector("#star")!;
    expect(star.textContent).toBe("☆");
    fireEvent.click(star);
    expect(star.textContent).toBe("★");
    expect(JSON.parse(localStorage.getItem(WATCH_KEY)!)).toEqual([K.sym]);
    fireEvent.click(star);
    expect(star.textContent).toBe("☆");
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    expect(() => fireEvent.click(star)).not.toThrow();
  });

  it("the tick moves the price in the hub and the side card to the latest quote and flashes it", async () => {
    h.qs = `view=stock&sym=${K.sym}`;
    route({ "/api/tick": () => ({ stocks: { [K.sym]: { ltp: 1234.5, pct: 0.5 } }, sectors: {}, nifty: { v: null }, time: "2026-11-12T11:15:30+05:30", market: "open" }) });
    const { container } = render(<SectorSection />);
    await flush();
    await flush(1000);
    expect(container.querySelector(`[data-k="sp:${K.sym}"]`)!.textContent).toBe("₹1,234.50");
    expect(container.querySelector(`[data-k="sr:${K.sym}"]`)!.textContent).toBe("₹1,234.50");
    expect(h.calls.find((c) => c.startsWith("/api/tick?syms="))).toContain(encodeURIComponent(K.sym));
  });

  it("escapes a <script> symbol everywhere", async () => {
    h.qs = "view=stock&sym=EVIL";
    const evil = structuredClone(K);
    evil.sym = "<script>x</script>";
    evil.peers[0].sym = "<img src=x onerror=y>";
    route({ "/api/stock": () => evil });
    const { container } = render(<SectorSection />);
    await flush();
    expect(container.querySelectorAll("script")).toHaveLength(0);
    expect(container.querySelectorAll("img")).toHaveLength(0);
    expect(container.querySelector("#side")!.textContent).toContain("<script>x</script>");
  });
});

describe("Search", () => {
  it("waits 180 ms after the last keystroke, then lists sectors and stocks", async () => {
    expect(SEARCH_DEBOUNCE_MS).toBe(180);
    const { container } = render(<SectorSection />);
    await flush();
    const input = container.querySelector("#q") as HTMLInputElement;
    const searches = () => h.calls.filter((c) => c.startsWith("/api/search")).length;
    fireEvent.change(input, { target: { value: "en" } });
    await flush(100);
    fireEvent.change(input, { target: { value: "ene" } });
    await flush(179);
    expect(searches()).toBe(0);
    await flush(1);
    expect(searches()).toBe(1);
    expect(h.calls.find((c) => c.startsWith("/api/search"))).toBe("/api/search?q=ene");
    const res = container.querySelector("#results") as HTMLElement;
    expect(res.style.display).toBe("block");
    expect(res.querySelectorAll("[data-sec]").length).toBeGreaterThan(0);
    expect(res.querySelectorAll("[data-sym]").length).toBeGreaterThan(0);
    // picking a result navigates and clears the box
    const push = vi.spyOn(window.history, "pushState");
    fireEvent.click(res.querySelector("[data-sec]")!);
    expect(push).toHaveBeenLastCalledWith(null, "", expect.stringContaining("/?view=sector&name="));
    expect(input.value).toBe("");
    expect(res.style.display).toBe("none");
  });

  it("an empty box closes the list and says 'Nothing found' for no hits", async () => {
    route({ "/api/search": () => [] });
    const { container } = render(<SectorSection />);
    await flush();
    const input = container.querySelector("#q") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "zzz" } });
    await flush(180);
    expect(container.querySelector("#results")!.textContent).toBe("Nothing found");
    fireEvent.change(input, { target: { value: "  " } });
    await flush(180);
    expect((container.querySelector("#results") as HTMLElement).style.display).toBe("none");
  });
});
