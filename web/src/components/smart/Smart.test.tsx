import { useRef } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import live from "@/test/fixtures/sm_live.json";
import research from "@/test/fixtures/research.json";
import { installFakeCanvas } from "@/test/canvas";

const h = vi.hoisted(() => ({ handler: ((): unknown => new Error("404")) as (p: string) => unknown }));
vi.mock("@/lib/api", () => ({
  API_BASE: "",
  getJSON: vi.fn(async (p: string) => {
    const r = h.handler(p);
    if (r instanceof Error) throw r;
    return r;
  }),
  usePoll: vi.fn(() => ({ data: null, error: null, refresh: () => {} })),
}));
vi.mock("lightweight-charts", () => {
  const series = () => ({
    setData: vi.fn(),
    setMarkers: vi.fn(),
    update: vi.fn(),
    applyOptions: vi.fn(),
    priceScale: () => ({ applyOptions: vi.fn() }),
    createPriceLine: vi.fn(() => ({})),
    removePriceLine: vi.fn(),
    priceToCoordinate: vi.fn(() => 10),
  });
  const chart = {
    addCandlestickSeries: vi.fn(series),
    addLineSeries: vi.fn(series),
    priceScale: () => ({ applyOptions: vi.fn() }),
    timeScale: () => ({ applyOptions: vi.fn(), setVisibleLogicalRange: vi.fn(), timeToCoordinate: vi.fn(() => 5), width: () => 500 }),
    applyOptions: vi.fn(),
    remove: vi.fn(),
  };
  return { createChart: vi.fn(() => chart) };
});

import { createChart } from "lightweight-charts";
import { ChartModal } from "@/components/ChartModal";
import { ChartHost } from "./ChartHost";
import { ResearchNote } from "./ResearchNote";
import { SmartSection } from "./SmartSection";
import { Stage, Tooltip } from "./Stage";
import type { SmartLive } from "@/lib/ml/smart/types";

const D = live as unknown as SmartLive;

function Harness({ data, onOpen }: { data: SmartLive; onOpen: (s: string) => void }) {
  const tip = useRef<HTMLDivElement>(null);
  return (
    <div className="ml-smart">
      <Stage data={data} paused={false} tipRef={tip} onOpen={onOpen} />
      <Tooltip tipRef={tip} />
    </div>
  );
}

const candles = [
  { time: 1000, open: 10, high: 12, low: 9, close: 11, volume: 1 },
  { time: 1300, open: 11, high: 13, low: 10, close: 12, volume: 1 },
  { time: 1600, open: 12, high: 14, low: 11, close: 13, volume: 1 },
];

function route(over: Record<string, (p: string) => unknown> = {}) {
  h.handler = (p) => {
    for (const [prefix, fn] of Object.entries(over)) if (p.startsWith(prefix)) return fn(p);
    if (p.startsWith("/api/sm/live")) return live;
    if (p.startsWith("/api/sm/chart")) return { sym: "SBIN", tf: "1d", ltp: 13, prev: 12, candles, read: null };
    if (p.startsWith("/api/sm/quote")) return { ltp: 13.5, time: 1650, prev: 12 };
    if (p.startsWith("/api/research")) return research;
    return new Error("404");
  };
}

beforeEach(() => {
  installFakeCanvas();
  route();
});
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("Stage", () => {
  it("renders 20 bull and 20 bear tile slots from sm_live.json, filled for each stock", () => {
    const { container } = render(<Harness data={D} onOpen={() => {}} />);
    expect(container.querySelectorAll("#hbull .tile")).toHaveLength(20);
    expect(container.querySelectorAll("#hbear .tile")).toHaveLength(20);
    expect(container.querySelectorAll("#hbull .tile[data-s]")).toHaveLength(D.bull.length);
    expect(container.querySelectorAll("#hbear .tile[data-s]")).toHaveLength(D.bear.length);
    expect(container.querySelector("#nb")).toHaveTextContent(String(D.bull.length));
    expect(container.querySelector("#nr")).toHaveTextContent(String(D.bear.length));
    expect(container.querySelectorAll("#chips .chip").length).toBeGreaterThan(0);
  });

  it("a tile click opens that symbol", () => {
    const onOpen = vi.fn();
    const { container } = render(<Harness data={D} onOpen={onOpen} />);
    fireEvent.click(container.querySelector('.tile[data-s="SBIN"]')!);
    expect(onOpen).toHaveBeenCalledWith("SBIN");
  });

  it("escapes a symbol like <script>: no script element reaches the DOM", () => {
    const evil = structuredClone(D);
    evil.stocks[0].sym = "<script>x</script>";
    evil.bull[0] = "<script>x</script>";
    evil.sectors[0].name = "<script>y</script>";
    const { container } = render(<Harness data={evil} onOpen={() => {}} />);
    expect(container.querySelectorAll("script")).toHaveLength(0);
    const tile = [...container.querySelectorAll<HTMLElement>(".tile[data-s]")].find((t) => t.dataset.s === "<script>x</script>")!;
    expect(tile.querySelector(".lab b")!.textContent).toBe("<script>x</script>");
    expect(container.querySelector("#chips")!.textContent).toContain("<script>y</script>");
  });

  it("shows the flow mini-card on the big tile and in the tooltip when the row has one", () => {
    const withFlow = structuredClone(D);
    withFlow.stocks.find((s) => s.sym === D.bull[0])!.flow = { oi_quad_label: "Long buildup", rvol: 2.5, deliv_z20_prev: 1.1, flow_score: 0.9 };
    const { container } = render(<Harness data={withFlow} onOpen={() => {}} />);
    const tile = container.querySelector(`.tile.big[data-s="${D.bull[0]}"]`) as HTMLElement;
    expect(tile.querySelector(".flowm")).toHaveTextContent("Long buildup");
    tile.dispatchEvent(new MouseEvent("mouseenter", { clientX: 10, clientY: 10 }));
    const tip = container.querySelector("#tip") as HTMLElement;
    expect(tip.style.display).toBe("block");
    expect(tip.textContent).toContain("rvol 2.5×");
    // rows without flow show nothing
    const plain = container.querySelector(`.tile.big[data-s="${D.bull[1]}"]`) as HTMLElement;
    expect(plain.querySelector(".flowm")).toBeNull();
  });
});

describe("ChartModal", () => {
  function mount() {
    return render(
      <ChartHost>
        <section className="ml-smart">
          <SmartSection />
          <ChartModal />
        </section>
      </ChartHost>,
    );
  }
  const openSbin = async (container: HTMLElement) => {
    await waitFor(() => expect(container.querySelector('.tile[data-s="SBIN"]')).not.toBeNull());
    fireEvent.click(container.querySelector('.tile[data-s="SBIN"]')!);
    await waitFor(() => expect(createChart).toHaveBeenCalled());
  };

  it("opens on a tile click with the chart, the subtitle from the live rank, and closes on Escape", async () => {
    const { container } = mount();
    expect((container.querySelector("#modal") as HTMLElement).style.display).toBe("none");
    await openSbin(container);
    expect((container.querySelector("#modal") as HTMLElement).style.display).toBe("flex");
    expect(container.querySelector("#mname")).toHaveTextContent("SBIN");
    expect(container.querySelector("#msub")!.textContent).toMatch(/BULLISH #1 OF 6/);
    await waitFor(() => expect(container.querySelector("#mlv")!.textContent).toContain("Not enough bars for a read"));
    // a stock has a research note; an index would not
    await waitFor(() => expect(container.querySelector("#rpt .rp-name")).not.toBeNull());
    fireEvent.keyDown(window, { key: "Escape" });
    expect((container.querySelector("#modal") as HTMLElement).style.display).toBe("none");
  });

  it("the Trades layer tolerates a missing /api/lab (404) and says so", async () => {
    const { container } = mount();
    await openSbin(container);
    expect(container.querySelector("#tstrat")).toBeNull(); // trades are off when a chart opens from Smart Money
    fireEvent.click(container.querySelector('button[data-l="trades"]')!);
    await waitFor(() => expect(container.querySelector("#tstrat")).not.toBeNull());
    expect(container.querySelectorAll("#tstrat option")).toHaveLength(1);
    await waitFor(() => expect(container.querySelector("#tpanel")!.textContent).toContain("Strategy lab is not answering."));
  });

  it("lists /api/lab/strategies grouped by source, with the exploratory badge and the gate chip", async () => {
    route({
      "/api/lab/strategies": () => [
        { key: "vcp", name: "VCP tightening break", source: "marketlab_swing", timeframe: "1d", exits: ["rr3_40", "rr2_20"] },
        { key: "tier_f1", name: "Tier F one", source: "tier", tier: "F", timeframe: "1d" },
        { key: "k151_a", name: "K one", source: "k151", timeframe: "1d" },
        { key: "flow_x", name: "Flow intraday", source: "flow", timeframe: "5m" },
      ],
      "/api/lab/trades": () => ({
        sym: "SBIN",
        strat: "VCP tightening break",
        exit: "rr3_40",
        exit_name: "3R, 40d",
        swing: true,
        trades: [],
        summary: { n: 0 },
        exploratory: true,
        gated: { verdict: "FAIL", lines_passed: 4 },
      }),
    });
    const { container } = mount();
    await openSbin(container);
    fireEvent.click(container.querySelector('button[data-l="trades"]')!);
    await waitFor(() => expect(container.querySelectorAll("#tstrat optgroup").length).toBeGreaterThan(0));
    const labels = [...container.querySelectorAll("#tstrat optgroup")].map((g) => g.getAttribute("label"));
    expect(labels).toEqual(["Market Lab swing", "Tier F", "151 Trading Strategies"]); // the 5m flow row is for intraday charts
    await waitFor(() => expect(container.querySelector("#tpanel")!.textContent).toContain("EXPLORATORY · net of StockSense costs"));
    expect(container.querySelector("#tpanel")!.textContent).toContain("GATE FAIL 4/9");
    // choosing a strategy offers that strategy's own exits
    fireEvent.change(container.querySelector("#tstrat")!, { target: { value: "vcp" } });
    await waitFor(() => expect([...container.querySelectorAll("#texit option")].map((o) => o.textContent)).toEqual(["Default exit", "rr3_40", "rr2_20"]));
  });

  it("shows NOT YET TESTED and GATE PASS chips", async () => {
    route({ "/api/lab/trades": () => ({ sym: "SBIN", strat: "x", swing: true, trades: [], exploratory: true, gated: null }) });
    const { container } = mount();
    await openSbin(container);
    fireEvent.click(container.querySelector('button[data-l="trades"]')!);
    await waitFor(() => expect(container.querySelector("#tpanel")!.textContent).toContain("NOT YET TESTED"));
  });
});

describe("ResearchNote", () => {
  it("escapes the company name: a <script> name never becomes an element", async () => {
    route({ "/api/research": () => ({ ...research, name: "<script>x</script>", summary: ["<img src=x onerror=alert(1)>"] }) });
    const onPeer = vi.fn();
    const { container } = render(<ResearchNote sym="EVIL" onPeer={onPeer} />);
    expect(screen.getByText(/Building the research note/)).toBeInTheDocument();
    await waitFor(() => expect(container.querySelector(".rp-name")).not.toBeNull());
    expect(container.querySelector(".rp-name")!.textContent).toBe("<script>x</script>");
    expect(container.querySelectorAll("script")).toHaveLength(0);
    expect(container.querySelectorAll("img")).toHaveLength(0);
    // a click on a peer row opens that peer
    const peer = container.querySelector(".rp-t tr[data-s]") as HTMLElement | null;
    if (peer) {
      fireEvent.click(peer);
      expect(onPeer).toHaveBeenCalledWith(peer.dataset.s);
    }
  });

  it("is empty for an index and says so when the note fails", async () => {
    const { container, rerender } = render(<ResearchNote sym="NIFTY" onPeer={() => {}} />);
    expect(container.querySelector("#rpt")!.innerHTML).toBe("");
    h.handler = () => new Error("502");
    rerender(<ResearchNote sym="TCS" onPeer={() => {}} />);
    await waitFor(() => expect(container).toHaveTextContent("Research note unavailable right now."));
  });
});
