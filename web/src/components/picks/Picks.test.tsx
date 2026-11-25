import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import picks from "@/test/fixtures/picks.json";
import history from "@/test/fixtures/picks_history.json";
import { LabActionsProvider } from "@/lib/ml/actions";

const h = vi.hoisted(() => ({ picks: null as unknown }));
vi.mock("@/lib/api", () => ({
  API_BASE: "",
  getJSON: vi.fn(async (p: string) => {
    if (p === "/api/picks") return h.picks;
    if (p.startsWith("/api/picks/history")) return history;
    throw new Error("404");
  }),
}));

import { PicksPage } from "./PicksPage";
import { evidencePct, EXPLORATORY_LABEL, FOOTER, headerNote, type PicksReply } from "@/lib/picks";

afterEach(cleanup);

describe("daily picks page", () => {
  it("renders gated cards and a separate exploratory section carrying the label", async () => {
    h.picks = picks;
    const openChart = vi.fn();
    render(
      <LabActionsProvider value={{ openChart }}>
        <PicksPage />
      </LabActionsProvider>,
    );
    const gated = await screen.findByRole("heading", { name: "Gate-passing picks" });
    const g = gated.closest("section")!;
    expect(within(g).getAllByText("tier_b_vcp").length).toBe(2);
    expect(within(g).getAllByText(/Paper qty/).length).toBe(2);
    expect(within(g).queryByText(EXPLORATORY_LABEL)).toBeNull();
    expect(within(g).queryByText("ZOMATO")).toBeNull();
    const e = document.getElementById("exploratory")!;
    expect(within(e).getByText(/ZOMATO/)).toBeInTheDocument();
    expect(within(e).getAllByText(EXPLORATORY_LABEL).length).toBe(1 + 2); // section label + one per card
    expect(e.querySelectorAll(".pk-card.expl").length).toBe(2);
    expect(g.querySelectorAll(".pk-card.expl").length).toBe(0);
    fireEvent.click(within(g).getAllByRole("button", { name: "Open chart" })[0]);
    expect(openChart).toHaveBeenCalledWith("TATASTEEL");
    expect(screen.getByText(FOOTER)).toBeInTheDocument();
  });

  it("shows the header strip and the track record", async () => {
    h.picks = picks;
    render(<PicksPage />);
    expect((await screen.findByText("2026-11-24", { selector: "b" }))).toBeInTheDocument();
    expect(await screen.findByText("SBIN")).toBeInTheDocument();
    expect(screen.getByText("-0.80")).toBeInTheDocument();
  });

  it("with no passing strategy shows the n_trials note and no gated cards", async () => {
    h.picks = { ...picks, gated: [], n_strategies_passing: 0 };
    render(<PicksPage />);
    expect(await screen.findByRole("note")).toHaveTextContent(/No strategy has passed the gate yet \(n_trials_total = 1840\)/);
    expect(document.querySelectorAll("#gated .pk-card").length).toBe(0);
    expect(document.querySelectorAll("#exploratory .pk-card.expl").length).toBe(2);
  });
});

describe("pick helpers", () => {
  it("scales the evidence bar against the strongest pick", () => {
    const g = (picks as unknown as PicksReply).gated;
    expect(evidencePct(g[0], g)).toBe(100);
    expect(evidencePct(g[1], g)).toBe(56);
  });
  it("prefers the server note", () => {
    expect(headerNote({ ...(picks as unknown as PicksReply), note: "custom" })).toBe("custom");
  });
});
