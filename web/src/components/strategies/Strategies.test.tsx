import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import strategies from "@/test/fixtures/lab_strategies.json";
import matrix from "@/test/fixtures/lab_matrix.json";

const h = vi.hoisted(() => ({ calls: [] as string[], lib: null as unknown }));
vi.mock("@/lib/api", () => ({
  API_BASE: "",
  getJSON: vi.fn(async (p: string) => {
    h.calls.push(p);
    if (p.startsWith("/api/lab/strategies")) return h.lib;
    if (p.startsWith("/api/lab/matrix")) return matrix;
    if (p.startsWith("/api/search")) return [{ type: "stock", sym: "S02", name: "S02" }];
    throw new Error("404");
  }),
}));

import { StrategiesPage } from "./StrategiesPage";

afterEach(cleanup);

describe("strategies page", () => {
  it("lists the library, filters it, and shows the tier text with no measured order yet", async () => {
    h.lib = strategies;
    render(<StrategiesPage />);
    await waitFor(() => expect(screen.getByText("260 of 260")).toBeInTheDocument());
    expect(screen.getByText(/Claimed order F < C < B; measured order not available yet/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Tier"), { target: { value: "F" } });
    expect(screen.getByText("3 of 260")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Verdict"), { target: { value: "pass" } });
    expect(screen.getByText("0 of 260")).toBeInTheDocument();
  });

  it("shows gate lines and Sharpe when the verdict is an object", async () => {
    const rows = (strategies as { key: string }[]).map((s, i) =>
      i === 0 ? { ...s, gated: { verdict: "PASS", lines_passed: 9, net_sharpe_5l: 1.23, sharpe_2x: 0.8, trades_oos: 150, positive_years: 4 } } : s,
    );
    h.lib = rows;
    render(<StrategiesPage />);
    const cell = await screen.findByText("PASS");
    const tr = cell.closest("tr")!;
    expect(within(tr).getByText("9/9")).toBeInTheDocument();
    expect(within(tr).getByText("1.23")).toBeInTheDocument();
    expect(within(tr).getByRole("link", { name: "eval" })).toHaveAttribute("href", expect.stringContaining("/api/lab/strategies?key="));
  });

  it("loads the per-stock matrix with the multiple-testing banner", async () => {
    h.lib = strategies;
    render(<StrategiesPage />);
    await screen.findByText("260 of 260");
    fireEvent.change(screen.getByLabelText("Symbol"), { target: { value: "s02" } });
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    await screen.findByRole("note");
    expect(screen.getByRole("note").textContent).toMatch(/174 tests.*3\.41/);
    expect(h.calls).toContain("/api/lab/matrix?sym=S02&tf=1d");
    expect(document.querySelectorAll("#matrix tbody tr").length).toBeGreaterThan(10);
  });
});
