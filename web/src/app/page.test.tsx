import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("@/lib/api", () => ({
  API_BASE: "",
  getJSON: vi.fn(async () => ({})),
  usePoll: vi.fn(() => ({ data: null, error: null, refresh: () => {} })),
}));

import Home from "@/app/page";
import { Nav } from "@/components/Nav";

describe("shell", () => {
  it("renders the three section ids with their scope classes", () => {
    const { container } = render(<Home />);
    for (const id of ["smart", "sector", "lab"]) {
      const el = container.querySelector(`section#${id}`);
      expect(el).not.toBeNull();
      expect(el).toHaveClass(`ml-${id}`);
    }
    expect(container.querySelectorAll(".divider")).toHaveLength(2);
  });

  it("renders the nav links in order", () => {
    render(<Nav />);
    const labels = ["Daily Picks", "Smart Money", "Sector Map", "Swing Lab", "Money Flow", "Order Flow", "Strategies", "Learn"];
    const links = screen.getAllByRole("link").map((a) => a.textContent);
    expect(links).toEqual(labels);
    expect(screen.getByRole("link", { name: "Learn" })).toHaveAttribute("href", expect.stringMatching(/^\/learn\/?$/));
  });
});
