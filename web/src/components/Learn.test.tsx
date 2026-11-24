import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

let topic: string | null = null;
vi.mock("next/navigation", () => ({ useSearchParams: () => ({ get: () => topic }) }));

import { Markdown, noteHref } from "@/components/Markdown";
import { LearnView } from "@/components/LearnView";

const FIXTURE = `# Title one

## Section two

| Col A | Col B |
|-------|-------|
| 1     | 2     |

See [VWAP](06-vwap.md) and [site](https://example.com).
`;

describe("learn", () => {
  it("renders headings and a table from markdown", () => {
    render(<Markdown source={FIXTURE} />);
    expect(screen.getByRole("heading", { level: 1, name: "Title one" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: "Section two" })).toBeInTheDocument();
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Col B" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "2" })).toBeInTheDocument();
  });

  it("maps note links to the topic route", () => {
    expect(noteHref("06-vwap.md")).toBe("/learn/?topic=06-vwap");
    expect(noteHref("./06-vwap.md#x")).toBe("/learn/?topic=06-vwap#x");
    expect(noteHref("https://example.com")).toBe("https://example.com");
  });

  const notes = [
    { slug: "00-a", title: "Alpha", body: "# Alpha\n\ntext" },
    { slug: "01-b", title: "Beta", body: "# Beta" },
  ];
  it("lists the notes, then shows one by topic", () => {
    topic = null;
    const { unmount } = render(<LearnView notes={notes} />);
    expect(screen.getByRole("link", { name: "Beta" })).toBeInTheDocument();
    unmount();
    topic = "00-a";
    render(<LearnView notes={notes} />);
    expect(screen.getByRole("heading", { name: "Alpha" })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Beta" })).toBeNull();
  });
});
