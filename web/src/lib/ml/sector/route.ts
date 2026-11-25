// Port of sector.html routing (lines 209-221). Hash routes (#/sector/Energy) become query routes:
//   /?view=sector&name=Energy&range=today      /?view=stock&sym=RELIANCE
export type Range = "today" | "w1" | "m1" | "m3";
export type View = "horizon" | "sector" | "stock";
export const RL: Record<Range, string> = { today: "Today", w1: "1 week", m1: "1 month", m3: "3 months" };
export const RANGES: Range[] = ["today", "w1", "m1", "m3"];

export interface SectorRoute {
  view: View;
  name: string | null;
  sym: string | null;
  range: Range;
}

interface ParamsLike {
  get(key: string): string | null;
}

export function parseRoute(p: ParamsLike | null | undefined): SectorRoute {
  const v = p?.get("view");
  const r = p?.get("range");
  const range: Range = r && (RANGES as string[]).includes(r) ? (r as Range) : "today";
  const view: View = v === "sector" && p?.get("name") ? "sector" : v === "stock" && p?.get("sym") ? "stock" : "horizon";
  return { view, name: view === "sector" ? p!.get("name") : null, sym: view === "stock" ? p!.get("sym") : null, range };
}

/** The URL of a view. `go('horizon')` -> "/", go('sector','Energy') -> "/?view=sector&name=Energy&range=today". */
export function viewUrl(view: string, arg?: string, range: Range = "today"): string {
  if (view === "sector") return `/?view=sector&name=${encodeURIComponent(arg ?? "")}&range=${range}`;
  if (view === "stock") return `/?view=stock&sym=${encodeURIComponent(arg ?? "")}${range === "today" ? "" : `&range=${range}`}`;
  return range === "today" ? "/" : `/?range=${range}`;
}

/** Client-side navigation: Next.js keeps useSearchParams in sync with history.pushState. */
export function navigate(url: string): void {
  if (typeof window !== "undefined") window.history.pushState(null, "", url);
}
