// Ports of Market Lab's tiny formatters (smart.html:181-182, sector.html:201-204 and 831-832, lab.html:133-134).
const ESC: Record<string, string> = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };

export const esc = (s: unknown): string => String(s ?? "").replace(/[&<>"]/g, (c) => ESC[c]);

export const pc = (x: number | null | undefined, d = 2): string =>
  x == null || isNaN(x) ? "–" : (x > 0 ? "+" : "") + Number(x).toFixed(d) + "%";

export const cls = (x: number | null | undefined): "up" | "down" | "" =>
  x != null && x > 0 ? "up" : x != null && x < 0 ? "down" : "";

export const rs = (x: number | null | undefined): string =>
  x == null
    ? "–"
    : "₹" + Number(x).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

export const sg = (v: number | string | null | undefined): string =>
  v == null ? "–" : (Number(v) > 0 ? "+" : "") + v;

export const rf = (v: number | null | undefined, d = 0): string =>
  v == null || !isFinite(v)
    ? "–"
    : Number(v).toLocaleString("en-IN", { maximumFractionDigits: d, minimumFractionDigits: d });

export const rsg = (v: number | null | undefined, d = 0): string =>
  v == null ? "–" : (v > 0 ? "+" : "") + rf(v, d);
