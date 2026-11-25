// OI quadrant chips: price direction x open-interest direction (flow/oi.py LABELS).
export interface QuadStyle {
  label: string;
  fg: string;
  bg: string;
  /** Price up (+1) or down (-1), OI up (+1) or down (-1). */
  price: 1 | -1;
  oi: 1 | -1;
}

export const QUADS: QuadStyle[] = [
  { label: "long buildup", fg: "#17603a", bg: "#e8f6ee", price: 1, oi: 1 }, // green
  { label: "short covering", fg: "#8a5a00", bg: "#fdf1d0", price: 1, oi: -1 }, // amber
  { label: "long unwinding", fg: "#b4501a", bg: "#fde6d6", price: -1, oi: -1 }, // orange
  { label: "short buildup", fg: "#9b1c2e", bg: "#fdecef", price: -1, oi: 1 }, // red
];

/** Colours of one quadrant label, or null for an unknown or missing label. */
export function quadStyle(label: string | null | undefined): QuadStyle | null {
  return QUADS.find((q) => q.label === label) ?? null;
}

/** Dot colour of a scatter point (the chip's foreground colour; grey when unlabelled). */
export const quadColor = (label: string | null | undefined): string => quadStyle(label)?.fg ?? "#8a93a6";
