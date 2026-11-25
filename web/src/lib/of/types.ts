// Response shapes of /api/of/* (docs/PLAN-M.md M4.1 to M4.8). Every reply carries method, data_source and estimated.

export interface OfMeta {
  sym?: string;
  date?: string;
  method: string;
  data_source: string;
  estimated: boolean;
}
export interface OfBar {
  time: number; // epoch seconds
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}
export type Series = (number | null)[]; // aligned with `bars`
export interface OfBars extends OfMeta {
  tf: number;
  bars: OfBar[];
  vwap?: Series;
  bands?: Partial<Record<"u1" | "l1" | "u2" | "l2", Series>>;
  anchored?: Series;
}
export interface VolumeProfile extends OfMeta {
  kind: "volume";
  bin: number;
  bins: number[]; // lower edges
  vol: number[];
  poc: number;
  vah: number;
  val: number;
  total: number;
}
export interface TpoProfile extends OfMeta {
  kind: "tpo";
  bin: number;
  bins: number[];
  letters: string[];
  counts: number[];
  periods: number;
  poc_tpo: number;
  vah: number;
  val: number;
  ib_high: number;
  ib_low: number;
  high: number;
  low: number;
  single_prints: number[];
  poor_high: boolean;
  poor_low: boolean;
  day_type?: string | null;
  open_type?: string | null;
}
export type ProfileReply = VolumeProfile | TpoProfile;
export interface DeltaReply extends OfMeta {
  ts: number[];
  delta: number[];
  cvd: number[];
  divergence: number[]; // +1 bullish, -1 bearish, 0 none
}
export interface FpRow {
  price: number;
  buy: number;
  sell: number;
  delta: number;
}
export interface FpStack {
  side: "buy" | "sell";
  from_price: number;
  to_price: number;
  n: number;
}
export interface FpBar {
  ts: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  rows: FpRow[];
  delta: number;
  poc: number;
  stacked_imbalances: FpStack[];
}
export interface FootprintReply extends OfMeta {
  tf: number;
  bin: number;
  bars: FpBar[];
}
export interface BigBar {
  ts: number;
  volume: number;
  baseline: number;
  ratio: number;
  side: number;
  ret_5m: number | null;
  ret_15m: number | null;
  ret_30m: number | null;
  mfe?: number | null;
  mae?: number | null;
}
export interface Absorb {
  ts: number;
  side: "bid" | "offer";
  level: string | number | boolean;
  price: number;
  volume_ratio: number;
  range_ratio: number;
}
export interface BigTradesReply extends OfMeta {
  big: BigBar[];
  absorption: Absorb[];
}
export interface HeatReply extends Partial<OfMeta> {
  empty?: boolean;
  reason?: string;
  times?: number[];
  prices?: number[];
  qty?: number[][]; // [time][price]
}
export interface TfRow {
  tf: string;
  trend: number;
  vs_value: number;
  vs_va: number;
  roc: number;
}
export interface TimescalesReply extends OfMeta {
  rows: TfRow[];
  score: number;
}
