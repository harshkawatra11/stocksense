// Shapes of /api/horizon, /api/sector, /api/stock, /api/search, /api/tick and /api/status as the Sector Map reads them.
export interface NewsItem {
  time: string;
  title: string;
  source: string;
  link: string;
  tag: string; // NEWS | RESULTS | POLICY (first letter picks the colour)
  sector?: string;
  since?: number | null;
}
export interface Pin {
  tag: string;
  title: string;
  source: string;
  link: string;
}
export interface RelChart {
  dates: string[];
  a: (number | null)[];
  b?: (number | null)[];
  pins?: Record<string, Pin[]>;
}

export interface HSector {
  name: string;
  index: string;
  group: string;
  v: number | null;
  size?: number;
  n?: number;
  rank?: number;
  rank_open?: number | null;
  trail?: number | null;
  live?: boolean;
  path: [number, number | null][];
}
export interface HorizonData {
  range: string;
  nifty: { v: number | null; last?: number };
  sectors: HSector[];
  groups: string[];
  climb: { name: string; from: number; to: number }[];
  themes: { name: string; parent?: string; n: number; beat: number; pct: number }[];
  news: NewsItem[];
  chart: RelChart | null;
  snaps: { m: number; nifty: number | null }[];
  error?: string;
}

export interface SRow {
  sym: string;
  name?: string;
  group: string;
  basic?: string;
  ltp: number | null;
  pct: number | null;
  mcap?: number | null;
  live?: boolean;
  w1?: number | null;
  m1?: number | null;
  m3?: number | null;
  above50?: boolean | null;
  near_high?: boolean | null;
  vol_ratio?: number | null;
  high52?: number | null;
  dma50?: number | null;
}
export interface SGroup {
  name: string;
  n: number;
  mid: number | null;
  above50: number;
  stocks: SRow[];
}
export interface SectorData {
  name: string;
  index: string;
  group?: string;
  ix: { last: number | null; pct: number; w1?: number | null; m1?: number | null; m3?: number | null; pe?: number | null; live?: boolean };
  nifty: { pct?: number | null; w1?: number | null; m1?: number | null; m3?: number | null; pe?: number | null };
  rank: number | null;
  rank_open: number | null;
  n: number;
  middle: number | null;
  groups: SGroup[];
  others: { sym: string; sector: string; ltp: number | null; pct: number | null }[];
  rank_hist: { date: string; rank: number }[];
  inside: { above50: number; above50_wk: number; rose_week: number; near_high: number; busy: number; n: number };
  movers: SRow[];
  laggards: SRow[];
  news: NewsItem[];
  chart: RelChart | null;
  error?: string;
}

export interface StockData {
  sym: string;
  row: SRow;
  cls: { sector?: string; industry?: string; basic?: string; pe?: number | null; mcap?: number | null; delivery?: number | null; high52?: number | null; low52?: number | null; name?: string };
  sector: { name: string; index: string } | null;
  peers: SRow[];
  against: { label: string; stock: number | null; sector: number | null; gap: number | null }[];
  rank: { pos: number; of: number } | null;
  lines: { stock: number[]; sector?: number[]; nifty?: number[]; m3: { stock: number | null; sector?: number | null; nifty?: number | null } } | null;
  news: NewsItem[];
  volume: { today?: number | null; normal?: number | null; ratio: number | null };
  delivery: number | null;
  dma50?: number | null;
  chart: RelChart | null;
  error?: string;
}

export type SearchHit = { type: "sector"; name: string } | { type: "stock"; sym: string; name?: string; pct: number | null };

export interface TickData {
  stocks: Record<string, { ltp: number | null; pct: number | null }>;
  sectors: Record<string, { v: number | null; last?: number }>;
  nifty: { v: number | null; last?: number };
  time: string;
  market: string;
}

export interface StatusData {
  classified?: number | boolean;
  progress?: string;
}
