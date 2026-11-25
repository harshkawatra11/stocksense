// Imperative island of the Sector Map (sector.html lines 236-595): loads the current view, draws the SVG views,
// owns the replay, the 1 s tick and the 30 s refresh. React renders the header, crumbs and search; this fills
// #view, #bottom and #side. Pure markup builders are in lib/ml/sector/*.
import { getJSON } from "@/lib/api";
import { esc, pc, rs, cls } from "@/lib/ml/format";
import { horizonSvg, hm, nowMinOf } from "@/lib/ml/sector/horizon";
import { applyTickToHorizon, onScreen, watchList, watchToggle } from "@/lib/ml/sector/live";
import type { SectorRoute } from "@/lib/ml/sector/route";
import type { HorizonData, SectorData, StatusData, StockData, TickData } from "@/lib/ml/sector/types";
import {
  bubbleTipHTML,
  groupTipHTML,
  hSideHTML,
  horizonViewHTML,
  lineChartHTML,
  pinTipHTML,
  sectorViewHTML,
  sSideHTML,
  stockSideHTML,
  stockTipHTML,
  stockViewHTML,
} from "@/lib/ml/sector/views";

export interface SectorEls {
  root: HTMLElement;
  view: HTMLElement;
  bottom: HTMLElement;
  side: HTMLElement;
  tip: HTMLElement;
}
export interface Crumb {
  view: SectorRoute["view"];
  sector?: string | null; // sector name to link back to (stock view) or to show (sector view)
  sym?: string;
}
export interface SectorHooks {
  go: (view: string, arg?: string) => void;
  openChart: (sym: string) => void;
  crumbs: (c: Crumb) => void;
  live: (t: { market: string; time: string } | null) => void;
}

const SVGNS = "http://www.w3.org/2000/svg";

export class SectorEngine {
  route: SectorRoute = { view: "horizon", name: null, sym: null, range: "today" };
  DATA: HorizonData | SectorData | StockData | null = null;
  AT: number | null = null;
  PLAY: ReturnType<typeof setInterval> | null = null;
  modalOpen = false;
  private SEEN: Record<string, number> = {};
  private seq = 0;
  private ctl: AbortController | null = null;
  private retry: ReturnType<typeof setTimeout> | null = null;
  private timers: ReturnType<typeof setInterval>[] = [];
  private off: (() => void)[] = [];
  private ro: ResizeObserver | null = null;
  private alive = true;

  constructor(
    private els: SectorEls,
    private hooks: SectorHooks,
  ) {}

  // ---- lifecycle ----
  start(): void {
    const { view, side, bottom } = this.els;
    const on = (el: HTMLElement, ev: string, fn: (e: Event) => void) => {
      el.addEventListener(ev, fn);
      this.off.push(() => el.removeEventListener(ev, fn));
    };
    for (const el of [view, side]) {
      on(el, "click", (e) => this.onClick(e));
      on(el, "mousemove", (e) => this.onMove(e as MouseEvent));
      on(el, "mouseleave", () => this.hideTip());
    }
    on(bottom, "click", (e) => this.onClick(e));
    on(bottom, "mousemove", (e) => this.onMove(e as MouseEvent));
    on(bottom, "mouseleave", () => this.hideTip());
    on(view, "input", (e) => {
      if ((e.target as HTMLElement).id === "slider") this.slide();
    });
    // prices on screen every second, the whole view every 30 s while not replaying and the chart is closed
    this.timers.push(setInterval(() => void this.tick(), 1000));
    this.timers.push(
      setInterval(() => {
        if (!this.PLAY && this.AT == null && !this.modalOpen) void this.load(false);
      }, 30000),
    );
    if (typeof ResizeObserver !== "undefined") {
      this.ro = new ResizeObserver(() => {
        if (this.DATA && this.route.view === "horizon") this.drawHorizon();
      });
      this.ro.observe(view);
    }
  }

  destroy(): void {
    this.alive = false;
    this.off.forEach((f) => f());
    this.off = [];
    this.timers.forEach(clearInterval);
    this.timers = [];
    this.stopPlay();
    this.ctl?.abort();
    if (this.retry) clearTimeout(this.retry);
    this.ro?.disconnect();
  }

  setRoute(r: SectorRoute): void {
    this.route = r;
    this.AT = null;
    this.stopPlay();
    void this.load(true);
  }

  // ---- loading ----
  async load(fresh: boolean): Promise<void> {
    const mySeq = ++this.seq;
    this.ctl?.abort();
    if (this.retry) clearTimeout(this.retry);
    const ctl = new AbortController();
    this.ctl = ctl;
    const { view } = this.els;
    const r = this.route;
    if (fresh) view.insertAdjacentHTML("beforeend", '<div class="loading" id="ld">Loading…</div>');
    try {
      const st = await getJSON<StatusData>("/api/status", ctl.signal);
      if (!this.alive || mySeq !== this.seq) return;
      // an API that does not report `classified` at all is taken as ready; only an explicit 0/false means "still building"
      if ("classified" in st && !st.classified) {
        view.innerHTML = `<div class="loading">Preparing the map: ${esc(typeof st.progress === "string" ? st.progress : "starting")}…</div>`;
        this.retry = setTimeout(() => void this.load(true), 4000);
        return;
      }
      const url =
        r.view === "horizon"
          ? `/api/horizon?range=${r.range}`
          : r.view === "sector"
            ? `/api/sector?name=${encodeURIComponent(r.name ?? "")}&range=${r.range}`
            : `/api/stock?sym=${encodeURIComponent(r.sym ?? "")}`;
      const d = await getJSON<HorizonData | SectorData | StockData>(url, ctl.signal);
      if (!this.alive || mySeq !== this.seq) return;
      if (d.error) throw new Error(d.error);
      this.DATA = d;
      this.announceCrumbs();
      if (r.view === "horizon") this.horizon(fresh);
      else if (r.view === "sector") this.sector();
      else this.stock();
      this.flash();
    } catch (e) {
      if (!this.alive || mySeq !== this.seq || ctl.signal.aborted) return;
      view.innerHTML = `<div class="loading">Could not load (${esc(e instanceof Error ? e.message : String(e))}). Retrying…</div>`;
    }
  }

  private announceCrumbs(): void {
    const d = this.DATA;
    const v = this.route.view;
    if (!d) return;
    if (v === "horizon") this.hooks.crumbs({ view: v });
    else if (v === "sector") this.hooks.crumbs({ view: v, sector: (d as SectorData).name });
    else {
      const s = d as StockData;
      this.hooks.crumbs({ view: v, sector: s.sector ? s.sector.name : null, sym: s.sym });
    }
  }

  flash(): void {
    this.els.root.querySelectorAll<HTMLElement>("[data-k]").forEach((el) => {
      const k = el.dataset.k ?? "";
      const v = Number(el.dataset.v);
      if (isNaN(v)) return;
      if (k in this.SEEN && this.SEEN[k] !== v) {
        el.classList.remove("flash-up", "flash-down");
        void el.getBoundingClientRect();
        el.classList.add(v > this.SEEN[k] ? "flash-up" : "flash-down");
      }
      this.SEEN[k] = v;
    });
  }

  // ---- tooltip ----
  private showTip(html: string, e: { clientX: number; clientY: number }): void {
    const t = this.els.tip;
    t.innerHTML = html;
    t.style.display = "block";
    t.style.left = Math.min(window.innerWidth - 240, e.clientX + 16) + "px";
    t.style.top = Math.min(window.innerHeight - 180, e.clientY + 12) + "px";
  }
  private hideTip(): void {
    this.els.tip.style.display = "none";
  }

  // ---- events (delegated) ----
  private onClick(e: Event): void {
    const t = e.target as HTMLElement;
    const go = t.closest<HTMLElement>("[data-gv]");
    if (go) return this.hooks.go(go.dataset.gv ?? "horizon", go.dataset.ga || undefined);
    const bub = t.closest<HTMLElement>(".bub[data-s]");
    if (bub) return this.hooks.go("sector", bub.dataset.s);
    const node = t.closest<HTMLElement>(".node[data-sym]");
    if (node) return this.hooks.go("stock", node.dataset.sym);
    const pin = t.closest<HTMLElement>(".pin");
    if (pin) {
      const l = pin.dataset.l ?? "";
      if (/^https?:\/\//i.test(l)) window.open(l, "_blank", "noopener");
      return;
    }
    if (t.closest("#play")) return this.play();
    if (t.closest("#openc") && this.route.view === "stock") return this.hooks.openChart((this.DATA as StockData).sym);
    if (t.closest("#star") && this.route.view === "stock") {
      const on = watchToggle((this.DATA as StockData).sym);
      const b = this.els.side.querySelector("#star");
      if (b) b.textContent = on ? "★" : "☆";
    }
  }

  private onMove(e: MouseEvent): void {
    const t = e.target as HTMLElement;
    const d = this.DATA;
    if (!d) return;
    const bub = t.closest<HTMLElement>(".bub[data-s]");
    if (bub && this.route.view === "horizon") {
      const h = bubbleTipHTML(d as HorizonData, bub.dataset.s ?? "", this.route.range);
      return h ? this.showTip(h, e) : this.hideTip();
    }
    const node = t.closest<HTMLElement>(".node[data-sym]");
    if (node && this.route.view === "sector") {
      const s = (d as SectorData).groups.flatMap((g) => g.stocks).find((x) => x.sym === node.dataset.sym);
      return s ? this.showTip(stockTipHTML(s), e) : this.hideTip();
    }
    const grp = t.closest<HTMLElement>(".grp");
    if (grp && this.route.view === "sector") {
      const gp = (d as SectorData).groups.find((g) => g.name === grp.dataset.g);
      return gp ? this.showTip(groupTipHTML(gp), e) : this.hideTip();
    }
    const pin = t.closest<HTMLElement>(".pin");
    if (pin) return this.showTip(pinTipHTML(pin.dataset.t ?? "", pin.dataset.s ?? ""), e);
    this.hideTip();
  }

  // ================= HORIZON =================
  private horizon(fresh: boolean): void {
    const v = this.els.view;
    if (fresh || !v.querySelector("#hsvg")) v.innerHTML = horizonViewHTML(this.route.range);
    this.drawHorizon();
    this.els.side.innerHTML = hSideHTML(this.DATA as HorizonData, this.route.range);
    this.bottom(`NIFTY 50 and the latest catalysts`, "— NIFTY 50 · pins = news (N), results (R), policy (P) · hover a pin for the headline", (this.DATA as HorizonData).chart);
  }

  drawHorizon(): void {
    const svg = this.els.view.querySelector<SVGSVGElement>("#hsvg");
    if (!svg || !this.DATA) return;
    const W = svg.clientWidth || 900;
    const H = svg.clientHeight || 690;
    const { g, tails, bubs } = horizonSvg(this.DATA as HorizonData, this.AT, W, H);
    const keep = svg.querySelector("#bubs");
    if (keep && svg.dataset.range === this.route.range) {
      svg.querySelector("#static")!.innerHTML = g;
      svg.querySelector("#tails")!.innerHTML = tails;
      // move existing bubbles (CSS transitions glide them), add new ones
      const tmp = document.createElementNS(SVGNS, "g");
      tmp.innerHTML = bubs;
      [...tmp.children].forEach((n) => {
        const name = (n as SVGElement).dataset.s;
        const old = [...keep.children].find((c) => (c as SVGElement).dataset.s === name) as SVGElement | undefined;
        if (old) {
          old.style.transform = (n as SVGElement).style.transform;
          old.innerHTML = n.innerHTML;
        } else keep.appendChild(n);
      });
    } else {
      svg.dataset.range = this.route.range;
      svg.innerHTML = `<g id="static">${g}</g><g id="tails">${tails}</g><g id="bubs">${bubs}</g>`;
    }
  }

  private slide(): void {
    const sl = this.els.view.querySelector<HTMLInputElement>("#slider");
    if (!sl || !this.DATA) return;
    const v = +sl.value;
    this.AT = v >= nowMinOf(this.DATA as HorizonData) ? null : v;
    const at = this.els.view.querySelector("#at");
    if (at) at.textContent = this.AT == null ? "Now · live" : "At " + hm(this.AT);
    this.drawHorizon();
  }

  private stopPlay(): void {
    if (this.PLAY) clearInterval(this.PLAY);
    this.PLAY = null;
  }

  play(): void {
    const d = this.DATA as HorizonData | null;
    const btn = this.els.view.querySelector("#play");
    if (this.PLAY) {
      this.stopPlay();
      if (btn) btn.textContent = "▶ Replay today";
      return;
    }
    if (!d || !d.snaps.length) return;
    this.AT = d.snaps[0].m;
    if (btn) btn.textContent = "⏸ Pause";
    this.PLAY = setInterval(() => {
      this.AT = (this.AT ?? 0) + 5;
      if (this.AT >= nowMinOf(d)) {
        this.AT = null;
        this.stopPlay();
        const b = this.els.view.querySelector("#play");
        if (b) b.textContent = "▶ Replay today";
      }
      const sl = this.els.view.querySelector<HTMLInputElement>("#slider");
      if (sl) sl.value = String(this.AT ?? 375);
      const at = this.els.view.querySelector("#at");
      if (at) at.textContent = this.AT == null ? "Now · live" : "At " + hm(this.AT);
      this.drawHorizon();
    }, 380);
  }

  // ================= SECTOR / STOCK =================
  private bottom(title: string, sub: string, chart: HorizonData["chart"]): void {
    const b = this.els.bottom;
    b.innerHTML = lineChartHTML(title, sub, chart, b.clientWidth ? b.clientWidth - 28 : 0);
  }

  private sector(): void {
    const d = this.DATA as SectorData;
    const v = this.els.view;
    v.innerHTML = sectorViewHTML(d, v.clientWidth || 1000, v.clientHeight || 690);
    this.els.side.innerHTML = sSideHTML(d);
    this.bottom(`${esc(d.name)} vs NIFTY 50`, `— ${esc(d.index)} — NIFTY 50 · both start at 100 · pins = news (N), results (R), policy (P)`, d.chart);
  }

  private stock(): void {
    const d = this.DATA as StockData;
    const v = this.els.view;
    v.innerHTML = stockViewHTML(d, v.clientWidth || 1000, v.clientHeight || 690);
    this.els.side.innerHTML = stockSideHTML(d, watchList().includes(d.sym));
    this.bottom(`${esc(d.sym)} vs NIFTY 50`, `— ${esc(d.sym)} — NIFTY 50 · both start at 100 · pins = news (N), results (R), policy (P)`, d.chart);
  }

  // ================= live: prices on screen every second =================
  async tick(): Promise<void> {
    const d = this.DATA;
    if (!d || document.hidden) return;
    const view = this.route.view;
    const mySeq = this.seq;
    try {
      const syms = onScreen(view, d);
      const t = await getJSON<TickData>("/api/tick?syms=" + encodeURIComponent(syms.join(",")));
      if (!this.alive || mySeq !== this.seq || this.DATA !== d) return;
      this.hooks.live({ market: t.market, time: t.time });
      if (view === "horizon" && this.route.range === "today" && this.AT == null && !this.PLAY) {
        applyTickToHorizon(d as HorizonData, t);
        this.drawHorizon();
      }
      const q = (k: string) => this.els.root.querySelector<HTMLElement>(`[data-k="${k.replace(/"/g, '\\"')}"]`);
      if (view === "sector") {
        const name = (d as SectorData).name;
        const x = t.sectors[name];
        if (x && x.v != null) {
          const el = q(`hub:${name}`);
          if (el) {
            el.textContent = pc(x.v);
            el.dataset.v = String(x.v);
          }
        }
        Object.entries(t.stocks).forEach(([s, p]) => {
          const el = q(`ss:${s}`);
          if (el && p.pct != null) {
            el.textContent = pc(p.pct);
            el.dataset.v = String(p.pct);
            el.setAttribute("fill", p.pct >= 0 ? "#22a05a" : "#dc3f54");
          }
        });
      }
      if (view === "stock") {
        const sym = (d as StockData).sym;
        const p = t.stocks[sym];
        if (p && p.ltp)
          [`sp:${sym}`, `sr:${sym}`].forEach((k) => {
            const el = q(k);
            if (el) {
              el.textContent = rs(p.ltp);
              el.dataset.v = String(p.ltp);
            }
          });
        Object.entries(t.stocks).forEach(([s, p2]) =>
          [`pp:${s}`, `pl:${s}`].forEach((k) => {
            const el = q(k);
            if (el && p2.pct != null) {
              el.textContent = pc(p2.pct);
              el.dataset.v = String(p2.pct);
              el.className = cls(p2.pct);
            }
          }),
        );
      }
      this.flash();
    } catch {
      /* the next tick tries again */
    }
  }
}
