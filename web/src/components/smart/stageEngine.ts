// Imperative island of the Smart Money stage (smart.html lines 238-539 glue): owns the canvases, the tile and chip
// markup and the hover state. React renders the empty skeleton; this class fills it.
import {
  animDone,
  drawBeads,
  drawFx,
  drawStage,
  geometry,
  hitSym,
  labelLeft,
  matchesH,
  nowMin,
  sectorLabelHTML,
  tween,
  type Anim,
  type Bead,
  type Conn,
  type Geometry,
  type Row,
} from "@/lib/ml/smart/stage";
import { ladder } from "@/lib/ml/smart/ladder";
import { chipsHTML, halfHTML } from "@/lib/ml/smart/tiles";
import { tipHTML, tipPos } from "@/lib/ml/smart/tip";
import type { Hover, SmartLive, SmartStock } from "@/lib/ml/smart/types";

export interface StageEls {
  root: HTMLElement;
  stage: HTMLElement;
  bg: HTMLCanvasElement;
  fx: HTMLCanvasElement;
  beads: HTMLCanvasElement;
  lagl: HTMLElement;
  leadl: HTMLElement;
  counts: HTMLElement;
  nb: HTMLElement;
  nr: HTMLElement;
  chips: HTMLElement;
  tiles: HTMLElement;
  hbull: HTMLElement;
  hbear: HTMLElement;
  tip: HTMLElement;
}

export class StageEngine {
  D: SmartLive | null = null;
  BY: Record<string, SmartStock> = {};
  G: Geometry | null = null;
  H: Hover = null;
  paused = false;
  private PREV: Record<string, [number, number]> = {};
  private ANIM: Anim = { t0: 0, from: {} };
  private PTS: Record<string, [number, number][]> = {};
  private ROWS: Row[] = [];
  private CONN: Conn[] = [];
  private BEADS: Bead[] = [];
  private SEEN: Record<string, number> = {};
  private tileEls: Record<string, HTMLElement> = {};
  private fx = { t0: 0, last: 0 };
  private raf = 0;
  private animRaf = 0;
  private ro: ResizeObserver | null = null;
  private off: (() => void)[] = [];

  constructor(
    private els: StageEls,
    private onOpen: (sym: string) => void,
  ) {}

  start(): void {
    const e = this.els;
    this.fx.t0 = performance.now();
    const on = <K extends keyof HTMLElementEventMap>(el: HTMLElement, ev: K, fn: (x: HTMLElementEventMap[K]) => void) => {
      el.addEventListener(ev, fn as EventListener);
      this.off.push(() => el.removeEventListener(ev, fn as EventListener));
    };
    on(e.stage, "mousemove", (ev) => {
      const t = ev.target as HTMLElement;
      if (t.closest(".tile") || t.closest(".chip")) return;
      const g = this.G;
      if (!g) return;
      const st = e.stage.getBoundingClientRect();
      const sym = hitSym(g, ev.clientX - st.left, ev.clientY - st.top, this.BEADS, this.ROWS, this.PTS);
      if (sym) {
        this.setH({ type: "stock", key: sym });
        this.tip(this.BY[sym], ev);
      } else if (this.H && this.H.type === "stock") this.setH(null);
    });
    on(e.stage, "mouseleave", () => this.setH(null));
    on(e.stage, "click", (ev) => {
      const t = ev.target as HTMLElement;
      if (t.closest(".tile") || t.closest(".chip")) return;
      if (this.H && this.H.type === "stock") this.onOpen(this.H.key);
    });
    if (typeof ResizeObserver !== "undefined") {
      this.ro = new ResizeObserver(() => this.render());
      this.ro.observe(e.stage);
    }
    if (typeof requestAnimationFrame !== "undefined") this.raf = requestAnimationFrame((n) => this.fxLoop(n));
  }

  destroy(): void {
    this.off.forEach((f) => f());
    this.off = [];
    this.ro?.disconnect();
    if (typeof cancelAnimationFrame !== "undefined") {
      cancelAnimationFrame(this.raf);
      cancelAnimationFrame(this.animRaf);
    }
  }

  setData(D: SmartLive): void {
    this.D = D;
    if (!this.paused) this.render();
  }

  setPaused(p: boolean): void {
    this.paused = p;
    if (!p) this.render();
    else {
      this.els.tip.style.display = "none";
      this.clearFx();
    }
  }

  // ---- hover ----
  setH(h: Hover): void {
    const same = JSON.stringify(h) === JSON.stringify(this.H);
    this.H = h;
    if (!h) this.els.tip.style.display = "none";
    if (same) return;
    const matches = matchesH(h);
    this.els.root.querySelectorAll<HTMLElement>(".tile[data-s]").forEach((t) => {
      const s = this.BY[t.dataset.s ?? ""];
      t.classList.toggle("dim", !!h && !(s && matches(s)));
      t.classList.toggle("hl", !!h && h.type === "stock" && !!s && s.sym === h.key);
    });
    this.els.root.querySelectorAll<HTMLElement>(".chip").forEach((c) => {
      c.classList.toggle("on", !!h && h.type === "sector" && c.dataset.sec === h.key);
      c.classList.toggle("dim", !!h && h.type === "sector" && c.dataset.sec !== h.key);
    });
    this.draw();
    this.beads();
  }

  tip(s: SmartStock | undefined, e: { clientX: number; clientY: number }): void {
    if (!s) return;
    const t = this.els.tip;
    t.style.display = "block";
    const p = tipPos(e.clientX, e.clientY, window.innerWidth, window.innerHeight);
    t.style.left = p.left + "px";
    t.style.top = p.top + "px";
    t.innerHTML = tipHTML(s);
  }

  // ---- rendering ----
  render(): void {
    const D = this.D;
    if (!D || !D.stocks || !D.stocks.length) return;
    const e = this.els;
    this.G = geometry(D, e.stage.clientWidth, e.stage.clientHeight, nowMin());
    this.BY = Object.fromEntries(D.stocks.map((s) => [s.sym, s]));
    e.nb.textContent = String(D.bull.length);
    e.nr.textContent = String(D.bear.length);
    this.tiles();
    this.chips();
    this.flash();
    this.animate();
    this.PREV = Object.fromEntries(D.stocks.map((x) => [x.sym, [x.from_open, x.disp] as [number, number]]));
    if (this.H && this.H.type === "stock") this.setH({ ...this.H });
  }

  private tiles(): void {
    const g = this.G!;
    const D = this.D!;
    const half = (g.Hs - 6) / 2;
    this.els.tiles.style.width = g.tilesW + "px";
    for (const [side, el, list] of [
      ["bull", this.els.hbull, D.bull],
      ["bear", this.els.hbear, D.bear],
    ] as const) {
      el.style.top = (side === "bull" ? 0 : half + 6) + "px";
      el.style.height = half + "px";
      el.innerHTML = halfHTML(side, list, this.BY);
    }
    this.tileEls = {};
    this.els.root.querySelectorAll<HTMLElement>(".tile[data-s]").forEach((t) => {
      const sym = t.dataset.s ?? "";
      this.tileEls[sym] = t;
      const s = this.BY[sym];
      const cv = t.querySelector("canvas");
      if (cv) ladder(cv, s && s.bars ? s.bars.slice(-36) : null, { step: t.classList.contains("big") ? 3 : 2.5 });
      t.onmouseenter = (ev) => {
        this.setH({ type: "stock", key: sym });
        this.tip(this.BY[sym], ev);
      };
      t.onmousemove = (ev) => this.tip(this.BY[sym], ev);
      t.onmouseleave = () => this.setH(null);
      t.onclick = () => this.onOpen(sym);
    });
  }

  private chips(): void {
    const g = this.G!;
    const D = this.D!;
    const c = this.els.chips;
    c.style.left = "0px";
    c.style.top = g.T + "px";
    c.style.height = g.B - g.T + "px";
    c.style.width = g.CH + "px";
    const room = Math.max(4, Math.floor((g.B - g.T + 2) / 21));
    c.innerHTML = chipsHTML(D.sectors, room);
    c.querySelectorAll<HTMLElement>(".chip").forEach((el) => {
      el.onmouseenter = () => this.setH({ type: "sector", key: el.dataset.sec ?? "" });
      el.onmouseleave = () => this.setH(null);
    });
  }

  /** Flash any [data-k] element whose [data-v] moved since the last render (card and tile prices). */
  flash(): void {
    this.els.root.querySelectorAll<HTMLElement>("[data-k]").forEach((el) => {
      const k = el.dataset.k ?? "";
      const v = Number(el.dataset.v);
      if (k in this.SEEN && this.SEEN[k] !== v) {
        el.classList.remove("flash-up", "flash-down");
        void el.offsetWidth;
        el.classList.add(v > this.SEEN[k] ? "flash-up" : "flash-down");
      }
      this.SEEN[k] = v;
    });
  }

  private animate(): void {
    this.ANIM = { t0: performance.now(), from: this.PREV };
    if (typeof requestAnimationFrame === "undefined") {
      this.draw();
      this.beads();
      return;
    }
    cancelAnimationFrame(this.animRaf);
    const step = () => {
      this.draw();
      this.beads();
      if (!animDone(this.ANIM, performance.now())) this.animRaf = requestAnimationFrame(step);
    };
    this.animRaf = requestAnimationFrame(step);
  }

  private ctx(cv: HTMLCanvasElement, w: number, h: number, dpr: number): CanvasRenderingContext2D | null {
    cv.width = w * dpr;
    cv.height = h * dpr;
    const x = cv.getContext("2d");
    if (!x) return null;
    x.setTransform(dpr, 0, 0, dpr, 0, 0);
    x.clearRect(0, 0, w, h);
    return x;
  }

  private env() {
    const { FO, DP } = tween(this.ANIM, performance.now());
    return { D: this.D!, BY: this.BY, g: this.G!, H: this.H, FO, DP };
  }

  draw(): void {
    if (!this.D || !this.G) return;
    const g = this.G;
    const x = this.ctx(this.els.bg, g.W, g.Hs, window.devicePixelRatio || 1);
    if (!x) return;
    const st = this.els.stage.getBoundingClientRect();
    const r = drawStage(x, {
      ...this.env(),
      now: new Date(),
      tileMid: (sym) => {
        const t = this.tileEls[sym];
        if (!t) return null;
        const b = t.getBoundingClientRect();
        return b.top - st.top + b.height / 2;
      },
    });
    this.PTS = r.PTS;
    this.ROWS = r.ROWS;
    this.CONN = r.CONN;
  }

  beads(): void {
    if (!this.D || !this.G) return;
    const g = this.G;
    const W = g.wfR;
    const cv = this.els.beads;
    cv.style.width = W + "px";
    const x = this.ctx(cv, W, g.beadsH, window.devicePixelRatio || 1);
    if (!x) return;
    const e = this.env();
    const r = drawBeads(x, { D: e.D, BY: e.BY, g: e.g, H: e.H, FO: e.FO });
    this.BEADS = r.BEADS;
    const put = (el: HTMLElement, kind: "lead" | "lag", s: typeof r.lead, xx: number) => {
      if (!s) return;
      el.style.left = labelLeft(xx, W) + "px";
      el.style.top = "0px";
      el.innerHTML = sectorLabelHTML(kind, s);
    };
    put(this.els.lagl, "lag", r.lag, r.lagX);
    put(this.els.leadl, "lead", r.lead, r.leadX);
    this.els.counts.style.left = W + 6 + "px";
  }

  private clearFx(): void {
    const g = this.G;
    if (!g) return;
    const x = this.els.fx.getContext("2d");
    if (x) x.clearRect(0, 0, g.W, g.Hs);
  }

  // ---- the 40 travelling lights, about 15 frames a second ----
  private fxLoop(now: number): void {
    this.raf = requestAnimationFrame((n) => this.fxLoop(n));
    if (document.hidden || !this.D || !this.G || now - this.fx.last < 66) return;
    this.fx.last = now;
    const g = this.G;
    const cv = this.els.fx;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    if (cv.width !== Math.round(g.W * dpr) || cv.height !== Math.round(g.Hs * dpr)) {
      cv.width = Math.round(g.W * dpr);
      cv.height = Math.round(g.Hs * dpr);
      cv.style.width = g.W + "px";
      cv.style.height = g.Hs + "px";
    }
    const x = cv.getContext("2d");
    if (!x) return;
    x.setTransform(dpr, 0, 0, dpr, 0, 0);
    x.clearRect(0, 0, g.W, g.Hs);
    if (this.paused) return;
    drawFx(x, { CONN: this.CONN, PTS: this.PTS, BY: this.BY, H: this.H, t0: this.fx.t0, now });
  }
}
