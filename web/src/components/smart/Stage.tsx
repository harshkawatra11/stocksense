"use client";
import { useEffect, useRef, type Ref } from "react";
import { pc } from "@/lib/ml/format";
import { ladder } from "@/lib/ml/smart/ladder";
import { px2 } from "@/lib/ml/smart/tiles";
import type { IndexKey, SmartLive } from "@/lib/ml/smart/types";
import { StageEngine } from "./stageEngine";

const KEYS: IndexKey[] = ["NIFTY", "BANKNIFTY", "SENSEX"];

export function Tooltip({ tipRef }: { tipRef: Ref<HTMLDivElement> }) {
  return <div id="tip" ref={tipRef} />;
}

export function IndexCards({ indices, onOpen }: { indices: SmartLive["indices"] | undefined; onOpen: (sym: string) => void }) {
  const cvs = useRef<Partial<Record<IndexKey, HTMLCanvasElement | null>>>({});
  useEffect(() => {
    for (const k of KEYS) {
      const cv = cvs.current[k];
      if (cv && indices?.[k]) ladder(cv, indices[k]!.bars, { step: 3 });
    }
  }, [indices]);
  return (
    <div className="cards" id="cards">
      {KEYS.map((k) => {
        const x = indices?.[k];
        if (!x) return <div className="icard" key={k} />;
        const r = x.read || {};
        const st = r.struct;
        const ob = r.ob;
        const liq = r.liq || {};
        const exp = x.expiry
          ? new Date(x.expiry + "T00:00:00").toLocaleDateString("en-GB", { day: "2-digit", month: "short" }).toUpperCase()
          : "–";
        return (
          <div className={`icard ${x.pct < 0 ? "neg" : ""}`} key={k} data-k={`i:${k}`} data-v={x.ltp} onClick={() => onOpen(k)}>
            <canvas
              id={`ic_${k}`}
              ref={(el) => {
                cvs.current[k] = el;
              }}
            />
            <div className="lab">
              <b>{k} INDEX</b>
              <span className={x.pct >= 0 ? "up" : "down"}>{pc(x.pct)}</span>
              {(x.delay ?? 0) > 300 && <span className="muted"> · 15 min late</span>}
              <br />
              {st && (
                <span className={`tag ${st.dir} ${st.type === "CHoCH" ? "choch" : ""}`}>
                  {st.type} {st.dir === "bull" ? "↑" : "↓"}
                </span>
              )}
            </div>
            <div
              className="px"
              style={{ position: "absolute", right: 0, top: 22, background: "var(--ink)", color: "#fff", fontSize: "10.5px", fontWeight: 700, padding: "1px 5px", zIndex: 2 }}
            >
              {px2(Number(x.ltp))} ◂
            </div>
            <div className="foot">
              <span>
                <span className="muted">OB</span> <b>{ob ? `${Math.abs(ob.dist_pct)}% ${ob.dist_pct < 0 ? "below" : "above"}` : "–"}</b>
                <br />
                <span className="muted">LIQ</span> <b>{liq.swept && liq.swept.length ? liq.swept.join(", ") + " swept" : "not swept"}</b>
              </span>
              <span className="exp">{exp}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export interface StageProps {
  data: SmartLive | null;
  paused: boolean;
  tipRef: React.RefObject<HTMLDivElement | null>;
  onOpen: (sym: string) => void;
}

/** The big stage: canvases, counts, sector chips and the 40 tiles. React renders the skeleton, StageEngine fills it. */
export function Stage({ data, paused, tipRef, onOpen }: StageProps) {
  const stage = useRef<HTMLDivElement>(null);
  const bg = useRef<HTMLCanvasElement>(null);
  const fx = useRef<HTMLCanvasElement>(null);
  const beads = useRef<HTMLCanvasElement>(null);
  const lagl = useRef<HTMLDivElement>(null);
  const leadl = useRef<HTMLDivElement>(null);
  const counts = useRef<HTMLDivElement>(null);
  const nb = useRef<HTMLDivElement>(null);
  const nr = useRef<HTMLDivElement>(null);
  const chips = useRef<HTMLDivElement>(null);
  const tiles = useRef<HTMLDivElement>(null);
  const hbull = useRef<HTMLDivElement>(null);
  const hbear = useRef<HTMLDivElement>(null);
  const engine = useRef<StageEngine | null>(null);
  const openRef = useRef(onOpen);
  useEffect(() => {
    openRef.current = onOpen;
  });

  useEffect(() => {
    const tip = tipRef.current;
    if (!tip || !stage.current) return;
    const eng = new StageEngine(
      {
        root: stage.current.closest<HTMLElement>(".ml-smart") ?? stage.current,
        stage: stage.current,
        bg: bg.current!,
        fx: fx.current!,
        beads: beads.current!,
        lagl: lagl.current!,
        leadl: leadl.current!,
        counts: counts.current!,
        nb: nb.current!,
        nr: nr.current!,
        chips: chips.current!,
        tiles: tiles.current!,
        hbull: hbull.current!,
        hbear: hbear.current!,
        tip,
      },
      (sym) => openRef.current(sym),
    );
    engine.current = eng;
    eng.start();
    return () => {
      eng.destroy();
      engine.current = null;
    };
  }, [tipRef]);

  useEffect(() => {
    engine.current?.setPaused(paused);
  }, [paused]);
  useEffect(() => {
    if (data) engine.current?.setData(data);
  }, [data]);

  return (
    <div id="stage" ref={stage}>
      <canvas id="bg" ref={bg} />
      <canvas id="fx" ref={fx} />
      <canvas id="beads" ref={beads} />
      <div className="slabel" id="lagl" ref={lagl} />
      <div className="slabel" id="leadl" ref={leadl} />
      <div className="counts" id="counts" ref={counts}>
        <div className="up">▲ BULLISH</div>
        <div className="n up" id="nb" ref={nb}>
          –
        </div>
        <div className="down">▼ BEARISH</div>
        <div className="n down" id="nr" ref={nr}>
          –
        </div>
      </div>
      <div id="chips" ref={chips} />
      <div id="tiles" ref={tiles}>
        <div className="half" id="hbull" ref={hbull} />
        <div className="half" id="hbear" ref={hbear} />
      </div>
    </div>
  );
}
