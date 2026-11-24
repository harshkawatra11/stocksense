"use client";
import { useEffect, useRef } from "react";
import { esc } from "@/lib/ml/format";
import { useLabActions } from "@/lib/ml/actions";
import { useChartHost } from "./ChartHost";
import { IndexCards, Stage, Tooltip } from "./Stage";
import { useSmartLive, type LiveState } from "./useSmartLive";
import type { SmartLive } from "@/lib/ml/smart/types";

function LiveBadge({ state, data }: { state: LiveState; data: SmartLive | null }) {
  if (state === "down") return <span className="live" id="live">server not reachable</span>;
  if (state === "warming") return <span className="live" id="live">loading the tape (about a minute)…</span>;
  if (!data)
    return (
      <span className="live" id="live">
        <span className="dot" />
        LIVE · –
      </span>
    );
  const t = new Date(data.time).toLocaleTimeString("en-GB", { timeZone: "Asia/Kolkata" });
  return (
    <span className="live" id="live">
      {data.market === "open" ? (
        <>
          <span className="dot" />
          LIVE · {t}
        </>
      ) : (
        <>
          MARKET {esc(data.market).toUpperCase()} · {t}
        </>
      )}
      {data.errors ? <span className="down"> · retrying</span> : null}
    </span>
  );
}

/** Port of smart.html: header, index cards, the stage and the tooltip. The pop-up chart is the page-level ChartModal. */
export function SmartSection() {
  const { data, state } = useSmartLive();
  const { req, setLive } = useChartHost();
  const { openChart } = useLabActions();
  const tip = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setLive(data);
  }, [data, setLive]);

  return (
    <>
      <header>
        <h1>SMART MONEY</h1>
        <span className="sub">Every F&amp;O stock and NIFTY, BANKNIFTY, SENSEX: displacement, structure, zones and liquidity, live</span>
        <span className="sp" />
        <a
          href="#sector"
          onClick={(e) => {
            e.preventDefault();
            document.getElementById("sector")?.scrollIntoView({ behavior: "smooth" });
          }}
        >
          Sector map ↓
        </a>
        <LiveBadge state={state} data={data} />
      </header>
      <main>
        <IndexCards indices={data?.indices} onOpen={openChart} />
        <Stage data={data} paused={!!req} tipRef={tip} onOpen={openChart} />
        <div className="note">
          Lines: every F&amp;O stock&apos;s move since the open (green above, red below), indices dashed. Waterfall: every stock now, strongest at the top; the 20
          strongest and 20 weakest connect to their tiles. Hover a line, a bar, a bead or a tile to follow one stock; hover a sector to see its stocks. Click to open its
          chart. Displacement = move since the open ÷ a normal day&apos;s range. Prices 2-3 s behind NSE (indices every 4 s, stocks every 12 s; Sensex 15 min late). A technical
          screen, not advice.
        </div>
      </main>
      <Tooltip tipRef={tip} />
    </>
  );
}
