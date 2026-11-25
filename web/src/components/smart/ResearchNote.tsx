"use client";
import { useEffect, useMemo, useState } from "react";
import { getJSON } from "@/lib/api";
import { isIdx } from "@/lib/ml/smart/chart";
import { researchHTML } from "@/lib/ml/smart/research";
import type { ResearchNote as Note } from "@/lib/ml/smart/types";

type Result = { sym: string; note: Note | null; failed: boolean };

/** The research note under the chart (smart.html rload/rdraw). Indices have none. */
export function ResearchNote({ sym, onPeer }: { sym: string; onPeer: (sym: string) => void }) {
  const [res, setRes] = useState<Result | null>(null);
  const idx = isIdx(sym);

  useEffect(() => {
    if (idx) return;
    const c = new AbortController();
    getJSON<Note>("/api/research?sym=" + encodeURIComponent(sym), c.signal).then(
      (r) => {
        if (!c.signal.aborted) setRes({ sym, note: r.error ? null : r, failed: false });
      },
      () => {
        if (!c.signal.aborted) setRes({ sym, note: null, failed: true });
      },
    );
    return () => c.abort();
  }, [sym, idx]);

  const cur = res && res.sym === sym ? res : null;
  const html = useMemo(() => (cur?.note ? researchHTML(cur.note) : ""), [cur]);

  if (idx) return <div id="rpt" />;
  if (!cur)
    return (
      <div id="rpt">
        <div className="rp">
          <div className="rp-load">
            <span className="rp-spin" />
            Building the research note for <b>{sym}</b> from NSE filings… (first time takes up to 20 s)
          </div>
        </div>
      </div>
    );
  if (cur.failed)
    return (
      <div id="rpt">
        <div className="rp">
          <div className="muted">Research note unavailable right now.</div>
        </div>
      </div>
    );
  return (
    <div
      id="rpt"
      dangerouslySetInnerHTML={{ __html: html }}
      onClick={(e) => {
        const tr = (e.target as HTMLElement).closest<HTMLElement>(".rp-t tr[data-s]");
        if (tr?.dataset.s) onPeer(tr.dataset.s);
      }}
    />
  );
}
