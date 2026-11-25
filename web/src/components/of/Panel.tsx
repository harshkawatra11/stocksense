import type { ReactNode } from "react";
import { learnHref, panelTitle } from "@/lib/of/draw";
import type { PanelState } from "./usePanel";

/** A titled panel: the title carries the method label and an info link to the learning note. */
export function Panel({
  name,
  method,
  source,
  estimated,
  topic,
  state,
  children,
}: {
  name: string;
  method?: string | null;
  source?: string | null;
  estimated?: boolean;
  topic: string;
  state: PanelState<unknown>;
  children?: ReactNode;
}) {
  return (
    <section className="of-panel" data-panel={topic}>
      <h2>
        <span>{panelTitle(name, method)}</span>
        <a href={learnHref(topic)} aria-label={`About ${name}`} title="What is this?">
          ⓘ
        </a>
      </h2>
      {(source || estimated) && (
        <div>
          {estimated && <span className="of-badge">estimated</span>}
          {source && <span className="of-badge">{source}</span>}
        </div>
      )}
      {state.loading ? (
        <div className="of-note">Loading…</div>
      ) : state.error ? (
        <div className="of-note">Not available: {state.error}</div>
      ) : (
        children
      )}
    </section>
  );
}
