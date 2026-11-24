import "@/styles/ml-smart.css";
import "@/styles/ml-sector.css";
import "@/styles/ml-lab.css";
import { Suspense } from "react";
import { ChartModal } from "@/components/ChartModal";
import { ChartHost } from "@/components/smart/ChartHost";
import { SmartSection } from "@/components/smart/SmartSection";
import { SectorSection } from "@/components/sector/SectorSection";

function Placeholder({ name, card }: { name: string; card: string }) {
  return (
    <div id="boot">
      {name} loads here (card {card}).
    </div>
  );
}

export default function Home() {
  return (
    <ChartHost>
      <section className="ml-smart section" id="smart">
        <SmartSection />
        <ChartModal />
      </section>
      <div className="divider" />
      <section className="ml-sector section" id="sector">
        {/* useSearchParams needs a Suspense boundary in a static export */}
        <Suspense fallback={<div id="boot">Sector Map loading…</div>}>
          <SectorSection />
        </Suspense>
      </section>
      <div className="divider" />
      <section className="ml-lab section" id="lab">
        <Placeholder name="Swing Lab" card="M6.4" />
      </section>
    </ChartHost>
  );
}
