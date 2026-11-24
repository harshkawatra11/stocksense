import "@/styles/ml-smart.css";
import "@/styles/ml-sector.css";
import "@/styles/ml-lab.css";
import { ChartModal } from "@/components/ChartModal";
import { ChartHost } from "@/components/smart/ChartHost";
import { SmartSection } from "@/components/smart/SmartSection";

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
        <Placeholder name="Sector Map" card="M6.3" />
      </section>
      <div className="divider" />
      <section className="ml-lab section" id="lab">
        <Placeholder name="Swing Lab" card="M6.4" />
      </section>
    </ChartHost>
  );
}
