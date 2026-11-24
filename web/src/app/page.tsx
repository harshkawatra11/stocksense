import "@/styles/ml-smart.css";
import "@/styles/ml-sector.css";
import "@/styles/ml-lab.css";
import { ChartModal } from "@/components/ChartModal";
import { LabActionsProvider } from "@/lib/ml/actions";

function Placeholder({ name, card }: { name: string; card: string }) {
  return (
    <div id="boot">
      {name} loads here (card {card}).
    </div>
  );
}

export default function Home() {
  return (
    <LabActionsProvider>
      <section className="ml-smart section" id="smart">
        <Placeholder name="Smart Money" card="M6.2" />
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
    </LabActionsProvider>
  );
}
