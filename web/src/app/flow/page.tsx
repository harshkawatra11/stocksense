import "@/styles/ml-smart.css";
import "@/styles/ml-sector.css";
import "@/styles/flow.css";
import { ChartModal } from "@/components/ChartModal";
import { ChartHost } from "@/components/smart/ChartHost";
import { FlowPage } from "@/components/flow/FlowPage";

export const metadata = { title: "Money Flow · StockSense" };

export default function Flow() {
  return (
    <ChartHost>
      <section className="ml-sector section flow-root">
        <div className="flow-page-wrap">
          <FlowPage />
        </div>
      </section>
      <section className="ml-smart section">
        <ChartModal />
      </section>
    </ChartHost>
  );
}
