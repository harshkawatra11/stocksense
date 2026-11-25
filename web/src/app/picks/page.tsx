import "@/styles/ml-smart.css";
import "@/styles/of.css";
import { ChartModal } from "@/components/ChartModal";
import { ChartHost } from "@/components/smart/ChartHost";
import { PicksPage } from "@/components/picks/PicksPage";

export const metadata = { title: "Daily picks · StockSense" };

export default function Page() {
  return (
    <ChartHost>
      <PicksPage />
      <section className="ml-smart section">
        <ChartModal />
      </section>
    </ChartHost>
  );
}
