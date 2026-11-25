import "@/styles/ml-sector.css";
import "@/styles/flow.css";
import { Suspense } from "react";
import { OptionsPage } from "@/components/flow/OptionsPage";

export const metadata = { title: "Options positioning · StockSense" };

export default function Options() {
  return (
    <section className="ml-sector section">
      {/* useSearchParams needs a Suspense boundary in a static export */}
      <Suspense fallback={<div id="boot">Options positioning loading…</div>}>
        <OptionsPage />
      </Suspense>
    </section>
  );
}
