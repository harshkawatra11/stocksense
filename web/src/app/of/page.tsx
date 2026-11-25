import "@/styles/of.css";
import { Suspense } from "react";
import { OrderFlowPage } from "@/components/of/OrderFlowPage";

export const metadata = { title: "Order flow · StockSense" };

export default function Page() {
  return (
    <Suspense fallback={<div id="boot">Order-flow workbench loading…</div>}>
      <OrderFlowPage />
    </Suspense>
  );
}
