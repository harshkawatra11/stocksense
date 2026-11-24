import type { Metadata } from "next";
import "./globals.css";
import { Nav } from "@/components/Nav";

export const metadata: Metadata = {
  title: "StockSense",
  description: "NSE research and paper-trading dashboard",
};

const FONTS =
  "https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link href={FONTS} rel="stylesheet" />
      </head>
      <body>
        <div id="disc">
          <b>Educational software · not investment advice.</b> Everything on this page is computed on your own
          computer from public data. Patterns and backtests are not recommendations. You are responsible for your
          own trades.
        </div>
        <Nav />
        {children}
      </body>
    </html>
  );
}
