# Footprint charts

## What it is

A footprint (or cluster) chart replaces each candle with a ladder: for every price inside the bar it shows the volume that traded on the bid and on the ask. It answers where, inside a bar, the buying and selling happened.

## How it is computed

For each display bar (for example 5 minutes), each 1-minute sub-bar's estimated buy and sell volume (bulk volume classification, note 02) is spread uniformly over the price bins the sub-bar spans, and bins are summed over the display bar. A stacked imbalance is three or more consecutive price bins where one side is at least three times the other side of the adjacent bin. The point of control (POC) of a bar is the bin with the most volume.

## What it says about flow and positioning

Stacked buy imbalances at the lows of a bar suggest buyers defended that price; stacked sell imbalances at highs suggest sellers did. Cells with high total volume but small delta show two-sided trade (a fight), cells with large delta show one side dominating.

## What NSE data allows and how StockSense computes it

A real footprint needs every trade with its side. We build an estimate from 1-minute bars (`orderflow/footprint.py`); buy plus sell volume equals bar volume by construction (tested). Because the split inside a minute is spread uniformly, imbalances finer than one minute are not meaningful, and the panel states "BVC estimate from 1-minute bars". History is limited to dates where 1-minute bars are stored, from 2026-03 with gaps, plus forward capture.

## Evidence

There is no academic validation of footprint imbalance rules as trading signals that we are aware of. The underlying classification step is validated for aggregate flow (Easley, Lopez de Prado, O'Hara 2012) but not at 1-minute, per-price-bin resolution on Indian stocks.

## How traders misuse it

Treating the uniform-spread estimate as if prices inside a minute were observed; trading "stacked imbalances" without measuring their forward returns; ignoring that thin stocks produce noisy cells.

## Where it appears in the dashboard

Order Flow workbench footprint panel, with the method label on the panel title.

## Sources

- https://doi.org/10.1093/rfs/hhs053
- https://global.oup.com/academic/product/trading-and-exchanges-9780195144703
