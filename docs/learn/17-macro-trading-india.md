# Macro trading for Indian equities

## What it is

Macro trading takes positions on the whole market (usually the NIFTY 50 index) from economy-wide variables: the rupee, crude oil, US yields, the dollar index, foreign investor flows, central bank dates. The question is when to be long the index and when to be in cash.

## How it is computed

- `tier_c_macro_trend`: long NIFTY for the next session if at least 3 of these hold at the close: USDINR 20-day change below 0, Brent 20-day change below 0, US 10-year yield 20-day change below 0, FII index-futures long ratio 5-day change above 0. Macro series are used only as of their `available_ts`.
- `tier_c_macro_fii`: long when the FII long ratio is above 0.5 and rising over 5 days.
- `k151_macro_mom` (paper section 19.2, adapted): mean of z-scores of the one-year changes in -USDINR, -US10Y and NIFTY return; long when above 0, decided monthly with a three-month minimum hold.
- `k151_announcement` (section 19.5): long NIFTY from open to close on RBI monetary policy decision days and on the Indian session after FOMC decisions, cash otherwise. Dates are kept in `config/macro_events.toml` with source URLs.

## What it says about flow and positioning

A weaker rupee, higher crude and higher US yields are conventionally bad for Indian equities (imported inflation, foreign outflows), so falls in those variables are read as supportive. FII positioning is a direct read on one large flow source.

## What NSE data allows and how StockSense computes it

Macro series come from the stored `macro_daily` dataset (BRENT, DXY, GOLD, NDX, SPX, US10Y, USDINR from stooq, daily). FII long ratios come from `participant_oi` since 2019-12. India has no CPI or GDP series in our data, so the paper's business-cycle variable is dropped. Index returns come from `index_daily`.

## Evidence

Lucca and Moench (2015) document large average US equity returns in the 24 hours before scheduled FOMC announcements, which motivates the announcement-day idea; whether an Indian-market version holds is tested here and is not assumed. Evidence that simple macro rules time the Indian index after costs is weak in our reading and the gated runs decide.

## How traders misuse it

Fitting the rule to a handful of past episodes; using revised macro numbers that were not known at the time (we use availability timestamps for this reason); treating correlation between the rupee and the index as causation.

## Where it appears in the dashboard

Strategies page entries for the macro keys, and the macro readings in each stock's research note.

## Sources

- https://doi.org/10.1111/jofi.12196
- https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx
