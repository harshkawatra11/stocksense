# Time scales and multi-timeframe alignment

## What it is

Prices show structure at several scales at once: a 5-minute pullback can sit inside a 60-minute uptrend inside a daily range. Multi-timeframe alignment asks whether trend, position relative to value and momentum agree across scales.

## How it is computed

For each timeframe (5m, 15m, 60m, daily, weekly) four signs are read: the direction of the last structure break (from `marketlab/structure.py`), close above or below VWAP (intraday) or the 20-period average (daily and weekly), position relative to the previous value area (above VAH, below VAL, inside), and the sign of the 10-bar rate of change. The score is the mean of all signs, from -1 to +1. 15-minute and 60-minute bars are built from 5-minute bars, and weekly from daily, by grouping on the floor of the time after removing the IST offset.

## What it says about flow and positioning

Agreement across scales means short-term traders and longer-term holders are leaning the same way; disagreement means a short-term move against the larger trend, which is more likely to fail or to be a pullback entry, depending on your rule.

## What NSE data allows and how StockSense computes it

All timeframes are derived from stored bars: `orderflow/timescales.py` returns the table and score. The test is trivial on synthetic data (a steadily rising series scores +1); whether alignment predicts returns is not claimed and is not a registered strategy beyond Market Lab's own trend filters (D, I, DI) used in the intraday library.

## Evidence

Technical-analysis rules based on moving averages have mixed evidence (Brock, Lakonishok and LeBaron 1992 found support in US data; Lo, Mamaysky and Wang 2000 found some pattern information; Gunasekarage and Power 2001 studied South Asian markets). Evidence for multi-timeframe scoring as such is thin.

## How traders misuse it

Cherry-picking whichever timeframe agrees with a trade already wanted; treating a high score as certainty; ignoring that lower timeframes have more noise and costs.

## Where it appears in the dashboard

Order Flow workbench timescales table.

## Sources

- https://doi.org/10.1111/j.1540-6261.1992.tb04681.x
- https://doi.org/10.1111/0022-1082.00265
- https://scholar.google.com/scholar?q=Gunasekarage+Power+profitability+of+moving+average+trading+rules+South+Asian+stock+markets
