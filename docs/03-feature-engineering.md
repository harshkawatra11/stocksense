# 03 — Feature Engineering

## What this component does

A raw candle is not predictive. `open=830, high=852, low=828, close=849, volume=1.2M` tells a model almost nothing on its own, because it carries no context: is that volume high for this stock? Is that close near a level that mattered? Is the sector doing the same thing?

The Feature Engine converts raw market data into the numeric context that makes prediction possible. It reads candles, F&O snapshots, and index data from the store, and emits **one versioned feature row per instrument per day**.

That row is literally what gets passed to `model.fit()`. Everything the model can ever know is in it, which is why this document is exhaustive: a category omitted here is a category the model is permanently blind to.

## The output contract

```
one row = (instrument_id, date, feature_schema_version, …N numeric columns)
```

Three properties are non-negotiable.

**Fixed schema.** Every row for a given schema version has the same columns in the same order. Models are trained against a schema, and inference must present an identical shape.

**Versioned.** Adding, removing, or redefining a feature bumps `feature_schema_version`. The model registry ([02-data-layer.md](02-data-layer.md)) records which version each model was trained against, so a model from three months ago remains reproducible against the feature definitions it actually saw — not against today's, which would silently change its meaning.

**Deterministic.** Recomputing features for 2019-03-14 must produce identical values today and next year. No wall-clock dependence, no randomness, no "days since today."

### Point-in-time correctness

This is the single most dangerous property in the whole engine, and it is worth more than a bullet point.

**A feature row for date D may only use information that was actually available at the end of date D.**

Violations of this are called lookahead leakage, and they are seductive because they make backtests look spectacular. Concrete ways it happens here:

- Computing a 20-day moving average *centered* on D, which includes days after D.
- Using a corporate action's adjustment retroactively on rows before the announcement was public.
- Normalizing a feature using statistics computed over the entire dataset, including the future.
- Joining F&O data timestamped after market close onto a feature row meant to represent the close.
- Including today's own close in a feature meant to predict today's close.

A model trained on leaked features will validate beautifully and fail immediately in live use, because in live use the future is not available. Every feature specified below must be implementable using a strictly backward-looking window, and any that cannot be is not built.

### Missing data

Features are computed on real market data, which has gaps: newly listed instruments have no 200-day history, illiquid names have zero-volume days, trading halts leave holes.

The rule is to **encode absence rather than invent values.** A 200-day feature on a 40-day-old listing is null, not zero and not forward-filled. LightGBM handles nulls natively and can learn from their pattern; a fabricated zero is a lie the model will treat as data.

## Feature categories

Seven categories. Each exists because it answers a question the others cannot.

---

### 1. Price

**Why it earns its place:** direction and momentum are the base signal. Everything else modulates this.

| Feature group | Contents |
|---|---|
| Returns | Close-to-close returns over 1, 2, 3, 5, 10, 20, 60 days |
| Momentum | Rate of change across the same windows; momentum rank within the universe |
| Acceleration | Change in momentum — whether the move is speeding up or tiring |
| VWAP relation | Distance from VWAP in absolute and percentage terms; days since last VWAP cross |
| Moving-average relation | Distance from 5/10/20/50/200 EMA and SMA; MA slope; MA alignment (are they stacked bullishly?); price position relative to the stack |
| Range position | Where the close sits within the day's range, the 5-day range, the 20-day range, the 52-week range |
| Breakout distance | Percentage distance from N-day highs and lows for several N; days since the last new high/low |
| Level proximity | Distance to recent swing highs and lows, to round numbers, and to the previous day's high, low, and close |

All computed on **corporate-action-adjusted** prices ([02-data-layer.md](02-data-layer.md)). Using unadjusted prices makes a 1:5 split indistinguishable from an 80% collapse.

---

### 2. Candlestick

**Why it earns its place:** two days can have identical closing returns and completely different stories. A day that opened at the low and closed at the high is not the same as a day that gapped up and sold off all session, even if both closed +2%.

| Feature group | Contents |
|---|---|
| Body | Body size relative to the day's range and to recent average range; body direction |
| Wicks | Upper and lower wick size relative to body and range; wick asymmetry |
| Gaps | Gap from previous close; whether the gap filled intraday; gap direction persistence |
| Sequences | Count of consecutive up or down closes; consecutive higher-highs / lower-lows |
| Patterns | Encodings for the small set of patterns worth including — engulfing, doji, hammer, inside/outside days — expressed as flags plus a strength measure rather than as opaque names |
| Volatility state | True range and ATR over multiple windows; ATR expansion/contraction ratio; realized volatility over several windows; volatility percentile within the instrument's own history |

The volatility features do double duty: they are predictive inputs and they are the primary raw material for Regime Labeling ([04-model-brain.md](04-model-brain.md)).

---

### 3. Volume

**Why it earns its place:** volume is the conviction behind the price. A breakout on half of average volume and a breakout on triple average volume are different events, and price alone cannot tell them apart.

| Feature group | Contents |
|---|---|
| Relative volume | Today's volume against the instrument's own 5/20/60-day average — critically, **relative to itself**, since raw volume only says whether a stock is large or small |
| Volume trend | Volume acceleration; consecutive rising-volume days |
| Volume–price interaction | Volume on up days vs down days; whether advances or declines are the higher-volume events |
| Divergence | Price making new highs while volume does not — the classic exhaustion tell |
| Spikes | Volume outliers measured in standard deviations of its own recent distribution |
| Turnover | Traded value, and liquidity rank within the universe |
| Delivery | Delivery percentage and its trend, where available — separates positional accumulation from intraday churn |

Liquidity features have a second role beyond prediction: they gate tradeability. A signal on an instrument that trades ₹40 lakh a day is not actionable regardless of how confident the model is, and the shortlister uses these features to enforce that ([04-model-brain.md](04-model-brain.md)).

---

### 4. Market structure

**Why it earns its place:** price and volume describe *what happened*; structure describes *where it happened relative to the map* the market has been drawing.

| Feature group | Contents |
|---|---|
| Trend structure | Higher-highs/higher-lows vs lower-highs/lower-lows classification over several lookbacks; trend maturity in days |
| Swing points | Distance to the most recent swing high and low; how many times each has been tested |
| Prior-day levels | Relationship to previous day's high, low, close; whether each was breached and whether the breach held |
| Opening range | Where the close sits relative to the first period's range (daily-resolution proxy: relationship of open to the rest of the session) |
| Range character | Today's range vs recent average; inside-day and outside-day flags; compression measures |
| Consolidation | Duration and tightness of the current base, if any; distance from the base's boundaries |
| Support/resistance | Distance to significant volume-weighted price levels; how many prior sessions found support or resistance nearby |

---

### 5. Market context

**Why it earns its place:** a stock up 2% on a day the Nifty is up 2.5% is underperforming. Without context, the model learns to chase market beta and calls it skill.

| Feature group | Contents |
|---|---|
| Index | Nifty and Sensex returns over matching windows; index trend and volatility state; India VIX level and change |
| Sector | Sector index return and trend; the instrument's rank within its sector |
| Relative strength | Instrument return minus index return, and minus sector return, over each window; RS trend; RS rank |
| Breadth | Advance/decline ratio; percentage of universe above its 20/50/200-day MA; new highs vs new lows |
| Correlation | Rolling correlation to index and sector; change in correlation — a name decoupling from its sector is a signal in itself |
| Beta | Rolling beta to the index |
| Regime context | The market-level regime label for the day, distinct from the instrument's own label |

---

### 6. F&O

**Why it earns its place:** the derivatives market frequently reveals positioning before the cash market reveals price. It is also where the sharpest participants express conviction, and where a signal that contradicts the cash chart is most worth knowing about.

| Feature group | Contents |
|---|---|
| Open interest | OI level, OI change, OI relative to its own recent average |
| Buildup classification | The standard four-quadrant read of price change against OI change |
| Futures basis | Futures price minus spot, in absolute and percentage terms; basis trend; annualized carry |
| Rollover | Rollover percentage near expiry; rollover cost |
| Options positioning | Put/call ratio by OI and by volume; PCR trend |
| Implied volatility | ATM IV level and change; IV percentile within its own history; IV skew; IV minus realized volatility |
| Strike structure | Max-pain distance; OI concentration at nearby strikes; whether price is approaching a heavy strike |
| Expiry | Days to expiry; expiry-week flag; day-of-expiry-cycle position |

The four-quadrant OI interpretation, stated explicitly since it is the backbone of several features:

| Price | OI | Interpretation |
|---|---|---|
| ↑ | ↑ | **Long buildup** — new money entering long. Genuine strength. |
| ↑ | ↓ | **Short covering** — rally driven by shorts exiting. Weaker, often fades once covering completes. |
| ↓ | ↑ | **Short buildup** — new money entering short. Genuine weakness. |
| ↓ | ↓ | **Long unwinding** — decline driven by longs exiting. Weaker, often exhausts. |

This distinction is exactly the kind of thing a price-only model cannot see: two identical +3% days mean opposite things depending on whether OI rose or fell.

F&O data exists only for instruments in the derivatives segment. For everything else these features are **null, not zero** — and the null pattern is itself informative, since F&O availability correlates with size and institutional interest.

---

### 7. News and events

**Why it earns its place:** it separates a move caused by market mechanics from a move caused by new information. Those two categories behave differently afterward, and a model that cannot tell them apart will keep predicting continuation into news-driven spikes that mean-revert.

This category is handled differently from the other six.

**A small numeric subset feeds LightGBM:**

| Feature | Purpose |
|---|---|
| Days since last material announcement | Recency of new information |
| Event type, categorically encoded | Earnings, dividend, board meeting, regulatory, M&A |
| Scheduled-event proximity | Days to next known earnings or board meeting — a stock two days from earnings behaves differently |
| Event density | Count of events in trailing windows |
| Historical event reaction | This instrument's average absolute move and direction bias following similar past event types |

**The qualitative substance goes to the Investigator instead.** Headlines, announcement text, and sentiment are passed to Layer 2 ([04-model-brain.md](04-model-brain.md)) rather than compressed into a numeric sentiment score for LightGBM.

The reasoning: a single "sentiment = 0.7" column throws away nearly everything that made the news meaningful, and it introduces a fragile dependency on whatever sentiment model produced the number. Reading "the CFO resigned two days after the auditor did" and understanding why that matters is a language task. Layer 2 exists to do language tasks. Forcing that judgment into a float and handing it to a decision tree is the worst of both worlds.

---

## Labels

Features are the input; labels are what the model is trained to predict. They live in the same pipeline and are subject to the same point-in-time discipline in reverse — **labels are the only thing permitted to look forward**, and only during training.

| Label | Definition |
|---|---|
| Direction | Sign of forward return over the prediction horizon |
| Magnitude | Forward return, for the expected-move output |
| Continuation | Whether the prevailing move persisted or reversed |
| Adverse excursion | Worst drawdown experienced before the horizon closed — supports risk-aware evaluation |
| Favorable excursion | Best gain available within the horizon |

The last two matter more than they first appear. A prediction that was "correct" at the horizon but drew down 6% along the way is not the same as one that never went against the position, and evaluating only the endpoint hides that difference entirely.

**The horizon is configurable, not hardcoded**, and it is recorded per prediction in the `predictions` table so that grading always compares like with like.

## Recomputation and cost

Nightly, only the new day's rows need computing — but features with long lookbacks depend on history, so the engine computes incrementally over a sufficient trailing window rather than rebuilding twenty-six years each night.

A **full rebuild** is required whenever the feature schema version changes, and it is a deliberate, Control Room–triggered operation rather than something that happens implicitly. After a full rebuild, every existing model is stale by definition — it was trained against different feature semantics — and the registry must reflect that rather than continuing to serve predictions from a model whose inputs have quietly changed meaning underneath it.
