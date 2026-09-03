# StockSense — Rebuild #3: a local quant stack that trades NSE intraday

**Date:** 2026-09-02 · **Branch:** `StockSense` (GitHub default) · **Executed by:** Claude Sonnet.
Written to be followed literally — exact module paths, exact formulas, exact DDL, exact test
names. Where a formula is given, type it as written. Where a file is marked **PROTECTED**, do
not edit it after it lands.

---

## Context — why this rebuild, and what is different this time

StockSense is a single-user NSE trading system for Harsh. **Account capital is read live from the
broker, never hardcoded — see "CAPITAL IS NEVER HARDCODED" below.** Intraday MIS gives up to 5×
leverage. Angel One is the broker, Upstox is the market-data
feed, an Electron desktop app is the face. It has been built and torn down twice. The entire
backend was wiped on 2026-09-02; the 42-commit research history survives on
`origin/phase0/prove-the-economics`.

> ### What this system IS, stated before anything else
>
> **An intraday engine. Everything is opened and closed inside one session, squared off by
> 15:10 IST. There is no overnight position and no multi-day hold anywhere in this plan.**
>
> The decision clock runs at **5 minutes** (with continuous sub-second stop/target monitoring),
> and label horizons are expressed in **bars and minutes, never in days**.
>
> The `h=10 days` figure in the table below is **historical record only** — the single
> configuration the *previous* build ever got through a gate. It is listed so we do not waste
> weeks re-deriving it. **It is not a target and nothing in Q1–Q10 builds toward it.**

**What the previous two builds proved, and must not be re-derived:**

| Result | Evidence |
|---|---|
| Cross-sectional ranker, h=10 days, point-in-time universe **PASSES** | `full_pit` +1.672% net alpha/rebalance, 25 folds, p=0.0001. mid +0.560%, small +0.694%. **large +0.060%, p=0.655 → FAIL.** Edge is mid/small-cap. Never confirmed forward. |
| **Intraday cross-sectional ranking is dead — 6 independent failures** | Decisive run: 15 folds, ~400k simulated trades, 91.2M real 1-min bars, **0/15 folds positive**, mean **gross** alpha −0.018%. No edge *before* costs. Hit rates 34–39%. |
| Confirmed by the real account | 441 intraday round trips (Aug 4–17, 2026): gross **+₹2,492**, charges **−₹3,785**, net **−₹1,293**. ~44 round trips/day was the killer, not per-trade cost. |
| MIS is *cheaper* than delivery | 8.3 bps vs 22.2 bps per ₹100k round trip, verified against a real charge sheet. |
| F&O positioning features | **FAIL** against the pre-registered gate. |

**The lesson that shapes this whole plan:** the failure was never compute or cost — it was
*ranking a large cross-section on generic ML features at intraday granularity*. Rebuilding that
with more GPU is the same experiment in a new costume. This plan is structurally different in
four ways: **(1)** market **microstructure** replaces generic technicals as the primary feature
source; **(2)** the system trades **event-conditioned, rare setups** on a small held book rather than
ranking hundreds; **(3)** Monte Carlo is used for **sizing and survival**, not for prediction;
**(4)** the multiple-testing guards are built **before** the strategy generator, not after.

---

## Findings from this session's research — five things that change the plan

### 1. Live auto-execution is IP-gated as of 2026-04-01 — this is the hard blocker

Effective 01-Apr-2026, **Angel One accepts API order execution only from a registered primary
static IP**, and this applies specifically to self-coded algos whose logic runs on the client's
machine — exactly our case ([Angel One](https://www.angelone.in/news/market-updates/what-s-changing-in-angel-one-s-smartapi-access-from-april-1-2026)).
NSE additionally mandates retail algos be hosted on the broker's server. SEBI's threshold is
**10 orders/second per exchange**; below that no algo registration is needed
([SEBI/NSE framework](https://www.quantinsti.com/articles/algorithmic-trading-india/)) — our
1–2 trades/day is nowhere near it, so registration is not the issue. **A dynamic home
broadband IP is.**

Consequence: **paper trading and all research are unaffected and start immediately.** Live
order placement is blocked until a static IP is arranged (commercial static-IP services for
Indian algo traders run ~₹499/mo) and whitelisted in the Angel One console. Phase Q0 probes
this on day one rather than discovering it after the execution engine is built.

### 2. Real HFT is not achievable here — but the HFT *mathematics* is, and it is where the money leaked

Genuine HFT is colocation, kernel-bypass networking, FPGAs and microseconds. Retail
REST/WebSocket over home broadband is 50–300 ms one way, and SEBI caps retail at 10 OPS. So
"HFT" as a latency race is off the table and this plan will not pretend otherwise.

What **is** achievable and directly valuable: the *quantitative content* of the five books
named — limit-order-book dynamics, bid-ask spread decomposition, market impact, execution
algorithms, tick-data analysis. The previous build's own post-mortem showed the account was
**gross positive and net negative**: the loss was in execution and turnover, not in direction.
Microstructure modelling attacks exactly that leak. This is the highest-expected-value part of
the entire plan and it is Phase Q3.

### 3. `finvizfinance` covers US markets only — repurposed, not discarded

Finviz screens NASDAQ/NYSE/AMEX. It has **no NSE coverage**, so it cannot screen Indian stocks.
Its honest use here: NSE's open gaps are strongly conditioned on overnight US/global moves, so
finviz's sector heatmap, gainers/losers and breadth become an **overnight global-regime feature
vector**, computed before 09:00 IST and fed to the gap/regime model. That is a real, testable
input — and it dovetails with the overnight–intraday reversal hypothesis below.
`yfinance` *does* cover NSE (`RELIANCE.NS`, `^NSEI`, `^INDIAVIX`) and is the free backup spine.

### 4. `gs-quant` needs institutional GS credentials for its pricing/data APIs

Confirmed from its README: "In order to access the APIs you will need a client id and secret.
These are available to institutional clients of Goldman Sachs." Its instrument/pricing layer is
also derivatives- and US-centric — of little direct use for NSE cash equity intraday.

What is genuinely usable offline, confirmed by listing the package:

- **`gs_quant/backtests/`** — a complete event-driven engine: `generic_engine.py`, `triggers.py`,
  `actions.py`, `action_handler.py`, `execution_engine.py`, `data_sources.py`, `strategy.py`,
  `predefined_asset_engine.py`, `backtest_objects.py`, `order.py`.
- **`gs_quant/timeseries/`** — a large pure-computation library: `algebra`, `analysis`,
  `econometrics`, `statistics`, `technicals`, `backtesting`, `datetime`, and notably
  **`tca.py` (transaction-cost analysis)** — directly relevant to Q3, which is where our money
  actually leaked.

**We adopt the architecture pattern** — `Strategy = f(Triggers → Actions) → ExecutionEngine` over
a `DataSource` — because it is a proven institutional design for exactly the "run thousands of
strategy configurations" problem, and we read `tca.py` and `technicals.py` as reference
implementations. We install `gs-quant` for `timeseries` and do not depend on Marquee for anything.

### 5. Machine inventory (measured, not assumed)

| | |
|---|---|
| CPU | Intel **i5**-13420H, 8 cores / 12 threads (you said i7 — it is an i5; sizing below uses the real number) |
| RAM | 15.7 GB |
| GPU | RTX 3050 **4 GB** Laptop, driver 596.36 |
| Disk | 147 GB free of 476 GB |
| Python | 3.14.5 — wheels confirmed available for numpy 2.5.2, pandas 3.0.5, duckdb 1.5.5, polars 1.44.1, lightgbm 4.7.0, numba 0.67.0, scipy 1.18.1, torch 2.13.0 |
| Node | v24.15.0, npm 11.12.1 (Electron ready) |
| Ollama | installed, v0.32.6 → auto-updating to 0.33.2; **only `qwen2.5` pulled**; server was not running |
| Claude CLI | present at `~/AppData/Roaming/npm/claude` |
| Obsidian | vault exists at `C:\Users\harsh\Documents\Obsidian Vault` (empty but for `Welcome.md`) |

**The 4 GB GPU is the binding constraint and it is shared.** Ollama and CuPy cannot both hold
it. Q6 introduces a GPU lease so they serialise instead of OOM-ing each other.

---

## THE COST WALL — the most important number in this plan

Computed during planning from the exact 2026 statutory rates, not estimated. It reshapes the
strategy priorities, and it is uncomfortable.

### The real round-trip cost

| Component | Rate | Side |
|---|---|---|
| Brokerage | ₹20 **or** 0.03%, whichever is **lower**, per order | both |
| STT | **0.025%** | **sell only** |
| Exchange txn (NSE) | 0.0030699% | both |
| SEBI turnover | 0.0001% | both |
| Stamp duty | 0.003% | **buy only** |
| GST | 18% on (brokerage + exchange txn) | both |

**Verified: a ₹100,000 round trip costs ₹82.64 = 8.26 bps**, confirming the 8.3 bps figure this
plan has been citing. But the figure is **not constant in position size**, because the flat ₹20
stops binding below ~₹66,667:

| Position size | Round trip | **bps** |
|---|---|---|
| ₹8,750 – ₹66,667 | proportional | **10.62** |
| ₹87,500 | ₹78.21 | 8.94 |
| ₹100,000 | ₹82.64 | 8.26 |
| ₹200,000 | ₹118.09 | 5.90 |

At realistic per-position sizes for a small account, **the true cost is 10.62 bps, not 8.3.**

### What that does to the strongest effect in the literature

Lou/Polk/Skouras, applying the McLean–Pontiff −58% haircut and real costs:

| Strategy | Gross/mo | After haircut | Cost/mo | **Net/mo** | |
|---|---|---|---|---|---|
| Intraday leg, **daily** round trips | 3.02% | 1.27% | 2.23% | **−0.96%** | **NEGATIVE** |
| Overnight leg, **daily** round trips | 3.47% | 1.46% | 2.23% | **−0.77%** | **NEGATIVE** |
| Close-to-close, **monthly** rebalance | 0.45% | 0.19% | 0.11% | **+0.08%** | viable, but thin |

**The best-evidenced effects in the academic literature lose money at retail costs when traded
daily.** 21 round trips a month at 10.62 bps is **2.23%/month in charges** — more than the
haircut edge itself.

**Break-even is a gross edge of ~14.2 bps per trade** (10 positions, 3× leverage). For reference,
the tug-of-war intraday leg is ~14.4 bps/day gross *before* decay and ~6.0 bps/day after — i.e.
**below the bar.**

### What this changes — and it does not kill the project

1. **Turnover is worth more than signal strength.** Halving turnover beats doubling edge, because
   cost scales linearly with trades while edge does not.
2. **It validates the event-conditioned family (family 5).** Rare setups with a large per-trade
   move are exactly the right shape: few round trips, high edge each. That family moves **up** the
   priority order.
3. **The cost hurdle (cascade stage 3) is the PRIMARY filter, not a formality.** Any daily-turnover
   candidate must clear ~10.6 bps gross per trade before it is worth walk-forwarding at all — and
   most will not.
4. **Family 1 keeps its slot but changes shape.** It runs first because it is cheap to test and
   settles a real question, but the *daily* intraday version is expected to fail on cost. The
   lower-turnover variants are what to watch.
5. **The wall recedes as the account grows** — 10.62 → 8.26 → 5.90 bps as positions get bigger. A
   strategy that is marginal now becomes viable later, which is another reason capital is read
   live rather than fixed.

**Honest caveat:** these are US large-cap value-weighted figures and Indian small/mid caps may
carry larger effects; the −58% haircut is an average, not a law. But the *direction* is not in
doubt, and pre-registering this expectation now is what stops a marginal backtest being talked
into looking good later.

---

## Three kinds of number, and only ONE of them should be intelligent

Prompted by a correct objection to a hardcoded ruin ceiling. Not every constant is the same kind
of thing, and treating them alike is how a system ends up either rigid or unsafe.

**1. DECISION parameters — must be intelligent, never constants.**
Leverage, position count, entry and exit rules, trade count, which names to hold. These are
*choices under uncertainty*, so they come from **maximising expected log growth given current
regime and current belief in the edge** (Q5). They adapt daily. No thresholds.

*Trade count is worth calling out:* `max_orders_per_day` is redundant as a decision rule, because
an objective that charges the real **10.62 bps** per round trip already refuses trades whose edge does not clear
cost. It survives only as a safety limit — category 3.

**2. STATISTICAL GATES — must be fixed, precisely BECAUSE they must not adapt.**
`DSR ≥ 0.95`, `PBO ≤ 0.5`, `binomial p ≤ 0.05`, `min_folds_required = 10`. These are
pre-registered significance conventions from the published literature. **A gate that adapts to
results is not a gate** — the entire value of pre-registration is that the bar cannot move once
the answer is visible. This project committed that error once and documented it. They stay frozen.

**3. SAFETY INTERLOCKS — must be fixed, dumb, and non-negotiable.**
Daily realised-loss limit, 15:10 square-off, per-order value cap, arming expiry, data-staleness
block. **These are deliberately NOT intelligent.** An adaptive kill switch is a kill switch that
can be reasoned around, and the one thing a kill switch must never do is find a clever argument
for not firing. They are dumb on purpose.

The failure modes are symmetric and both real: making category 1 rigid produces a system that
cannot adapt to a changing market; making category 2 or 3 adaptive produces a system that talks
itself into a bad trade. **Intelligence belongs in the decisions, discipline in the limits.**

---

## CAPITAL IS NEVER HARDCODED — a core architectural rule

**This was got wrong in earlier drafts and is corrected here.** A fixed ₹17,500 was written into
`core/config.py` and into ~8 places in this plan. That figure came from a transcript weeks old, and
**it is already stale**. Worse than stale, it is the wrong *shape*: a system that bakes in its own
account size cannot notice the account growing, so every derived quantity — position size, price
band, leverage headroom, ruin probability — silently drifts out of date while looking correct.

### The rule

> **No rupee figure for account capital appears anywhere in source, config, or strategy. Capital is
> READ from the broker at decision time, every time.**

The Q0.3 probe already proved this works: Angel One's **`rmsLimit`** endpoint returns live margin
and available cash, read-only, from this ISP (`rms status=True`).

### How it is implemented

```python
# src/stocksense/core/capital.py -- the single source of truth
@dataclass(frozen=True)
class AccountState:
    equity_inr: float          # live, from broker RMS
    available_margin_inr: float
    utilised_margin_inr: float
    as_of: datetime
    source: str                # "broker" | "cached" | "explicit"

def live_account_state(max_age_s: int = 300) -> AccountState:
    """Reads Angel One RMS limits. Cached for max_age_s so a 5-minute scan does
    not hammer the endpoint. On failure returns the last known value with
    source='cached' and a stale flag -- it must NEVER silently substitute a
    constant, because a wrong capital figure sizes every order wrongly."""
```

- `core/config.py` loses `equity_inr` as a default. Any offline/backtest caller must pass capital
  **explicitly**, so no code path can accidentally inherit a stale number.
- Every function that needs capital — `tradeable_price_band`, `whole_share_quantity`,
  `probability_of_ruin`, position sizing — already takes it as an **argument**. That stays, and no
  default is ever supplied.
- **If capital cannot be read and no fresh cache exists, the engine refuses to size an order** and
  says so, rather than guessing. An interlock covers this.

### Backtests are expressed in PERCENT, never rupees

This is the deeper fix. A backtest that reports "₹290 per cycle" is only true for one account size
and rots the moment the balance changes. Every result — alpha, drawdown, cost drag, expectancy —
is stored as a **fraction of deployed capital**. Rupee figures are *derived at display time* from
whatever the account currently holds.

Consequence: **the research layer never knows or cares what the balance is.** Only the execution
and reporting layers do, and they ask the broker.

### Where rupee examples still appear in this document

Only in two places, both explicitly hypothetical and neither load-bearing:
1. The historical record of what the *previous* account did (gross +₹2,492, charges −₹3,785) —
   that is a measured fact about the past, not a parameter.
2. Worked illustrations of a formula (tick drag, divisibility). These use a stated example figure
   purely to make the arithmetic legible, and the code takes capital as an argument.

The position-count and ruin tables in Q5 are likewise **ratios**, and hold at any account size —
what changes with capital is only the *upper* price bound, which widens as the balance grows.

---

## The arithmetic, stated once so it never has to be argued again

- A daily target of **8–11% of equity** (which is what ₹1,500–2,000/day meant at the balance then)
  is **1.7–2.3% per day of leveraged exposure, every day**. No system achieves this at any account
  size — which is the point of stating it as a RATIO: it stays true as the balance grows.
- A genuinely excellent outcome (Sharpe ≈ 1.5, net of the real 10.62 bps round trip) is on the
  order of **₹75–150/day average, with routine ±₹500–1,000 daily swings** and losing weeks.
- **"Never makes a loss" is not attainable.** A Sharpe-2 strategy still loses 35–40% of days.
  What *is* attainable and is what this plan optimises for: **bounded loss** (a hard daily stop
  that cannot be overridden) and **positive expectancy compounding over months**.
- The capital ladder is adopted, but NOT the 1–2 name concentration — see Q5, where measurement
  showed a single name at 5× carries an ~87% one-year chance of losing half the account. Once
  equity reaches ~₹30–40k, 1–2%/day becomes the target — which is itself still ambitious but is
  within an order of magnitude of reality.

**The system will report the number it actually earns. If that is ₹60/day, it will say ₹60/day.**

---

## Decisions taken (from your brief)

| Decision | Choice |
|---|---|
| Frontend | **Rebuilt from scratch** from your screen recordings (you chose this) — not restored from git |
| Startup | **No Windows scheduled tasks.** Everything starts from a CLI you invoke by prompting Claude Code, and every feed self-reports health so failures are visible, never silent |
| Deployment | Offline-first local Electron desktop app. No cloud, no hosted service |
| Market scope | **NSE cash equities only.** No F&O, no crypto, no forex. Intraday 09:00–15:10 IST |
| Money | Ungated after **week 1 of paper trading**, subject to the gate + interlocks in Q7 |
| LLMs | Ollama **local** (llama3.2:3b) + Ollama **Cloud** for synthesis. **No Claude CLI and no Anthropic API key in the runtime** — Claude is a development tool here, not a dependency |
| Knowledge | Obsidian vault as the knowledge graph, written by the app, read by Claude Code |
| **Leverage & position count** | **Neither is chosen and there is NO hard threshold.** Both are selected each day by maximising **expected log growth** under an edge that may be spurious — see Q5's "Intelligent sizing". Compounding prices ruin by itself, so no cutoff has to be invented. Measured: this picks **3x/8-10 names** on a confident edge and walks itself down to **1x** as the edge looks less real. It overrides the earlier "5x MIS, 1-2 names" assumption everywhere it appeared. |
| Capital | **Read live from the broker at decision time. Never hardcoded, never defaulted.** Research results are stored as percentages so they stay true as the balance changes |

---

## PROTECTED paths — do not edit after they land

```
src/stocksense/evaluation/gate.py
src/stocksense/evaluation/walkforward.py
src/stocksense/evaluation/vault.py
src/stocksense/evaluation/attempts.py
src/stocksense/execution/cost_model.py
src/stocksense/execution/interlocks.py
src/stocksense/brokers/angel_execute.py
research/*preregistration*.md
tests/unit/test_{leakage,determinism,gate,vault,interlocks,readonly}.py
```

---

# Q0 — Probes before building (half a day, and it may reshape everything)

Every previous build lost time to an assumption that a ten-minute probe would have killed. Each
probe writes `research/probes/<name>.md` with the raw result.

**Q0.1 — Angel One static-IP reality.** Log in to the Angel One console; find the SmartAPI
static-IP registration page; record the current public IP (`curl ifconfig.me`) and whether the
ISP assigns a static one. Then attempt one *test* order placement (smallest possible quantity,
immediately cancelled) and record the exact rejection or success. **This single result decides
whether Q7's live path is buildable now or parked behind a static-IP purchase.**

**Q0.2 — Upstox live feed.** Verify the `.env` token still authenticates; open a
`v3` market-data WebSocket; record ticks/second for 20 symbols over 60 seconds and the
wall-clock lag between exchange timestamp and local receipt. This is the real latency number —
everything in Q8 is sized against it, not against a guess.

**Q0.3 — Angel One read-only.** Fresh TOTP login (the previous build found session reuse across
processes silently fails on this SDK — always fresh-login), then `getHolding` + `getPosition` +
`getTradeBook`. Confirms the account-state path still works from this ISP.

**Q0.4 — GPU headroom.** `pip install cupy-cuda12x`; allocate and time a 50M-path float32 Sobol
draw; record peak VRAM with `nvidia-smi` while Ollama is (a) stopped and (b) running qwen2.5.
Produces the hard batch-size ceiling used by Q6.

**Q0.5 — Torch CUDA on Python 3.14.** `pip install torch --index-url .../cu124` and assert
`torch.cuda.is_available()`. If no cp314 CUDA wheel exists, the Q5 sequence model runs on CPU or
is deferred — decide from evidence, not hope.

**Q0.6 — News + finviz reachability.** Moneycontrol/ET RSS worked on this ISP previously;
re-verify. Then `pip install finvizfinance` and pull one screener page — confirm it is reachable
and confirm (as expected) that no NSE ticker resolves.

**Gate:** all six results written and read before a line of Q1 is committed.

---

# Q1 — Data spine (rebuild, better than before)

Nothing is salvageable on disk — `data_store/` (26 GB) was deleted. But the *shape* of the
previous spine was correct and is reproduced, with the two bugs that cost days already fixed.

**`src/stocksense/data/store.py`** — **BUILT. The design in the approved plan was wrong and was
corrected against a measurement.**

The plan said: open readers with DuckDB's `read_only=True` so they coexist with the writer.
**That does not work.** Measured on this machine: with one process holding the file, a second
process opening it `read_only=True` fails with

```
_duckdb.IOException: IO Error: Cannot open file "...": File is already open
```

which is *exactly* the failure that killed the previous build's nightly jobs. Adopting the
approved plan verbatim would have reproduced the bug it was written to prevent.

What was verified to work instead, in the same test: **three concurrent processes reading
Parquet through an in-memory DuckDB connection, while a fourth held the DuckDB file lock.**
Parquet takes no lock. So storage splits in two:

- **`Store`** — the single **writer**. Owns `stocksense.duckdb`, which now holds *only* small
  mutable state (`ingest_runs`, `corporate_actions`, and later the attempt registry, orders and
  arming). Bulk rows never live there.
- **`Reader`** — **lock-free**, any number of concurrent processes, reads only Parquet. This is
  what the 10 search workers, the API server and the UI all use. **Nothing outside `store.py`
  opens the DuckDB file.**

`Store.publish()` snapshots the small DuckDB tables to Parquet so readers have exactly one
access path for every dataset.

Bulk datasets are partitioned Parquet by year-month, so an upsert rewrites a **bounded** amount
of data (one month of bhavcopy ≈ 22 days × ~2,500 symbols). Each partition is written to a temp
file and atomically renamed, so a reader can never observe a half-written partition and a crash
mid-write leaves the previous one intact.

One further trap found by the tests and fixed in the write path: a python `date` round-trips
through Parquet as a pandas `Timestamp`, so the dtype silently depended on whether a frame came
from ingest or from disk — and a `date == Timestamp` comparison quietly evaluates False, which
would have dropped rows from every trading-calendar join. The time column is now normalised on
write, and `bhavcopy_bounds()` hands back plain `date` objects.

10 tests cover this, including `test_readers_work_while_a_writer_holds_the_lock`, which asserts
both the positive (Parquet readers succeed) and the negative (the DuckDB file reader is refused).

**Sources, in priority order:**

- `data/nse_bhavcopy.py` — daily EOD 2010→today, both format eras (legacy and UDiFF, cutover
  2024-07-08). **Two known traps, fix on first write:** (a) `pd.read_csv(..., keep_default_na=False,
  na_values=[""])` — `"NA"` is a real NSE series code for bond instruments and pandas silently
  nulls it, which crashed an entire backfill; (b) `LAG(close)` per symbol is **not** "yesterday" —
  illiquid names skip days, so every day-over-day computation must join the market calendar.
- `data/corporate_actions.py` — NSE corporate-actions endpoint, quarterly windows, free-text
  `subject` parser. The previous parser reached **96.2%** agreement against yfinance-implied
  adjustment jumps; reproduce its grammars, including the three split phrasings and the genuine
  `"Splt"`/`"Frm"` typo variant. Known unfixable-from-text gaps to record, not hide: rights
  issues, buybacks, demergers, schemes of arrangement.
- `data/adjust.py` — back-adjusted `adj_close` on both price and total-return bases. **Critical:**
  the anomaly detector must flag *adjusted-price jumps with no matching corporate-action record*,
  **not** `adj_close` vs raw `close` — the latter quarantined RELIANCE, TCS, INFY, HDFCBANK and
  ~600 other blue chips for having genuine splits.
- `data/upstox_hist.py` — 1-minute bars, 2022→today, ~250 most-liquid names. Generator +
  incremental write (resumable); content-hash disk cache; 31-day max window per request.
- `data/yfinance_src.py` — `.NS` tickers, `^NSEI`, `^INDIAVIX`, plus `^GSPC`/`^IXIC`/`^VIX` and
  USDINR for the overnight-global vector.
- `data/finviz_src.py` — US sector heatmap + breadth, fetched pre-09:00 IST. **US only, by design.**
- `data/news.py` — Moneycontrol/ET RSS, per-symbol and per-sector, timestamped so nothing
  published after a decision can leak into it.
- `data/universe_pit.py` — point-in-time tradeable universe (liquidity + price floors resolved
  *as of* each date). This is the single most important anti-survivorship control and it must be
  wired into every fold, not just a display command.

**Tests:** `test_data_spine.py` — the `"NA"` series row survives; a calendar-gap symbol is not
treated as consecutive; a known split (ECLERX 1:2, PASHUPATI 1:10) yields a continuous adjusted
series; interrupting a backfill loses no committed rows.

---

# Q2 — Measurement and guards, built BEFORE any strategy generator

This ordering is not negotiable and is the reason the previous build's Phase K existed.

> **After 1,000 independent backtests the expected best Sharpe ratio is 3.26 — even when the true
> edge is exactly zero.** (Bailey & López de Prado)

You have asked for *thousands* of strategies. At N=5,000 the expected best Sharpe from pure noise
is higher still. Without the guards below, the search does not find alpha; it finds the prettiest
noise, faster. Build the referee before the players.

**`evaluation/factor_metrics.py`**
```python
def cross_sectional_ic(scored, method="spearman") -> pd.Series
    # ONE IC per rebalance date (per-date rank IC — the industry standard).
    # Drop dates with < 10 non-NaN pairs.
def icir(ic_series) -> float
    # mean(IC) / std(IC, ddof=1). nan if len < 3 or std == 0.
    # THIS IS THE SEARCH'S OBJECTIVE FUNCTION.
def decay_curve(...) -> pd.DataFrame   # columns: horizon, mean_ic, std_ic, icir, n_dates
    # ONE fitted model, ONE fixed scoring-date list, scored once; then per horizon
    # look up that horizon's label over the SAME dates. Never re-subsample per horizon.
def half_life(curve) -> float          # horizon where mean_ic <= 0.5 * peak, linearly interpolated
def sharpe / sortino / max_drawdown / calmar
```
Calibration to state in the docstring: **real equity factors run IC 0.02–0.05. IC > 0.15 is an
overfitting red flag, not a win.** Grinold: IR ≈ IC × √breadth.

**`evaluation/robustness.py`** (needs `scipy>=1.11` declared *explicitly* in `pyproject.toml` —
a transitive-only dependency has bitten this repo before)
```python
EULER_MASCHERONI = 0.5772156649015329

def expected_max_sharpe(n_trials, trial_sharpe_std, trial_sharpe_mean=0.0) -> float:
    """SR0 = mean + std * [ (1-g)*Z^-1(1 - 1/N) + g*Z^-1(1 - 1/(N*e)) ]"""
    # PRE-VERIFIED: N=1000, std=1.0, mean=0.0 -> 3.2551 (published 3.26).
    # Reference: N=10 -> 1.5746, N=100 -> 2.5306, N=10000 -> 3.8607.

def deflated_sharpe_ratio(observed_sharpe, n_trials, trial_sharpe_std,
                          sample_length, skew, kurtosis) -> float:
    """DSR = Z[ (SR - SR0) * sqrt(T - 1)
                / sqrt(1 - skew*SR + ((kurtosis - 1)/4) * SR**2) ]
    kurtosis is the RAW fourth moment (normal = 3.0), NOT excess. >= 0.95 is the bar."""

def probability_of_backtest_overfitting(performance: pd.DataFrame, s: int = 16) -> dict:
    """CSCV. rows = time slices, cols = strategy configs.
    Split rows into s contiguous subsets (s even, >= 4); for every combination of s/2
    as IS, complement is OOS; n_star = best IS column; w = rank of n_star in OOS / (n_cols+1);
    logit = ln(w/(1-w)); PBO = fraction of combinations with logit <= 0.
    PBO <= 0.5 is the bar."""
```

**`evaluation/vault.py`** — sealed holdout. `VAULT_SEAL_DATE = date(2025, 7, 1)` (later than the
previous build's 2025-01-01, because the search will be far larger this time and needs a bigger
untouched sample). Enforced at the single `load_candles` choke point: without an `UnsealToken`,
rows on/after the seal are dropped and an INFO line records how many. `unseal()` refuses unless
the pre-registration file is **committed to git**, the attempt exists in the registry, and no
prior unseal exists for that `hypothesis_id`. **One unseal per hypothesis, ever.**

**`evaluation/attempts.py`** — append-only registry. **Every single backtested configuration
registers here, no exceptions.** This is what turns `n_trials` in the Deflated Sharpe from a
guess into a counted fact.

**`evaluation/gate.py`** — the promotion gate. Pre-registered thresholds, binomial hit-rate test
on positive folds, `min_folds_required=10`. **PROTECTED.** No threshold may be adjusted after
seeing a result — this project committed that error once, documented it, and rebuilt from
statistical principle. That discipline carries.

**Decision authority — write this table into both module docstrings:**

| Metric | Role |
|---|---|
| ICIR, decay half-life | Search objective + first screen |
| `alpha_net` via `gate.py` | Promotion gate — unchanged, untouched |
| DSR ≥ 0.95, PBO ≤ 0.5 | Final gate, on the sealed vault only |
| IC, Sharpe, Sortino, Calmar, maxDD | Diagnostics — never a gate alone |

---

# Q2.5 — Labelling and sampling: the AFML layer

`labels/` and `features/sampling.py`. Missing from the first draft and added because it is
standard practice at a real desk, and because one of these techniques targets your measured
failure directly.

**Triple-barrier labelling** (López de Prado). Replace fixed-horizon returns with three barriers:
profit target, stop loss, and a time limit — the first one touched is the label. This is the
"first-touch" label the plan already called for, named properly. It matters because a
fixed-horizon label credits a trade with a return it would never have realised: with a 1.5% stop,
whether price touched −1.5% *before* +2% decides everything. Barriers are scaled by realised
volatility, not fixed in percent, so the same rule adapts across names and regimes. Session-bounded
— never crosses 15:10.

**Meta-labelling — the single most on-point technique in this plan.** A primary model says
*direction*; a **secondary** model says *whether to act at all*, and sizes the bet. Your account is
the textbook case for it: **gross +₹2,492, charges −₹3,785, net −₹1,293** across 441 round trips.
The direction was roughly fine; the problem was taking too many marginal trades. Meta-labelling is
precisely the method for raising precision and cutting trade count — and cutting trade count is
what turns that ledger positive. It also produces a calibrated P(win) that feeds Q5's sizing
directly.

**CUSUM event sampling.** Do not sample every bar. A CUSUM filter fires only when cumulative
price movement breaches a threshold, which concentrates training on structural breaks instead of
noise, and *naturally produces the "rare, event-conditioned setups" strategy family* rather than
bolting that on. Fewer, better-conditioned samples.

**Sample uniqueness weights + sequential bootstrap.** Overlapping label windows mean rows are not
independent, which silently inflates every significance test in the pipeline. Weight each sample
by its average uniqueness and bootstrap sequentially. This interacts with the gate: without it,
the walk-forward's p-values are optimistic for a reason that has nothing to do with the signal.

**Fractional differentiation.** Prices are non-stationary; returns are stationary but memoryless.
Fractional differencing takes the minimum `d` that passes an ADF test while retaining maximum
memory — keeping level information a plain return throws away.

**Volume / dollar bars.** Time bars sample the quiet lunch hour as heavily as the open. Bars
sampled on traded value have better statistical properties and align with how information
actually arrives. Derived from the 1-minute spine, never fetched separately.

---

# Q3 — Microstructure: the HFT mathematics, applied where the money actually leaks

This is the genuinely new pillar and, on the evidence, the highest-expected-value work in the
plan. Your account was **gross positive, net negative**. That is an execution problem, and
execution is the one thing the previous builds never modelled properly.

### The constraint that shapes this whole phase (researched, not assumed)

The literature is unambiguous that **order-flow imbalance is the strongest short-horizon
predictor there is.** Cont, Kukanov & Stoikov show price changes over short intervals are driven
by OFI with a *linear* relation whose slope is inversely proportional to depth, robust to
intraday seasonality and stable across stocks and time scales
([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1712822)); single-level OFI explains
**65–87% of short-term mid-price variance**.

And there is an NSE-specific result that is almost custom-written for this project
([*Information content of order imbalance in an order-driven market: Indian Evidence*](https://www.sciencedirect.com/science/article/abs/pii/S1544612320316779)):

- Predictability is **strong for the first five minutes and gone within thirty**.
- The 1-minute coefficient is **positive**; every longer horizon is **negative** — the
  return/order-imbalance relation *reverses*.
- Information shows up in **deeper book levels**, implying informed traders beyond the best quote.

That reversal is a real, Indian, structural effect and it is the same family as the
overnight–intraday reversal hypothesis in Q4. It is worth testing directly.

**But here is the blocker, and it is decisive:** **historical depth data does not exist for us.**
Upstox serves only OHLCV bars historically; depth (D5, or D30 in `FULL_D30`) is **live-stream
only**. So true OFI **cannot be backtested** on 2022–2026. Building the microstructure pillar on
OFI alone would produce a signal with no validation history — exactly the mistake this project
keeps being asked not to repeat.

### The resolution: two tracks, honestly separated

**Track A — what IS computable from the 1-minute OHLCV we actually have, and therefore
backtestable today.** There is a mature low-frequency literature for exactly this situation
(recovering microstructure from transaction prices when quotes are unavailable):

- **EDGE** — Ardia, Guidotti & Kroencke, *JFE* 161 (2024) 103916. The current state of the art:
  asymptotically unbiased, optimally combines open/high/low/close, and the authors state it
  applies at **both low and high frequency**. Python package `bidask` (`edge`, `edge_rolling`,
  `edge_expanding`). **This is the effective-spread estimator we use**, per symbol per
  time-of-day bucket — intraday spread is U-shaped and a flat constant is a lie.
- **Roll (1984)** serial-covariance spread and **Corwin–Schultz (2012)** high-low, as
  cross-checks. Report all three; disagreement between them is itself a data-quality signal.
- **Amihud illiquidity**, **Kyle's lambda**, realised volatility, and tick-rule signed volume
  (a bar-level approximation of order flow — weaker than true OFI, but backtestable).

**Track B — true OFI, recorded live from today, usable later.** The live WebSocket carries depth.
`microstructure/flow.py` records it to Parquet from day one so that in a few months there is a
genuine depth history to test on. **It is a feature nothing decides on until it has been
validated against a real sample** — no exceptions, same gate as everything else.

Stating it plainly: **Track A ships and can be validated now; Track B accrues and gets tested
later.** Any claim that this system "trades on order flow" before Track B has history would be
false.

### Modules

**`microstructure/spread.py`** — EDGE (primary), Roll and Corwin–Schultz (cross-checks), Amihud,
Kyle's lambda. Per symbol, per time-of-day bucket.

**`microstructure/impact.py`** — square-root market-impact law and the Almgren–Chriss optimal
execution schedule. At this account's order sizes on liquid names impact is small but **not
zero**, and the previous
cost model used a flat 5 bps slippage constant with no participation check at all.

**`microstructure/lob.py` — limit order book simulator** (Numba `@njit`). Price-time priority
matching, L2 reconstruction, queue-position estimation. Its honest role is **execution research,
not alpha**: replay Track B's recorded depth to measure what a limit order at the touch actually
achieves versus a market order. This is where the "code the concepts" content of Cartea/Jaimungal
and Bouchaud lives, and it directly answers whether passive entry can claw back part of the
8.3 bps.

**`microstructure/flow.py`** — live-depth recorder + OFI, Lee–Ready trade signing, volume-clock
bars, VPIN. Records now, decides nothing yet.

**Tests:** EDGE recovers a planted spread on a synthetic series to within tolerance; the three
estimators agree in order of magnitude on a real liquid symbol; the impact model is monotone in
size; a hand-built book produces a known fill; OFI is sign-correct on a constructed buy-pressure
sequence.

---

# Q3.5 — The intraday horizon, made explicit

Because "5-minute prediction" can mean two very different systems, and only one of them works.

### Two clocks, and they are not the same thing

| | Cadence | What happens |
|---|---|---|
| **Evaluation clock** | **every 5 min**, 09:15→15:10 | The frozen model re-scores the watchlist on fresh 5-min bars. Runs ~75× a day. |
| **Trade clock** | **1–2 fills a day** | A signal only fires when it clears the setup precondition, the interlocks and the cost hurdle. Most evaluations produce nothing. |

**Re-scoring every 5 minutes is correct. Emitting a trade every 5 minutes is not.** 75 signals a
day at the real 10.62 bps round-trip is ~8.0% of exposure per day burned in charges before any edge exists —
which is arithmetically the same mistake your account already made: 441 round trips over two
weeks, gross **+₹2,492**, charges **−₹3,785**, net **−₹1,293**. Trade count is the variable that
killed it, so trade count is a *constraint*, not an output: `max_orders_per_day = 8`,
`max_open_positions` (swept 2–10 per Q5, then FIXED by the winning config), enforced in
`execution/interlocks.py`.

### The fact that constrains the design

The previous build tested exactly "predict every 5 minutes": 5-minute bars, 244 symbols, 91.2M
real 1-minute bars, 15 folds, ~400,000 simulated trades. **0 of 15 folds positive. Mean *gross*
alpha −0.018%** — no edge before a single rupee of cost. Cross-sectionally ranking a large
universe on price-derived features at 5-minute granularity is the one thing we know does not work,
and no amount of GPU, leverage or re-tuning rescues a signal with no gross edge.

So the 5-minute clock stays; **what it evaluates changes**.

### Data availability decides the running order

Audited against the vault seal (2025-07-01):

| Data | Research window | Sessions | Status |
|---|---|---|---|
| **Daily bhavcopy** | 2010-01-04 → 2025-07-01 | **~3,800** | ingesting now, ~60% done |
| 1-minute intraday | 2022-01-01 → 2025-07-01 | **~870** | **blocked on the Upstox token** |
| Delivery % | ~2021 → 2025-07-01 | ~1,100 | with the daily backfill |

**The intraday window is only 3.5 years and is mostly one regime** — it contains the 2022
correction and the 2023–24 bull run, but no 2020-style crash. With CPCV at `n_folds=10` that is
~87 sessions per fold: workable, but thin, and it means an intraday result deserves more suspicion
than a daily one carrying 15.5 years.

**This is a third independent reason family 1 (tug of war) runs first**: it needs only daily bars,
so it gets 15.5 years and ~4× the sample of anything requiring minute data.

### Horizons, in bars — the config the search actually sweeps

```python
BAR_MINUTES        = 5            # research grain, resampled from the 1-min spine
HORIZON_BARS       = (1, 3, 6, 12, 24)   # 5m, 15m, 30m, 1h, 2h  -- NEVER days
MAX_HOLD_BARS      = 72           # 6h: an MIS position cannot outlive the session
SESSION_BOUNDED    = True         # no label or holding window crosses 15:10
```

Every label is a **triple barrier** (Q2.5) in these units: volatility-scaled profit target,
volatility-scaled stop, and a time barrier in bars. Nothing is close-to-close, and nothing
survives the session.

### What we evaluate at 5 minutes, in priority order

1. **Opening-range breakout** with volume confirmation against a *same-time-of-day* baseline —
   fires in the first 30–60 minutes, holds 15 min to a few hours.
2. **Intraday momentum** — first half-hour predicts last half-hour (Gao/Han/Li/Zhou). One
   decision, late-session entry.
3. **Overnight–intraday reversal** — the strongest untested lead the project owns; enter at the
   open, exit by 15:10. A once-a-day decision that the 5-min clock manages, not re-decides.
4. **Event-conditioned gap** — rare structural preconditions, few trades a year by design. The
   deliberate opposite of the flat-signal-across-a-large-universe failure.
5. **Order-flow imbalance** — on NSE, predictability is strong for the **first five minutes** and
   **reverses** beyond it, which is the most promising 5-minute signal in the literature. It is
   **live-recorded now and gated until it has history** (Q3, Track B), because we cannot backtest
   depth we never captured.

The honest summary: **the 5-minute loop is the heartbeat; the tradeable events are rare.** A day
on which it evaluates 75 times and trades zero is the system working correctly.

---

# Q4 — Strategy space and the search over it

**`strategies/base.py`** — one protocol, adopting the gs-quant pattern:
```python
class Strategy(Protocol):
    def triggers(self, state: MarketState) -> list[Trigger]: ...
    def actions(self, trigger: Trigger, state: MarketState) -> list[Action]: ...
    params: dict          # every free parameter, declared — this is what the search sweeps
    hypothesis: str       # the economic story. Required. A strategy with no mechanism is rejected.
```

**Families to implement** (each a parameterised template, not a single strategy):

1. **Overnight–intraday reversal** — rank by `open/prev_close - 1`, long the overnight losers /
   short the winners at the open, exit at the close. Documented at ~5× conventional short-term
   reversal (Della Corte & Kosowski). **Pre-registered but never run** by the previous build, and
   computable from bhavcopy alone — no minute bars needed. India twist: post-2011 NSE shows
   positive overnight and negative intraday drift, which cuts for the short leg. **This is the
   single strongest untested lead the project owns and it runs first.**
2. **Opening-range breakout**, gated on OFI confirmation and a volume spike measured against a
   *same-time-of-day* baseline (a flat baseline fires at every open and close).
3. **Intraday mean reversion** on microstructure dislocation — price far from VWAP *with* order
   flow already reverting.
4. **Market intraday momentum** — first half-hour predicts last half-hour (Gao/Han/Li/Zhou).
5. **Event-conditioned gap** — trade only on a rare structural precondition (gap beyond a
   threshold + delivery-% regime shift + global overnight vector). Deliberately few trades/year.
   The prior failure mode was a *flat signal across a large universe*; this is its opposite.
6. **Pairs / cointegration** on NSE sector baskets.
7. **Regime-gated wrappers** — an HMM (`hmmlearn`, 2–4 states on returns + realised vol + India
   VIX) that stands the whole book down in hostile regimes.

### Family 1 specified exactly — the first one that runs, copy-paste level

The others follow this template. Every field below is swept, and the space is what "thousands of
strategies" concretely means. Exit parameters are **conditional** on `exit_rule`, so this is not a
naive full cross product:

```
base    = demean(2) x winsorise(3) x side(3) x n_positions(5) x cap_band(4) x min_move(3) = 1,080
exits   = session_close(1) + trail(3) + giveback(3) + hard_stop(3)                        =    10
FAMILY 1 TOTAL                                                                            = 10,800
```

**10,800 configurations from this one family**, before any other family or any ranker config.
Against the Q4.0 engine that is **6 min at 2 positions, 13 min at 5, 25 min at 10** — comfortably
inside a nightly window.

```python
# src/stocksense/strategies/overnight_reversal.py
@dataclass(frozen=True)
class OvernightReversalConfig:
    # --- signal -----------------------------------------------------------
    #   signal = open / prev_adj_close - 1, cross-sectionally demeaned per date.
    #   BOTH legs use ADJUSTED prices via data/adjust.with_prev_adjusted_close,
    #   and rows are restricted to prev_gap_sessions == 1. Using bhavcopy's raw
    #   prev_close here would read every split ex-date as a -50% overnight move.
    demean: bool = True                       # sweep: (True, False)
    winsorise_pct: float = 0.01               # sweep: (0.0, 0.01, 0.025)  clip tails
    # --- selection --------------------------------------------------------
    side: str = "long"                        # sweep: ("long", "short", "long_short")
    n_positions: int = 5                      # sweep: (2, 3, 5, 8, 10)  -- see Q5 ruin table
    cap_band: str = "mid"                     # sweep: ("small", "mid", "large", "full_pit")
    min_overnight_move: float = 0.01          # sweep: (0.005, 0.01, 0.02)  ignore noise
    # --- exits (Q8's searched exit families) -------------------------------
    exit_rule: str = "session_close"          # sweep: ("session_close","trail","giveback","hard_stop")
    trail_pct: float = 0.015                  # sweep: (0.01, 0.015, 0.025)
    giveback_pct: float = 0.30                # sweep: (0.2, 0.3, 0.5) of peak unrealised
    hard_stop_pct: float = 0.015              # sweep: (0.01, 0.015, 0.02)
    # --- costs (NOT swept -- fixed at the verified figure) -----------------
    charges_bps: float = 8.3                  # compute_charges, equity_intraday
```

**Entry** at the open (filled via `fill_model`, never at the signal bar's own close).
**Exit** by `exit_rule`, and unconditionally at 15:10.
**Economic story, required by `Strategy.hypothesis`:** overnight moves are driven by order
imbalance accumulated while the market is shut, which is absorbed and partly reversed once
continuous trading resumes. Documented at ~5× conventional short-term reversal (Della Corte &
Kosowski). India twist: post-2011 NSE shows persistently positive overnight and negative intraday
drift, which cuts **for** the short leg and **against** the long leg — hence all three `side`
variants are measured and reported, whatever the outcome.

**Why this one is first:** it needs no minute bars, no Upstox token and no static IP —
`open` and `prev_close` are already in the ingested bhavcopy (verified null-free and positive
across 1,290,511 rows). It is the shortest path from "no brain" to a measured PASS/FAIL.

### The remaining families, same shape

| Family | Signal | Key swept parameters | Data needed |
|---|---|---|---|
| 2 · Opening-range breakout | break of first-`k`-bar high/low | `k ∈ (3,6,12)` bars, `vol_mult ∈ (1.2,1.5,2.0)` vs same-time-of-day baseline, index-filter on/off | 1-min spine |
| 3 · Intraday mean reversion | distance from VWAP in σ | `z_entry ∈ (1.5,2,2.5)`, `hold_bars ∈ (3,6,12)` | 1-min spine |
| 4 · Intraday momentum | first-half-hour return predicts last-half-hour | entry time, `min_move` | 1-min spine |
| 5 · Event-conditioned gap | gap × delivery-% shift × overnight global vector | `gap_min ∈ (2%,3%,5%)`, `deliv_z ∈ (1,2)` | daily + delivery |
| 6 · Pairs / cointegration | OU spread on sector baskets | lookback, `z_entry`, `z_exit`, half-life filter | daily |
| 7 · Regime gate (wrapper) | HMM state on returns + realised vol + India VIX | `n_states ∈ (2,3,4)`, which states permit trading | daily + VIX |

Family 7 **wraps** any of 1–6 rather than standing alone, which multiplies the space again — and
is exactly why every evaluation must register in `attempts.py`.

**`search/space.py`** — the parameter grid, plus a deterministic seeded expression generator over
`rank, zscore, delta(n), ts_mean(n), ts_std(n), ratio, neg`, max depth 3. Reproducible, no LLM.

## The evidence base — what the literature actually supports

This directly answers the "generator is the weak half" critique. The search space must contain
effects with **published magnitudes and a stated mechanism**, not indicator permutations. Below is
what survived a deep read, ranked by strength of evidence. Every number is from the paper.

### A. Overnight vs intraday "tug of war" — STRONGEST, and tradeable from data we already have

Lou, Polk & Skouras, *JFE* 134 (2019) 192–213.

Sort stocks into value-weighted deciles on their **past one-month overnight return**
(`open ÷ prev close − 1`), rebalanced monthly. The long-winner/short-loser portfolio earns:

| Component | 3-factor alpha | t-stat |
|---|---|---|
| **Overnight** | **+3.47%/month** | **16.83** |
| **Intraday** | **−3.02%/month** | **−9.74** |

Across 14 strategies they find profits are earned **entirely overnight or entirely intraday,
typically with opposite signs**. Sorting on past one-month *intraday* return instead gives a
+2.19%/month intraday excess return (t = 6.72).

**The intraday-tradeable form for us:** long the overnight-*loser* decile, short the overnight-
*winner* decile, **hold open→close only**. That harvests the −3.02% leg as a positive.

**Mechanism (required by `Strategy.hypothesis`):** investor clienteles. Institutional ownership
rises with *intraday* returns — institutions trade during the session, individuals around the
open. Two clienteles push in opposite directions at different times of day.

**Why India should amplify it:** NSE has far higher retail participation than the US, and the
mechanism is *precisely* clientele separation. More retail → sharper separation.

**Needs only `open`, `close`, `prev_close` — all in the bhavcopy already on disk.** No minute
bars, no Upstox token, no static IP. This becomes **strategy family 1**, replacing the vaguer
"overnight–intraday reversal" entry.

### B. Gamma hedging drives intraday momentum — STRONG, with a hard conditioning variable

Baltussen, Da, Lammers & Van Iwaarden, *JFE* 142 (2021) 377–403. 60+ futures, 1974–2020.

The **last 30 minutes** before close is predicted by the return over the **rest of the day**
(previous close → last 30 min). Asset-class Sharpe **0.87–1.73**. Note `r_ROD` beats the
first-half-hour predictor `r_ONFH` of Gao et al. — better out-of-sample R².

The finding that matters most is conditional (their Table 7, S&P 500):

| Dealer net gamma exposure | β_ROD | t-stat | R² |
|---|---|---|---|
| **NGE ≥ 0** (long gamma) | 0.82 | 1.03 | **0.05%** |
| **NGE < 0** (short gamma) | **6.63** | **4.78\*\*\*** | **3.58%** |

**Same strategy, 8× the coefficient and 70× the R², switched by one observable.** Mechanism:
option market makers are typically net short gamma (retail buys puts/calls, dealers write them),
so hedging forces them to **buy as prices rise and sell as they fall** — mechanically manufacturing
momentum. When they are net *long* gamma they trade against the move and momentum vanishes or
inverts. Hedging concentrates near the close for liquidity reasons (the U-shaped volume curve).

**Why this is a genuinely India-shaped opportunity:** NSE is the world's largest index-options
market by contracts. If gamma-hedging flow drives intraday momentum anywhere, it should be
strongest here. NGE is computable from the NSE options chain (open interest × gamma per strike,
signed by assumed dealer positioning).

**This is NOT the F&O attempt that failed.** That one fed open interest as features into a
cross-sectional ranker. This uses gamma positioning as a **regime switch on a time-series
momentum trade** — different signal, different mechanism, different trade.

### C. Intraday seasonality — half-hour-of-day continuation

Heston, Korajczyk & Sadka, *Journal of Finance* 65 (2010).

Returns continue at **half-hour intervals that are exact multiples of a trading day**, persisting
**40 trading days**. A stock strong in the 10:00–10:30 bucket tends to be strong in that same
bucket on later days. Not explained by volume, order imbalance, volatility or spreads; not a
day-of-week effect; not driven by size or index membership. Needs the 1-minute spine.

### D. Order-flow imbalance — strongest short-horizon signal, but not yet backtestable

Cont, Kukanov & Stoikov: single-level OFI explains **65–87%** of short-horizon mid-price variance,
linearly, with slope inversely proportional to depth. NSE-specific evidence: predictability is
**strong for the first five minutes, gone within thirty**, the 1-minute coefficient is **positive
while longer horizons turn negative**, and information appears in **deeper book levels**.
Recorded live from day one; **gated until it has history** (Q3, Track B).

### E. Calibration: expect published effects to shrink — by how much, precisely

McLean & Pontiff, *Journal of Finance* 71 (2016). Across 97 published predictors:

- **−26%** out-of-sample (the data-mining component)
- **−58%** post-publication (a further −32% from publication-informed trading)
- Decay is **larger for predictors with the highest in-sample returns**
- Surviving returns **concentrate in low-liquidity, high-idiosyncratic-risk stocks**

Two consequences, both pre-registered before any result:

1. The tug-of-war's −3.02%/month should be **haircut to roughly −1.3%/month** as the honest prior.
   If our backtest reproduces 3%, that is a red flag for a bug, not a triumph.
2. That last bullet is **independent evidence for the small/mid-cap focus** — the same place the
   previous build's gate found edge (mid +0.560%, small +0.694%, **large FAILED at p = 0.655**).

### F. Counter-evidence, recorded so it is not quietly ignored

- **"Emerging markets fuel trend-following opportunities, whereas developed markets favour
  mean-reversion."** The tug-of-war trade is mean-reversion-flavoured, so India may favour the
  momentum families (B, C) over it. This tempers the prior on A and is a reason to run both.
- **"Return anomalies may not persist in emerging or thinly traded markets."** Directly relevant.
- Overnight returns exceed intraday returns in **23 of 29 countries**, so the *decomposition* is
  global even where the tradeable spread is not.

### G. Demoted: delivery-% and FII/DII flows

Both were treated as promising Indian signals in earlier drafts. A deliberate search for
peer-reviewed evidence found **essentially none** — delivery-% is a practitioner folk indicator
("high delivery = accumulation") with no published return-predictability study behind it, and on
FII/DII the standing view is that *"daily flow data tells you what happened, not what will happen
tomorrow."*

They stay in the feature registry as cheap candidates. They lose their privileged status, and the
plan no longer calls delivery-% "the most plausible genuinely Indian intraday signal" — that claim
was unsupported.

---

## An honest critique of this plan, before the spec

Written because the failure mode here is self-congratulation, and because the executing model
should know where the risk actually is.

**The referee is the strong half and it is already built.** DSR, PBO, the sealed vault, purged
walk-forward, the attempt registry — that is genuinely institutional machinery, and most retail
systems have none of it.

**The generator is the weak half, and it is where this plan can still fail.** A search finds only
what its space contains. If the space is "RSI thresholds × MACD periods × stop distances", then
5,000 configurations of nothing is still nothing, and the referee will simply certify that
rigorously. **Running 5,000 backtests instead of 40 does not create edge** — it finds an edge
faster *if one exists in the space*, and finds noise faster if it does not.

The previous build did not fail because its referee was weak. It failed because its hypothesis
space was exhausted: cross-sectional ranking on price-derived technicals, the most crowded idea in
retail quant. **Repeating that with a bigger sweep would fail again, more expensively.**

Three specific weaknesses to hold in view:

1. **The strongest documented signal is currently un-backtestable.** Order-flow imbalance explains
   65–87% of short-horizon mid-price variance and, on NSE specifically, is strongly predictive for
   the first five minutes. We have **no historical depth data**, so it can only be recorded going
   forward. That is a real hole, not a detail.
2. **Whether a retail intraday edge exists at all** at retail order sizes with a 10.62 bps round trip is
   genuinely unresolved. The evidence so far leans negative.
3. **The most likely single outcome remains "nothing passes."** The guards exist so that answer can
   be believed — but it is still the modal outcome, and no amount of compute changes that.

What actually improves the odds is the *content* of the space: structural and economic hypotheses
(overnight reversal, event-conditioned rare setups, meta-labelling on trade selection) rather than
parameter sweeps over indicators everyone already mines.

---

## The search engine is the product — this is the core of the rebuild

**The problem with the previous two builds was throughput, and it is worth naming exactly.**
They tested **one hypothesis at a time**: form an idea, build features for it, run a 15-fold
purged walk-forward costing hours, get a FAIL, write it up, start again. Across the *entire
project history* that produced roughly **40 configurations** and six null results over months.
At that rate you cannot search a strategy space — you can only sample it, and a space this size
sampled 40 times will almost always return nothing.

**This build inverts that.** The unit of work is not "a strategy" but "a nightly sweep of
thousands of strategies", and the expensive validation is spent only on the handful that survive
cheap screening.

### Why you cannot simply run 5,000 full walk-forwards

A full purged/embargoed walk-forward with costed portfolio simulation is minutes per config.
5,000 × 2 min ≈ **7 days of compute** for one night's sweep. So the engine is a **cascade**: each
stage is orders of magnitude cheaper than the next and kills most of what enters it.

| Stage | What it does | Cost per config | Survivors |
|---|---|---|---|
| **0 · Batch signal eval** | All configs' signals computed as batched matrix ops over one feature panel, on GPU | ~ms | all |
| **1 · Rank-IC / ICIR screen** | Per-date cross-sectional rank IC on in-sample blocks only | ~ms | ~5% |
| **2 · Decay screen** | Reject `half_life < 3 bars` — the video's "decays in two days" filter, made numeric | ~ms | ~2% |
| **3 · Cost hurdle** | Net expectancy vs the **10.62 bps** real round trip (see THE COST WALL) at the config's own turnover. **This is the PRIMARY filter, not a formality** — a daily-turnover candidate needs ~14.2 bps gross per trade just to break even | ~ms | ~0.5% |
| **4 · Purged walk-forward + `gate.py`** | The expensive, honest test | minutes | ~10–30 configs |
| **5 · Sealed vault, one unseal** | DSR ≥ 0.95 **and** PBO ≤ 0.5 | minutes | **0 or 1 champion** |

**Stage 0 is where the machine earns its keep.** The 5-min research panel is ~250 symbols ×
~1,000 sessions × 75 bars ≈ **18.75 M rows**, which is ~75 MB per float32 feature column — it fits
GPU memory whole. Signals for a *batch* of configs are then one matmul against that resident
panel, not 5,000 separate passes over pandas. Batch size is set from Q0.4's measured ceiling
(~30 configs per batch keeps peak VRAM inside the 2.5 GB budget). Stages 1–3 are reductions over
the same resident arrays.

Stage 4 is CPU: `multiprocessing` over **10 of 12 threads** (2 reserved for the UI and OS), each
worker with its own lock-free `Reader`, results streamed to Parquet.

**Budget: 2,000–5,000 configurations per nightly run**, versus ~40 across the whole history of the
previous builds.

### The other half — and it is not optional

Searching 5,000 configs instead of 40 fixes throughput and *creates* a new problem:

> After 1,000 independent backtests the expected best Sharpe is **3.26** with **zero** true edge.
> At 5,000 it is higher still.

So a mass search **will** hand you a beautiful equity curve on the first night. Without Q2's
guards that result is indistinguishable from a real edge, and acting on it costs real money. The
throughput fix and the false-positive fix are different fixes and this build needs both:

- **Every config registers in `attempts.py`** — all 5,000, not the survivors. That count *is* the
  `n_trials` in the Deflated Sharpe, so a bigger sweep automatically raises its own bar.
- **PBO via CSCV** asks the direct question: does the config that wins in-sample rank above median
  out-of-sample? For a noise-driven search it does not, and PBO says so.
- **The sealed vault** (2025-07-01 →) is never touched by stages 0–4. One unseal per hypothesis, ever.

**Stopping rule, pre-registered before the first run:** fixed iteration budget, fixed survivor
count, written down in advance. Never "stop when something looks good" — that is p-hacking
wearing a lab coat, and at 5,000 configs a night it is p-hacking at industrial scale.

### Why this is worth doing even though most nights return nothing

Because a night that ends with **"5,000 configs, 0 champions"** is now a *cheap, informative*
result that took one unattended night, instead of an expensive month. And the recorded failure
reasons — `low_ic | unstable_ic | fast_decay | cost_drag | capacity` — are what the next sweep
refines against. That feedback loop is the thing the previous builds never had.

**`search/selection.py`** — the funnel above, in that exact order, with the failure reason
recorded for every rejected config.

---

# Q4.0 — Performance engineering: 5,000 backtests in under 3 minutes

**All numbers below are MEASURED on this laptop during planning, not estimated.** This section
exists because the first naive design would have taken **25 hours per nightly sweep**, and that
would have been discovered at 3 a.m. instead of now.

### The measurement that changed the architecture

A path-dependent exit (trailing stop) simulated over every symbol on every day:

```
50 configs x 2,000 symbols x 75 bars  = 7.5M steps in 0.221s  -> 34M steps/sec (1 thread)
=> 5,000 configs x 4,100 sessions      = 1,507 minutes  (25 HOURS)   <-- unusable
```

**The fix is architectural, not hardware.** You only need to simulate the *path* of positions the
strategy actually **takes** — at most 2 a day — not of every symbol it merely *considered*:

```
naive : 2,000 symbols x 4,100 days = 8.2M paths
real  :     2 taken   x 4,100 days = 8.2k paths      -> 1,000x less work
```

### The resulting two-stage engine, with measured timings

| Stage | Work | Implementation | Measured, 5,000 configs |
|---|---|---|---|
| **A · Select** | signal → cross-sectional rank → top-N per date | pure NumPy on a `(4100 x 2000)` float32 panel; **`np.argpartition`**, never a double `argsort` | **0.61 min** (10 threads) |
| **B · Simulate** | path-dependent exits on *taken* positions only | Numba `@njit(parallel=True)` | **2.2 min** (10 threads) |
| | | **TOTAL** | **≈ 2.8 min** |

Stage B scales linearly with the position count (Stage A does not — ranking cost is independent of
how many names you take). Measured extrapolation:

| positions/day | paths simulated | Stage B | **total** |
|---|---|---|---|
| 2 | 8,200 | 2.2 min | **2.8 min** |
| 5 | 20,500 | 5.5 min | **6.1 min** |
| 10 | 41,000 | 11.0 min | **11.6 min** |

Even at 10 positions a 5,000-config sweep finishes in under 12 minutes.

**Two specific optimisations, each measured:**

1. **`argpartition` beats double `argsort` by 6.7×** — 493 ms → 73 ms per config, *identical
   results*, because we need the top 2, not a fully ordered 2,000.
2. **Stage A belongs on the CPU, not the GPU.** Measured: CuPy 17.7 ms/config single-stream =
   1.47 min for 5,000, versus **0.61 min on 10 CPU threads**. Ranking is *memory-bound*; the PCIe
   round trip costs more than the kernel saves. **The GPU's job is Monte Carlo (compute-bound),
   not selection.** This is the opposite of the intuitive allocation, which is why it was measured.

**Memory:** the panel is `4,100 x 2,000 = 8.2M` cells, **33 MB per float32 feature** — the whole
thing lives resident in RAM. Do not materialise `(n_configs x n_dates x n_symbols)`; at 5,000
configs that is 40 GB. Configs are streamed in batches of 50.

### On `vectorbt`

Its core idea is right and worth stating: *"one backtest and ten thousand backtests are, to the
machine, almost the same operation"* — parameters become an extra array dimension, hot loops are
Numba-compiled, and it runs ~1000× faster than event-driven `backtrader`. It resolves cleanly on
Python 3.14 (v1.1.0).

**We adopt the pattern, not the dependency.** It pulls matplotlib, plotly, ipython and the whole
Jupyter widget stack, and its full-materialisation model is exactly the 40 GB shape avoided above.
Our two-stage split is both leaner and faster for this specific problem.

---

# Q4.05 — The infrastructure blueprint: what to take, and four bugs not to copy

From `trading_infrastructure_blueprint.md`. Several instincts in it are correct and are adopted;
the code has real defects that must not be reproduced.

### Adopted

1. **Zero-copy DuckDB → Arrow → Polars.** `conn.execute(...).arrow()` then `pl.from_arrow(...)`
   moves data with no serialisation. Genuinely faster than `.fetchdf()` and now the default read
   path for the live loop.
2. **Heterogeneous delegation** — CPU for screening/filtering/state estimation, GPU for Monte
   Carlo. This **independently agrees with the measurement above**, arrived at by different
   reasoning, which is a good sign.
3. **Monte Carlo only on the ~20 pre-filtered candidates**, never all 2,300 — the same insight as
   the Stage A/B split.
4. **ACO / genetic algorithms are OFFLINE ONLY.** A correction this plan needed: Q4.5's
   metaheuristics belong in the *nightly* search and must never run inside the 5-minute loop,
   where their sequential branching would thrash cores and blow the window.
5. **No RL agents** — correct on 4 GB VRAM.
6. **Kalman filter** for denoising live 5-minute bars — genuinely useful, **new to this plan**, and
   added to the feature layer as an adaptive alternative to moving averages.
7. **Hierarchical Risk Parity** — adopted, but **deferred, and here is the honest reason**: HRP
   allocates across *many* correlated assets by building a covariance hierarchy. At **1–2
   positions there is no covariance structure to exploit**, so it would be decoration today. It
   becomes genuinely useful once the account grows enough to hold 5+ names, and it is specified
   now so that transition needs no redesign. Until then, Kelly plus the equity-at-risk cap does
   the whole job.

**Kalman filter, specified** — `features/kalman.py`, local-level (optionally local-trend) model:

```
state:        x_t = x_{t-1} + w_t ,  w_t ~ N(0, Q)     # latent "true" price
observation:  y_t = x_t       + v_t ,  v_t ~ N(0, R)     # the printed 5-min close
```
The Kalman gain `K_t = P_t / (P_t + R)` blends prediction and observation optimally, so it adapts
faster than any fixed-window moving average and has **no lag parameter to overfit**. `Q` and `R`
are estimated online from rolling windows of returns and price deviations, and the ratio `Q/R`
(signal-to-noise) is the one swept parameter. Emits `kalman_level` and `kalman_slope` as features
— never as a standalone trading signal.

### Four bugs to fix, not copy

1. **BLAS thread env vars are set *after* `import numpy`.** They only take effect if set *before*
   the first NumPy/Polars import, so as written they do nothing. Also `OMP_NUM_THREADS="auto"` is
   invalid — it takes an integer. Correct form, at the very top of the entry point:
   ```python
   import os
   os.environ["OMP_NUM_THREADS"] = "10"     # 10 of 12 threads; 2 for UI + OS
   os.environ["MKL_NUM_THREADS"] = "10"
   os.environ["NUMEXPR_NUM_THREADS"] = "10"
   import numpy as np                        # AFTER, never before
   ```
2. **`shift(1)` and `rolling_std` without `.over("ticker")`.** On a long `(ticker, timestamp)`
   frame these run across ticker boundaries, so the first row of each symbol silently inherits the
   previous symbol's close. Every window op must be `.over("ticker")`.
3. **`.sort("returns").limit(20)` takes 20 ROWS, not 20 tickers** — on a long frame those rows can
   all be the same symbol. Aggregate to one row per ticker *before* selecting.
4. **`duckdb.connect(path, read_only=True)` while another process writes fails.** Already proven
   on this machine — cross-process it raises `IOException`, which is exactly the bug that killed
   the previous build's nightly jobs. Our Parquet read surface exists precisely for this.

Also: the "never write a `for` loop" directive is wrong for this codebase. A **Numba** loop is the
*fastest correct* tool for path-dependent exits — measured at 34M steps/sec — and a blanket ban
would forbid the best solution. The real rule is narrower: **no interpreted Python loops over
market data**; `@njit` loops are encouraged.

Finally, the hardware profile says i7; it is an **i5-13420H (8 cores / 12 threads)**.

---

# Q4.6 — `models/ranker.py`: the LightGBM ranker, specified exactly

Written at copy-paste level so nothing has to be inferred. **This is one candidate brain among
several, not "the" brain** — it competes with the rule families on the identical gate.

### What it does, mechanically

One row per `(symbol, date)`. Features `X` from the registry. Label `y` = forward return over the
horizon, **cross-sectionally demeaned within each date**. The demeaning is load-bearing: without
it the model learns "the market rose that day", not "this stock beat its peers", and every score
is then dominated by market direction we cannot trade.

Each tree greedily picks a feature and threshold that most reduce squared error
(`if vol_ratio > 1.8 → leaf +0.004`); boosting fits tree *k* to the residual of trees 1..*k*−1.
Prediction sums every tree's leaf value into one score; stocks are ranked by score **within each
date** and the top *N* are taken.

Trees rather than a neural net because the data is tabular, the sample is modest, and — decisive
here — **feature importance is readable**, which is how a data bug masquerading as alpha gets
caught. That happened in the previous build.

### Exact interface

```python
# src/stocksense/models/ranker.py
from dataclasses import dataclass, field

@dataclass(frozen=True)
class RankerConfig:
    objective: str = "regression"     # "regression" | "lambdarank"
    horizon_bars: int = 6             # from Q3.5's HORIZON_BARS
    n_estimators: int = 400
    learning_rate: float = 0.03
    num_leaves: int = 31              # keep < 2**max_depth
    max_depth: int = 6                # SHALLOW on purpose: financial data is
                                      # low signal-to-noise; deep trees memorise
    min_data_in_leaf: int = 200       # large: a leaf built on 5 rows is noise
    feature_fraction: float = 0.7
    bagging_fraction: float = 0.7
    bagging_freq: int = 1
    lambda_l1: float = 0.0
    lambda_l2: float = 1.0
    min_gain_to_split: float = 0.0
    seed: int = 7                     # determinism is asserted by a test
    top_n: int = 5                    # positions taken; swept (2,3,5,8,10) -- Q5

def demean_by_date(labeled: pd.DataFrame, label_col: str) -> pd.Series:
    """y_rel = y - mean(y within that date). THE step that makes the model
    learn relative performance instead of market direction."""

def fit_ranker(feats, labeled, feature_cols, cfg) -> lgb.Booster:
    """objective="regression" -> LGBMRegressor on the demeaned label.
       objective="lambdarank" -> LGBMRanker with group = rows per date, and
       the label discretised into 5 quantile buckets per date (lambdarank needs
       ordinal relevance grades, not continuous returns). Optimises NDCG@top_n
       directly, which matches "we only buy the top 1-2" better than squared
       error over the whole cross-section. Both variants are swept."""

def score(model, feats, feature_cols) -> pd.Series
def rank_within_date(scores: pd.Series, dates: pd.Series) -> pd.Series
def feature_importance(model, feature_cols) -> pd.DataFrame   # gain + split count
```

### Non-negotiables, each with a test

- **Determinism.** Same seed and same data must give a bit-identical model. Set
  `deterministic=True`, `force_row_wise=True`, `num_threads` fixed.
- **No leakage.** The label window must lie strictly after the feature window; the leakage suite
  asserts a shuffled label destroys IC.
- **Fit inside the fold only.** Every scaler/encoder statistic comes from training rows alone.
- **Importance drift is monitored.** A feature the model ignored for months suddenly dominating is
  a data bug far more often than an insight.
- **Calibration.** Real equity factors give IC 0.02–0.05. If a fitted model reports IC > 0.15 on
  its own training period, the correct reaction is to hunt for leakage, not to celebrate.

### How it enters the sweep

`RankerConfig` fields are swept exactly like a rule strategy's parameters, so a learned ranker and
a hand-written rule are directly comparable on one gate. Both register in `attempts.py`, both face
the same DSR with the same `n_trials`.

---

# Q4.5 — Metaheuristic search: ant colony, particle swarm, genetic

`search/metaheuristics.py`. This was missing from the first draft of the plan and is added
because it is a named requirement, not as decoration.

**What they are for.** Q4.3's deterministic enumeration covers a bounded expression space
exhaustively. It does not scale to the strategy space you actually want searched — thousands of
strategies across seven families, each with continuous parameters (thresholds, lookbacks, stop
and target distances, regime cut-offs). That is a high-dimensional, non-convex, noisy objective
with no gradient, which is precisely what population metaheuristics are for.

- **ACO** (Ant Colony Optimization) — pheromone-weighted construction over the *discrete*
  choices: which factor, which operator, which composition. Natural fit for building alpha
  expressions as paths through a graph, and the pheromone trail is a readable record of which
  building blocks keep appearing in survivors.
- **PSO** (Particle Swarm) — the *continuous* parameters. Each particle is one parameter vector;
  the swarm converges on regions of the space with high fitness.
- **GA** — crossover/mutation over whole strategy genomes when you want structural jumps rather
  than local refinement.

Run ACO for structure and PSO for parameters in a nested loop: ACO proposes an expression, PSO
tunes its constants, the fitness returned to ACO is the tuned score.

**Fitness = ICIR on research data only**, with the vault enforced automatically at the loader —
the same objective as Q4, so metaheuristics and enumeration are directly comparable.

**GPU acceleration.** Population methods are embarrassingly parallel and the fitness evaluation
is the bottleneck, not the swarm update. Evaluate the whole population's signals as one batched
CuPy matmul over the feature panel (population × features × dates), reusing the Q5 memory-budget
and streaming machinery. The swarm/pheromone update is tiny and stays on CPU.

### The caveat, and it is the important part

**A better optimiser overfits harder.** That is not a reason to skip it, but it is the reason
the guards come first. Brute enumeration of 5,000 configs already implies an expected best
Sharpe of ~3.9 from pure noise; ACO/PSO will find higher in-sample peaks *faster*, because
finding peaks is exactly what they are good at, and a noise peak is still a peak.

So the discipline is non-negotiable and slightly counter-intuitive:

- **Every fitness evaluation registers in `attempts.py`** — not every *survivor*, every
  *evaluation*. A 200-particle swarm over 50 generations is **10,000 trials**, and that is the
  `n_trials` that goes into the Deflated Sharpe. Metaheuristics therefore make the DSR bar
  *harder* to clear, which is correct and is the honest way to use them.
- The iteration budget, population size and generation count are **pre-registered before the
  run**, so the trial count cannot be quietly grown until something passes.
- Survivors still face the identical funnel: purged walk-forward → `gate.py` → one vault unseal
  → DSR ≥ 0.95 and PBO ≤ 0.5. A metaheuristic gets no shortcut.

**Where these methods are unambiguously good, with no overfitting risk at all:** the *execution*
problems in Q3/Q5, where the objective is a real engineering cost rather than an estimated edge —
Almgren–Chriss schedule optimisation, participation-capped order slicing, and constrained
position sizing. Use them there without hesitation.

---

# Q5 — Monte Carlo, optimised for a 4 GB GPU and a 12-thread CPU

You asked specifically for this to be optimised to the maximum with least load. The largest win
is **mathematical, not hardware**: variance reduction cuts the paths needed for a target accuracy
by 10–100×, which beats any amount of brute-force throughput.

**`simulation/montecarlo.py`**

**Measured on your machine before any of this was designed (Q0.4):** 3,965 MB free; at the
2,500 MB budget, **2.62 M float32 paths × 250 steps**; **1 M paths × 250 steps in 0.183 s** on
GPU versus ~3.5 s on CPU (≈19×), after a **1.9 s one-time CUDA context warm-up** that the live
engine pays at startup and never on the decision path. The CPU fallback does 100 k paths in
0.35–0.52 s, so a busy GPU degrades the system rather than stopping it.

- **Sobol quasi-random sequences** (`scipy.stats.qmc.Sobol`, scrambled) instead of pseudo-random —
  near O(1/N) convergence versus O(1/√N) for smooth low-dimensional integrands.
- **Brownian-bridge path construction — the non-obvious one, and the one that makes QMC actually
  work here.** A 250-step path is a 250-dimensional integral, and plain Sobol degrades badly in
  high dimension. Building the path by bridge (fill t=T first, then T/2, then quarters…)
  concentrates almost all the variance into the first few Sobol dimensions, where the low-
  discrepancy property is strongest, collapsing the *effective* dimension to single digits. This
  matters specifically for us because our labels are **path-dependent** (first-touch stop/target),
  which is exactly the case plain QMC handles worst.
- **Multilevel Monte Carlo** for the barrier/first-touch estimators — coarse paths for the bulk of
  the variance, few fine paths for the correction; ~20× reported in comparable settings.
- **Antithetic variates** — pair each draw `u` with `1-u`. Cheap, and effective when the payoff is
  near-linear in the driver.
- **Control variates** — use a quantity with a known expectation (e.g. the index return over the
  same window) to subtract variance from the estimate.
- **Stationary block bootstrap** on *real* returns as the default generator, not GBM. Markets are
  not lognormal; the fat tails are the entire point of running the simulation.
- **float32 on GPU** (2× throughput, half the memory), with a periodic float64 CPU run to bound
  the discretisation error and prove the two agree.
- **Chunked streaming**: batch size derived from Q0.4's measured ceiling, hard-capped so peak VRAM
  stays **≤ 2.5 GB** (leaving room for display + Ollama). Pinned host memory, 2 CUDA streams to
  overlap host-to-device copy with compute.
- **CPU fallback** via `numba.prange` whenever the GPU lease is held by Ollama — the pipeline must
  never stall on GPU contention.

**Block bootstrap, specified.** The default generator is the **stationary bootstrap** (Politis &
Romano 1994) over *real* returns, not GBM — markets are not lognormal and the fat left tail is the
entire reason for simulating. Block length comes from **Politis & White (2004)** automatic
selection with the **Patton, Politis & White (2009)** correction; the stationary variant is chosen
specifically because it is *less sensitive to block-length misspecification* than a fixed-block
bootstrap. Implement as `simulation/bootstrap.py::optimal_block_length(returns) -> float` and
`stationary_bootstrap(returns, n_paths, horizon, block_len, seed)`.

**`simulation/risk.py`** — VaR / CVaR at 95 and 99, the full **drawdown distribution** (not just
the historical max), and **probability of ruin** over a 250-day horizon.

### How many positions? 1–2 was WRONG. Measured during planning.

The brief said "find 1–2 stocks, invest all the capital", and earlier drafts encoded that as a
requirement. **It does not survive its own arithmetic.** Holding the same edge and changing only
how it is spread, at a realistic pairwise correlation of ρ = 0.3 and 5× leverage. **These are
ratios and hold at ANY account size** — only the upper price bound moves with the balance:

| Positions | Portfolio vol | **P(lose 50% of capital in 1 year)** | Grinold IR gain | Capital efficiency @₹900 |
|---|---|---|---|---|
| 1 | 2.50% | **86.7%** | 1.00× | 99.8% |
| 2 | 2.02% | **61.5%** | 1.41× | 98.7% |
| 3 | 1.83% | 54.0% | 1.73× | 98.7% |
| 5 | 1.66% | 51.5% | 2.24× | 97.7% |
| **8** | **1.56%** | **45.8%** | **2.83×** | **98.7%** |
| 10 | 1.52% | 46.4% | 3.16× | 92.6% |
| 15 | 1.47% | 37.3% | 3.87× | 92.6% |

Two independent forces both point the same way:

1. **Survival.** At 5× on a single name there is an **86.7% chance of losing half the account
   within a year even with a positive edge**. Two names only reaches 61.5%. That is a survival
   problem, not a strategy problem.
2. **Grinold's fundamental law, IR ≈ IC × √breadth.** Going 2 → 8 names multiplies risk-adjusted
   return by **2.0×** *for identical signal quality*. Breadth is free performance.

**Divisibility does not bind where it was assumed to.** At ₹900/share, 8 positions is 12 shares
each at **98.7%** capital efficiency; it only degrades past ~15 names.

**Consequences for the plan:**
- `max_open_positions` moves from a fixed **2** to a **swept parameter over (2, 3, 5, 8, 10)**,
  decided by measured risk-adjusted return, not by preference.
- `probability_of_ruin` becomes a **hard constraint, not a diagnostic**: any configuration whose
  1-year P(−50%) exceeds a pre-registered ceiling is rejected regardless of its backtest return.
- The **correlation must be measured, not assumed**. ρ = 0.3 is illustrative; NSE small/mid caps
  co-move far more than that on bad days — exactly when it matters. `simulation/risk.py` estimates
  the realised correlation of the actual candidate set each day and re-runs the ruin calculation
  against it.
### Intelligent sizing — an objective, not a threshold

An earlier draft rejected configurations against a hardcoded `MAX_RUIN_1Y = 0.10`. **That number
was arbitrary and it is removed.** A constant cannot adapt to regime, to correlation, or to how
much the edge is actually believed. The replacement is a proper objective.

**Maximise expected LOG growth.** Not arithmetic return — log. This is the whole trick: compounding
already punishes drawdowns savagely and a wiped account is `log(0) = -inf`, so **ruin is priced
into the objective without any threshold being declared.**

Measured on this machine, edge = 8 bps/day, ρ = 0.3, no constraint applied:

| leverage | n=2 | n=5 | n=8 | n=10 |
|---|---|---|---|---|
| 1× | +0.154 | +0.161 | +0.170 | +0.167 |
| 2× | +0.195 | +0.260 | +0.280 | +0.293 |
| **3×** | +0.153 | +0.292 | +0.334 | **+0.344** |
| 5× | **−0.276** | +0.148 | +0.237 | +0.260 |

Log-utility picks **3× / 10 names on its own**, and correctly marks **5× on 2 names as
wealth-DESTROYING (−0.276) despite a positive edge** — conclusions reached by computation, not by
a constant someone typed.

**The uncertainty that matters is not an error bar — it is "does this edge exist at all?"** A
symmetric posterior on the mean barely moves the optimum (tested: it did not). What moves it is
the live possibility that the whole edge is an artifact of selection, which is exactly what
McLean–Pontiff's −58% decay describes. Modelled as a mixture, `P(spurious)`:

| P(edge is fake) | chosen leverage | positions | log growth |
|---|---|---|---|
| 0% | 3× | 8 | +0.334 |
| 30% | 3× | 10 | +0.173 |
| 50% | **2×** | 10 | +0.079 |
| 70% | **1×** | 8 | +0.031 |
| 90% | 1× | 8 | **−0.010** → do not trade |

**Sizing walks itself down as belief in the edge weakens, and turns itself off entirely when the
edge is probably fake.** No schedule, no thresholds, no hand-tuning.

**`P(spurious)` is not a guess — it is already computed.** `PBO` is literally the probability the
in-sample winner is a below-median performer out of sample, and `1 − DSR` is the probability the
Sharpe is a multiple-testing artifact. Both feed straight in:

```python
# simulation/sizing.py
def p_spurious(pbo: float, dsr: float, n_graded: int) -> float:
    """Probability this edge is not real. Blends the two guards we already
    compute with how much FORWARD evidence exists -- a backtest-only edge is
    never fully believed, however good its numbers."""

def choose_sizing(edge_dist, vol, corr, p_spurious, leverage_grid, n_grid) -> Sizing:
    """Argmax of expected log growth over the (leverage x positions) surface,
    integrating over the possibility that the edge is spurious. Recomputed
    DAILY from live volatility and the realised correlation of the actual
    candidate set -- never from stored constants."""
```

Three things are therefore adaptive rather than fixed: **regime** (σ and ρ measured daily from
live data), **belief** (PBO/DSR/forward record), and **the resulting size**. The only fixed input
left is the 5× MIS ceiling, which is the broker's limit, not our choice.

*Note this cuts against the "invest everything in one mover" instinct. The single-name path has
the highest expected return and the highest chance of not being there to collect it.*

### Kelly, and why this project must use a *small* fraction — with the evidence

`fractional_kelly` already defaults to 0.25. The literature says that is right, and says why:

- **Estimation error is the binding practical limitation**, not the formula. Kelly assumes the edge
  is *known*; ours is *estimated* from a backtest.
- Measured in the literature: a **10% overestimate of edge costs ~19% of long-run compound growth**
  under full Kelly.
- **Thorp — who took Kelly from theory to practice — recommends half-Kelly or less.** Full Kelly
  routinely draws down more than half of capital.
- Half-Kelly approximates the true optimum *once parameter uncertainty is priced in*.

**Our edge estimate is far worse than 10% overestimated**, for three compounding reasons:
McLean–Pontiff decay (−26% out-of-sample, −58% post-publication), selection bias from a
5,000-config search, and thin diversification at a small position count. So:

```python
# simulation/sizing.py -- the rule, not a suggestion
KELLY_FRACTION = 0.25            # quarter-Kelly. Never raise without a written argument.
EDGE_HAIRCUT   = 0.42            # keep 42% of backtested edge (McLean-Pontiff -58%)

def sized_fraction(win_prob, win_loss_ratio, *, haircut=EDGE_HAIRCUT):
    """Kelly is computed on the HAIRCUT edge, never the raw backtest edge.
    Applying Kelly to an un-discounted backtest number is how a 'mathematically
    optimal' sizing rule bankrupts an account."""
```

This is also why `probability_of_ruin` exists and why the plan expects it to argue for **less than
5× leverage** on 1–2 concentrated names. Kelly maximises *growth*; it says nothing about surviving
the path, and concentration plus leverage is where the path kills you.

**`simulation/sizing.py`** — this is where MC actually earns its place. Given the measured edge
distribution, 5× MIS leverage, and **1–2 concentrated names**, compute fractional-Kelly and
volatility-targeted sizes and the resulting P(ruin). Concentration plus leverage is precisely the
configuration where the arithmetic of ruin bites hardest. **Expect the honest answer to be "use
less than 5×", and let it say so.**

---

# Q6 — LLM layer, tightly bounded, plus Obsidian

**The load-bearing rule, carried over unchanged: no language model ever produces a number that
decides a trade.** The model decides *what*; Python decides size, stop, target and cost; the LLMs
explain, diagnose and propose — never author a figure that reaches an order.

## Q6.1 — The news layer: a four-stage pipeline where the LLM is the smallest part

`data/news.py` + `llm/sentiment.py`. Researched rather than assumed, because the obvious design
("feed headlines to an LLM, get a sentiment score") is the weak one.

**The two findings that shape it:**

1. **Freshness beats sentiment.** A *fresh* story moves price ~**39 bps** over a day against
   ~**23 bps** for a *stale* one, and it takes ~4 days for fresh news to be fully incorporated.
   Daily news predicts returns for only 1–2 days. RSS is dominated by the same story
   republished across outlets, so **deduplication is the highest-value stage**, not classification.
2. **We are structurally too late for intraday news.** At RSS latency the move has happened. But
   we place trades at **09:30**, so **overnight news processed before the open** is where our
   latency is genuinely fine. The news layer is therefore weighted to the pre-open window, and it
   feeds the same gap/regime model as the finviz global vector and the overnight-reversal
   hypothesis.

**The stages, in order of value:**

| Stage | Job | Tool | Why not an LLM |
|---|---|---|---|
| 1. Entity resolution | headline → NSE symbol(s) | dictionary + alias/fuzzy match | It is a lookup, not a judgement. Deterministic and auditable |
| 2. **Novelty / dedup** | is this story new, or the 6th rehash? | `nomic-embed-text` embeddings + cosine clustering per symbol per day | The highest-value stage, per finding #1 |
| 3. Sentiment score | signed intensity | **FinBERT / TinyFinBERT** | Purpose-built on financial text (~85–97% acc), ~20 docs/s on one CPU thread, deterministic, no hallucination, and it does not contend for the 4 GB GPU |
| 4. Event type | earnings / order-win / regulatory / downgrade / M&A | **`llama3.2:3b`**, constrained to a fixed enum | The one job a small general LLM does well: coarse classification with a closed output set |

### Two tiers: small-and-local for the hot path, Ollama **Cloud** for synthesis

**Corrected after checking the hardware.** MiniMax M2 and Nemotron Ultra cannot run on this
laptop — not "slowly", *at all*. MiniMax M2 is 229B parameters and needs **83 GB even at the most
aggressive Q2_K quantisation** (130 GB at 4-bit); Nemotron Ultra 253B needs **126.5 GB at INT4**.
The RTX 3050 has **4 GB**. That is a 20–30× shortfall, not a tuning problem.

**But running them locally was never the ask — Ollama Cloud is.** Verified on this machine:
Ollama **0.33.2** is installed, the server is up, and `ollama signin` / `ollama signout` exist, so
cloud models are supported. Those run on Ollama's hardware and the 4 GB ceiling stops applying.
That is the right answer to the cost concern, and it is adopted.

| Tier | Where | Jobs | Why |
|---|---|---|---|
| **Local** `llama3.2:3b` (Q4_K_M, ~2–3 GB) | this GPU | stage-4 event classification; any per-item, high-volume call | Runs offline, costs nothing, no rate limit. Small models are documented as adequate for "tool routing, classification, structured extraction with a tight system prompt" — exactly and only what we ask |
| **Cloud** MiniMax M2 / Nemotron via `ollama` | Ollama's servers | nightly synthesis, the **"where is the mistake, why didn't it make profit"** diagnosis, proposing strategy code | Once a day, low volume, genuinely benefits from a frontier-class model. Free/subscription rather than per-token Claude billing |

**`llm/claude_cli.py` is dropped from the runtime.** You are right that it should not be burning
your Claude budget, and the system does not need it: Python computes every figure, and narration
is presentation. Claude stays a *development* tool in this repo, not a *runtime* dependency.

**One honesty note on cloud.** It is the only part of this design where data leaves the machine —
scorecard facts, strategy metrics, symbol names, your P&L. That is your own trading data going to
Ollama's servers. It does not compromise the trading path, because the load-bearing rule is
unchanged: **no language model, local or cloud, ever produces a number that decides a trade.** The
cloud tier is fed Python-computed facts and writes prose. If it is unreachable the pipeline runs
exactly as before — the LLM layer already fails soft.

**Action for you:** `ollama signin`, then tell me which cloud model you want wired
(`ollama list` after signing in shows what your plan exposes). Until then the local tier covers
everything and nothing is blocked.

Whichever tier is used, it does **not** produce the sentiment score — that stays on the CPU
encoder, which is deterministic, hallucination-free, and does not contend for the GPU.

**What reaches a trade:** stages 1–4 produce **feature columns** — `news_novelty`,
`news_sentiment`, `news_count_zscore`, `event_type` one-hots — and nothing more. They go through
the identical gate as every other feature and get **dropped if their IC does not earn the
column**. The honest prior is that headline sentiment carries little tradeable edge at retail
latency; novelty and abnormal news *volume* are the more plausible survivors.

**`llm/ollama.py`** — local runtime wrapper. **You must run `ollama pull nomic-embed-text`
(274 MB)** — only `qwen2.5` is present and it has no embedding capability. Fails soft: returns
`None` when Ollama is down, and the whole pipeline runs normally with it switched off entirely.

## Q6.2 — OpenBB and TradingView: what each is actually worth here

Both were raised as possible integrations. Researched, and the honest answers differ.

### OpenBB (`OpenBB-finance/OpenBB`) — **use as a reference, not a dependency**

72.6k stars, genuinely excellent, and the Bloomberg-alternative framing is fair. Two facts decide
it for *this* project:

1. **Its Indian coverage is `yfinance` wearing a wrapper.** The full provider list is
   `alpha_vantage, benzinga, biztoc, bls, cboe, cftc, congress_gov, deribit, ecb, econdb, eia,
   famafrench, federal_reserve, finra, finviz, fmp, fred, government_us, imf, intrinio, multpl,
   nasdaq, oecd, sec, seeking_alpha, stockgrid, tiingo, tmx, tradier, tradingeconomics, wsj,
   yfinance` — **there is no NSE/BSE-native provider at all.** We already call yfinance directly,
   *and* we pull the NSE bhavcopy archive, which is strictly better for our purpose: primary
   source, point-in-time, with delivery-% and the full listed universe rather than a survivor set.
   OpenBB would add an abstraction layer over data we already get more directly and more completely.
2. **It is AGPL v3, on every file.** For a personal, non-distributed, single-user desktop app the
   copyleft obligations do not trigger — there is no conveying and no network service to third
   parties. But importing it would mean that *if this is ever shared with anyone*, the whole
   application inherits AGPL. That is a strategic door worth not closing by accident for a
   dependency whose Indian coverage we do not need.

**What is genuinely worth taking from it — read the code, copy the ideas:**
its **provider abstraction** (many heterogeneous sources normalised behind one standardised
model per data type) is precisely the problem our `data/` layer has with bhavcopy + Upstox +
yfinance + finviz + RSS, and it is a well-tested design. Same for its router/registry pattern.
Study `openbb_platform/core` and the provider packages; write our own thin equivalent.

**How to clone it (you offered, and yes — but not into this repo).** Clone it *beside* the
project as a read-only reference, e.g. `../_reference/OpenBB`, and add nothing to
`pyproject.toml`. Keeping it outside the working tree matters for two reasons: it avoids
vendoring ~70k stars' worth of AGPL source into a repo we may later want to license differently,
and it keeps `git status`, the test collector and the search workers from ever walking it. The
value is in reading `openbb_platform/core` and two or three provider packages to see how they
standardise heterogeneous sources — an afternoon of reading, not an integration.

### TradingView — **valuable in one specific direction only**

TradingView has no official retail market-data API, and scraping it is both fragile and a ToS
question. So it is **not** a data source; we have Upstox and the NSE archive.

What *is* supported, documented and genuinely useful is the **webhook alert**: an alert fires,
TradingView POSTs a JSON payload to a URL you control. That makes it a legitimate *signal input*
and a *human-in-the-loop* channel — you spot something on a chart, the alert reaches the engine,
and it is recorded alongside everything else the system saw.

Constraints, measured rather than assumed:
- **Only ports 80 and 443** are accepted; anything else is rejected outright.
- The endpoint must be **publicly reachable**, so a local desktop app needs a tunnel
  (`cloudflared` or `ngrok`). That is a genuine attack surface on a machine holding broker
  credentials, so the receiver is **signed-payload, allow-listed, and inert by default**: a
  TradingView alert can *create a candidate*, never place an order. It enters the same interlock
  chain as any other signal.
- Webhooks require a paid plan and 2FA on the TradingView account.

**Verdict:** worth building as `server/webhooks.py` at low priority, after the live engine exists.
The other honest use is as an idea source — porting a Pine Script strategy into the Q4 registry
where it must clear the same gate as everything else.

### On connecting a TradingView MCP — yes, but NOT as a backtest engine

Offered, and worth accepting for two of its three plausible uses. Checked what these servers
actually expose: Pine Script validation, the full v6 reference (457 functions, 427 variables),
chart control and analysis — and, in the larger builds, "strategy optimization".

**Accept it for:**
- **Chart reading** — visually confirming a signal looks like what the numbers claim.
- **Idea sourcing** — porting a Pine strategy into `strategies/` where it faces *our* gate.

**Refuse it for validation, for a structural reason and a discipline reason.**

*Structural:* TradingView's strategy tester has an "**inability to backtest multiple instruments
at once (portfolio backtesting)**" and tests **one timeframe** at a time. Our core strategies are
**cross-sectional** — the tug-of-war, the ranker, every top-N selection requires ranking ~2,000
stocks *on the same date simultaneously*. Pine Script runs per-chart. It cannot express our
strategies at all; this is not a quality gap, it is a category mismatch. It also "does not analyse
every single price movement within a candle" — approximated fills, where our whole Q3 effort is to
make fills honest.

*Discipline — and this is the one that actually matters:* a config tested on TradingView **never
registers in `attempts.py`**. That count is the `n_trials` fed to the Deflated Sharpe. Testing
strategies in a second, unlogged place makes `n_trials` an undercount, which silently inflates
every DSR the system reports. **It would quietly break the exact guard that distinguishes this
build from the two that failed.** One backtest engine, one registry, one truth.

### What I actually need from you — the short list

Rather than accept every tool offered, these are the ones that unblock real work, in order:

| | Why | Blocks |
|---|---|---|
| **1. Upstox OAuth token refresh** | expires daily ~03:30 IST; currently 401 | the 1-min spine **and** the whole live advisor (R7) |
| **2. Telegram bot token + chat id** (@BotFather) | outbound alerts | Q6.3 |
| **3. `ollama signin`** | cloud synthesis tier | Q6 narration/diagnosis only |
| **4. Static IP decision** | Angel One rejects API orders from unregistered IPs since 2026-04-01 | live *execution* only — not the advisor |

**Not needed:** an external database MCP. We have DuckDB + Parquet locally, and adding a second
store would fight the single-writer design that was measured into existence. Obsidian is already
connected with Dataview installed — nothing further required there.

### RSI, MACD and friends

Deterministic functions of price, no agent and no LLM. They live in
`features/technical.py`, registered in the factor registry, and clear the same gate as anything
else. Noting the trap explicitly: RSI is the most data-mined indicator in existence, which is
exactly the *factor homogenisation* failure AlphaAgent documents — a generator left unchecked
rediscovers it and lands on crowded, fast-decaying signals. The originality penalty in Q4.5
exists to push against that. Their honest role is as **conditioning variables**, not signals.

**`llm/synthesis.py`** — one interface, two backends, selected by config: `local` (llama3.2:3b) or
`cloud` (a model exposed by `ollama signin`). Jobs: nightly narration of the scorecard from
Python-computed facts, carrying a **numeric tripwire** that flags any figure in the output not
traceable to those facts; the **"where is the mistake, why didn't it make profit"** diagnosis; and
*proposing* new strategy code, which then clears the identical gate as everything else.
**No Claude CLI, no Anthropic API key** anywhere in the runtime.

**`llm/obsidian.py` — the knowledge graph.** **The vault moves into the repo**, at
`stocksense/vault/` — your request, and the better arrangement anyway: notes version alongside the
code and results that produced them, so a strategy note and its commit travel together. Add
`vault/.obsidian/` to `.gitignore` (workspace UI state) but **track the notes themselves**.

*One action for you:* Obsidian → **Open folder as vault** → pick
`…\WEB-DEV-PROJECTS\stocksense\vault`. The old `Documents\Obsidian Vault` can stay or go; nothing
reads it. **Dataview is already installed** (confirmed), so queryable tables of every strategy and
verdict work from the first note written.

The app writes Markdown with YAML frontmatter and `[[wikilinks]]` into `vault/`:
```
StockSense/
  Daily/2026-09-15.md            what it traded, why, P&L, what went wrong
  Strategies/<slug>.md           hypothesis, params, ICIR, DSR, PBO, verdict, [[links]]
  Hypotheses/<slug>.md           pre-registration, frozen before results
  Verdicts/<slug>.md             PASS/FAIL, unmodified
  Symbols/<SYMBOL>.md            your history in it, corporate actions, microstructure profile
  Incidents/<date>-<slug>.md     feed failures, rejected orders, disarms
```
Wikilinks between notes *are* the graph — Obsidian's built-in graph view renders it with no
plugin. Claude Code reads this folder to reconstruct context across sessions, and a pointer goes
in `CLAUDE.md`.

**What you must do manually (this is the whole list):**
1. Open Obsidian → Settings → Community plugins → install **Dataview** (gives queryable tables of
   every strategy/verdict) and optionally **Templater**.
2. `ollama pull nomic-embed-text`.
3. Confirm the vault path above is the one you want; say so if you moved it.

Everything else is plain file writes — no API, no sync service, no plugin development.

---

# Q6.3 — Telegram: eyes on everything, without sitting at the screen

`notify/telegram.py`. A bot token plus your chat id in `.env`; one outbound HTTPS POST per
message. No inbound webhook, no public endpoint, no tunnel — so it adds **no attack surface** to a
machine holding broker credentials, unlike the TradingView receiver.

This is the right channel precisely because the engine now scans all session: you should not have
to watch a terminal for seven hours to learn that something started moving at 12:00.

**What it sends** (each independently switchable, because a channel that cries wolf gets muted,
and a muted channel is worthless):

| Priority | Event |
|---|---|
| **Critical** | interlock breach / auto-DISARM · daily loss limit hit · broker rejection · **feed dead during market hours** |
| **Action** | new entry candidate (symbol, entry, stop, target, qty) · **exit now** on a held position · 15:10 square-off warning |
| **Watch** | a held name moving abnormally fast · a name entering the shortlist mid-session · regime stand-down triggered |
| **Daily** | 08:00 pre-open brief · post-close scorecard with net P&L after charges and tax |

**Rules that keep it trustworthy:**
- **Read-only.** Telegram never places, modifies or cancels an order. It reports; you decide, or
  the armed engine acts through its own interlocks. A reply of "buy" does nothing.
- **Rate-limited and deduplicated** — one alert per symbol per state change, not per 5-minute scan.
- **Fails soft.** If Telegram is unreachable the engine trades on unaffected and the failure shows
  in the health panel. A notification channel must never be able to halt the system.

*Action for you:* create a bot via **@BotFather**, then put `STOCKSENSE_TELEGRAM_BOT_TOKEN` and
`STOCKSENSE_TELEGRAM_CHAT_ID` in `.env`.

---

# Q7 — Execution: paper first, live behind interlocks

**`execution/cost_model.py`** (**PROTECTED**) — exact Indian charges to the paisa. Brokerage,
STT (intraday: 2.5 bps, **sell leg only**), exchange txn charges, SEBI turnover fee, stamp duty
(**buy leg only**), GST on brokerage+txn. Verified previously against a real charge sheet at
**8.3 bps** per ₹100k MIS round trip — reproduce and re-verify against your latest contract note.

**`execution/fill_model.py`** — fills at **next-bar open + half the measured effective spread**,
never the signal bar's own close. Participation capped against the bar's real volume. Circuit-band
locked bars excluded. Per-symbol MIS leverage table (not every stock gets 5×).

**`execution/algos.py`** — TWAP, VWAP, POV, Implementation Shortfall, sized by Q3's impact model.
At retail order sizes these rarely bind, but "rarely" is measured against the live account, not assumed.

**`execution/paper_broker.py`** — full simulated account: orders, fills, positions, daily NAV,
charged through the real cost model. **Week 1 is paper-only, per your instruction.**

**`execution/interlocks.py`** (**PROTECTED**) — checked in this exact order before every order:
```
armed_state == ARMED
  -> strategy_id matches the armed strategy
  -> inside 09:15–15:10 IST
  -> per-order value cap
  -> open-position cap (the armed config's swept value)
  -> orders-per-day cap
  -> daily realised-loss limit in ₹  (HARD. non-overridable.)
  -> data staleness check (no decision on a stale tick)
  -> participation / circuit check
  -> duplicate-order check
  -> static-IP present and matches the registered one
```
Any breach → **auto-DISARM** + manual re-arm required. Every rejection logged with its reason; a
rejected order must never vanish silently. **Mandatory square-off at 15:10**, ahead of the
broker's ~15:20 MIS auto-square-off.

**`brokers/angel_execute.py`** (**PROTECTED**) — the *only* module in the codebase permitted to
reference an order-placement call. A test asserts that fact by AST-scanning every other file.

**Arming.** Default **DISARMED**. All must hold to arm: gate PASS on research data; DSR ≥ 0.95 and
PBO ≤ 0.5 on the vault; ≥ 5 trading days of paper record; ≥ 30 graded predictions; static IP
registered and verified; and an explicit typed confirmation phrase at the CLI. No scheduled job,
no agent, no API route may arm. **Arming expires after 5 trading days and must be renewed.**

---

# Q8 — The live engine and the day

**`live/engine.py`** — three separate clocks, never blocking each other:

| Clock | Cadence | Job |
|---|---|---|
| Tick | ms–s | Upstox WebSocket ingest → in-memory ring buffer → LOB state → risk interlocks |
| Decision | 5 s – 5 min | Re-score the **frozen** model on fresh bars; emit/withdraw signals |
| Research | nightly | Everything in Q4/Q5 |

**The rule in one line: re-scoring during the session is fine; re-training during the session is
forbidden.** A model retrained on this morning's data to trade this afternoon has seen the
answer — it looks brilliant in testing and loses money live. The model is frozen at 09:15 and does
not change until the nightly run.

### What this actually is: StockSense talking to you, all session

The product is **a companion that watches the whole market continuously and tells you what it
sees**, not a generator that emits a list and goes quiet. Concretely, every 5 minutes from 09:15
to 15:10 it re-reads **every liquid NSE stock**, and it speaks to you on Telegram in three voices:

1. **About what you already hold** — position-aware **buy / sell / hold** on your *actual* Angel One
   positions. Not generic advice: "you are long X at ₹88, it is now ₹95 and the trail sits at ₹93".
2. **About what you could hold** — names entering or leaving the candidate set, at whatever time
   they start moving. 12:00 was an example, not a schedule; a name that comes alive at 10:40 or
   14:05 is caught on the same clock.
3. **About itself** — a feed died, an interlock tripped, the regime turned hostile, 15:10 is close.

### This unlocks a v1 that ships *before* the execution blockers clear

Worth stating plainly, because it changes what is reachable soon:

**The advisor needs neither the static IP nor order-placement rights.** Angel One's **read-only**
sync already works from this ISP (verified in Q0.3: fresh TOTP login in 831 ms, holdings /
positions / tradebook / RMS all readable). So:

| | Needs | Blocked by |
|---|---|---|
| **v1 — Advisor** | live prices + read-only holdings + Telegram | **only the Upstox token refresh** |
| **v2 — Armed engine** | everything in v1 + order placement | static IP registration, *and* a passed gate |

v1 is a genuinely useful product on its own — it watches all 2,000+ names so you do not have to,
knows what you are holding, and pings you when something changes. You place the orders. That is
also the honest sequencing: **advice you act on manually, while the system builds the forward track
record that would justify ever letting it act by itself.**

**Non-negotiable labelling:** until a strategy clears the gate, every Telegram suggestion carries
its status — `UNVALIDATED` — and the running hit-rate of past suggestions. A companion that talks
confidently about signals with no measured edge is worse than silence, because it is persuasive.

### The engine hunts ALL SESSION. It does not shortlist at 09:30 and go quiet.

**This corrects the earlier draft, which read as "produce 1–2 names at 09:30 and stop". That was
wrong, and the objection is right:** a name that starts moving at 12:00 and runs to 14:00 is worth
more than a 09:30 pick that bled ₹200, and a system that stopped scanning at 09:30 would never see
it. **The whole session is the opportunity window.**

What still holds is a cap on *entries*, not on *attention*:

**To remove a likely misreading: "we only take 2 positions" is NOT "we only look at 2 stocks."**
Every one of ~2,000 names is scored on every 5-minute sweep. The position cap governs how many of
those scores turn into *trades*. Separately, the backtest optimisation in Q4.0 skips simulating the
intraday *path* of names the strategy never bought — if you did not hold it, its wiggle cannot
affect your P&L. Full analysis; selective holding. (And per Q5, the cap itself is now swept over
2–10 rather than fixed at 2.)

- **Scanning is continuous and universe-wide** — every 5 minutes across **every liquid NSE stock**,
  09:15 → 15:10. Roughly **75 sweeps a day over ~2,000 names**, never over a frozen 09:30 shortlist.

  *Feasibility, separated into the two things people confuse:* the **compute** is trivial — 2,000
  symbols × a few dozen features is a vectorised operation in well under a second, and the machine
  has 12 threads idle during market hours. The real constraint is the **feed subscription limit**:
  how many instruments Upstox's WebSocket will stream at once, and in which mode. That is a
  measurable fact, not a guess, so it gets a probe (**Q0.7**) the moment the token is refreshed —
  full-depth for the names in play, and lighter LTP-only for the long tail, if the limits require
  tiering.
- **Entering is selective** — a setup fires only when it clears its precondition, the interlocks
  and the cost hurdle. `max_orders_per_day` stays a hard cap, because ~44 round trips/day is
  arithmetically what turned your account's gross **+₹2,492** into net **−₹1,293**.

**Watch everything all day; act rarely.** Those are not in tension.

**The day, revised:**
- **08:00** — pre-open: overnight global vector, news sweep, gap candidates, feed health check.
- **09:00–09:15** — pre-open session data; silent.
- **09:15–09:30** — no *entries* (the noisiest fifteen minutes of the day), but the scanner is
  already running and building opening-range state.
- **09:30 onward** — entries become possible, and **the shortlist is recomputed every 5 minutes**.
  Names enter and leave it all day. A stock that was nothing at 09:30 and breaks out at 12:00 is a
  first-class candidate at 12:00.
- **Continuously** — open positions re-evaluated on the same clock: trail, hold, or exit.
- **15:10** — hard square-off.
- **After close** — scorecard, net of charges *and* intraday tax (speculative business income at
  slab rate; the previous build reported ₹0 intraday tax, which is simply wrong).

### Exit policy is a SEARCHED parameter, not a fixed rule

The IFCI day is the reason this matters: opened ~₹88, ran to ₹95, fell back to ₹93 — where a naive
stop or a "lock in the gain" rule would have exited — then closed at ₹98 for ~₹4,000. A fixed stop
would have cost most of that.

The honest counter is equally true: "hold through the drawdown" is also exactly how accounts blow
up, and one good day is an anecdote, not evidence. So neither belief is hardcoded. The exit rule
becomes part of the **search space** and the data decides:

| Exit family | Searched parameter |
|---|---|
| Hard stop | fixed % or ATR multiple |
| **Trailing stop** | trail distance; whether it only activates after +X% |
| **Give-back cap** | exit after surrendering Y% of peak unrealised gain |
| Time stop | exit after N bars with no progress |
| Target | fixed R-multiple, or none — let 15:10 end it |

My prior, written down in advance so it can be checked against the result: **trailing and
give-back exits should beat both a hard stop and a hard target**, because they are the only
families that can capture an IFCI-shaped day *and* still bound the loss on a bad one.

### Universe: small and mid cap, which the evidence already supports

Your conclusion is right and the previous build's gate agrees — though for the opposite reason to
the one given. Large caps are *less* volatile than small caps, not more; what kills them here is
that they are the most efficiently priced, so there is no edge left to find. The measured result
was blunt: **large `h10 n10` = +0.060%, p = 0.655 → FAIL**, while mid (+0.560%) and small
(+0.694%) both passed at p = 0.0001.

The scan therefore treats small and mid cap as the primary hunting ground and reports large cap
separately as a control. The cap band stays a pre-registered split, never tuned after the fact.

**Latency budget** (against Q0.2's measured numbers, not guesses): tick→state < 50 ms;
state→signal < 500 ms; signal→order submitted < 1 s. Achieved by keeping hot state in memory
(never a DB round-trip on the decision path), Numba-compiling the LOB update, and doing all I/O
on a separate thread from decisioning.

---

# Q9 — Frontend: the terminal from your recordings

**I watched all nine clips** (`20260902-1246-01` … `20260902-1305-12`, 4–15 s each, silent, ~520×390).
They are captioned feature demos of a quant research terminal branded "PSTL / AI Trading Agent".
Each caption names the feature, and together they specify a complete product. **First housekeeping
task: move them to `docs/reference/ui/` and add `*.mp4` to `.gitignore`** — right now they sit
untracked in the repo root.

### The design language, as observed

Near-black ground (#07090B-ish), 1px hairline borders, very small monospace type, extremely dense
information grid. Muted teal/green/amber accents; red/green strictly reserved for sign. A
persistent **top status strip** and **bottom status strip**, both packed with tiny live readouts.
**Every screen is three columns:** left = navigation + parameter sliders, centre = the primary
visual, right = facts, formulas, tables and streaming logs. Content is never centred in whitespace;
it fills the frame edge to edge. This is a Bloomberg-terminal idiom, not a SaaS dashboard, and it
is what "PROPER LIVE" means in practice.

### The nine screens, and what each maps to in the backend

| # | Caption (verbatim) | What it shows | Backend it renders |
|---|---|---|---|
| 1 | *(Strategy Lab)* | Left: **strategy deck** — cards with sparkline + live metrics ("Volume-weighted mean reversion", "Cross-sectional momentum", "Vol-targeted trend", "Cointegrated pair, OU spread", "Variance risk premium", "Index volatility filter") and parameter sliders. Centre: badge + strategy title + description, a 6-metric KPI row, a 4-metric row, then a full-width equity curve (2015→2026). Right: **kernel / alpha definition** as a live formula block, key-parameters table, prose rationale, bullet findings. A second view is a full **tearsheet**: underwater drawdown curve, returns series, returns histogram, monthly-returns heatmap by year × month | Q4 strategy families + Q2 `factor_metrics` / `gate` |
| 2 | "Added volatility clustering simulation tab for calculating position size for tail risk" | Particle/force-graph volatility-cluster visual; right panel of tail statistics (annual vol, excess kurtosis, VaR 1%/5%, ES, skew, autocorrelation, tail dependence); bottom: amber vol time-series + green ACF decay curve | **Q5 Monte Carlo + `simulation/sizing.py`** — this is literally the MC sizing screen |
| 3 | "Data pipeline from external data sources to multiple trading signals which can do self-recovering" | **DAG**: SOURCES → INGEST → COMPUTE → RISK → SIGNALS with animated flowing edges; left: pipeline health counters (24/24, 44.5 s); bottom: **operations log** streaming `retry 1/3 → recovering`, `retry 2/3 failed → back-off 1.6s`, `healthy again — resumed`; right: selected-node detail + recent-run table | **Q1 data spine + the reason you dropped scheduled tasks.** This screen *is* the answer: every feed failure is visible and self-heals in front of you |
| 4 | "Machine learning experiments lab for agent to route prediction problem to explainable ml model" | Model DAG as 3D nodes, code/log pane, 3D response surface, right-hand **model leaderboard** (LightGBM runs with metrics), bottom feature-importance bars | Q4 `search/engine.py` + LightGBM; feature importance is the drift audit |
| 5 | "Just added market sentiment dashboard which shows realtime sentiment volume and it's temperature" | Source chips across the top (Stocktwits, Trends, Consumer, Reddit, YouTube, News auto, Blogs) wired by edges down to **per-ticker circular sentiment gauges**; a red→green temperature gradient bar; right: whole-board stats + live sentiment-tagged news feed | Q6 `llm/ollama.py` news scoring + `data/news.py`. **Ours is NSE**: Moneycontrol/ET RSS, not Stocktwits |
| 6 | "Agent to generate alpha and simulate it on a coordinate plane" | A red↔green 2-D field with candidate alpha trajectories drawn over it; a natural-language **ALPHA SEARCH** prompt bar with RUN / CLOSE | Q4 `search/space.py` + Q6 `llm/synthesis.py` |
| 7 | "Automatically generating alphas by natural language and selecting the closest pair of stocks for arbitrage trading" | Prompt → generated alpha code + a plain-English explanation panel → cointegrated pair selection | Q4 family #6 (pairs/cointegration) driven by Q6 |
| 8 | "Integrated experimental tools into a single robust alpha research product" | Tearsheet, a **"how much factor is observed"** research table (one row per factor, with a *Learn* button each), and a long-form research/wiki reading view | Q2 metrics + the Obsidian knowledge graph, surfaced in-app |
| 9 | "Each data processing steps and its realtime status now showing up to my pipeline monitor with the self healing systems" | Force-directed graph of every processing step with realtime status, node inspector, per-stage cards | Q1/Q8 health — the same system as #3, zoomed out |

### Framework decision: Electron. Nothing is cloud-hosted.

**The architecture, stated plainly so it cannot be misread:** an Electron shell spawns a Python
FastAPI process on a free `127.0.0.1` port, which reads DuckDB/Parquet from local disk. **No
Vercel, no hosting, no external service, no telemetry.** The only bytes that leave the machine
are outbound market-data fetches to NSE, Upstox, Angel One and the news feeds. Pull the network
cable and the app still opens, still shows every historical result, and says which feeds are dead.

This is exactly what `phase0/prove-the-economics` did, and its `desktop/main.js` is worth lifting
rather than rewriting: it finds a free port, spawns the API as a child, waits for the port to
*listen* (not for HTTP 200 — a busy database is a correct 503, and waiting on 200 once stopped
the window opening at all), and kills the child on **every** exit path, because on Windows a
detached child does not die with its parent.

**Electron vs Tauri, honestly.** Tauri genuinely wins the headline numbers: 3–10 MB bundle
against 120–200 MB, roughly 5× less RAM, ~380 ms cold start against ~1,420 ms. Those are real.
They just do not buy anything *this* project needs:

| | Electron | Tauri | Which matters here |
|---|---|---|---|
| RAM | ~300–400 MB | ~80–150 MB | **Neither.** 15.7 GB total; the binding constraints are 10 of 12 CPU threads and 4 GB VRAM. Saving 250 MB changes nothing. |
| Bundle | ~150 MB | ~5 MB | **Neither.** One machine, one user, no distribution. |
| Cold start | ~1.4 s | ~0.4 s | Marginal — started once a day. |
| Toolchain | Node 24, already installed | **Rust + WebView2** | **Electron.** Smart App Control already blocked scipy's wheels on this machine; adding a new native build chain is fresh surface for exactly that class of failure. |
| Rendering | Chromium, identical everywhere | OS WebView2 | **Electron.** The reference UI is *extremely* dense — hairline borders, tiny monospace, canvas heatmaps, an L2 depth ladder. WebView2/Chromium CSS differences bite hardest on precisely that kind of design. |
| Proven here | `phase0`'s shell ran on this exact laptop | unproven | **Electron.** |
| DevTools | full Chromium devtools | weaker | **Electron.** Building a live terminal is much faster with them. |

Tauri would be the right call if we were shipping to many users, targeting a 4–8 GB machine, or
needed a small installer. None of those apply. **Electron, and your preference is also the
correct engineering answer here.**

### Build order for the UI

Electron + FastAPI + **WebSocket push** (the UI never polls — that is what makes it feel live).
Rendering: **uPlot** for time-series (it redraws a 100k-point series in ~1 ms, which is what the
tick tape needs), **D3-force** for the pipeline/alpha DAGs, plain canvas for the heatmaps and
gauges. No React needed; the existing app was vanilla ES modules and that was the right call.

1. **Health / Pipeline monitor** (screens 3 + 9) — **first**, because it is your stated reason for
   dropping scheduled tasks, and because it is useful the moment Q1 exists.
2. **Live** — tape, per-symbol sparklines, L2 depth ladder, tick-by-tick P&L, positions.
3. **Strategy Lab** (screen 1) — needs Q4.
4. **Risk / Vol-clustering** (screen 2) — needs Q5.
5. **ML Lab** (screen 4), **Sentiment** (screen 5), **Alpha Agent** (screens 6+7), **Research/Wiki**
   (screen 8).
6. **Scorecard** and **Portfolio forensics** (Q10) — your real trades, win rate, where you lost.

**Offline-first, always.** The app opens and is fully usable with every feed down; it says which
feed is dead and when it last had data, rather than showing a stale number as if it were live.

**One honest note on the reference:** those clips show US tickers (NVDA, TSLA, AAPL, META, SPX) and
US sentiment sources. We are taking the *design and the feature set*, not the market — every screen
gets NSE data. And the numbers in those demos are that product's, not a promise about ours.

---

# Q10 — Portfolio forensics (your real trades)

`portfolio/forensics.py`, rebuilt from the previous `statements/` package: fuzzy-column XLSX
parsers for the Angel One tradebook, FIFO position reconstruction, win rate, average win vs
average loss, expectancy, real cost drag, max drawdown, and behavioural diagnostics (revenge
trading, sizing chaos, overtrading). Plus the counterfactual engine you asked for: **where you
lost, and where you would have won** under a different exit rule. The previous build reconciled to
within **1.6%** of the broker's own reported P&L — hold the rebuild to that same bar.

Every sizing rule downstream calibrates to *these measured numbers*, never to the profit target.

---

# Where today actually stands — 2026-09-03, 08:00 IST

**Trading live today is not possible, and I would rather say so at 08:00 than at 09:30.** What is
missing is not polish:

| Needed to trade | State |
|---|---|
| Live price feed | **Upstox token expired** (HTTP 401). No OAuth refresh, no live prices. |
| Live order placement | **Static IP not registered** with Angel One. Orders are rejected since 2026-04-01. |
| Adjusted prices | `adjust.py` written **an hour ago, untested, uncommitted**. |
| Universe / vault / attempts / walk-forward / gate | **Not built.** |
| Any strategy, or the search engine | **Not built.** |
| Data spine | ~60% ingested (through 2022-04), **286 days failed** in a DNS outage and need a repair pass. |

Trading on an untested pipeline with real leverage would be the exact failure this rebuild exists
to prevent — and the previous build's own evidence says an unvalidated intraday signal loses money
(0/15 folds, negative *gross* alpha; your account: gross +₹2,492, charges −₹3,785).

**What today can realistically produce:** a finished, repaired data spine; tested and committed
adjusted prices; the point-in-time universe; the vault and attempt registry; and — if the day goes
well — **the first real pre-registered backtest**, on daily data, needing neither the Upstox token
nor a static IP. That is a genuine result to look at tonight, rather than a live trade placed on
faith.

---

# Phase R — the next block of work, in dependency order

**Context.** Q0 is done, and Q1/Q2 are half-built: the store, the daily spine and the
corporate-action parser exist, as do the multiple-testing guards, the factor metrics and the
sizing layer (127 tests). What is missing is everything between *raw prices* and *a defensible
backtest*. Until that exists, the mass-search engine has nothing honest to search with.

**The sequencing insight that shapes this phase:** the first hypothesis needs **no minute bars at
all**. Overnight–intraday reversal is `open/prev_close - 1` in and `close/open - 1` out — both
columns are in bhavcopy, both are **verified null-free and non-negative across 1,290,511 rows
already ingested**. It is a genuine intraday trade (enter at the open, exit by 15:10) that is
fully backtestable on daily data *today*, while the Upstox token is still expired.

So Phase R builds the research foundation and runs one real, pre-registered experiment on it —
rather than waiting on the intraday spine.

### R1 · `data/adjust.py` — adjusted prices *(blocks everything)*

Raw bhavcopy makes a 1:10 split look like a −90% day. Cumulative back-adjustment: factor 1.0 at
the latest date, multiplied backwards through each ex-date, on two bases — `price`
(splits/bonuses) and `total` (also dividends). Consumes `Reader.corporate_actions()` and the
`factor_price` already parsed.

**The anomaly detector must flag adjusted-price jumps with NO matching corporate-action record —
NOT `adj_close` vs raw `close`.** The latter is the check that quarantined RELIANCE, TCS, INFY,
HDFCBANK and ~600 other blue chips in a previous build, for the crime of having had real splits.
It also inverts survivorship bias into something worse: "never had a corporate action".

Day-over-day work joins the trading calendar via `core.clock.trading_days`, never row adjacency —
an illiquid name's previous row can be weeks earlier.

*Acceptance:* a known split (ECLERX 1:2, PASHUPATI 1:10) yields a continuous adjusted series with
no residual jump; a symbol with a genuine bonus is **not** quarantined; every unexplained jump is
enumerated, never aggregated away.

### R2 · `data/universe_pit.py` — point-in-time tradeable universe

The single most important anti-survivorship control, and in the previous build it was correct but
wired into nothing except a display command. `universe_as_of(d)` resolves liquidity and price
filters using **only rows dated strictly before `d`**.

Reuses `simulation.sizing.tradeable_price_band` for the price bounds, so the universe is defined
by what this account can actually trade (₹250–₹2,187 today) rather than an arbitrary constant.

*Acceptance:* a symbol that only becomes liquid in 2015 is absent from the 2010 universe while
its later rows survive; no query reads a row dated on or after the as-of date.

### R3 · `evaluation/vault.py` — the sealed holdout **(PROTECTED)**

**`VAULT_SEAL_DATE = date(2025, 7, 1)`** — your decision, and the balanced one. It withholds
~14 months / ~290 trading days (~8% of history) while leaving 2010 → 2025-H1 for research,
which still spans the 2020 crash and the 2021–24 bull run. Enough holdout that a single DSR/PBO
test on it means something; not so much that the search is blind to the modern regime.

Enforced at the single `load_candles` choke point every research path funnels through. Without an
`UnsealToken`, rows on/after the seal are dropped and the withheld count is logged. `unseal()`
refuses unless the pre-registration is **committed to git**, the attempt is registered, and no
prior unseal exists for that hypothesis. **One unseal per hypothesis, ever** — that is what makes
the holdout a holdout rather than a slower test set.

### R4 · `evaluation/attempts.py` — the attempt registry **(PROTECTED)**

Append-only. **Every configuration evaluated registers here, not just survivors** — that count
*is* the `n_trials` fed to `robustness.deflated_sharpe_ratio`, which is what makes a wider sweep
raise its own bar instead of gaming itself.

### R5 · `evaluation/walkforward.py` + `gate.py` **(PROTECTED)** — exact spec

**Purging and embargoing, defined precisely** (López de Prado). Two distinct operations, both
required — dropping either one leaks:

- **Purge** — remove training observations whose **label window overlaps the test set**. With a
  10-day forward label, a training row dated 5 days before the test set still "knows" 5 days of
  test-period returns. `purge_size = horizon_bars` observations, applied on **both** sides of every
  test block.
- **Embargo** — additionally drop training observations in the window **immediately after** the
  test block, killing serial correlation that purging alone leaves. **`embargo_pct = 0.01`** (the
  AFML convention) → on 4,100 sessions that is **41 sessions**.

```python
# src/stocksense/evaluation/walkforward.py    ** PROTECTED **
@dataclass(frozen=True)
class CVConfig:
    n_folds: int = 10           # groups the timeline is cut into
    n_test_folds: int = 2       # groups held out per split
    purge_bars: int | None = None      # None -> horizon_bars of the strategy
    embargo_pct: float = 0.01          # 41 sessions on a 4,100-session history
    min_folds_required: int = 10       # gate refuses to rule on fewer
    session_bounded: bool = True       # a fold boundary may NEVER fall inside a session
```

**Use Combinatorial Purged CV, not a single walk-forward.** With `n_folds=10, n_test_folds=2` you
get `C(10,2) = 45` train/test splits, which recombine into **`k·C(N,k)/N = 9` distinct backtest
paths** instead of one. That matters twice over: nine equity curves are far harder to fool than
one, and **PBO needs a matrix of paths as its input** — a single walk-forward cannot feed it.

**`evaluation/gate.py` (PROTECTED, frozen on landing).** One-sided binomial test on the count of
positive folds:

```python
GATE = dict(
    min_folds_required   = 10,
    min_mean_alpha_net   = 0.0,     # AFTER compute_charges, never gross
    max_binomial_p       = 0.05,    # H0: positive folds ~ Binomial(n, 0.5)
    max_drop_fraction    = 0.15,    # >15% of folds unusable -> inconclusive, not a pass
)
```

Verdict is `PASS | FAIL | INCONCLUSIVE`. **No threshold may change after a result is seen** — this
project committed that error once, documented it, and rebuilt from statistical principle.

---

### R5.5 · `features/technical.py` — the exact feature list

Named explicitly so nothing has to be invented. Every one is computed **per symbol with
`.over("symbol")`** and joined to the trading calendar, never by row adjacency.

| Group | Features |
|---|---|
| **Returns** | `ret_1/5/10/20d`, `overnight_ret = open/prev_adj_close − 1`, `intraday_ret = close/open − 1`, `ret_1m_overnight_mean`, `ret_1m_intraday_mean` *(the tug-of-war sorts)* |
| **Volatility** | `realised_vol_5/20d`, `atr_14`, `parkinson_vol` (high/low), `vol_of_vol` |
| **Volume** | `vol_ratio_vs_20d`, `vol_ratio_vs_same_time_of_day` *(intraday volume is U-shaped — a flat baseline fires at every open and close)*, `turnover_inr`, `amihud_illiquidity` |
| **Trend / oscillator** | `rsi_14`, `macd(12,26,9)`, `dist_from_vwap_sigma`, `dist_from_sma_20/50`, `opening_range_position` |
| **Microstructure** | `edge_spread` (Ardia–Guidotti–Kroencke), `roll_spread`, `corwin_schultz_spread`, `kyle_lambda`, `tick_rule_signed_volume` |
| **Regime** | `india_vix`, `nifty_ret_1d`, `hmm_state`, `overnight_global_vector` (S&P/Nasdaq/VIX close, USDINR) |
| **Fundamental-ish** | `delivery_pct`, `delivery_pct_zscore_20d` *(demoted — see §G, no published evidence)* |
| **Denoised** | `kalman_level`, `kalman_slope` — adaptive state estimate of price, from the blueprint |
| **News** | `news_novelty`, `news_sentiment`, `news_count_zscore`, `event_type` one-hots |

**Every feature must pass the leakage suite**: shuffling the label must collapse its IC to ~0, and
no feature may read a row dated on or after the prediction timestamp.

### R6 · First pre-registered experiment: overnight–intraday reversal

`research/preregistration_overnight_reversal.md`, **committed before any result exists**, fixing:
signal definition, the three variants (long-only / short-only / long-short), cap bands, cost model
(`compute_charges` at the real 10.62 bps round trip for this position size), fold parameters, iteration budget, and
pass criteria. Then run it, and commit the verdict **unmodified**, PASS or FAIL.

India-specific prior worth stating in advance: post-2011 NSE shows persistently positive overnight
and negative intraday drift, which cuts *for* the short leg and *against* the long leg. Measuring
all three variants is what makes that testable rather than assumed.

**A FAIL here is a real result and gets committed as one.** It costs one night, not a month, which
is the entire point of the rebuild.

### R-phase code specifications — copy-paste level

Written so nothing has to be inferred. Signatures, algorithms, edge cases and test names are all
given.

#### `src/stocksense/data/universe_pit.py`  (R2)

```python
"""Point-in-time tradeable universe. THE anti-survivorship control.

The rule that makes it point-in-time: every filter is computed from rows dated
STRICTLY BEFORE the as-of date. A single `<=` here silently leaks tomorrow's
liquidity into today's universe and inflates every downstream result.
"""

@dataclass(frozen=True)
class UniverseFilter:
    min_avg_turnover_inr: float = 5_000_000.0   # ~50 lakh/day liquidity floor
    lookback_days: int = 60                     # calendar days for the average
    min_price_inr: float | None = None          # None -> from tradeable_price_band
    max_price_inr: float | None = None
    series: str = "EQ"
    min_observations: int = 20                  # needs history, not 2 prints

def universe_as_of(reader, as_of, flt, equity_inr=None) -> list[str]:
    """Symbols tradeable AS OF `as_of`, using only prior data.

    Algorithm -- one SQL pass, no python loop:
      1. window = [as_of - lookback_days, as_of)     <-- END EXCLUSIVE. Critical.
      2. SELECT symbol, avg(turnover_inr) AS t, count(*) AS n,
                last(close ORDER BY date) AS px
         FROM bhavcopy_eq
         WHERE series = ? AND date >= ? AND date < ?    <-- '<' never '<='
         GROUP BY symbol
      3. keep t >= min_avg_turnover_inr AND n >= min_observations
      4. keep min_price <= px <= max_price. If bounds are None derive them from
         simulation.sizing.tradeable_price_band(equity_inr) -- so the universe is
         defined by what THIS account can trade, and widens as it grows.
      5. return sorted(symbols)
    """

def universe_membership(reader, dates, flt, equity_inr=None) -> pd.DataFrame:
    """[date, symbol] across many dates. ONE query per date, never per row."""

def filter_panel(panel: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Inner-join a long [symbol,date,...] frame to membership, preserving all
    other columns unchanged."""
```

**Tests** — `tests/unit/test_universe_pit.py`:
`test_symbol_illiquid_before_and_liquid_after_is_excluded_from_the_earlier_date` ·
`test_window_end_is_exclusive_no_lookahead` (a row dated exactly `as_of` must not influence it) ·
`test_price_bounds_default_to_the_account_band` · `test_membership_is_one_query_per_date` ·
`test_filter_panel_preserves_extra_columns`

#### `src/stocksense/evaluation/vault.py`  (R3) — **PROTECTED**

```python
VAULT_SEAL_DATE = date(2025, 7, 1)     # ~14 months withheld

class VaultSealed(RuntimeError): ...

@dataclass(frozen=True)
class UnsealToken:
    unseal_id: str
    attempt_id: str
    hypothesis_id: str
    preregistration_path: str
    preregistration_sha256: str
    issued_at: datetime

def apply_seal(df, date_col="date", token=None) -> pd.DataFrame:
    """Drop rows on/after VAULT_SEAL_DATE unless a token is presented.
    Logs `vault_ceiling_applied` at INFO with the dropped row count; with a token
    logs at WARNING with the unseal_id -- an unseal must be visible in the log."""

def unseal(store, *, attempt_id, hypothesis_id, preregistration_path,
           reason, requested_by="user") -> UnsealToken:
    """Refuses unless ALL hold:
       1. the preregistration file EXISTS, is COMMITTED, and is UNMODIFIED --
          `git ls-files --error-unmatch <path>` returns 0 AND
          `git diff HEAD -- <path>` is empty.
       2. attempt_id exists in evaluation_attempts.
       3. NO prior vault_unseals row for this hypothesis_id, else
          raise VaultSealed(f"hypothesis {hypothesis_id} already used its unseal")
    Then write the vault_unseals row and return the token."""
```

DDL — append to `store.SCHEMA`, and add `"vault_unseals"` to `SMALL_TABLES`:

```sql
CREATE TABLE IF NOT EXISTS vault_unseals (
    unseal_id              VARCHAR   NOT NULL PRIMARY KEY,
    attempt_id             VARCHAR   NOT NULL,
    hypothesis_id          VARCHAR   NOT NULL,
    preregistration_path   VARCHAR   NOT NULL,
    preregistration_sha256 VARCHAR   NOT NULL,
    issued_at              TIMESTAMP NOT NULL,
    requested_by           VARCHAR   NOT NULL,
    reason                 VARCHAR   NOT NULL
);
```

**Tests:** `test_seal_drops_rows_on_or_after_the_seal_date` ·
`test_unseal_requires_a_committed_preregistration` ·
`test_unseal_requires_an_unmodified_preregistration` (committed then edited → raises) ·
`test_second_unseal_for_same_hypothesis_raises` · `test_every_unseal_is_recorded` ·
`test_token_lifts_the_ceiling`

#### `src/stocksense/evaluation/attempts.py`  (R4) — **PROTECTED**

```python
"""Append-only attempt registry. EVERY evaluated configuration registers here.

That count IS the n_trials fed to robustness.deflated_sharpe_ratio, which is what
makes a wider sweep raise its own bar instead of gaming itself. Registering only
survivors would understate n_trials and silently inflate every DSR.
"""

def register_attempt(store, *, hypothesis_id, config_hash, config_json, family) -> str:
    """Insert one attempt, return attempt_id. Idempotent on
    (hypothesis_id, config_hash) so a resumed sweep never double-counts."""

def record_result(store, attempt_id, *, verdict, metrics_json, fail_reason) -> None:
    """verdict: screened_out | gate_fail | gate_pass | vault_fail | promoted
    fail_reason: low_ic | unstable_ic | fast_decay | cost_drag | capacity | None"""

def count_attempts(reader, hypothesis_id=None) -> int:
    """n_trials for the DSR. Counts EVERY attempt, not just survivors."""

def trial_sharpe_std(reader, hypothesis_id) -> float:
    """Cross-sectional std of Sharpe across attempts -- the other DSR input.
    MEASURED from the sweep, never assumed to be 1.0."""
```

```sql
CREATE TABLE IF NOT EXISTS evaluation_attempts (
    attempt_id    VARCHAR   NOT NULL PRIMARY KEY,
    hypothesis_id VARCHAR   NOT NULL,
    family        VARCHAR   NOT NULL,
    config_hash   VARCHAR   NOT NULL,
    config_json   VARCHAR   NOT NULL,
    registered_at TIMESTAMP NOT NULL,
    verdict       VARCHAR,
    fail_reason   VARCHAR,
    metrics_json  VARCHAR,
    UNIQUE (hypothesis_id, config_hash)
);
```

**Tests:** `test_every_config_registers_not_just_survivors` ·
`test_registration_is_idempotent_on_config_hash` · `test_count_attempts_is_the_dsr_n_trials` ·
`test_trial_sharpe_std_is_measured_not_assumed`

#### `src/stocksense/search/engine.py` — the two-stage core

```python
"""The cascade measured in Q4.0. Stage A never leaves NumPy; Stage B never
leaves Numba."""

@dataclass(frozen=True)
class Panel:
    """Dense (n_dates x n_symbols) float32 arrays, resident in RAM (33MB each).
    Built ONCE per sweep, shared read-only by every config."""
    dates: np.ndarray          # datetime64[D], sorted ascending
    symbols: np.ndarray
    adj_open: np.ndarray
    adj_high: np.ndarray
    adj_low: np.ndarray
    adj_close: np.ndarray
    volume: np.ndarray
    tradeable: np.ndarray      # bool, from universe_pit.universe_membership
    # __post_init__ asserts every array is (len(dates), len(symbols)) and float32

def build_panel(reader, start, end, flt) -> Panel:
    """Long -> wide pivot. NaN where a symbol did not trade; selection uses the
    `tradeable` mask, never NaN-checking."""

def stage_a_select(panel, signal, top_n, side) -> np.ndarray:
    """(n_dates x n_symbols) signal -> (n_dates x top_n) int32 symbol indices.

    MUST use np.argpartition, NOT np.argsort -- measured 6.7x faster (73ms vs
    493ms per config over the full panel) for identical results, because we need
    the top N, not a fully ordered 2,000.

        masked = np.where(panel.tradeable, signal, -np.inf)
        idx    = np.argpartition(-masked, top_n, axis=1)[:, :top_n]

    side='short' negates the signal; 'long_short' returns both ends."""

@njit(parallel=True, cache=True)
def stage_b_simulate(paths, exit_code, trail, giveback, hard_stop):
    """paths: (n_configs, n_positions_total, n_bars) float32 ADJUSTED prices for
    POSITIONS ACTUALLY TAKEN only -- never every symbol. That distinction is the
    1,000x saving measured in Q4.0.

    Returns (n_configs, n_positions_total) float32 gross fractional return.
    exit_code: 0=session_close 1=trail 2=giveback 3=hard_stop.
    Every path is forced flat at the final bar (the 15:10 square-off)."""

def run_sweep(panel, configs, *, batch_size=50, n_workers=10) -> pd.DataFrame:
    """Streams configs in batches so the (n_configs x n_dates x n_symbols) cube is
    NEVER materialised -- at 5,000 configs that would be 40GB. Registers EVERY
    config in attempts.py BEFORE evaluating it.

    Returns one row per config: [attempt_id, config_hash, mean_ic, icir,
    half_life, gross_ret, net_ret, n_trades, fail_reason]."""
```

**Tests:** `test_argpartition_selection_matches_argsort_exactly` ·
`test_untradeable_symbols_are_never_selected` ·
`test_stage_b_trailing_stop_exits_at_the_right_bar` (hand-built path) ·
`test_every_path_is_flat_at_the_final_bar` ·
`test_sweep_registers_every_config_including_screened_out_ones` ·
`test_sweep_peak_memory_stays_under_batch_bound`

#### `src/stocksense/strategies/base.py`

```python
class Strategy(Protocol):
    family: str          # "overnight_reversal" | "opening_range" | ...
    hypothesis: str      # REQUIRED economic story. A strategy with no stated
                         # mechanism is REJECTED at registration -- this is the
                         # AlphaAgent anti-homogenisation rule, enforced.
    params: dict[str, Any]

    def signal(self, panel: Panel) -> np.ndarray:
        """(n_dates x n_symbols) float32; higher = more attractive; NaN where
        undefined. MUST NOT read any row dated on or after the decision bar."""

    def exit_spec(self) -> ExitSpec:
        """Which exit family and its parameters, consumed by stage_b_simulate."""

def config_hash(strategy: Strategy) -> str:
    """sha256 over (family, sorted params). The idempotency key for attempts."""
```

---

### R7 · Advisor v1 — the companion, in two layers

Unblocks the moment the Upstox token is refreshed; needs **no** static IP and **no** order rights.

**Layer 1 — factual monitoring. Needs no validation at all, because it asserts nothing.**
It reports things that are simply true: what you hold (from the read-only Angel sync), what it is
worth now, how far it has moved, whether volume is abnormal, whether a feed has died, how close
15:10 is. This is useful on day one and cannot mislead, because it makes no prediction.

**Layer 2 — suggestions. Labelled `UNVALIDATED` until a strategy clears the gate.**
Every buy/sell/hold carries its status and the running hit-rate of past suggestions, so the
track record accumulates in the open. This is what turns into a real forward record — and it is
the same ledger that would later justify arming anything.

Components: `live/scanner.py` (5-minute universe sweep), `brokers/angel_readonly.py` (positions,
already probe-verified), `notify/telegram.py`, and `live/state.py` holding hot state in memory so
the decision path never touches the database.

### Also in this phase

- **Repair the 286 failed days.** A DNS outage killed 285 consecutive fetches around 2020-12;
  they are recorded as `failed`, so re-running `backfill-daily` retries exactly those. Also widen
  the retry backoff — 3 attempts over ~6 s cannot ride out a multi-minute network drop.
- **Refresh the Upstox OAuth token** (*yours* — they expire daily ~03:30 IST). This is the single
  blocker on both the 1-minute spine and the whole of R7.
- Run `backfill-corporate-actions` once the daily backfill releases the write lock.
- Move the Obsidian vault to `stocksense/vault/` and open it in Obsidian.
- `ollama signin` for the cloud synthesis tier; `@BotFather` for the Telegram token.

---

# Implementation status (2026-09-02)

**Done and committed** (`83ed49c` on `StockSense`, GitHub default branch — **not yet pushed, the
push was blocked by the auto-mode classifier and needs your approval**):

- Package skeleton, `pyproject.toml`, `.venv` on Python 3.14.5, full dependency stack importing.
- `core/config.py` (typed settings + secret redaction), `core/clock.py` (IST, session, calendar).
- `probes/` + `stocksense probe {network,compute,upstox,angel,all}` and `stocksense doctor`.
- All six Q0 probes run; `research/probes/` written. Findings folded into this plan.

**Done, uncommitted:**

- `data/store.py` — rewritten around the Parquet/DuckDB split above, after the measurement
  falsified the approved design. `Store` (writer) + `Reader` (lock-free).
- `tests/unit/test_store.py` — **10 tests, all passing**, including the cross-process
  writer-vs-reader property and the `date`/`Timestamp` dtype regression.
- `data/nse_bhavcopy.py` — both era parsers, disk cache, resumable `backfill()`. **Written, not
  yet exercised against a real range.** Live-verified during design: UDiFF 2026-08-28 → 3,612 rows
  (2,629 EQ); legacy 2020-01-01 → 1,910 rows with `SERIES == "NA"` present, confirming the trap;
  delivery file is plain CSV, 3,460 rows.

**Immediate next steps:** unit-test `nse_bhavcopy.py` against cached fixtures, then run the
2010→today backfill (hours, unattended, resumable), then Q2's guards.

---

# Build order

```
Q0 probes ──> Q1 data spine ──> Q2 guards ──> Q2.5 labelling ──┬─> Q3 microstructure ──> Q4 search
                                                               │      ──> Q4.5 metaheuristics
                                                               │      ──> Q5 Monte Carlo
                                                               └─> Q10 forensics (start early)
Q6 LLM + Obsidian ── in parallel from Q1 onward
Q7 execution: paper immediately; live path blocked on Q0.1's static-IP answer
Q8 live engine ── after Q7 paper works
Q9 frontend ── health/pipeline monitor first, then Live, then the rest
Q6.2 TradingView webhook receiver ── low priority, after Q8
```

# Verification

- `pytest tests/unit` green after every phase, plus dedicated leakage and determinism suites.
- `git diff` on every **PROTECTED** path is empty across the whole build.
- Pre-registration commit timestamps precede their first results — checked mechanically.
- `attempts` contains one row per backtested configuration; `n_trials` in every DSR matches it.
- `vault_unseals` contains at most one row per hypothesis.
- A killed backfill or training run resumes with no loss of committed progress.
- An AST scan proves no order-placement call exists outside `brokers/angel_execute.py`.
- Paper NAV is reproducible from the fill log alone.

# Honest risks, in the order I would worry about them

1. **The search finds nothing.** The most likely single outcome, and the guards exist so that
   answer can be believed rather than argued with.
2. **The search finds something that is not real.** Thousands of configs guarantee a great-looking
   best from noise alone. This is why DSR/PBO/vault come *before* the generator.
3. **Live execution stays blocked** on the static-IP requirement. Paper trading and the entire
   research stack are unaffected, but "money ungated after week 1" depends on Q0.1.
4. **Costs kill a real signal.** Your own account already proved this shape: gross +₹2,492,
   charges −₹3,785. Q3 attacks it directly; it may still win.
5. **Concentration is the biggest sizing risk**, and Q5 now quantifies it: at 5× on a single name
   there is an ~87% chance of losing half the account within a year even WITH a positive edge.
   Position count and leverage are both swept, with probability-of-ruin as a hard constraint.
6. **The 4 GB GPU is shared** with Ollama and the display. The lease design prevents OOM but
   serialises work — nightly runs will be longer than a dedicated-GPU estimate suggests.

# Bibliography — every claim in this plan, traceable

Cited so that no number here is folklore, and so a later reader can check the source rather than
trust the plan.

**Strategy evidence**
- Lou, Polk & Skouras (2019), *A tug of war: Overnight versus intraday expected returns*, **JFE
  134**, 192–213 — overnight-sorted deciles: overnight α **+3.47%/mo (t=16.83)**, intraday α
  **−3.02%/mo (t=−9.74)**. Mechanism: investor clienteles.
- Baltussen, Da, Lammers & Van Iwaarden (2021), *Hedging demand and market intraday momentum*,
  **JFE 142**, 377–403 — last-30-min predicted by rest-of-day; asset-class Sharpe 0.87–1.73;
  NGE<0 β=6.63 (t=4.78), R²=3.58% vs NGE≥0 β=0.82 (t=1.03), R²=0.05%.
- Gao, Han, Li & Zhou (2018), *Market intraday momentum*, **JFE** — first half-hour predicts last
  half-hour; superseded as a predictor by `r_ROD` above.
- Heston, Korajczyk & Sadka (2010), *Intraday patterns in the cross-section of stock returns*,
  **Journal of Finance 65** — half-hour-of-day continuation persisting 40 trading days.
- Cont, Kukanov & Stoikov, *The price impact of order book events* — OFI explains 65–87% of
  short-horizon mid-price variance.
- *Information content of order imbalance in an order-driven market: Indian evidence*,
  **Finance Research Letters** — NSE: strong for 5 minutes, gone by 30; 1-min coefficient positive,
  longer horizons negative; information in deeper book levels.

**Discipline and calibration**
- Bailey & López de Prado, *The Deflated Sharpe Ratio* — E[max Sharpe] = **3.26** at N=1,000 with
  zero true edge.
- Bailey, Borwein, López de Prado & Zhu, *The probability of backtest overfitting* — CSCV/PBO.
- López de Prado (2018), *Advances in Financial Machine Learning* — purging, embargo (**1%**),
  CPCV, triple-barrier labelling, meta-labelling, sample uniqueness, fractional differentiation.
- McLean & Pontiff (2016), *Does academic research destroy stock return predictability?*,
  **Journal of Finance 71** — **−26%** out-of-sample, **−58%** post-publication across 97
  predictors; decay largest where in-sample returns were largest; survivors concentrate in
  **low-liquidity, high-idiosyncratic-risk** stocks.

**Microstructure and execution**
- Ardia, Guidotti & Kroencke (2024), *Efficient estimation of bid-ask spreads from open, high, low
  and close prices*, **JFE 161**, 103916 — the **EDGE** estimator; Python package `bidask`.
- Roll (1984); Corwin & Schultz (2012) — spread estimators used as cross-checks.
- Almgren & Chriss — optimal execution; square-root market-impact law.

**Sizing and simulation**
- MacLean, Thorp & Ziemba, *The Kelly Capital Growth Investment Criterion* — full Kelly's
  drawdowns; **half-Kelly or less** in practice; a 10% edge overestimate costs ~19% of long-run
  growth.
- Politis & Romano (1994) stationary bootstrap; Politis & White (2004) automatic block length,
  with the Patton, Politis & White (2009) correction.

**Counter-evidence, deliberately recorded**
- Emerging markets favour **trend-following**, developed markets **mean-reversion** — tempers the
  prior on the (mean-reversion-flavoured) tug-of-war trade in India.
- "Return anomalies may not persist in emerging or thinly traded markets."
- Overnight > intraday returns in **23 of 29 countries** — the decomposition is global even where
  the tradeable spread is not.

---

# Session handoff — 2026-09-03, executed by Claude Sonnet

Context verified before execution: 129/129 tests passing, git clean except `data/adjust.py`
(matches the R1 spec below already, untested/uncommitted) and the user's reference blueprint.
Data spine at 7.28M rows / 3,740 sessions through 2026-04, 286 known-failed days from a recorded
DNS outage awaiting a repair re-run. Proceeding through Phase R in order: test+commit R1, repair
the backfill, R2 universe_pit → R3 vault → R4 attempts → R5 walkforward/gate → R5.5 features →
R6 the first pre-registered experiment. No deviation from the specs above; auto mode, minimal
narration, commits at each landed module.

---

# Not doing

Real-HFT latency claims · F&O, crypto or forex · Windows scheduled tasks · any Anthropic API key ·
any LLM producing a number that decides a trade · editing `gate.py` or a pre-registration after
seeing a result · unsealing the vault twice for one hypothesis · arming without your typed
confirmation · promising a daily profit figure.

---
---

# SESSION LOG — 2026-09-03: audit + Phase S (the search engine)

**From this point on, this file (`docs/MASTER_PLAN.md`) is THE plan for this repo,
for every Claude Code session, forever.** No new plan files. Read this file first.
Edit it in place. Append a new dated `SESSION LOG` section per session; never
overwrite the sections above. See `CLAUDE.md` at the repo root for the mechanism
and commit rules.

## PART 1 — AUDIT RESULT (completed 2026-09-03)

### Verified by running, not by reading

| Check | Result |
|---|---|
| Test suite | **292/292 pass** (`python -m pytest tests/ -q`) |
| Daily spine | **8,204,812 rows**, 2010-01-04 -> 2026-09-02, **4,117 sessions**, 7,786 symbols |
| Sessions/year | 243-251 every year -- **complete**, no holes (the 451 "failed" ingest_runs were retried OK) |
| Delivery data | 4,172,120 rows |
| `data_store/` | 340 MB parquet + 779 MB cache + 1.8 MB duckdb |
| Real source LOC | **4,500 lines** across 27 substantive modules |

### Built and matching the plan

`data/` (store, bhavcopy, corporate_actions parser, adjust, universe_pit) ·
`evaluation/` (factor_metrics, robustness DSR/PBO, walkforward CPCV, gate, vault, attempts) ·
`microstructure/` (lob, spread, impact, flow) · `probes/` (all Q0) · `simulation/sizing.py` ·
`core/` (config, clock) · `cli/main.py` (version, doctor, 2 backfills, data-status, 5 probes) ·
`strategies/` (base + overnight_reversal) · `search/runner.py`.

### FINDING 1 -- CRITICAL, BLOCKS ALL RESEARCH: `corporate_actions` has **0 rows**

`adjust.py` exists and is tested, but with an empty corporate-actions table `cum_factor` is
1.0 everywhere, so `adj_open == open` and `adj_close == close`. Every split and bonus is
therefore **unadjusted**. Measured on real 2025-Q1 data, the 15 most negative overnight gaps
are **all corporate actions, not market moves**:

```
KAMDHENU    479.30 -> 48.25  = -89.9%   (1:10 split, factor 0.10)
JAIBALAJI   828.70 -> 163.00 = -80.3%   (1:5  split)
SHRIRAMFIN 2809.85 -> 566.00 = -79.9%   (1:5  split)
GARFIBRES  4650.95 -> 958.50 = -79.4%   (4:1  bonus, factor 0.20)
IGL/SENCO/NAVA/KSOLVES ~ -50%           (1:1 bonus)
```

`overnight_reversal` with `side="long"` buys the **most negative** signal. Today it would
therefore pick **100% phantom crashes** -- trades on price moves that never happened. This is
precisely the trap `adjust.py`'s own docstring warns about, and the protection is wired but
the fuel tank is empty.

**The fetcher works.** Verified live this session: `fetch_window(session, 2025-01-01,
2025-03-31)` returned **291 records**, parsed correctly (KAMDHENU -> split factor 0.100000,
GARFIBRES -> bonus factor 0.200000, JAGSNPHARM -> split 5->2 factor 0.400000). The CLI command
`backfill-corporate-actions` exists. **It was simply never run.** Fixed in phase S1.

### FINDING 2 -- plan violation: capital IS hardcoded

`src/stocksense/core/config.py:67-71` contains `equity_inr = 17_500.0`, `max_leverage = 5.0`,
`max_open_positions = 2`, `daily_loss_limit_inr = 700.0`. This document's own "CAPITAL IS NEVER
HARDCODED" section states these must not exist as defaults, and the decisions table states
leverage and position count are chosen daily by expected-log-growth, not fixed. Fixed in S0.

### FINDING 3 -- the pipeline has a hole in the middle

`features/`, `labels/`, `execution/`, `brokers/`, `live/`, `llm/`, `portfolio/`, `server/`
are **all empty 0-byte `__init__.py`**. Consequences that matter now:

- **`execution.compute_charges` does not exist**, though `gate.py`'s docstring names it as the
  gate's net-of-charges input. `overnight_reversal.py` hardcodes a flat `10.62` bps instead --
  which is *wrong*, because cost is **not constant in position size** (10.62 bps below
  Rs66,667, 8.26 at Rs100k, 5.90 at Rs200k -- see THE COST WALL above). Fixed in S2.
- **`search/space.py` does not exist** although `strategies/base.py` references it in prose.
  Fixed in S3.
- `labels/` and `features/` being empty does **not** block family 1, which computes its signal
  inline. Deliberately deferred; they are needed for families 2-7, not for the first verdict.

### FINDING 4 -- `overnight_reversal` limitations (known, not bugs)

- Does not implement the `Strategy` protocol (functional pipeline instead). Acceptable for now.
- `exit_rule` raises for anything but `"session_close"`, so `trail_pct`, `giveback_pct`,
  `hard_stop_pct` are declared-but-dead. Fixed in S4.
- **No intraday bars exist** (`intraday_bars` dataset MISSING). So path-dependent exits cannot
  be simulated exactly -- only approximated from daily OHLC. S4 specifies the exact
  conservative convention and flags it in every metric it produces.

### Verdict

The **foundation is sound and matches the plan** -- data spine complete, all statistical guards
built, PROTECTED files in place, 292 tests green. What is missing is the thing actually asked
for: **the search engine that backtests thousands of strategies**. That is `search/`
(currently 31 lines) plus its two prerequisites (corporate actions, cost model). Phase S below
builds exactly that, and ends with one real, pre-registered verdict.

---

## PART 2 -- PHASE S: exact build spec

**This section is written to be executed by a smaller/cheaper model with minimal
inference. Every file path, function signature, constant, formula, test name and
shell command below is exact. Do not redesign; type what is written. If something
here contradicts what you find in the repo, STOP and report -- do not silently
pick one.**

### Rules that apply to every phase below

1. **TDD, no exceptions.** Write the test file first. Run it. See it fail with
   `ModuleNotFoundError` or `AssertionError`. Only then write the implementation.
2. After each phase: `python -m pytest tests/ -q` must be fully green before committing.
3. Commit after each phase. Message style: what was built, what a test caught, test count.
   **Author is the user via `gh` auth. No `Co-Authored-By: Claude` trailer, no
   "Generated with Claude Code" line — see CLAUDE.md.**
4. Activate the venv in every bash call: `source .venv/Scripts/activate 2>/dev/null;`
5. Never edit a **PROTECTED** file after it lands: `evaluation/{gate,walkforward,vault,attempts}.py`,
   `execution/cost_model.py` (created in S2), `research/*preregistration*.md`.
6. If a phase's acceptance check fails, STOP and report. Do not proceed to the next phase.
7. **Append progress to THIS file** (a new dated `SESSION LOG` section, or a status
   line under this one) at the end of your session. Never create a separate plan file.

---

### S0 -- Remove the hardcoded capital *(small, do first)*

**File:** `src/stocksense/core/config.py`

Replace lines 65-71 (the "capital and risk" block) with:

```python
    # ---- capital and risk --------------------------------------------------
    # CAPITAL IS NEVER HARDCODED. There is deliberately no `equity_inr` default:
    # a system that bakes in its own account size cannot notice the account
    # growing, so every derived quantity (position size, price band, leverage
    # headroom, ruin probability) silently rots while still looking correct.
    # Capital is READ from the broker at decision time (core/capital.py, not yet
    # built) and passed explicitly to every function that needs it.
    #
    # Leverage and position count are likewise NOT constants: both are chosen
    # per-day by maximising expected log growth (Q5 above). The two entries
    # below are SAFETY INTERLOCKS -- deliberately dumb ceilings that a decision
    # rule may never exceed -- not the decision itself.
    max_leverage_ceiling: float = 5.0        # MIS regulatory max; NOT a target
    max_orders_per_day: int = 8              # hard interlock, see Q3.5
    daily_loss_limit_pct: float = 0.04       # fraction of equity, NOT rupees
```

Then `grep -rn "equity_inr\|max_leverage\b\|max_open_positions\|daily_loss_limit_inr" src/ tests/`
and fix every hit. `simulation/sizing.py` already takes `equity_inr` as an argument -- leave it.

**Test file:** `tests/unit/test_config_no_hardcoded_capital.py`

```python
def test_settings_has_no_equity_default()          # not hasattr(s, "equity_inr")
def test_settings_has_no_rupee_loss_limit()        # not hasattr(s, "daily_loss_limit_inr")
def test_loss_limit_is_a_fraction_not_rupees()     # 0 < s.daily_loss_limit_pct < 1
def test_leverage_ceiling_is_named_as_a_ceiling()  # hasattr(s, "max_leverage_ceiling")
```

**Acceptance:** `grep -rn "17_500\|17500" src/` returns nothing.

---

### S1 -- Ingest corporate actions *(CRITICAL BLOCKER)*

No new code. Run the existing command, then prove the fix with a regression test.

**Step 1 -- run the backfill** (long-running; ~34 six-month windows, 2010->today):

```bash
source .venv/Scripts/activate 2>/dev/null
python -m stocksense.cli.main backfill-corporate-actions --start 2010-01-01
```

Run it in the background. When it completes, expect roughly **30,000-35,000**
actions (the previous build recorded 34,829) and ~900 `unparsed` (rights/buyback/demerger --
a documented, deliberate gap, not a failure).

**Step 2 -- verify the data landed:**

```bash
python -c "
from stocksense.data.store import Reader
r = Reader('data_store/parquet')
print(r.sql('SELECT count(*) n FROM {corporate_actions}'))
print(r.sql('SELECT action_type, count(*) n FROM {corporate_actions} GROUP BY 1 ORDER BY 2 DESC'))
print(r.sql(\"SELECT * FROM {corporate_actions} WHERE symbol='KAMDHENU' AND ex_date >= DATE '2025-01-01' AND ex_date < DATE '2025-02-01'\"))
"
```
The KAMDHENU row must show `action_type='split'`, `factor_price=0.1`.

**Step 3 -- the regression test that must exist forever after.**

**Test file:** `tests/unit/test_no_phantom_splits.py`

```python
"""The audit found family 1 selecting 100% stock splits as its "overnight
losers" because corporate_actions was empty. This test is the tripwire that
makes that failure loud instead of silent, forever."""

import pytest
from datetime import date
import pandas as pd
from stocksense.data.store import Reader
from stocksense.data.adjust import adjusted_prices, with_prev_adjusted_close

REAL_STORE = "data_store/parquet"

@pytest.mark.skipif(not Reader(REAL_STORE).exists("corporate_actions"),
                    reason="real store not present")
def test_corporate_actions_table_is_not_empty():
    r = Reader(REAL_STORE)
    n = int(r.sql("SELECT count(*) n FROM {corporate_actions}").iloc[0]["n"])
    assert n > 20_000, f"only {n} corporate actions -- adjustment is a no-op, see audit FINDING 1"

@pytest.mark.skipif(not Reader(REAL_STORE).exists("corporate_actions"),
                    reason="real store not present")
def test_known_split_is_adjusted_away():
    # KAMDHENU 1:10 split, ex-date 2025-01-08. Raw prev_close 479.30 -> open 48.25.
    r = Reader(REAL_STORE)
    px = adjusted_prices(r, symbols=["KAMDHENU"], start=date(2024, 12, 1), end=date(2025, 2, 1))
    px = with_prev_adjusted_close(px)
    row = px[px["date"] == pd.Timestamp("2025-01-08")].iloc[0]
    overnight = row["adj_open"] / row["prev_adj_close"] - 1
    assert abs(overnight) < 0.25, f"split still reads as a {overnight:.1%} overnight move"

@pytest.mark.skipif(not Reader(REAL_STORE).exists("corporate_actions"),
                    reason="real store not present")
def test_extreme_overnight_gaps_are_rare_after_adjustment():
    """Before the fix, >0.1% of rows showed a <-40% overnight move, all of them
    splits. After adjustment this must be vanishingly rare."""
    r = Reader(REAL_STORE)
    px = adjusted_prices(r, start=date(2025, 1, 1), end=date(2025, 4, 1))
    px = with_prev_adjusted_close(px)
    px = px[px["prev_gap_sessions"] == 1]
    gap = px["adj_open"] / px["prev_adj_close"] - 1
    frac_extreme = (gap < -0.40).mean()
    assert frac_extreme < 0.0005, f"{frac_extreme:.4%} of rows gap below -40% -- splits still leaking"
```

**Acceptance:** all three pass, plus the full suite stays green. **Do not start S2 until this
passes** -- every downstream result is invalid without it.

---

### S2 -- `execution/cost_model.py` *(PROTECTED once landed)*

The gate's stated input. Cost is **not** constant in position size; the flat `10.62` in
`overnight_reversal.py` must be replaced by a call into this module.

**Statutory 2026 rates -- type exactly:**

```python
BROKERAGE_FLAT_INR   = 20.0          # per order
BROKERAGE_PCT        = 0.0003        # 0.03%, whichever is LOWER
STT_SELL_PCT         = 0.00025       # 0.025%, SELL leg only
EXCHANGE_TXN_PCT     = 0.000030699   # 0.0030699%, both legs
SEBI_TURNOVER_PCT    = 0.000001      # 0.0001%, both legs
STAMP_DUTY_BUY_PCT   = 0.00003       # 0.003%, BUY leg only
GST_PCT              = 0.18          # 18% on (brokerage + exchange txn)
```

**Formula per leg** (`turnover = price * qty`):
```
brokerage     = min(BROKERAGE_FLAT_INR, turnover * BROKERAGE_PCT)
exchange_txn  = turnover * EXCHANGE_TXN_PCT
sebi_turnover = turnover * SEBI_TURNOVER_PCT
stt           = turnover * STT_SELL_PCT     if side == "sell" else 0.0
stamp_duty    = turnover * STAMP_DUTY_BUY_PCT if side == "buy"  else 0.0
gst           = GST_PCT * (brokerage + exchange_txn)
total         = brokerage + exchange_txn + sebi_turnover + stt + stamp_duty + gst
```

**Public API:**
```python
@dataclass(frozen=True)
class Charges:
    brokerage: float; stt: float; exchange_txn: float
    sebi_turnover: float; stamp_duty: float; gst: float; total: float

def leg_charges(turnover_inr: float, side: str) -> Charges
    # side in ("buy", "sell"); raises ValueError otherwise; turnover < 0 raises

def compute_charges(position_inr: float) -> Charges
    # one full round trip (buy leg + sell leg at the same notional).
    # THIS is the function evaluation/gate.py's docstring refers to.

def round_trip_bps(position_inr: float) -> float
    # compute_charges(position_inr).total / position_inr * 10_000

def breakeven_move_bps(position_inr: float, price: float, spread_ticks: float = 2.0) -> float
    # round_trip_bps(position_inr) + spread_ticks * simulation.sizing.tick_drag_bps(price)
    # REUSE tick_drag_bps from src/stocksense/simulation/sizing.py -- do not reimplement.
```

**Test file:** `tests/unit/test_cost_model.py` -- these four expected values are **verified
arithmetic**, type them as given:

```python
def test_hundred_thousand_round_trip_matches_the_verified_charge_sheet():
    assert compute_charges(100_000).total == pytest.approx(82.645, abs=0.01)   # 8.26 bps

def test_eighty_seven_thousand_five_hundred_round_trip():
    assert compute_charges(87_500).total == pytest.approx(78.21, abs=0.02)     # 8.94 bps

def test_fifty_thousand_round_trip_is_ten_point_six_two_bps():
    assert round_trip_bps(50_000) == pytest.approx(10.62, abs=0.01)

def test_cost_in_bps_falls_as_position_size_grows():
    assert round_trip_bps(50_000) > round_trip_bps(100_000) > round_trip_bps(200_000)

def test_brokerage_is_capped_at_twenty_rupees():
    assert leg_charges(10_000_000, "buy").brokerage == 20.0

def test_brokerage_is_proportional_below_the_cap():
    assert leg_charges(10_000, "buy").brokerage == pytest.approx(3.0)

def test_stt_is_charged_on_the_sell_leg_only():
    assert leg_charges(100_000, "buy").stt == 0.0
    assert leg_charges(100_000, "sell").stt == pytest.approx(25.0)

def test_stamp_duty_is_charged_on_the_buy_leg_only():
    assert leg_charges(100_000, "sell").stamp_duty == 0.0
    assert leg_charges(100_000, "buy").stamp_duty == pytest.approx(3.0)

def test_gst_applies_to_brokerage_plus_exchange_txn_only():
    c = leg_charges(100_000, "buy")
    assert c.gst == pytest.approx(0.18 * (c.brokerage + c.exchange_txn))

def test_invalid_side_raises():
    with pytest.raises(ValueError): leg_charges(1000, "hold")

def test_negative_turnover_raises():
    with pytest.raises(ValueError): leg_charges(-1, "buy")
```

**Then wire it in.** In `src/stocksense/strategies/overnight_reversal.py`:
- Delete the `DEFAULT_CHARGES_BPS = 10.62` constant.
- Change `OvernightReversalConfig.charges_bps: float = DEFAULT_CHARGES_BPS` to
  `position_inr: float | None = None  # None -> caller must pass charges_bps to daily_pnl`.
- Keep `daily_pnl(positions, prices, charges_bps)` signature **unchanged** (it stays a pure
  function of a bps number) but update its docstring to say the number must come from
  `execution.cost_model.round_trip_bps(position_inr)`, never a literal.
- Update `tests/unit/test_overnight_reversal.py` accordingly; the existing
  `test_daily_pnl_subtracts_round_trip_charges` keeps its literal 10.62 as a *unit* test input.

**Acceptance:** full suite green; `grep -rn "10.62" src/` returns only `cost_model.py` docs.

---

### S3 -- `search/space.py`: the 10,800-config grid

**File:** `src/stocksense/search/space.py`

```python
"""The parameter space the nightly sweep enumerates.

Deterministic and ordered: the same call always yields the same configs in the
same order, so an interrupted sweep resumes exactly where it stopped and
attempts.config_hash gives a stable identity per config.

Family 1 arithmetic, stated so it can be checked:
    base  = demean(2) x winsorise(3) x side(3) x n_positions(5)
            x cap_band(4) x min_overnight_move(3)                  = 1,080
    exits = session_close(1) + trail(3) + giveback(3) + hard_stop(3) =    10
    TOTAL                                                            = 10,800
Exits are CONDITIONAL, not a full cross product: a session_close config carries
no trail_pct, so the space is 1,080 x 10 and not 1,080 x 3 x 3 x 3.
"""

FAMILY1_BASE_GRID: dict[str, tuple] = {
    "demean":             (True, False),
    "winsorise_pct":      (0.0, 0.01, 0.025),
    "side":               ("long", "short", "long_short"),
    "n_positions":        (2, 3, 5, 8, 10),
    "cap_band":           ("small", "mid", "large", "full_pit"),
    "min_overnight_move": (0.005, 0.01, 0.02),
}

FAMILY1_EXIT_VARIANTS: tuple[dict, ...] = (
    {"exit_rule": "session_close"},
    {"exit_rule": "trail",      "trail_pct":     0.010},
    {"exit_rule": "trail",      "trail_pct":     0.015},
    {"exit_rule": "trail",      "trail_pct":     0.025},
    {"exit_rule": "giveback",   "giveback_pct":  0.20},
    {"exit_rule": "giveback",   "giveback_pct":  0.30},
    {"exit_rule": "giveback",   "giveback_pct":  0.50},
    {"exit_rule": "hard_stop",  "hard_stop_pct": 0.010},
    {"exit_rule": "hard_stop",  "hard_stop_pct": 0.015},
    {"exit_rule": "hard_stop",  "hard_stop_pct": 0.020},
)

def expand_grid(base: dict[str, tuple], variants: tuple[dict, ...]) -> Iterator[dict]
    # itertools.product over sorted(base) keys -- SORTED, for determinism --
    # then for each base combo yield {**base_combo, **variant} per variant.

def family1_configs() -> Iterator[dict]
    # expand_grid(FAMILY1_BASE_GRID, FAMILY1_EXIT_VARIANTS)

def count_configs(base: dict[str, tuple], variants: tuple[dict, ...]) -> int
    # prod(len(v) for v in base.values()) * len(variants)
```

**Test file:** `tests/unit/test_search_space.py`

```python
def test_family1_yields_exactly_10800_configs()          # sum(1 for _ in family1_configs()) == 10800
def test_count_configs_agrees_with_enumeration()
def test_every_config_carries_an_exit_rule()
def test_session_close_configs_carry_no_trail_pct()      # "trail_pct" not in cfg
def test_trail_configs_carry_a_trail_pct()
def test_enumeration_order_is_deterministic()            # two calls -> identical lists
def test_every_config_is_accepted_by_OvernightReversalConfig()
    # for cfg in itertools.islice(family1_configs(), 50):
    #     OvernightReversalConfig(**cfg)   # must not raise (needs S4's exit support)
def test_config_hashes_are_unique_across_the_whole_family()
    # len({attempts.config_hash("overnight_reversal", c) for c in family1_configs()}) == 10800
```

Note the second-to-last test **will fail until S4 lands** (`OvernightReversalConfig` currently
raises for non-`session_close` exits). Write it now, mark it
`@pytest.mark.xfail(reason="exit rules land in S4", strict=True)`, and **remove the xfail in
S4** -- that is how S4 proves it did its job.

---

### S4 -- Path-dependent exits, and the honest OHLC caveat

**The constraint:** there are **no intraday bars**. `intraday_bars` is MISSING from the store.
So trail / giveback / hard_stop cannot be simulated exactly -- the intra-session price path is
unknown. Only `session_close` is exact.

**The convention -- implement exactly this, and document it in the module docstring:**

> With only daily OHLC, the order in which the high and the low were touched is unknowable.
> We resolve it **pessimistically**: assume the ADVERSE extreme is reached first. For a long,
> that is the low; for a short, the high. A stop that the low would have triggered therefore
> triggers, even if the high came first and the trade would really have been profitable.
> This biases every path-dependent exit's measured return DOWNWARD. That is the correct
> direction to be wrong in: it can produce a false FAIL, never a false PASS, and a false PASS
> is the only kind of error that costs money.

**File:** `src/stocksense/strategies/exits.py` (new)

```python
def session_close_return(side, adj_open, adj_close) -> float
    # side * (adj_close / adj_open - 1). Exact; no approximation.

def hard_stop_return(side, adj_open, adj_high, adj_low, adj_close, hard_stop_pct) -> float
    # adverse = adj_low if side == 1 else adj_high
    # if side * (adverse / adj_open - 1) <= -hard_stop_pct: return -hard_stop_pct
    # else: return session_close_return(...)

def trail_return(side, adj_open, adj_high, adj_low, adj_close, trail_pct) -> float
    # favourable = adj_high if side == 1 else adj_low
    # peak_gain  = side * (favourable / adj_open - 1)
    # if peak_gain <= 0: return hard-exit at close (no trail ever armed)
    # PESSIMISTIC ORDERING: assume the adverse extreme came first, so the trail
    # is measured from the OPEN, not from the peak, unless the peak alone clears
    # the trail distance:
    #   if peak_gain > trail_pct: return peak_gain - trail_pct
    #   else: return session_close_return(...)

def giveback_return(side, adj_open, adj_high, adj_low, adj_close, giveback_pct) -> float
    # peak_gain as above; if peak_gain <= 0: return session_close_return(...)
    # exit when the position gives back `giveback_pct` OF THE PEAK:
    #   floor = peak_gain * (1 - giveback_pct)
    #   close_gain = session_close_return(...)
    #   return floor if close_gain < floor else close_gain

def apply_exit(side, adj_open, adj_high, adj_low, adj_close, exit_rule, **params) -> float
    # dispatch table; raises ValueError on an unknown exit_rule.

EXACT_EXIT_RULES = frozenset({"session_close"})
APPROXIMATE_EXIT_RULES = frozenset({"trail", "giveback", "hard_stop"})
def is_exact(exit_rule: str) -> bool
```

**Test file:** `tests/unit/test_exits.py` -- hand-built bars with known answers:

```python
def test_session_close_long_is_close_over_open_minus_one()
def test_session_close_short_inverts_the_sign()
def test_hard_stop_fires_when_the_low_breaches_it_for_a_long()      # ret == -hard_stop_pct
def test_hard_stop_does_not_fire_when_the_low_stays_above_it()
def test_hard_stop_for_a_short_uses_the_HIGH_not_the_low()
def test_trail_returns_peak_minus_trail_when_the_peak_clears_it()
def test_trail_falls_back_to_close_when_the_peak_never_cleared_it()
def test_giveback_floors_the_return_at_peak_times_one_minus_giveback()
def test_giveback_keeps_the_close_when_the_close_beat_the_floor()
def test_unknown_exit_rule_raises()
def test_pessimistic_ordering_a_bar_that_hit_both_stop_and_target_records_the_STOP()
    # THE load-bearing test for the convention. open=100, low=98, high=105, close=104,
    # side=1, hard_stop_pct=0.015 -> must return -0.015, NOT +0.04.
def test_is_exact_only_for_session_close()
```

**Then update `OvernightReversalConfig.__post_init__`:** replace the
`if self.exit_rule != "session_close": raise` with validation against
`exits.EXACT_EXIT_RULES | exits.APPROXIMATE_EXIT_RULES`, and update
`daily_pnl` to call `exits.apply_exit` per position (it needs `adj_high`/`adj_low` from
`prices` now -- add them to the merge). Remove the `xfail` from S3's
`test_every_config_is_accepted_by_OvernightReversalConfig`.

---

### S5 -- `search/engine.py`: the vectorised sweep

This is the throughput core. Design constraints, all from measurement in Q4.0 above:

- Panel is `(n_dates x n_symbols)` float32 -- 4,117 x 7,786 ~= 32 M cells ~= **128 MB per column**.
  Restrict the universe to keep this sane (see `build_panel` below).
- **Use `np.argpartition`, never a double `argsort`** -- 6.7x faster (493 ms -> 73 ms/config),
  identical results, because we need the top N, not a fully ordered 7,786.
- **Selection stays on CPU.** Measured: CuPy 17.7 ms/config vs 0.61 min total on 10 CPU
  threads. Ranking is memory-bound; the PCIe round trip costs more than the kernel saves.
  The GPU's job is Monte Carlo, not selection. Do **not** put this on CuPy.
- **Never materialise `(n_configs x n_dates x n_symbols)`** -- at 10,800 configs that is >40 GB.
  Stream configs in batches of 50.

```python
@dataclass(frozen=True)
class Panel:
    dates:      np.ndarray   # (D,)   datetime64[D], sorted ascending, unique
    symbols:    np.ndarray   # (S,)   <U20, sorted
    adj_open:   np.ndarray   # (D, S) float32, np.nan where the symbol did not trade
    adj_high:   np.ndarray   # (D, S) float32
    adj_low:    np.ndarray   # (D, S) float32
    adj_close:  np.ndarray   # (D, S) float32
    prev_adj_close: np.ndarray  # (D, S) float32
    gap_ok:     np.ndarray   # (D, S) bool  -- prev_gap_sessions == 1
    in_universe: np.ndarray  # (D, S) bool  -- from data.universe_pit
    turnover:   np.ndarray   # (D, S) float32 -- for the cap_band split

def build_panel(reader, start: date, end: date, *, equity_inr: float,
                cap_band: str = "full_pit") -> Panel
    """One pass over adjusted_prices + with_prev_adjusted_close + universe_membership,
    pivoted to dense arrays. REUSE, do not reimplement:
        data.adjust.adjusted_prices
        data.adjust.with_prev_adjusted_close
        data.universe_pit.universe_membership
        evaluation.vault.apply_seal   <-- MANDATORY: call it on the long frame
                                          BEFORE pivoting. Without a token this
                                          drops every row on/after 2025-07-01.
    cap_band buckets by trailing-60d median turnover, computed point-in-time:
        small  = below the 33rd percentile that date
        mid    = 33rd-67th
        large  = above the 67th
        full_pit = no cap filter (the whole PIT universe)
    """

def signal_matrix(panel: Panel, demean: bool, winsorise_pct: float) -> np.ndarray
    """(D, S) float32. adj_open / prev_adj_close - 1, np.nan where not gap_ok
    or not in_universe. Winsorise per ROW (per date) with np.nanpercentile,
    then subtract the per-row nanmean if demean."""

def select_topn(signal: np.ndarray, n: int, side: str, min_move: float
                ) -> tuple[np.ndarray, np.ndarray]
    """Returns (idx, sides): idx is (D, k) int32 column indices of the selected
    symbols per date (-1 padding where fewer than n were eligible), sides is
    (D, k) int8 of +1/-1. k = n for long/short, 2n for long_short.
    Eligibility: |signal| >= min_move and not nan.
    MUST use np.argpartition(masked, kth=n)[:, :n], not argsort."""

def simulate(panel: Panel, idx: np.ndarray, sides: np.ndarray,
             exit_rule: str, exit_param: float, charges_bps: float) -> np.ndarray
    """(D,) float32 of daily equally-weighted net portfolio return.
    Only the SELECTED positions are simulated -- at most 10/day, not 7,786:
    that is the 1,000x saving that turns a 25-hour sweep into a 3-minute one.
    Numba @njit(parallel=True) over dates. NaN on dates with zero positions."""

def run_config(panel: Panel, config: dict) -> np.ndarray
    """signal_matrix -> select_topn -> simulate. Returns the (D,) daily series."""
```

**Test file:** `tests/unit/test_search_engine.py`

```python
def test_build_panel_shapes_match_dates_by_symbols()
def test_build_panel_applies_the_vault_seal()          # no date >= 2025-07-01 without a token
def test_signal_matrix_matches_the_pandas_implementation_row_for_row()
    # THE equivalence test. Build a small panel, run compute_overnight_signal from
    # strategies/overnight_reversal.py on the same data, assert allclose. If the
    # fast path and the reference path disagree, the fast path is wrong.
def test_signal_is_nan_where_gap_is_not_one_session()
def test_select_topn_long_picks_the_n_smallest_signals()
def test_select_topn_matches_a_reference_argsort_implementation()
def test_select_topn_pads_with_minus_one_when_too_few_are_eligible()
def test_select_topn_respects_min_move()
def test_simulate_session_close_matches_daily_pnl_from_the_strategy_module()
    # second equivalence test, this time end-to-end against the tested slow path
def test_simulate_subtracts_charges_once_per_position_per_day()
def test_run_config_is_deterministic()
```

The two equivalence tests are the point of this phase: `strategies/overnight_reversal.py` is
already tested and correct but slow; `search/engine.py` is fast but new. **The fast path is
only trustworthy if it reproduces the slow path exactly.** If they disagree, fix the engine.

---

### S6 -- `search/selection.py`: the six-stage cascade

Each stage is orders of magnitude cheaper than the next and kills most of what enters it.
Every rejected config records **why**.

| Stage | Test | Reject reason recorded |
|---|---|---|
| 0 | run_config -> daily returns | -- |
| 1 | ICIR screen: `evaluation.factor_metrics.icir` on in-sample dates only, reject `icir < 0.10` | `unstable_ic` |
| 2 | Decay: `factor_metrics.half_life`, reject `half_life < 3` bars | `fast_decay` |
| 3 | **Cost hurdle -- the PRIMARY filter.** Reject if mean gross per-trade edge < `cost_model.round_trip_bps(position_inr) * 1.34` (the 14.2 bps break-even at 10.62 bps cost) | `cost_drag` |
| 4 | Purged CPCV + `evaluation.gate.evaluate_gate` | `gate_fail` |
| 5 | Sealed vault, one unseal: `DSR >= 0.95` AND `PBO <= 0.5` | `vault_fail` |

```python
@dataclass(frozen=True)
class CascadeConfig:
    min_icir: float = 0.10
    min_half_life_bars: float = 3.0
    cost_hurdle_multiple: float = 1.34
    is_fraction: float = 0.7          # in-sample fraction for stages 1-3

@dataclass(frozen=True)
class CascadeResult:
    config: dict
    config_hash: str
    stage_reached: int
    verdict: str            # screened_out | gate_fail | gate_pass | vault_fail | promoted
    fail_reason: str | None # low_ic | unstable_ic | fast_decay | cost_drag | capacity | None
    metrics: dict

def screen_config(panel, config, cascade_cfg, folds) -> CascadeResult
def run_sweep(panel, configs, store, *, hypothesis_id, cascade_cfg, folds,
              max_configs=None) -> list[CascadeResult]
    """EVERY config registers via attempts.register_attempt BEFORE it is screened,
    and records its outcome via attempts.record_result -- all 10,800, not the
    survivors. That count IS n_trials in the Deflated Sharpe, so a wider sweep
    automatically raises its own bar. Registering only survivors would understate
    n_trials and silently inflate every DSR the system reports."""
```

**Test file:** `tests/unit/test_selection.py`

```python
def test_every_config_registers_even_when_screened_out_at_stage_1()
    # THE load-bearing test. 20 configs, 19 rejected at stage 1 ->
    # attempts.count_attempts(reader, hypothesis_id) == 20, not 1.
def test_low_icir_config_is_rejected_with_unstable_ic()
def test_fast_decaying_config_is_rejected_with_fast_decay()
def test_config_below_the_cost_hurdle_is_rejected_with_cost_drag()
def test_a_config_that_survives_all_screens_reaches_stage_4()
def test_stage_reached_is_recorded_for_every_result()
def test_run_sweep_is_resumable()   # re-running yields the same attempt_ids (idempotent)
def test_sweep_never_touches_sealed_dates()
```

---

### S7 -- CLI + the first real pre-registered verdict

**Add to `src/stocksense/cli/main.py`:**

```python
@app.command("sweep")
def sweep_cmd(
    family: str = typer.Option("overnight_reversal"),
    hypothesis_id: str = typer.Option(...),
    start: str = typer.Option("2010-01-01"),
    end: str = typer.Option(None),
    max_configs: int = typer.Option(None, help="cap for a smoke run"),
    equity_inr: float = typer.Option(..., help="REQUIRED. Capital is never defaulted."),
) -> None:
    """Run the nightly sweep. Reports: configs run, survivors per stage, elapsed."""

@app.command("leaderboard")
def leaderboard_cmd(hypothesis_id: str = typer.Option(None), limit: int = 25) -> None:
    """Read evaluation_attempts, print the top configs by mean_alpha_net with
    their verdicts and fail reasons. Read-only; uses Reader, never Store."""
```

**Then the first real experiment.**

1. Write `research/preregistration_overnight_reversal.md` and **`git commit` it BEFORE
   running anything.** It must fix, in advance: the signal definition; the three `side`
   variants; the cap bands; the cost model (`execution.cost_model.compute_charges`); the fold
   parameters (`CVConfig(n_folds=10, n_test_folds=2, embargo_pct=0.01)`); the iteration budget
   (10,800 configs, no more); and the pass criteria (`gate.GATE` unchanged, then DSR >= 0.95
   and PBO <= 0.5 on one vault unseal). State the India prior explicitly: post-2011 NSE shows
   positive overnight and negative intraday drift, which cuts **for** the short leg and
   **against** the long leg.
2. Smoke run: `--max-configs 50` -- confirm it completes and the attempt count is 50.
3. Full run: all 10,800.
4. **Commit the verdict unmodified, PASS or FAIL.** A FAIL is a real result and costs one
   night rather than a month; that is the entire point of the rebuild. Do not adjust a
   threshold after seeing a number -- this project committed that error once and rebuilt from
   statistical principle.

---

## Deliberately NOT in phase S, and why

- **`features/`, `labels/`** -- family 1 computes its signal inline and needs neither. They
  are prerequisites for families 2-7 (which need the 1-minute spine), not for the first verdict.
- **`intraday_bars` ingestion** -- blocked on the Upstox token (Q0.2 probe). All of phase S
  runs on daily bars by design.
- **`brokers/`, `live/`, `execution/interlocks.py`** -- blocked on the Angel One static-IP
  registration (Q0.1). Paper trading and research are unaffected.
- **`llm/`, `server/`, the Electron frontend (Q9)** -- these present results. There are no
  results to present until S7 produces one.
- **GPU Monte Carlo (`simulation/montecarlo.py`)** -- sizing, which matters after a strategy
  passes, not before.

---

## Verification -- run these, in order, to confirm phase S landed

```bash
source .venv/Scripts/activate 2>/dev/null

# 1. every test green
python -m pytest tests/ -q

# 2. the blocker is actually fixed
python -m pytest tests/unit/test_no_phantom_splits.py -v

# 3. cost model matches the verified charge sheet
python -c "from stocksense.execution.cost_model import compute_charges, round_trip_bps; \
print(compute_charges(100_000).total, round_trip_bps(50_000))"
# expect: 82.645...  10.62...

# 4. the space is the right size
python -c "from stocksense.search.space import family1_configs; print(sum(1 for _ in family1_configs()))"
# expect: 10800

# 5. fast path == slow path (the equivalence tests)
python -m pytest tests/unit/test_search_engine.py -v

# 6. smoke sweep, then check EVERY config registered
python -m stocksense.cli.main sweep --hypothesis-id smoke_v1 --max-configs 50 --equity-inr 17500
python -c "from stocksense.data.store import Reader; from stocksense.evaluation import attempts; \
print(attempts.count_attempts(Reader('data_store/parquet'), 'smoke_v1'))"
# expect: 50

# 7. the real run
python -m stocksense.cli.main sweep --hypothesis-id overnight_reversal_v1 --equity-inr 17500
python -m stocksense.cli.main leaderboard --hypothesis-id overnight_reversal_v1
```

**Phase S is done when step 7 prints a verdict -- PASS or FAIL -- that has been committed
without editing a single threshold.**
