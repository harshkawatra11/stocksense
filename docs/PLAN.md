# StockSense — clean-slate rebuild: an autonomous NSE quant research and paper-trading system

Plan version 2, written 2026-11-11 after a research pass on the real data and the real APIs.
Executors: Sonnet-class subagents, one task card at a time. In Phase 0 this file is copied into the new repo as `docs/PLAN.md` and becomes the only plan.

How to read this file: Parts A–C are background every executor reads once. Part D is the shared rulebook (conventions and data contracts). Part E is the work, as numbered task cards. Parts F–H are the failure table, verification, and the status checklist.

---

# PART A — Context

## A.1 What is being built

Throw away every previous attempt (four rebuilds, eight branches, a second worktree, all code, tests, logs, plans, instruction files, `node_modules`, `.venv`) and build one system from zero. Keep only the downloaded market data. The system must:

1. start by itself every trading morning and run all day without anyone watching it;
2. keep its data current;
3. read whatever capital is in the broker account and size trades from it;
4. find trades with a real ML/quant pipeline and record them on paper;
5. say why each stock was picked;
6. include a news-driven component;
7. test **Kronos** (the open-source foundation model for financial candlesticks, `github.com/shiyu-coder/Kronos`; this is the model the user means, not Amazon's Chronos), fine-tune it on NSE data, and use it only if it earns its place;
8. train its own models on NSE data and get better every day;
9. stop future AI sessions from being pulled back to the old code.

## A.2 Why earlier attempts failed, and what this plan does about it

Every new session read the old branches, the old plan, the old instruction files and the old memory notes, and rebuilt the same thing. Phase 0 removes every one of those channels: local git history, remote branches, the Codex worktree, Claude's project memory, any scheduled Windows task, and the old system's state tables inside `data_store`.

## A.3 Three things this plan will not pretend

- **Nobody can promise profit, and nobody profits every single day.** Professional quant funds lose on roughly 45% of days; what they have is a small positive edge after costs, repeated many times. The engineering target here is: positive expected return after all Indian trading costs, proven on data the model never saw, then proven again in forward paper trading. This plan builds the machinery that finds such an edge if it exists and refuses to fool itself if it does not. If no model passes the test, the system still runs all day, still paper-trades its best candidate (labelled unproven), and keeps learning. Real money is outside this plan.
- **FPGA / Arduino / ESP32 hardware does not make these models better.** FPGAs matter for microsecond order latency at exchange co-location. This system decides on daily and minute data. The only sensible use is an optional ESP32 status light (Phase 15).
- **No finance "skills" need installing.** The installed `finance` plugin is corporate accounting and is unrelated to trading research.

## A.4 Defaults chosen (say so at approval if any is wrong)

| Decision | Default | Why |
|---|---|---|
| Backup before wipe | Done already: `C:\Users\harsh\Desktop\HARSH\_stocksense_archive_2026-11-11\` (git bundle of all refs, zip of working tree, stash patch, env copy). No AI session is ever pointed at it. | The wipe is irreversible. Delete the folder by hand any time. |
| GitHub | Same repo `harshkawatra11/stocksense`; new orphan `main` with one commit; every other remote branch deleted. | Matches "delete every file on every branch and publish a proper commit". Residual: old commits stay reachable on GitHub only by exact SHA until garbage-collected, and old PR pages persist. Task T0.9 is the total-erasure alternative. |
| Trading mode | Paper only. No order-placement code exists anywhere; a test enforces it. | "capture them on paper". Also, since 2026-04-01 SEBI requires a whitelisted static IP and an exchange Algo-ID for API orders. |
| Strategy families | (S1) multi-day delivery book, bought near the close and sold at the open; (S2) same-day open→close book decided from the pre-open auction; (S3) event/news sleeve; (S4) minute-bar intraday, later. | Chosen from what the data showed (Part B.3). |
| LLM | None in the decision path. | User: "I do not want this." |
| Where it runs | This laptop, via Windows Task Scheduler. Laptop on and logged in on trading days; missed days are caught up automatically. | GPU and data are here. |
| Commit style | Author `Harsh Kawatra <harshkawatra11@gmail.com>`, no AI attribution lines. | The one rule carried over from the old instruction file because it is a personal preference. Say so if it should be dropped. |
| New instruction files | Phase 0 writes a new short `CLAUDE.md` and identical `AGENTS.md`. | Their only job is to tell every future session "there is no history; read `docs/PLAN.md`". |

## A.5 Things only the user can do (none blocks Phase 0–7)

1. ~~Upstox Analytics Token~~ — already present in `.env` (`STOCKSENSE_UPSTOX_ANALYTICS_TOKEN`, checked 2026-11-11). It expires a year after it was generated; renew it from the Upstox Developer Apps page → Analytics tab.
2. ~~Angel credentials~~ — present in `.env` (names mapped in T1.2).
3. Windows power settings: "When I close the lid: Do nothing" and sleep = never while plugged in.
4. Optional: a Telegram bot token and chat id for a morning and evening message.
5. Check one real Angel One contract note for the delivery brokerage actually charged (Angel's own pages disagree: ₹0 on one page, "lower of ₹20 or 0.1%, minimum ₹5" on another). T5.1 uses the costlier figure until told otherwise.

---

# PART B — Facts established on 2026-11-11

Everything in this part was observed directly (read-only) on that date. Executors rely on it and do not re-derive it.

## B.1 Repository and machine state right now

- **Phase 0 is complete (2026-11-11).** The repo contains only the fresh-start commit plus later work; there is no `.venv` (T1.1 creates it); `data_store` was reorganized per T0.6.
- Remote `origin = https://github.com/harshkawatra11/stocksense.git`; default branch `claude/loving-brahmagupta-sQm9O`; other remote branches `StockSense`, `phase0/prove-the-economics`, `research/preregistration-intraday`.
- Machine: Windows 11, 12 logical CPUs, 16 GB RAM, RTX 3050 Laptop 4 GB (driver 596.36), 93 GB free on C:.
- Pythons installed: 3.11, 3.12 (`C:\Users\harsh\AppData\Local\Programs\Python\Python312\python.exe`), 3.14. `uv 0.12.13` is installed. PyTorch CUDA wheels exist for cp312 at `https://download.pytorch.org/whl/cu128` (2.9.1).
- `gh` CLI and `hf` CLI are logged in. `.mcp.json` registers the two official NSE MCP servers.

## B.2 Data on disk (`data_store/`, 2.3 GB, git-ignored)

| Dataset under `data_store/parquet/` | Rows | Coverage | Columns |
|---|---|---|---|
| `bhavcopy_eq/part-YYYY-MM.parquet` | 8,270,877 | 2010-01-04 → 2026-09-29; 4,135 sessions; 7,918 symbols; series EQ 6.60M, BE 0.57M, SM 0.33M, GB, BZ, ST, GS, N2 … | `symbol, series, date(TIMESTAMP), open, high, low, close, prev_close, last_price, volume, turnover_inr, n_trades, era` (`legacy` ≤ 2024-07-05, `udiff` ≥ 2024-07-08); `n_trades` null for 2010 and part of 2011 |
| `bhavcopy_delivery/part-YYYY-MM.parquet` | 4,235,252 | 2019-08-23 → 2026-09-29 | `symbol, series, date, deliv_qty, deliv_pct` |
| `corporate_actions/part-all.parquet` | 35,319 | ex_date 2010-01-04 → 2026-09-18 | `symbol, ex_date(DATE), action_type, ratio_num, ratio_den, factor_price, dividend_amount, face_before, face_after, subject_raw, parse_status`; types: dividend 20,661; noise 12,106; bonus 575; split 558; buyback 351; rights 348; distribution 308; other 171; demerger 106; scheme_of_arrangement 85; bonus+split 32; 1,372 rows `unparsed` |
| `fo_bhavcopy/part-YYYY-MM.parquet` | 8,372,243 | 2024-07-08 → 2026-09-18 | `symbol, instrument_type(STO/STF/IDO/IDF), expiry_date, strike_price, option_type, date, open, high, low, close, prev_close, settle_price, oi, change_in_oi, underlying_price` |
| `india_vix/part-all.parquet` | 4,103 | 2010-01-04 → 2026-09-17 | `date(DATE), close` |
| `intraday_bars/part-YYYY-MM.parquet` | 9,434,028 | 5-minute, 470 symbols, 2024-09-13 → 2026-09-17; 75 bars per full session (09:15 … 15:25) | `symbol, ts(TIMESTAMPTZ), interval('FIVE_MINUTE'), open, high, low, close, volume` |
| `nse_snapshots/<kind>/YYYY-MM-DD.parquet` | small | 2026-09-18 → 2026-10-21; kinds fii_dii, gainers, losers, heatmap, high_52w, low_52w, market_turnover, most_active_value, most_active_volume, oi_spurts, pre_open, volume_gainers | per kind; `pre_open` has `indicative_price, final_price, buy_quantity, sell_quantity` |

Also kept: `data_store/cache/` (1.5 GB raw NSE downloads: `cm_legacy/YYYY/YYYY-MM-DD.zip`, `cm_udiff/YYYY/YYYY-MM-DD.zip`, `corpact`, `delivery`, `fo_bhavcopy`, `instruments`) and `data_store/news/` (17 days of captures).

Old-system state to archive and never read: `parquet/{evaluation_attempts, hypothesis_registry, ingest_runs, live_quote_snapshots, paper_events, paper_positions, paper_signals, vault_unseals}`; directories `_quarantine, account, experiments, market_days, market_lens, nse, operations, pytest-temp-20260929 (has a permission-locked subfolder), releases, research_snapshots, runtime, ui`; file `stocksense.duckdb`.

## B.3 What the data itself showed (development period only, ≤ 2025-09-30)

These five findings shape the design. Each was measured on the stored data; none used data after 2025-09-30.

**Finding 1 — NSE's `prev_close` is NOT adjusted for splits or bonuses.** On 1,064 split/bonus ex-dates, `prev_close` equals the raw previous close in 99.4–100% of cases. Raw exchange file confirms: KAMDHENU, 2025-01-08, 10-for-1 split: `PrvsClsgPric = 479.30`, `ClsPric = 44.50`. Consequence: returns must be adjusted from the corporate-action table. That table is imperfect: its factor agrees with the real overnight price ratio (within 10%) on 93% of splits and only 85% of bonuses; 95 split/bonus rows have no trading row on the ex-date; rights, demergers and 1,372 unparsed rows carry no factor. The fix is the hybrid "table + price evidence" algorithm in T3.2.

**Finding 2 — 19 trading sessions are missing from the stored data.** They are special sessions on weekends or holidays (budget days, Muhurat trading, disaster-recovery drills). Detected because on the following session more than 80% of symbols have `prev_close` ≠ the previous stored close. The missing dates:
`2010-02-06, 2012-01-07, 2012-03-03, 2012-04-28, 2012-09-08, 2013-05-11, 2013-11-03, 2014-03-22, 2015-02-28, 2016-10-30, 2019-10-27, 2020-02-01, 2020-11-14, 2023-11-12, 2024-01-20, 2024-03-02, 2024-05-18, 2025-02-01, 2026-02-01`.
The files exist on NSE (checked for 2020-02-01, 2024-01-20, 2025-02-01). Consequence: ingestion must try **every calendar day**, weekends included, and treat "404" as "no session" (T2.3).

**Finding 3 — Overnight returns are strongly positive and intraday returns strongly negative, every year.** Equal-weighted mean log return per day in basis points (bps), top-500 liquid stocks, 2012 → 2025-09:

| Liquidity rank | Overnight (prev close → open) | Intraday (open → close) | Total |
|---|---|---|---|
| 1–50 | +15.9 | −13.0 | +2.8 |
| 51–100 | +16.8 | −13.2 | +3.6 |
| 101–200 | +19.5 | −16.1 | +3.4 |
| 201–350 | +24.8 | −21.2 | +3.6 |
| 351–500 | +29.6 | −25.3 | +4.3 |

For the top 100 the overnight figure was positive and the intraday figure negative in each of the 14 years. The independent 5-minute bars (470 symbols, 2024-09 → 2026-09) agree: overnight +14.4, open→close −11.3, of which −4.9 falls in the first five minutes (09:15–09:20), −1.4 in 09:20–09:30, −2.4 in 09:30–10:00, −0.5 in 10:00–15:00, −2.2 in 15:00–15:30.
Consequences: (a) a multi-day position must be **bought near the close, never at the open**: a 5-session holding entered at the close averaged +35 bps against +15 bps when entered at the open, a 20 bps difference per trade for free; (b) sell at the open; (c) the pattern survives because capturing it directly costs more than it pays (delivery tax alone is 20 bps a round trip).

**Finding 4 — The opening-gap fade is large at the opening price and nearly gone five minutes later.** Top-200 liquid stocks, next-day open→close return by decile of the opening gap: bottom decile +30 bps, rising smoothly to −54 bps for the top decile. Shorting the top decile at the open and covering at the close: gross +40 to +58 bps per day in each year 2014–2025, positive on 60–69% of days. But on 5-minute bars (2024-09 → 2025-09): entry at the opening price +50.7 bps; entry at 09:20 +11.0 bps; entry at 09:30 +8.3 bps. An intraday round trip costs roughly 15–20 bps. So this edge exists only for orders filled in the opening auction itself. Consequence: strategy S2 decides from the pre-open auction's indicative prices at 09:07 and is tracked in two books forever: `auction` (filled at the official open) and `auction_strict` (filled at the first quote after 09:16). The gap between those two books is the real answer and only forward data can give it.

**Finding 5 — Classic signals do carry information on NSE.** Mean daily rank IC × 100 (t-statistic), top-500 with a ₹1 crore turnover floor, 2012 → 2025-09. Labels: `oc1` = next day open→close; `c2o5` = next day's close → open five sessions later; `c2o20` likewise 20 sessions.

| Signal (value on day t) | oc1 | c2o5 | c2o20 |
|---|---|---|---|
| low 21-day volatility | 6.18 (21.3) | 2.82 (8.8) | 3.68 (11.9) |
| low max daily return, 21 days | 5.13 (22.2) | 2.31 (9.4) | 2.76 (11.7) |
| low sum of last 5 overnight returns | 5.43 (38.4) | −0.20 | 0.43 |
| small opening gap on day t | 4.56 (32.0) | −0.37 | 0.22 |
| delivery-share z-score (2019+) | 3.12 (17.3) | 2.18 (12.4) | 1.51 (9.1) |
| 5-day reversal | 1.81 (8.6) | 2.83 (13.8) | 0.84 (4.2) |
| low sum of last 5 intraday returns | −0.55 | 2.91 (14.9) | 0.59 (3.1) |
| 12-month momentum skipping last month | 0.10 | 2.44 (9.3) | 4.11 (14.1) |
| 6-month momentum | −0.36 | 1.53 (6.2) | 3.12 (12.0) |
| close near the day's high | 2.00 (13.0) | −1.36 (−9.2) | −0.06 |
| low turnover shock | 1.90 (12.3) | 1.16 (7.9) | −0.32 |
| 1-month momentum | −0.84 | −0.74 | 0.17 |

Top-decile 12-month momentum earned 122 bps per 20 sessions gross against 62 for the universe and −67 for the bottom decile. These are gross, long-known effects, not a finished strategy; they show the raw material is there. Honesty note: this exploration counts as prior trials; the trial registry is seeded with 30 trials (T6.3) so later statistics are deflated for it.

Other data facts: the liquid universe was thin before 2014 (stocks with median daily turnover ≥ ₹1 crore: 288–347 in 2011–2013, 486+ from 2014, 1,254 in 2025; ≥ ₹5 crore: 163–256 until 2016, 852 in 2025). The stored 5-minute bars appear to be split-adjusted as of their download date (bar opens match bhavcopy opens at the median but differ hugely on average); T2.12 verifies and handles this. The opening 5-minute bar carries about 4–5% of the day's volume.

## B.4 External facts (checked live 2026-11-11)

**NSE archive files (plain GET with a browser User-Agent and `Referer: https://www.nseindia.com/`; all returned 200 unless noted):**

| What | URL pattern | Notes |
|---|---|---|
| Equity bhavcopy, UDiFF (2024-07-08 →) | `https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip` | header: `TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1..4`; 404 on non-sessions |
| Equity bhavcopy, legacy (→ 2024-07-05) | `https://nsearchives.nseindia.com/content/historical/EQUITIES/YYYY/MON/cmDDMONYYYYbhav.csv.zip` (`MON` = `JAN`…`DEC`) | works for 2020-02-01; header `SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN` |
| Full bhavcopy with delivery | `https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_DDMMYYYY.csv` | header (note leading spaces): `SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER`; `-` for missing delivery. **Trap: requesting a holiday (10-11-2026) returned the previous session's file (rows dated 09-Nov-2026). Always check `DATE1` equals the requested date.** |
| F&O bhavcopy, UDiFF | `https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip` | same header family |
| All indices, daily | `https://nsearchives.nseindia.com/content/indices/ind_close_all_DDMMYYYY.csv` | header `Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,Closing Index Value,Points Change,Change(%),Volume,Turnover (Rs. Cr.),P/E,P/B,Div Yield`; exists for 05-01-2015, 404 for 02-01-2012 (true start is between); old names differ (`CNX Nifty` = `Nifty 50`) |
| Symbol changes | `https://nsearchives.nseindia.com/content/equities/symbolchange.csv` | header `NAME OF THE COMPANY,OLD SYMBOL,NEW SYMBOL,EFFECTIVE DATE` (date `30-OCT-2019`) |
| Listed equities | `https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv` | `SYMBOL,NAME OF COMPANY, SERIES, DATE OF LISTING, PAID UP VALUE, MARKET LOT, ISIN NUMBER, FACE VALUE` |
| ETF list | `https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv` | `Symbol,Underlying Asset,SecurityName,DateofListing,…` |
| Sector map | `https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv`, `…/ind_niftytotalmarket_list.csv` | `Company Name,Industry,Symbol,Series,ISIN Code` |
| Price bands (circuit limits), today | `https://nsearchives.nseindia.com/content/equities/sec_list.csv` | `Symbol,Series,Security Name,Band,Remarks` |
| Participant-wise F&O open interest | `https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_DDMMYYYY.csv` | Client / DII / FII / Pro rows |
| F&O lot sizes (also = current F&O stock list) | `https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv` | |
| Daily volatility file | `https://nsearchives.nseindia.com/archives/nsccl/volt/CMVOLT_DDMMYYYY.CSV` | |
| Bulk / block deals, today | `https://nsearchives.nseindia.com/content/equities/bulk.csv`, `…/block.csv` | `Date,Symbol,Security Name,Client Name,Buy/Sell,Quantity Traded,Trade Price / Wght. Avg. Price,Remarks` |

**NSE JSON APIs (`https://www.nseindia.com/api/...`; returned 200 with browser headers even though the home page returned 403 to the cookie warm-up):**

| What | Endpoint | Fields seen |
|---|---|---|
| Holidays | `/api/holiday-master?type=trading` | object keyed by segment (`CM`, `CBM`, …): `tradingDate` (`26-Jan-2026`), `weekDay`, `description` |
| Announcements | `/api/corporate-announcements?index=equities&from_date=DD-MM-YYYY&to_date=DD-MM-YYYY` | `symbol, desc` (category), `an_dt` (`09-Nov-2026 23:45:59`), `attchmntText`, `attchmntFile`, `seq_id`, `sm_name`, `sm_isin`, `exchdisstime`, `sort_date`, `hasXbrl`. **History confirmed back to January 2011** (149 rows for 3 days in 2011; 315 in 2015; 1,938 for 4 days in 2026) |
| Corporate actions | `/api/corporates-corporateActions?index=equities&from_date=…&to_date=…` | `symbol, series, exDate` (`02-Nov-2026`), `recDate, subject, faceVal, comp, isin` |
| Board meetings | `/api/corporate-board-meetings?index=equities&from_date=…&to_date=…` | `bm_symbol, bm_date, bm_purpose, bm_desc, bm_timestamp, sm_isin` |
| Financial results index | `/api/corporates-financial-results?index=equities&period=Quarterly` | `symbol, broadCastDate, filingDate, exchdisstime, fromDate, toDate, audited, consolidated, relatingTo, xbrl` (URL), `seqNumber` |
| FII/DII cash flows, latest day | `/api/fiidiiTradeReact` | `category, date, buyValue, sellValue, netValue` |

**Kronos** (`shiyu-coder/Kronos`, MIT, 47.5k stars, AAAI 2026, arXiv 2508.02739, last push 2026-04-13). Decoder-only Transformer pre-trained on 12+ billion candles from 45+ exchanges. A tokenizer turns each bar (open, high, low, close, volume, amount) into two discrete tokens (`s1`, `s2`); the model predicts the next bar's tokens. Weights on Hugging Face under `NeoQuasar/`: `Kronos-mini` (4.1M parameters, 16 MB, context 2048, tokenizer `Kronos-Tokenizer-2k`), `Kronos-small` (24.7M, 99 MB, context 512, tokenizer `Kronos-Tokenizer-base`), `Kronos-base` (102.3M, 409 MB, context 512, same tokenizer). Not a pip package: code is the repo's `model/` folder (`__init__.py`, `kronos.py`, `module.py`), needing `torch`, `einops`, `huggingface_hub`, `safetensors`, `pandas`, `numpy`, `tqdm`. Its `requirements.txt` pins `pandas==2.2.2` and must not be installed. Read from the source:
- `KronosPredictor(model, tokenizer, device=None, max_context=512, clip=5)`; `predict(df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True)`; `predict_batch(df_list, x_timestamp_list, y_timestamp_list, pred_len, …)` (all series same length).
- Pre-processing inside `predict`: columns `[open, high, low, close, volume, amount]` as float32; per-series mean and std over the lookback rows; `x = (x − mean)/(std + 1e-5)`; clip to ±5; time features from timestamps in this order: `minute, hour, weekday, day, month`.
- `auto_regressive_inference(tokenizer, model, x, x_stamp, y_stamp, max_context, pred_len, clip, T, top_k, top_p, sample_count)`: repeats each series `sample_count` times, `tokenizer.encode(x, half=True)` → `(s1_ids, s2_ids)`, loops `pred_len` steps calling `model.decode_s1(s1, s2, stamp)` then `model.decode_s2(context, sampled_s1)`, samples with temperature/top-p, then `tokenizer.decode([s1, s2], half=True)`, reshapes to `[n_series, sample_count, seq, 6]` and **returns the mean over `sample_count`** (`np.mean(preds, axis=1)`).
- Fine-tuning recipe in `finetune/train_predictor.py` and `finetune/config.py`: window = `lookback_window (90) + predict_window (10) + 1` bars; normalize the whole window with the mean/std of the first `lookback_window` rows, clip ±5; `token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)` under `no_grad`; inputs `[:, :-1]`, targets `[:, 1:]`; `logits = model(token_in0, token_in1, batch_x_stamp[:, :-1, :])`; `loss, s1_loss, s2_loss = model.head.compute_loss(logits[0], logits[1], token_out0, token_out1)`; AdamW lr 4e-5, betas (0.9, 0.95), weight decay 0.1; `OneCycleLR(max_lr=4e-5, pct_start=0.03, div_factor=10)`; gradient clip 3.0; batch 50; 30 epochs of 2,000 steps; tokenizer lr 2e-4; their inference settings `T=0.6, top_p=0.9, sample_count=5`. The scripts need Qlib and multi-GPU `torchrun` and are demos by the authors' own disclaimer.
- The repo ships a regression test: `tests/test_kronos_regression.py` with `tests/data/regression_input.csv`, `regression_output_256.csv`, `regression_output_512.csv`.
- Independent evidence is thin: the paper reports large rank-IC gains over general time-series models; one 2026 study found no significant accuracy difference between Kronos and a small trained network; no independent trading results exist. Treat as an open question to test.

**Upstox (read from the official docs):** historical candles V3 `GET https://api.upstox.com/v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}` with `Authorization: Bearer <token>`; `unit` ∈ minutes (1–300), hours (1–5), days, weeks, months; minute and hour data from January 2022; 1–15-minute intervals: at most 1 month per request; >15 minutes: 1 quarter; days: from January 2000, 1 decade per request; candle array `[timestamp, open, high, low, close, volume, open_interest]`; error codes UDAPI1147 (bad interval), UDAPI1148 (bad date range). Rate limits: 50/second, 500/minute, 2,000 per 30 minutes. The **Analytics Token** is long-lived (1 year), read-only, needs no OAuth redirect and no static IP, and covers historical data and market quotes.

**Angel One SmartAPI:** the docs site is client-rendered and could not be read by tools. Known from the forum: `getCandleData` documented at 3 requests/second, with widespread reports of spurious `AB1021 Too many requests`. Everything else about Angel is a probe task (T2.11) with a defined fallback.

**Charges (equity cash segment, 2026):** STT delivery 0.1% on buy and on sell; STT intraday 0.025% on sell only. Stamp duty on buy only: 0.015% delivery, 0.003% intraday. NSE transaction charge 0.00297% of turnover, both sides. SEBI fee ₹10 per crore (0.0001%), both sides. GST 18% on brokerage + exchange + SEBI fee. DP charge ₹20 + GST per scrip per sell day, delivery only. Angel One brokerage (their knowledge-centre page): lower of ₹20 or 0.1% per executed order, minimum ₹5, for both delivery and intraday.

**SEBI retail algo framework:** mandatory from 2026-04-01. API orders only from a whitelisted static IP, each carrying an exchange Algo-ID. Affects real orders only.

---

# PART C — How this plan is executed

## C.1 Roles

- **Orchestrator**: the main session. It reads this whole file once, then works down the task list in Part E. For each task it dispatches one Sonnet subagent with the prompt template in C.3, waits for its report, runs the acceptance command itself, and only then marks the task done in Part H and moves on. It never does a task's implementation itself unless a subagent fails twice.
- **Executor**: a Sonnet subagent. It implements exactly one task card.

## C.2 Order and parallelism

Tasks are numbered `T<phase>.<n>`. Run phases in the order of the checklist in Part H. Inside a phase, run tasks in number order unless the card says `Parallel-safe: yes`, in which case the orchestrator may run it alongside the other parallel-safe cards of the same phase (each in its own git worktree, merged one at a time with the suite re-run after each merge). Never run two tasks that write the same file at once.

## C.3 Executor prompt template (the orchestrator fills the braces)

```
You are implementing task {TASK_ID} of the StockSense project.
Working directory: C:\Users\harsh\Desktop\HARSH\WEB-DEV-PROJECTS\stocksense
1. Read docs/PLAN.md Part D completely (conventions and contracts). Read the task card {TASK_ID} in Part E completely. Read the cards it lists under "Depends on" only for their "Provides" lines.
2. Follow the card exactly. Write the tests listed first, run them, confirm they fail because the code is missing, then write the code.
3. Do not look for any older code, branch, archive or data_store/_archive_v1. Do not change config/gate.toml. Do not write any function that places, modifies or cancels a broker order.
4. If a fact in the plan turns out false when you test it (URL, column name, API field), record the observed fact in docs/DECISIONS.md under the task id, adapt the smallest piece of code, and continue.
5. Finish with: the acceptance command from the card and its real output; `python -m pytest tests -q` and its real last line; `git status --short`. If something fails and you cannot fix it in two attempts, stop and report exactly what fails and what you tried.
6. Commit with message "{TASK_ID}: {TITLE}". Author is already configured. Add no attribution lines.
```

## C.4 Task card format

Every card has: **Goal**, **Depends on**, **Provides** (the public functions and files later tasks use), **Files**, **Spec**, **Edge cases** (each one becomes a test), **Tests** (named, with concrete numbers), **Acceptance** (one command and what it must print), **Parallel-safe**.

## C.5 The ten rules (copied verbatim into `CLAUDE.md` and `AGENTS.md` in T0.7)

1. **There is no history.** The repository begins at the first commit on `main`. Do not search for, fetch, or ask about earlier branches, bundles, archives, or `data_store/_archive_v1/`. If `docs/PLAN.md` does not answer a design question, decide from first principles, write the decision in `docs/DECISIONS.md`, and continue.
2. **Paper only.** No function that places, modifies or cancels a broker order may exist in `src/`. `tests/unit/test_no_orders.py` enforces it.
3. **No look-ahead.** Every feature passes the truncation test. Every model is fitted only on rows whose label was fully known before the first prediction date, with purge and embargo.
4. **The gate is written before the models.** `config/gate.toml` is committed in Phase 1 and never edited after a result has been seen. A fail is a result; record it and move on.
5. **Test first.** Write the test, see it fail for the stated reason, then write the code. Whole suite green before every commit.
6. **Capital is never a constant.** It is read from the broker at decision time and passed as an argument. Backtests run on a capital grid and store percentages.
7. **Every job is idempotent and keyed by (job name, trading date).**
8. **Secrets live only in `.env`.** Never print, log or commit one.
9. **When a probe contradicts the plan,** record the fact in `docs/DECISIONS.md`, adapt the smallest piece, keep going. Do not redesign.
10. **One commit per task,** message `T<phase>.<n>: <title>`. Push after each phase passes.

---

# PART D — Shared rulebook: conventions and contracts

Every executor reads this part before its task card. If a card and this part disagree, the card wins and the executor notes it in `docs/DECISIONS.md`.

## D.1 Environment and commands

- Shell: PowerShell. Python: 3.12 in `.venv`, managed with `uv`. Always run Python as `.venv\Scripts\python.exe` (never bare `python`).
- Tests: `.venv\Scripts\python.exe -m pytest tests -q`. One file: add the path. One test: `-k name`.
- Lint: `.venv\Scripts\python.exe -m ruff check src tests` (must be clean before commit); format: `ruff format src tests`.
- CLI: `.venv\Scripts\python.exe -m stocksense <command>`.
- Environment variables set for every run (in `scripts/*.ps1`, the scheduled task, and `tests/conftest.py`): `PYTHONUTF8=1`, `POLARS_MAX_THREADS=8`, `OMP_NUM_THREADS=8`, `TOKENIZERS_PARALLELISM=false`, `HF_HUB_DISABLE_TELEMETRY=1`.
- Never install a package without adding it to `pyproject.toml`. Never `pip install -r` a third-party requirements file.

## D.2 Code conventions

- `src/` layout, package `stocksense`. Python ≥ 3.12 syntax. Full type hints on public functions. `from __future__ import annotations` at the top of every module.
- **polars** for all tabular work (lazy where the data is large). **pandas** appears only inside `models/kronos_model.py` (Kronos needs it) and nowhere else; a test greps for `import pandas` outside that file and `third_party/`.
- **DuckDB** only for ad-hoc SQL over parquet inside `data/status.py` and research scripts. **SQLite** for all mutable state.
- No global mutable state. Anything needing "now" takes a `Clock`. Anything needing settings takes a `Settings`. Anything doing HTTP takes a client object. This is what makes everything testable without the network.
- Errors: define `StockSenseError(Exception)` in `core/errors.py` with subclasses `DataNotAvailable` (expected, e.g. file not yet published), `DataValidationError`, `BrokerError`, `ConfigError`, `LookaheadError`, `LockboxError`. Never `except Exception: pass`. Catch broad exceptions only at the job boundary in the daemon, where they are logged with traceback and recorded.
- Logging: `log = get_logger(__name__)`; structured: `log.info("ingest_done", dataset="bhavcopy_eq", date=str(d), rows=n)`. Never log a secret (the logger redacts by key name; still do not pass them).
- Floats: prices and returns are `Float64`. Never compare floats with `==` in logic; use tolerances given in the cards.
- File I/O: always `encoding="utf-8"`; all parquet writes go through `data/io.py` (atomic).
- Random: every function that uses randomness takes a `seed` argument; tests fix it.
- Public function docstrings: one line saying what it returns and the units (fraction vs percent vs bps). **Returns are fractions everywhere in code (0.01 = 1%).** Basis points appear only in config values whose key ends in `_bps` and in report text.

## D.3 Time and identity

- Time zone: `Asia/Kolkata` (IST) everywhere. `date` columns are IST session dates with polars dtype `Date`. `ts` columns are `Datetime("us", "Asia/Kolkata")`. The raw adopted `bhavcopy_eq`, `bhavcopy_delivery`, `fo_bhavcopy` store `date` as a naive timestamp; `data/io.read` casts any `date` column to `Date` on read, so no other code ever sees a timestamp there.
- **Session**: a date on which NSE's cash market traded. Sessions can fall on weekends or holidays (special sessions).
- **Decision date `t`**: the session whose end-of-day data is the newest information used. Features for `t` may use anything published by 20:00 IST on `t`.
- **Entry points** (used in labels, backtest, and paper broker): `O(d)` = official open of session `d`; `C(d)` = official close of session `d`.
- `sid` (Int32): the project's permanent id for a company, stable across symbol renames (T3.1). `symbol` is the NSE ticker valid on that date.
- Ids: `decision_id = sha1(f"{book}|{strategy}|{date}")[:16]`; `order_id = sha1(f"{decision_id}|{sid}|{side}|{when}")[:16]`; `model_id = f"{family}-{target}-{spec_hash8}-{train_end:%Y%m%d}"`; `spec_hash8 = sha1(canonical_json(spec))[:8]` where canonical JSON has sorted keys and no whitespace.

## D.4 Directory layout

Repository (created in Phase 1):

```
CLAUDE.md  AGENTS.md  README.md  pyproject.toml  uv.lock  .gitignore  .env.example  .mcp.json
config/    default.toml  universe.toml  costs.toml  gate.toml  strategies.toml  risk.toml  schedule.toml
docs/      PLAN.md  DECISIONS.md  DATA.md  RUNBOOK.md  API.md  RESULTS-G1.md (later)
scripts/   install-task.ps1  uninstall-task.ps1  run-daemon.ps1
src/stocksense/
  __init__.py  __main__.py  cli.py
  core/        config.py clock.py calendar.py paths.py log.py state.py errors.py ids.py
  data/        catalog.py io.py nse_http.py status.py validate.py
               ingest_eq.py ingest_fo.py ingest_index.py ingest_corpact.py ingest_reference.py
               ingest_announcements.py ingest_intraday.py ingest_macro.py ingest_misc.py ingest_all.py
  panel/       security_master.py adjust.py returns.py universe.py market.py
  features/    base.py registry.py price.py volume.py delivery.py risk.py overnight.py derivs.py market.py
               calendar_feats.py events.py intraday.py kronos_feats.py build.py
  labels/      forward.py
  costs/       india.py slippage.py
  backtest/    engine.py fills.py metrics.py
  validation/  splits.py trials.py dsr.py pbo.py gate.py lockbox.py walkforward.py
  models/      base.py linear.py gbm.py seqnet.py kronos_model.py kronos_finetune.py ensemble.py meta.py store.py
  news/        sources.py entity.py taxonomy.py textmodel.py eventstudy.py signals.py
  portfolio/   construct.py risk.py
  brokers/     angel.py upstox.py quotes.py instruments.py
  paper/       ledger.py broker.py
  live/        daemon.py jobs.py schedule.py preflight.py keepawake.py notify.py
  learn/       nightly.py promote.py drift.py alpha_search.py
  explain/     decision.py report.py labels_text.py
  api/         app.py
tests/         conftest.py  unit/  integration/  fixtures/synth.py  fixtures/files/
third_party/kronos/   __init__.py  model/  LICENSE  COMMIT  tests/
```

Data (`data_store/`, never committed):

```
parquet/   RAW, adopted:  bhavcopy_eq/ bhavcopy_delivery/ corporate_actions/ fo_bhavcopy/ india_vix/ intraday_bars/ nse_snapshots/
           RAW, new:      index_daily/ symbol_changes/ listed_equities/ etf_list/ sectors/ price_bands/ holidays/
                          announcements/ board_meetings/ results_index/ macro_daily/ participant_oi/ fii_dii/
                          deals/ spreads/ preopen/ quotes_1m/ bars_1m/ news_items/
cache/     raw downloads exactly as fetched (kept forever for bhavcopy; 400 days for the rest)
news/      raw news captures
panel/     security_master.parquet  daily/part-YYYY.parquet  universe/part-YYYY.parquet  market.parquet  adj_events.parquet
features/  v<N>/part-YYYY.parquet  v<N>/manifest.json   kronos/<tag>/YYYY-MM-DD.parquet   seq/v<N>/…npy
labels/    v<N>/part-YYYY.parquet
models/    <model_id>/…
preds/     <model_id>/part-YYYY.parquet
state/     stocksense.sqlite  lockbox_touches.jsonl  heartbeat.json  daemon.lock  gpu.lock
reports/   YYYY-MM-DD.md  YYYY-MM-DD.json
logs/      daemon-YYYY-MM-DD.log
_archive_v1/   old-system state. Never read. A test fails if any source file contains the string "_archive_v1".
```

## D.5 Dataset contracts

Types are polars dtypes. "Key" = uniqueness key enforced on write. Partitioning: `month` = `part-YYYY-MM.parquet` by the date column; `year` = `part-YYYY.parquet`; `all` = `part-all.parquet`; `day` = `YYYY-MM-DD.parquet`.

**RAW adopted (schemas as stored; do not rewrite history files except by upsert of new rows):**

| Dataset | Key | Partition | Columns |
|---|---|---|---|
| `bhavcopy_eq` | symbol, series, date | month | symbol Utf8, series Utf8, date Date*, open, high, low, close, prev_close, last_price Float64, volume Int64, turnover_inr Float64, n_trades Float64, era Utf8; new nullable: isin Utf8, name Utf8, avg_price Float64 |
| `bhavcopy_delivery` | symbol, series, date | month | symbol, series, date Date*, deliv_qty Float64, deliv_pct Float64 (0–100) |
| `corporate_actions` | symbol, ex_date, subject_raw | all | as in B.2; new nullable: record_date Date, series Utf8, isin Utf8, source Utf8, rights_price Float64 |
| `fo_bhavcopy` | symbol, instrument_type, expiry_date, strike_price, option_type, date | month | as in B.2 (strike null and option_type null for futures) |
| `india_vix` | date | all | date Date, close Float64 |
| `intraday_bars` | symbol, ts, interval | month | as in B.2; new nullable: source Utf8, adj_asof Date |

(*stored as timestamp in old partitions; cast on read.)

**RAW new:**

| Dataset | Key | Partition | Columns |
|---|---|---|---|
| `index_daily` | index_name, date | year | index_name Utf8 (canonical upper-case, see T2.6 map), date, open, high, low, close Float64, pe, pb, div_yield Float64, volume Float64, turnover_cr Float64 |
| `symbol_changes` | old_symbol, new_symbol, change_date | all | company Utf8, old_symbol, new_symbol Utf8, change_date Date |
| `listed_equities` | symbol, as_of | all | symbol, name, series Utf8, listing_date Date, isin Utf8, face_value Float64, as_of Date |
| `etf_list` | symbol, as_of | all | symbol, underlying, name Utf8, as_of Date |
| `sectors` | symbol, as_of | all | symbol, industry, company Utf8, isin Utf8, source Utf8, as_of Date |
| `price_bands` | symbol, series, date | month | symbol, series Utf8, band Utf8 (e.g. `2`,`5`,`10`,`20`,`No Band`), date Date |
| `holidays` | date | all | date Date, description Utf8, fetched_at Datetime |
| `announcements` | seq_id | month (by ts) | seq_id Utf8, symbol Utf8, ts Datetime (from `an_dt`), exch_ts Datetime nullable (from `exchdisstime`), category Utf8 (from `desc`), summary Utf8 (from `attchmntText`), attachment_url Utf8, company Utf8, isin Utf8, has_xbrl Boolean, first_seen_ts Datetime nullable (set only in live mode) |
| `board_meetings` | symbol, meeting_date, announced_ts | all | symbol Utf8, meeting_date Date, purpose Utf8, description Utf8, announced_ts Datetime |
| `results_index` | seq_number | all | seq_number Utf8, symbol Utf8, period_from Date, period_to Date, filing_ts Datetime, exch_ts Datetime nullable, audited Utf8, consolidated Utf8, relating_to Utf8, xbrl_url Utf8 |
| `macro_daily` | series, date | all | series Utf8, date Date, close Float64, available_ts Datetime |
| `participant_oi` | date, client_type | year | date, client_type Utf8, fut_idx_long, fut_idx_short, fut_stk_long, fut_stk_short, opt_idx_call_long, opt_idx_put_long, opt_idx_call_short, opt_idx_put_short Float64 (plus stock-option columns as present) |
| `fii_dii` | date, category | all | date, category Utf8, buy_value, sell_value, net_value Float64 |
| `deals` | date, symbol, client, side, kind, qty, price | year | date, kind Utf8 (`bulk`/`block`), symbol, client Utf8, side Utf8, qty Float64, price Float64 |
| `spreads` | symbol, ts | month | symbol Utf8, ts Datetime, bid, ask, ltp Float64, bid_qty, ask_qty Float64, source Utf8 |
| `preopen` | symbol, ts | month | symbol Utf8, ts Datetime, prev_close, indicative_price, final_price Float64 nullable, buy_qty, sell_qty Float64, source Utf8 |
| `quotes_1m` | symbol, ts | day | symbol Utf8, ts Datetime, ltp, open, high, low, close_prev Float64, volume Float64, bid, ask Float64 nullable, source Utf8 |
| `bars_1m` | symbol, ts | month | symbol Utf8, ts Datetime, open, high, low, close Float64, volume Int64, source Utf8, adj_asof Date |
| `news_items` | source, raw_id | month (by ts_seen) | source Utf8, raw_id Utf8, ts_published Datetime nullable, ts_seen Datetime, title Utf8, summary Utf8, url Utf8 |

**DERIVED:**

`panel/security_master.parquet` — key (sid, valid_from): `sid Int32, symbol Utf8, valid_from Date, valid_to Date, isin_last Utf8, name_last Utf8, first_date Date, last_date Date`.

`panel/daily/part-YYYY.parquet` — key (sid, date). One row per company per session, series EQ/BE/BZ only:
`sid Int32, date Date, symbol Utf8, series Utf8, open, high, low, close, prev_close Float64, volume Int64, turnover_inr Float64, n_trades Float64, deliv_qty Float64, deliv_pct Float64, gap_factor Float64, gap_src Utf8, adj Float64, adj_open, adj_high, adj_low, adj_close Float64, adj_volume Float64, ret, ret_co, ret_oc, div_ret, tret Float64, sessions_since_prev Int32, flag_bigmove, flag_unexplained_gap, flag_stale, flag_one_price, flag_locked_up, flag_locked_dn Boolean, industry Utf8`.

`panel/universe/part-YYYY.parquet` — key (date, sid): `date, sid, turnover_rank Int32, med_turn_60 Float64, u500 Boolean, u200 Boolean, in_fno Boolean`.

`panel/market.parquet` — key (date): `date, ew500_ret, ew500_co, ew500_oc, ew200_ret, nifty50_ret, nifty500_ret, n_u500 Int32, breadth_200 Float64, xs_dispersion Float64`.

`panel/adj_events.parquet` — audit of every adjustment applied: `sid, date, symbol, gap_factor, gap_src, r_obs, f_ca, ca_types Utf8, note Utf8`.

`features/v<N>/part-YYYY.parquet` — key (date, sid): `date, sid`, then for each cross-sectional feature `<name>` (rank-scaled Float32 in [−0.5, 0.5]) and `<name>__raw` (Float32); market-state features under their own names (Float32, same value for all sids on a date). `manifest.json`: `{"version": N, "created": iso, "git_sha": str, "features": [{"name","group","kind":"xs"|"mkt","version":int,"first_valid_date":"YYYY-MM-DD","coverage":float,"label":"plain-English text"}]}`.

`labels/v<N>/part-YYYY.parquet` — key (date, sid): `date, sid, y_oc, y_c2o_5, y_c2o_10, y_c2o_20, y_c2c_1, y_o2o_5 Float64 (raw fractions, null when undefined), end_oc, end_5, end_10, end_20 Date (when each label became known), entry_blocked_open, entry_blocked_close Boolean, label_delisted Boolean`, and for every `y_*`: `z_<name>` Float32 (cross-sectional robust z, clipped ±5) and `r_<name>` Float32 (rank in [−0.5, 0.5]).

`preds/<model_id>/part-YYYY.parquet` — key (date, sid): `date, sid, score Float32, made_at Datetime, mode Utf8 ('walkforward'|'live')`.

## D.6 SQLite schema (`data_store/state/stocksense.sqlite`)

Created by numbered migrations in `core/state.py`. `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=30000; PRAGMA foreign_keys=ON`. All timestamps ISO-8601 text with `+05:30`.

```sql
-- migration 1 (T1.5)
CREATE TABLE job_runs(job TEXT, trade_date TEXT, attempt INTEGER, status TEXT CHECK(status IN('running','done','failed','skipped')),
  started_at TEXT, finished_at TEXT, detail TEXT, PRIMARY KEY(job, trade_date, attempt));
CREATE TABLE kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);
-- migration 2 (T6.3)
CREATE TABLE trials(trial_id TEXT PRIMARY KEY, created_at TEXT, family TEXT, target TEXT, spec_json TEXT, spec_hash TEXT,
  status TEXT, metrics_json TEXT, returns_path TEXT, git_sha TEXT, note TEXT);
-- migration 3 (T7.7)
CREATE TABLE models(model_id TEXT PRIMARY KEY, family TEXT, target TEXT, spec_hash TEXT, train_end TEXT, created_at TEXT,
  status TEXT CHECK(status IN('candidate','shadow','champion','retired')), validated INTEGER DEFAULT 0, path TEXT, note TEXT);
CREATE TABLE model_events(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, model_id TEXT, event TEXT, detail TEXT);
CREATE TABLE live_ic(model_id TEXT, decision_date TEXT, target TEXT, ic REAL, n INTEGER, PRIMARY KEY(model_id, decision_date, target));
-- migration 4 (T10.1)
CREATE TABLE decisions(decision_id TEXT PRIMARY KEY, book TEXT, strategy TEXT, trade_date TEXT, created_ts TEXT,
  capital_ref_inr REAL, capital_stale INTEGER, risk_json TEXT, payload_json TEXT);
CREATE TABLE orders(order_id TEXT PRIMARY KEY, decision_id TEXT, book TEXT, sid INTEGER, symbol TEXT, side TEXT CHECK(side IN('BUY','SELL')),
  product TEXT CHECK(product IN('delivery','intraday')), when_kind TEXT CHECK(when_kind IN('open','close','now')), trade_date TEXT,
  qty INTEGER, ref_price REAL, created_ts TEXT, status TEXT CHECK(status IN('pending','filled','cancelled')), reason TEXT, cancel_reason TEXT);
CREATE TABLE fills(fill_id TEXT PRIMARY KEY, order_id TEXT, ts TEXT, price REAL, qty INTEGER, slip_assumed REAL, quote_bid REAL, quote_ask REAL,
  quote_ltp REAL, source TEXT, official_ref REAL, reconciled INTEGER DEFAULT 0);
CREATE TABLE fees(fill_id TEXT PRIMARY KEY, brokerage REAL, stt REAL, exchange REAL, sebi REAL, stamp REAL, gst REAL, dp REAL, total REAL);
CREATE TABLE cash_events(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, book TEXT, kind TEXT, amount_frac REAL, ref TEXT);
CREATE TABLE corp_adjust(id INTEGER PRIMARY KEY AUTOINCREMENT, book TEXT, sid INTEGER, date TEXT, factor REAL, note TEXT);
CREATE TABLE marks(book TEXT, date TEXT, nav REAL, cash_frac REAL, gross REAL, n_pos INTEGER, capital_ref_inr REAL, day_ret REAL, PRIMARY KEY(book, date));
CREATE TABLE corrections(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, fill_id TEXT, old_price REAL, new_price REAL, reason TEXT);
-- migration 5 (T11.2)
CREATE TABLE drift(date TEXT, kind TEXT, name TEXT, value REAL, threshold REAL, breached INTEGER, PRIMARY KEY(date, kind, name));
```

Book accounting is in **NAV units**: every book starts at `nav = 100.0`. An order's quantity is computed from the day's broker capital, but the ledger stores the trade's weight as a fraction of book NAV (`amount_frac`), so book performance is a percentage series independent of how capital changes. Rupee figures in reports are `fraction × capital_ref_inr` of that day.

## D.7 Configuration files (full initial content)

`config/default.toml`
```toml
[paths]
data_root = "data_store"
[time]
tz = "Asia/Kolkata"
decision_cutoff = "20:00"
[research]
dev_start = "2011-01-03"
wf_test_start = "2014-01-01"
lockbox_start = "2025-10-01"
[runtime]
threads = 8
gpu = true
```

`config/universe.toml`
```toml
series_allowed = ["EQ"]
min_history_sessions = 252
min_traded_of_last_60 = 57
min_close = 20.0
min_med_turnover_inr = 1.0e7      # ₹1 crore
u500_size = 500
u500_exit_rank = 600
u200_size = 200
u200_exit_rank = 250
u200_min_med_turnover_inr = 1.0e8 # ₹10 crore
etf_name_regex = "(?i)(ETF|BEES|LIQUID|GILT|NIFTY|SENSEX|GOLD|SILVER|IETF|ADD$)"
```

`config/costs.toml` (T5.1 verifies brokerage and DP and rewrites `verified_on`)
```toml
verified_on = "2026-11-11"
[delivery]
stt_buy = 0.001
stt_sell = 0.001
stamp_buy = 0.00015
exchange = 0.0000297
sebi = 0.000001
gst = 0.18
dp_per_scrip_sell_inr = 20.0
brokerage_rate = 0.001
brokerage_cap_inr = 20.0
brokerage_min_inr = 5.0
[intraday]
stt_buy = 0.0
stt_sell = 0.00025
stamp_buy = 0.00003
exchange = 0.0000297
sebi = 0.000001
gst = 0.18
dp_per_scrip_sell_inr = 0.0
brokerage_rate = 0.001
brokerage_cap_inr = 20.0
brokerage_min_inr = 5.0
[slippage]
impact_k = 0.5
open_penalty_bps = 3.0
close_penalty_bps = 2.0
half_spread_bps_by_rank = [[50, 1.5], [100, 2.5], [200, 4.0], [350, 7.0], [500, 12.0]]
[grid]
capital_inr = [25000, 100000, 500000, 2500000]
```

`config/gate.toml` (frozen in T1.7; SHA-256 pinned by a test)
```toml
[research]
min_mean_rank_ic        = 0.02
min_ic_tstat            = 3.0
min_net_sharpe          = 1.0
min_net_sharpe_2x_slip  = 0.5
min_excess_tstat        = 2.0
min_frac_positive_years = 0.70
min_dsr                 = 0.95
max_pbo                 = 0.30
max_drawdown            = 0.25
reference_capital_inr   = 500000
[lockbox]
min_net_sharpe          = 0.5
min_net_return          = 0.0
[paper]
min_sessions            = 60
min_net_sharpe          = 0.5
max_drawdown            = 0.10
max_ic_shortfall        = 0.5
[promotion]
min_shadow_sessions     = 20
max_p_value             = 0.10
[auction]
min_strict_capture      = 0.5
```
Meaning of each line is in T6.6. `[auction].min_strict_capture`: an S2 model counts as validated only when, over ≥ 60 paper sessions, the `auction_strict` book's net return is at least half of the `auction` book's and itself passes `[paper]`.

`config/strategies.toml`
```toml
[s1]                         # multi-day delivery book
product = "delivery"
entry = "close"              # buy at C(t+1)
exit = "open"                # sell at O(t+1+h)
targets = ["y_c2o_5", "y_c2o_10", "y_c2o_20"]
universe = "u500"
long_only = true
k_max = 20
[s2]                         # same-day auction book
product = "intraday"
entry = "open"               # O(t+1), decided 09:07 from pre-open indicative prices
exit = "close_minus"         # squared off 15:18
targets = ["y_oc"]
universe = "u200"
long_only = false
shorts_require_fno = true
k_max = 10
[s3]
enabled = false              # event sleeve, switched on in Phase 8
max_capital_frac = 0.30
```

`config/risk.toml`
```toml
target_vol_annual = 0.12
max_gross = 1.0
max_name_weight = 0.15
max_industry_weight = 0.30
max_adv_participation = 0.01
max_fixed_cost_bps = 15.0
min_position_inr_floor = 10000
stop_min = 0.08
stop_vol_mult = 3.0
book_dd_half = 0.10
book_dd_stop = 0.15
no_entry_sessions_before_results = 1
no_entry_sessions_before_split = 2
s2_free_cash_frac = 1.0
```

`config/schedule.toml` — full content in T10.4.

## D.8 Testing conventions

- `tests/unit/` runs with no network, no GPU, no real data, in under 90 seconds total. `tests/integration/` may use the synthetic market and the full pipeline; mark with `@pytest.mark.integration`. Tests that need real `data_store` are marked `@pytest.mark.realdata` and skipped when `data_store/parquet/bhavcopy_eq` is absent. Tests that need CUDA are marked `@pytest.mark.gpu` and skipped when unavailable. `pytest.ini_options`: `markers` registered, `timeout = 300`, `addopts = "-q -m 'not realdata and not gpu'"` (the acceptance commands of data and GPU tasks pass `-m realdata` or `-m gpu` explicitly).
- HTTP is tested with `httpx.MockTransport` and files in `tests/fixtures/files/` (small real samples saved by the task that writes the parser: 20 lines of each file type).
- Time is tested with `FixedClock` / `SteppingClock`.
- Every test asserts concrete numbers computed by hand in a comment above the assertion.

**The synthetic market (`tests/fixtures/synth.py`, built in T1.9) is shared by every later test.** Contract:

```python
@dataclass(frozen=True)
class SynthMarket:
    sessions: list[date]                 # 600 weekday sessions from 2018-01-01, plus ONE Saturday special session at index 120
    eq: pl.DataFrame                     # raw bhavcopy_eq schema (prices UNADJUSTED, prev_close = raw previous close)
    delivery: pl.DataFrame               # bhavcopy_delivery schema
    corporate_actions: pl.DataFrame      # corporate_actions schema
    symbol_changes: pl.DataFrame
    listed: pl.DataFrame                 # listed_equities schema
    etf: pl.DataFrame
    sectors: pl.DataFrame                # 4 industries
    truth: pl.DataFrame                  # (symbol_key, date) -> true total return, hidden alpha; for assertions
def make_market(n=40, n_sessions=600, seed=7, planted_ic: float = 0.0) -> SynthMarket: ...
def write_market(m: SynthMarket, root: Path) -> None:   # writes into a temp data_root in the real partition layout
```
Return model: `r_i,t = beta_i × m_t + e_i,t + planted_ic_term`, `m_t ~ N(0.0004, 0.01²)`, `e ~ N(0, 0.02²)`, `beta ~ U(0.5, 1.5)`; open = previous close × exp(N(0.0015, 0.008²)), close from the day return, high/low = max/min(open, close) × (1 ± |N(0, 0.004)|); volume lognormal with per-stock scale so median turnover spans ₹0.2 crore to ₹200 crore; `planted_ic`: each stock-day has hidden `alpha ~ N(0,1)` and the NEXT session's return gets `+ planted_ic × 0.02 × alpha` (so a feature equal to `alpha` has IC ≈ `planted_ic`).
Scripted events (symbols `S00`…`S39`):

| Stock | Event |
|---|---|
| S03 | split 1:5 effective at session 200 (raw price ÷5, volume ×5; corporate_actions row `split`, `factor_price=0.2`, `parse_status='ok'`) |
| S05 | bonus 1:1 at session 250 (`factor_price=0.5`) |
| S06 | bonus 1:1 at session 260 whose corporate_actions row has the WRONG factor 0.667 (tests the price cross-check) |
| S07 | renamed to `S07N` from session 300 (row in `symbol_changes`) |
| S08 | renamed to `S08N` from session 310 with NO row in `symbol_changes` (tests auto-detection) |
| S09 | suspended (no rows) sessions 310–330 |
| S11 | delisted: last row at session 400 |
| S13 | dividend ₹5 at session 150 |
| S14 | split 1:2 at session 220 with NO corporate_actions row (tests the suspected-ratio rule) |
| S15 | locked at upper circuit sessions 450–452 (`open=high=low=close=prev×1.05`) |
| S16 | demerger at session 330: raw price drops 30% overnight, corporate_actions row `demerger`, no factor |
| S17 | symbol `NIFTYBEES`, in `etf` |
| S19 | new listing at session 350 (no rows before) |
| S21 | price below ₹20 throughout |
| S23 | series switches EQ → BE at session 380 and back at 420 |
| S25 | rights issue 1:4 at premium ₹40 (face ₹10) at session 280, subject `Rights 1:4 @ Premium Rs 40/-`, `parse_status='unparsed'` |

A second helper `make_tiny_market()` returns 3 stocks × 12 sessions with hand-written prices for exact-arithmetic tests.

## D.9 Definition of done for any task

1. Tests listed on the card exist and pass. 2. Whole suite passes. 3. `ruff check` clean. 4. Acceptance command output pasted in the report. 5. `docs/DECISIONS.md` updated if anything deviated. 6. One commit.

---

# PART E — The work

## PHASE 0 — Reset (destructive; the orchestrator runs this itself, in order; no subagent)

Run in PowerShell from `C:\Users\harsh\Desktop\HARSH\WEB-DEV-PROJECTS\stocksense`. After each step run its check; stop if a check fails.

**T0.1 Stop the old system — DONE 2026-11-11** (no scheduled task, no process).

**T0.2 Backup — DONE 2026-11-11.** `C:\Users\harsh\Desktop\HARSH\_stocksense_archive_2026-11-11\` holds `stocksense-all-refs.bundle` (13.4 MB, `git bundle verify` reported a complete history), `working-tree.zip` (11.7 MB), `stash0.patch`, `env.backup`.

**T0.3 Remove the Codex worktree — DONE 2026-11-11.**

**T0.4 Delete every file except the keep-list — RESUME HERE.** Already removed: `.git`, `.claude`, `.pytest_cache`, `.ruff_cache`. The command is safe to repeat.
```powershell
$here = (Get-Location).Path
if ($here -ne 'C:\Users\harsh\Desktop\HARSH\WEB-DEV-PROJECTS\stocksense') { throw "wrong directory: $here" }
$keep = @('data_store', '.env', '.mcp.json')
Get-ChildItem -Force | Where-Object { $keep -notcontains $_.Name } | ForEach-Object {
  try { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction Stop }
  catch { "FAILED: $($_.Name): $($_.Exception.Message)" }
}
Get-ChildItem -Force | Select-Object Name
```
If a line prints `FAILED` for `.venv` or `node_modules` (a file in use): run `Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|node' } | Select-Object ProcessId,Name,CommandLine`, `taskkill /F /T /PID <pid>` for any that reference this folder, then repeat the command. Long-path failures under `node_modules`: `cmd /c "rmdir /s /q node_modules"`.
Check: the listing shows exactly `.env`, `.mcp.json`, `data_store`.

**T0.5 Fresh git.**
```powershell
git init -b main
git config user.name 'Harsh Kawatra'
git config user.email 'harshkawatra11@gmail.com'
git remote add origin https://github.com/harshkawatra11/stocksense.git
```
Check: `git status` says "No commits yet" on branch `main`.

**T0.6 Reorganize `data_store` (moves only).**
```powershell
$D = 'data_store'; $X = "$D\_archive_v1"
New-Item -ItemType Directory -Force "$X\parquet" | Out-Null
'evaluation_attempts','hypothesis_registry','ingest_runs','live_quote_snapshots','paper_events','paper_positions','paper_signals','vault_unseals' |
  ForEach-Object { if (Test-Path "$D\parquet\$_") { Move-Item "$D\parquet\$_" "$X\parquet\$_" } }
'_quarantine','account','experiments','market_days','market_lens','nse','operations','releases','research_snapshots','runtime','ui' |
  ForEach-Object { if (Test-Path "$D\$_") { Move-Item "$D\$_" "$X\$_" } }
if (Test-Path "$D\stocksense.duckdb") { Move-Item "$D\stocksense.duckdb" "$X\stocksense.duckdb" }
if (Test-Path "$D\pytest-temp-20260929") {
  takeown /F "$D\pytest-temp-20260929" /R /D Y | Out-Null
  icacls "$D\pytest-temp-20260929" /grant "${env:USERNAME}:F" /T /C | Out-Null
  Remove-Item -Recurse -Force "$D\pytest-temp-20260929"
}
'panel','features','labels','models','preds','state','reports','logs' | ForEach-Object { New-Item -ItemType Directory -Force "$D\$_" | Out-Null }
Get-ChildItem "$D\parquet" | Select-Object Name
```
Check: `data_store\parquet` lists exactly `bhavcopy_delivery, bhavcopy_eq, corporate_actions, fo_bhavcopy, india_vix, intraday_bars, nse_snapshots`. (`pytest-temp-20260929` is old test scratch, not data; if the delete still fails, leave it and move on.)

**T0.7 Write the first files.**
- `.gitignore`:
  ```
  data_store/
  .venv/
  .env
  *.env.local
  __pycache__/
  *.pyc
  *.egg-info/
  .pytest_cache/
  .ruff_cache/
  .hypothesis/
  *.log
  *.parquet
  *.sqlite*
  *.duckdb*
  node_modules/
  dist/
  Thumbs.db
  .DS_Store
  ```
- `.env.example`: every key name from `.env` with its value blanked (PowerShell: `Get-Content .env | ForEach-Object { if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=') { "$($Matches[1])=" } else { $_ } } | Set-Content -Encoding utf8 .env.example`), then append `UPSTOX_ANALYTICS_TOKEN=`, `TELEGRAM_BOT_TOKEN=`, `TELEGRAM_CHAT_ID=` if absent. Open the result and confirm no value survived.
- `docs/PLAN.md`: this file, unchanged. `docs/DECISIONS.md`: a title, the defaults table from A.4, and an empty section per phase.
- `CLAUDE.md` and `AGENTS.md` (identical): title "StockSense", the ten rules of C.5 verbatim, then: "Plan and status: `docs/PLAN.md` (checklist in Part H). Conventions: Part D. Run tests: `.venv\Scripts\python.exe -m pytest tests -q`. Commits: author Harsh Kawatra, no attribution lines."
- `README.md`: what the project is in five lines; pointer to `docs/PLAN.md`.

**T0.8 Publish.**
```powershell
git add -A
git status --short     # must NOT list .env or anything under data_store
git commit -m "StockSense: fresh start"
git push --force -u origin main
gh repo edit harshkawatra11/stocksense --default-branch main
gh api repos/harshkawatra11/stocksense/branches --paginate --jq '.[].name'
# for every name printed other than main:
gh api -X DELETE "repos/harshkawatra11/stocksense/git/refs/heads/<name>"     # URL-encode a slash in the name as %2F is NOT needed; pass the name as is
gh pr list --repo harshkawatra11/stocksense --state open --json number --jq '.[].number'
# for every number printed:
gh pr close <number> --repo harshkawatra11/stocksense
```
Check: `git log --oneline` prints one line; `git branch -a` prints `main` and `remotes/origin/main`; the branches API prints only `main`.

**T0.9 Alternative, only if the user asks for total erasure on GitHub.** `gh auth refresh -s delete_repo`; `gh repo delete harshkawatra11/stocksense --yes`; `gh repo create harshkawatra11/stocksense --private --source . --push`.

**T0.10 Reset Claude's project memory** at `C:\Users\harsh\.claude\projects\c--Users-harsh-Desktop-HARSH-WEB-DEV-PROJECTS-stocksense\memory\`. Delete every `*.md` there except `MEMORY.md`. Write two files with the standard frontmatter:
- `fresh-start-2026-11-11.md` (type `project`): "On 2026-11-11 the repo was wiped and restarted from one commit on `main`. No earlier code, branch, plan or memory is a valid reference. The plan is `docs/PLAN.md`. **Why:** four earlier rebuilds failed because sessions copied old designs. **How to apply:** never look for history; decide from the plan or first principles."
- `feedback-windows-process-and-model-name.md` (type `feedback`): "Kill processes with `taskkill /F /T` before restarting and check python PIDs first. When the user says 'Chronos' for this project they mean Kronos (`shiyu-coder/Kronos`). **Why:** stated by the user on 2026-11-11. **How to apply:** never plan Amazon Chronos here."
Replace `MEMORY.md` with two index lines pointing at them.

**Phase 0 done when:** all checks pass and a brand-new terminal prints `1` for `git log --all --oneline | Measure-Object -Line | Select-Object -ExpandProperty Lines`.

---

## PHASE 1 — Foundation

### T1.1 Environment and project skeleton
**Goal:** a working Python 3.12 environment with CUDA PyTorch, and the empty package tree.
**Depends on:** Phase 0. **Parallel-safe:** no.
**Files:** `pyproject.toml`, `uv.lock`, every `__init__.py` in the D.4 tree, `src/stocksense/__main__.py`, `tests/conftest.py`, `scripts/run-daemon.ps1`.
**Spec:**
1. `pyproject.toml`:
```toml
[project]
name = "stocksense"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "polars>=1.30", "duckdb>=1.3", "pyarrow>=19", "numpy>=2.1", "pandas>=2.2", "scipy>=1.14",
  "scikit-learn>=1.6", "statsmodels>=0.14", "lightgbm>=4.5",
  "torch>=2.9", "einops>=0.8", "huggingface_hub>=0.33", "safetensors>=0.5", "tqdm>=4.66",
  "sentence-transformers>=3.3",
  "httpx>=0.27", "tenacity>=9.0", "pydantic>=2.9", "pydantic-settings>=2.6", "typer>=0.15", "rich>=13",
  "pyotp>=2.9", "mcp>=1.2", "fastapi>=0.115", "uvicorn>=0.32", "rapidfuzz>=3.10", "feedparser>=6.0",
  "lxml>=5.3", "tzdata>=2024.2",
]
[project.optional-dependencies]
dev = ["pytest>=8.3", "pytest-timeout>=2.3", "hypothesis>=6.120", "ruff>=0.8", "freezegun>=1.5"]
[project.scripts]
stocksense = "stocksense.cli:app"
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
packages = ["src/stocksense", "third_party/kronos"]
[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
[tool.pytest.ini_options]
testpaths = ["tests"]
timeout = 300
addopts = "-q -m 'not realdata and not gpu'"
markers = ["integration: uses the synthetic market end to end", "realdata: needs data_store", "gpu: needs CUDA"]
[tool.ruff]
line-length = 110
target-version = "py312"
extend-exclude = ["third_party"]
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```
2. Commands: `uv venv --python 3.12 .venv`; `uv sync --extra dev`; if the lock fails on `torch`, change the index URL to `…/whl/cu126` and retry, and record which worked.
3. `third_party/kronos/__init__.py` empty for now (filled in T7.5) so the wheel target exists.
4. `tests/conftest.py`: sets the D.1 environment variables with `os.environ.setdefault`; fixture `tmp_data_root(tmp_path)` returning a `Settings` whose `data_root` is a temp folder with the D.4 subfolders created; fixture `fixed_clock` at `2026-11-12 10:00 IST`.
5. `scripts/run-daemon.ps1`: sets the D.1 variables, `Set-Location` to the repo, runs `.venv\Scripts\python.exe -m stocksense run`.
**Tests:** `tests/unit/test_env.py::test_imports` imports polars, duckdb, lightgbm, torch, einops, httpx, typer; `::test_torch_cuda` (marked `gpu`) asserts `torch.cuda.is_available()` and that the device name contains `3050`.
**Acceptance:** `.venv\Scripts\python.exe -c "import torch, polars, lightgbm; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` prints a `2.9.x+cu12x` version, `True`, and the RTX 3050 name.

### T1.2 Settings
**Goal:** one typed settings object. **Depends on:** T1.1. **Provides:** `core.config.Settings`, `load_settings(config_dir=Path("config"), env_file=Path(".env")) -> Settings`.
**Files:** `core/config.py`, `core/errors.py`, all seven `config/*.toml` files with the D.7 content (`schedule.toml` may be a stub with a comment until T10.4), `tests/unit/test_config.py`.
**Spec:** `Settings` is a frozen pydantic model with nested models `Paths(data_root)`, `Time(tz, decision_cutoff)`, `Research(dev_start, wf_test_start, lockbox_start)`, `Runtime(threads, gpu)`, `Universe(...)`, `Costs(delivery, intraday, slippage, grid)`, `Gate(...)`, `Strategies(s1, s2, s3)`, `Risk(...)`, and `Secrets`. `Secrets` fields are all `str | None`: `angel_api_key, angel_client_id, angel_pin, angel_totp_secret, upstox_analytics_token, telegram_bot_token, telegram_chat_id`. Secrets are read from the environment / `.env`. **Observed on 2026-11-11 (all set):** `STOCKSENSE_ANGEL_API_KEY, STOCKSENSE_ANGEL_CLIENT_CODE, STOCKSENSE_ANGEL_PASSWORD` (the PIN), `STOCKSENSE_ANGEL_TOTP_SECRET, STOCKSENSE_UPSTOX_API_KEY, STOCKSENSE_UPSTOX_API_SECRET, STOCKSENSE_UPSTOX_ACCESS_TOKEN` (daily, expires), `STOCKSENSE_UPSTOX_ANALYTICS_TOKEN` (1-year, read-only: the one to use), `STOCKSENSE_TELEGRAM_BOT_TOKEN, STOCKSENSE_TELEGRAM_ALLOWED_CHAT_ID, STOCKSENSE_TELEGRAM_ALLOWED_USER_ID`. Mapping (first non-empty wins): `angel_api_key ← STOCKSENSE_ANGEL_API_KEY`; `angel_client_id ← STOCKSENSE_ANGEL_CLIENT_CODE`; `angel_pin ← STOCKSENSE_ANGEL_PASSWORD`; `angel_totp_secret ← STOCKSENSE_ANGEL_TOTP_SECRET`; `upstox_analytics_token ← STOCKSENSE_UPSTOX_ANALYTICS_TOKEN, then STOCKSENSE_UPSTOX_ACCESS_TOKEN`; `telegram_bot_token ← STOCKSENSE_TELEGRAM_BOT_TOKEN`; `telegram_chat_id ← STOCKSENSE_TELEGRAM_ALLOWED_CHAT_ID`. Implement with pydantic `AliasChoices`. `Settings.require(*names)` raises `ConfigError("missing secret: angel_pin")` listing every missing one; it is called by broker code, never at import. `repr(Secrets)` shows `***` for every set value.
**Tests:** defaults load from the real `config/`; an env var overrides an alias; `require` raises with the names of all missing; `repr` contains no secret value; a wrong type in a toml raises `ConfigError` naming the key.
**Acceptance:** `.venv\Scripts\python.exe -c "from stocksense.core.config import load_settings; s=load_settings(); print(s.research.lockbox_start, s.costs.delivery.stt_buy, s.secrets)"` prints `2025-10-01 0.001` and a masked secrets repr.

### T1.3 Clock
**Provides:** `core.clock.Clock` (Protocol: `now() -> datetime`, tz-aware IST; `today() -> date`; `sleep(seconds)`), `SystemClock`, `FixedClock(dt)`, `SteppingClock(start, step=timedelta(seconds=60))` (each `now()` call returns the current value; `sleep(s)` advances by `s` without waiting; `advance(td)`), and `IST = ZoneInfo("Asia/Kolkata")`.
**Files:** `core/clock.py`, `tests/unit/test_clock.py`, `tests/unit/test_no_naked_now.py`.
**Spec:** `SystemClock.now()` = `datetime.now(IST)`; `SystemClock.sleep` = `time.sleep`. The guard test walks `src/stocksense/**/*.py` except `core/clock.py` and fails on any match of the regex `\b(datetime\.now|datetime\.utcnow|date\.today|time\.time|time\.sleep)\s*\(` printing file and line.
**Tests:** `FixedClock` returns the same aware datetime; `SteppingClock.sleep(90)` advances exactly 90 s; naive datetime passed to `FixedClock` raises `ValueError`; the guard test passes on the tree and fails on a temp file containing `datetime.now()` (test the checker function directly).

### T1.4 Trading calendar
**Provides:** `core.calendar.TradingCalendar` with `from_store(settings, clock) -> TradingCalendar`, `is_session(d)`, `next_session(d)` (strictly after), `prev_session(d)` (strictly before), `sessions(a, b)` inclusive list, `shift(d, n)` (n sessions forward/back from session `d`; if `d` is not a session, raise `ValueError`), `session_index(d) -> int`, `last_completed_session(now)`, `session_times(d) -> (open_dt, close_dt)`.
**Files:** `core/calendar.py`, `tests/unit/test_calendar.py`.
**Spec:**
- Past sessions = distinct dates in `bhavcopy_eq` (read only the `date` column lazily). These are the truth, including weekend special sessions.
- Future (after the last stored date): a weekday is a session unless it is in `holidays`; a weekend day is a session only if listed under `special_sessions` in `config/schedule.toml` (`[[special_sessions]] date="2027-02-01" open="09:15" close="15:30"`).
- `holidays` dataset is filled by T2.9; before it exists the calendar treats all future weekdays as sessions and logs one warning.
- `session_times`: default 09:15–15:30 IST; overridden per date by `special_sessions` (Muhurat sessions are evening sessions; the daemon does not trade them: `tradable=false` flag on the entry makes the daemon run only the data jobs that day).
- `last_completed_session(now)`: the latest session whose close time is ≤ `now`.
- The object is immutable; build once per process and after each `eod_ingest`.
**Edge cases / Tests:** (1) synthetic market: the Saturday special session at index 120 is a session and `next_session(Friday before) == that Saturday`; (2) `shift(d, 5)` then `shift(_, -5)` returns `d` for 200 random sessions (hypothesis); (3) a future holiday in a fake `holidays` frame is skipped; (4) `prev_session(next_session(d)) == d` for any session `d`; (5) `sessions(a, b)` with `a > b` returns `[]`; (6) `shift` on a non-session raises; (7) `last_completed_session` at 15:29 on a session returns the previous session, at 15:30 returns that day.
**Acceptance:** unit tests pass.

### T1.5 Paths, logging, state, ids
**Provides:** `core.paths.Paths(settings)` with a method per D.4 folder and `dataset_dir(name)`; `core.log.get_logger(name)` and `configure_logging(settings, clock, console=True)`; `core.state.State(settings)` with `connect() -> sqlite3.Connection`, `migrate()`, `job_start(job, trade_date) -> attempt`, `job_finish(job, trade_date, attempt, status, detail)`, `job_status(job, trade_date) -> str | None` (status of the latest attempt), `kv_get/kv_set`; `core.ids` with the D.3 id functions and `canonical_json`.
**Files:** the four modules, `tests/unit/test_state.py`, `test_log.py`, `test_ids.py`.
**Spec:**
- Logging: stdlib `logging` with a JSON formatter (`ts, level, logger, event, **fields`), a `TimedRotatingFileHandler`-free design: the file name contains the date from the clock at configure time and the daemon reconfigures at midnight. A `RedactFilter` replaces the value of any field whose key matches `(?i)(token|secret|pin|password|totp|authorization|api_key|jwt)` with `"***"`, and also scrubs any substring of the message equal to a known secret value from `Settings.secrets`.
- State: migrations are a list of `(version, sql)`; table `schema_version(version)`; `migrate()` applies missing ones in one transaction each; idempotent. Only migration 1 exists now; later tasks append theirs (D.6 says which).
- `job_start` inserts `attempt = max(existing)+1`, status `running`. A `running` row older than 6 hours is treated as failed by `job_status`.
**Tests:** redaction of a field and of a secret inside a message; two migrations applied twice leave `schema_version` unchanged; a reader connection can `SELECT` while a writer holds an open transaction (WAL); `job_status` returns `done` after finish, `failed` for a stale `running`; id functions are deterministic and 16 hex chars; `canonical_json({"b":1,"a":2}) == '{"a":2,"b":1}'`.

### T1.6 Parquet I/O
**Provides:** `data.io.read(settings, dataset, start=None, end=None, columns=None) -> pl.LazyFrame`; `write_partition(settings, dataset, key, df)`; `upsert(settings, dataset, df)` (routes rows to partitions by the dataset's partition rule and merges on the dataset key); `list_partitions(settings, dataset)`; `data.catalog.CATALOG: dict[str, DatasetSpec]` with `DatasetSpec(name, layer, key, date_col, partition, schema: dict[str, pl.DataType], freshness)` for every D.5 dataset.
**Files:** `data/io.py`, `data/catalog.py`, `tests/unit/test_io.py`.
**Spec:**
- `read`: build one `pl.scan_parquet(file)` per partition file that can overlap `[start, end]` (decide from the file name), cast each to the catalog schema (add missing columns as typed nulls; cast a Datetime `date` to `Date`; ignore unknown extra columns), `pl.concat(how="vertical")`, then filter on the date column. Returns an empty LazyFrame with the right schema when no file exists.
- `write_partition`: validate columns against the catalog (missing required key column → `DataValidationError`), sort by key, write to `<final>.tmp-<pid>` with `compression="zstd"`, then `os.replace`. On `PermissionError` (Windows: the target is open by another process) retry 5 times, 0.5 s apart, then raise.
- `upsert`: for each affected partition: read existing (if any), `pl.concat([old, new], how="diagonal_relaxed")`, `unique(subset=key, keep="last", maintain_order=True)`, write. Rows in `new` with a null in any key column → `DataValidationError`.
**Edge cases / Tests:** upserting the same frame twice gives identical bytes of row content (compare frames); a new row with an existing key replaces the old row; a month partition lacking the `isin` column reads together with one that has it, `isin` null for the old rows; raising inside the write (monkeypatch `os.replace`) leaves the old file intact and removes the temp file; `read` with `start`/`end` returns only that range; empty dataset returns schema with zero rows; old-style timestamp `date` is returned as `Date`.

### T1.7 Frozen gate and the guard tests
**Files:** `config/gate.toml` (D.7 content, exact), `tests/unit/test_gate_frozen.py`, `test_no_orders.py`, `test_no_pandas_outside_kronos.py`, `test_no_archive_refs.py`.
**Spec:** `test_gate_frozen` computes SHA-256 of `config/gate.toml` bytes (after normalizing `\r\n` to `\n`) and compares with a constant in the test; the executor computes the hash once and pastes it. `test_no_orders` walks `src/` and fails on any case-insensitive match of `placeorder|modifyorder|cancelorder|place_order|modify_order|cancel_order|/order/place|/order/modify|/order/cancel|/v1/placeOrder|gttCreateRule`. `test_no_pandas_outside_kronos` fails on `^\s*(import pandas|from pandas)` in any file under `src/` except `models/kronos_model.py`. `test_no_archive_refs` fails on the substring `_archive_v1` in `src/`.
**Tests:** each guard also has a self-test that feeds the checker a bad temp file and expects a failure.

### T1.8 CLI skeleton and `doctor`
**Provides:** `stocksense.cli.app` (typer). Commands, each a thin wrapper calling a function in the owning module: `doctor`, `data-status`, `ingest`, `build-panel`, `build-features`, `build-labels`, `train`, `validate`, `backtest`, `run`, `job`, `report`, `resume`. Unimplemented ones print `not implemented yet (phase N)` and exit with code 2.
**Spec for `doctor`:** prints Python version, torch version and CUDA device, free disk GB on the data drive, data root, for each catalog dataset that exists its last date and row count (cheap: read parquet metadata, not data), SQLite schema version, heartbeat age if `heartbeat.json` exists, and which secrets are set (names only). Exit code 0 unless the data root is missing.
**Tests:** `typer.testing.CliRunner` on `doctor` with `tmp_data_root` exits 0 and prints `data root`; an unimplemented command exits 2.

### T1.9 The synthetic market fixture
**Provides:** `tests.fixtures.synth.make_market`, `make_tiny_market`, `write_market` exactly as specified in D.8.
**Files:** `tests/fixtures/synth.py`, `tests/unit/test_synth.py`.
**Spec additions:** raw prices are rounded to 0.05 (tick). Raw `prev_close` is the previous session's raw `close` for that symbol (NOT adjusted, mirroring NSE; see B.3 Finding 1); for a new listing `prev_close` = the first open. `turnover_inr = volume × (open+high+low+close)/4`. `n_trades = volume/50`. Delivery: `deliv_pct ~ clip(N(45, 15), 5, 95)`, `deliv_qty = volume × pct/100`. The Saturday special session is a normal session date in `sessions`. The `truth` frame holds per (symbol_key = original symbol before any rename, date): `true_ret` (total return including dividend, i.e. what an investor earned), `alpha`.
**Tests:** shape (≈ 40 × 600 rows minus suspension/listing/delisting gaps); S03 raw close ratio across session 200 ≈ 0.2 × (1 + true_ret) within tick rounding; S07 has rows under `S07` up to 299 and `S07N` from 300; S09 has no rows in 310–330; with `planted_ic=0.05`, the daily Spearman correlation between `alpha[t]` and `true_ret[t+1]` averages between 0.035 and 0.065; same seed → identical frames.

**Phase 1 done when:** suite green; `stocksense doctor` prints the seven adopted datasets with last dates matching B.2.

---

## PHASE 2 — Data: adopt, repair, extend, keep current

Rules for every ingester: (1) download raw bytes to `data_store/cache/<source>/…` first and parse from the cache, so a parser fix never needs a re-download; (2) if the cache file exists, do not download again; (3) upsert by the catalog key; (4) return `IngestResult(dataset, date, rows, status)` with status in `ok | not_published | no_session | skipped | failed`; (5) never raise for "not published yet": return the status.

### T2.1 NSE HTTP client
**Provides:** `data.nse_http.NseClient(clock, transport=None)` with `get_archive(url) -> bytes | None` (None on 404), `get_api(path, params, referer_path) -> Any` (parsed JSON), `close()`.
**Files:** `data/nse_http.py`, `tests/unit/test_nse_http.py`.
**Spec:**
- One `httpx.Client(http2=False, timeout=httpx.Timeout(30, connect=15), follow_redirects=True)`.
- Headers on every request: `User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36`, `Accept-Language: en-US,en;q=0.9`, `Accept-Encoding: gzip, deflate`. Archive requests add `Accept: */*`, `Referer: https://www.nseindia.com/`. API requests add `Accept: application/json, text/plain, */*`, `Referer: https://www.nseindia.com/<referer_path>`.
- Rate limit: at most 1 request per 0.6 s across the client (token spacing using `clock`, plus 0–0.2 s jitter from a seeded RNG).
- API cookie handling: try the API call directly (observed to work without cookies on 2026-11-11). On 401/403: GET `https://www.nseindia.com/` then `https://www.nseindia.com/market-data/live-equity-market` (ignore their status codes, keep cookies), wait 1 s, retry the API call. At most 2 warm-ups per call.
- Retries (tenacity): 5 attempts, exponential 2 → 60 s, on `httpx.TransportError`, 429, 500–599. A 404 on an archive URL returns `None` immediately (no retry).
- An archive response whose body starts with `<!DOCTYPE html` or `<html` is NSE's "not found" page served with status 200 on some paths: treat as 404.
- An API response that is not valid JSON after warm-up raises `DataNotAvailable`.
**Tests (MockTransport):** archive 404 → None; archive HTML body with 200 → None; API 403 then 200 after warm-up → JSON returned and exactly two warm-up GETs made; 500, 500, 200 → success with 3 calls; spacing ≥ 0.6 s between calls measured on `SteppingClock`; headers present.

### T2.2 Validation and `data-status`
**Provides:** `data.validate.validate_dataset(settings, name, start=None, end=None) -> list[Issue]`; `data.status.status_table(settings, calendar, clock) -> pl.DataFrame`; CLI `data-status`.
**Files:** `data/validate.py`, `data/status.py`, tests, `docs/DATA.md` (baseline).
**Spec — checks per dataset:** schema matches catalog; no duplicate keys; date column non-null. Price datasets additionally: `low ≤ min(open, close) + 1e-6`, `high ≥ max(open, close) − 1e-6`, prices > 0, `volume ≥ 0`. Session completeness for `bhavcopy_eq`: every calendar session has ≥ 500 rows, and per-session EQ row count within ±15% of the trailing 20-session median. **Missing-session detector** (the Finding 2 query): for each session `d`, among EQ rows that also have a row on the previous stored session, the share with `|prev_close/prev_stored_close − 1| > 0.0005`; if > 0.5, report `missing_session_before:<d>`.
`status_table` columns: dataset, first_date, last_date, rows, sessions_missing (for daily datasets: calendar sessions between first and last with no rows), fresh (bool: last_date ≥ last completed session, per the catalog freshness rule), issues (count).
**`docs/DATA.md` baseline:** run `data-status` on the adopted data and paste the table; it must show the 19 `missing_session_before` issues listed in B.3. This is the "before" picture.
**Tests:** on the synthetic market written to a temp root: zero issues; after deleting one session's rows from the frame: one `missing_session_before` for the next session; a row with `high < close` → one issue; duplicate key → one issue.
**Acceptance:** `stocksense data-status` prints 7 datasets; `bhavcopy_eq` shows last date 2026-09-29 and 19 missing-session issues.

### T2.3 Equity bhavcopy and delivery ingester, plus the repair backfill
**Goal:** one ingester for the cash market that handles both file eras, fills the 19 missing sessions, and brings the data to today.
**Depends on:** T2.1, T2.2. **Provides:** `data.ingest_eq.ingest_eq_day(ctx, d) -> list[IngestResult]`, `ingest_eq_range(ctx, a, b)`, where `ctx = IngestContext(settings, clock, calendar, nse, state)`.
**Files:** `data/ingest_eq.py`, `tests/unit/test_ingest_eq.py`, `tests/fixtures/files/udiff_cm_sample.csv`, `legacy_cm_sample.csv`, `sec_bhavdata_full_sample.csv` (first 20 lines of real files, saved by the executor from the cache / a download).
**Spec:**
1. **Era:** `d ≥ 2024-07-08` → UDiFF URL; else legacy URL (B.4). Cache paths (existing convention): `cache/cm_udiff/YYYY/YYYY-MM-DD.zip`, `cache/cm_legacy/YYYY/YYYY-MM-DD.zip`.
2. **UDiFF parse:** read the single CSV inside the zip with `pl.read_csv(..., infer_schema_length=0)` (all strings), then cast. Map: `TckrSymb→symbol, SctySrs→series, TradDt→date (format %Y-%m-%d), OpnPric→open, HghPric→high, LwPric→low, ClsPric→close, PrvsClsgPric→prev_close, LastPric→last_price, TtlTradgVol→volume, TtlTrfVal→turnover_inr, TtlNbOfTxsExctd→n_trades, ISIN→isin, FinInstrmNm→name`; `era='udiff'`. Keep every row (all series), as the stored data does.
3. **Legacy parse:** header `SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN` followed by a trailing comma that creates an empty last column: drop any column whose name is empty or starts with `_duplicated`/`Unnamed`. `TIMESTAMP` format `%d-%b-%Y` (`01-FEB-2020`; parse case-insensitively by title-casing the month). `TOTALTRADES` and `ISIN` are absent in the oldest files: null. `era='legacy'`.
4. **First action of the task — the probe:** parse the cached file for 2026-09-29 and for 2020-02-03 and compare with the stored rows for those dates on `symbol, series, open, high, low, close, prev_close, volume` (exact for integers, 1e-6 for floats, `turnover_inr` relative 1e-6). They must match exactly; if not, fix the mapping until they do and record what was wrong.
5. **Delivery and VWAP:** `sec_bhavdata_full_DDMMYYYY.csv`. Strip whitespace from header names and from every string value. **Reject the file if any `DATE1` value (format `%d-%b-%Y`) differs from the requested date** (holiday trap, B.4) → status `no_session`. Map `SYMBOL, SERIES, DELIV_QTY→deliv_qty, DELIV_PER→deliv_pct`, with `-` → null; upsert into `bhavcopy_delivery`. Also take `AVG_PRICE` and upsert it into `bhavcopy_eq.avg_price` for that date (join on symbol, series). Cache: `cache/delivery/YYYY/YYYY-MM-DD.csv`. This file exists back to at least 2020-02 (stored delivery starts 2019-08-23; do not try to extend delivery history earlier than that in this task).
6. **Which days to try:** `ingest_eq_range(a, b)` iterates **every calendar day** including weekends. Skip a day that already has ≥ 500 rows in `bhavcopy_eq`. A 404 on a weekday that the calendar expects as a session and is ≤ 1 day old → `not_published`. A 404 on any other day → `no_session`; if the day is more than 3 days old, remember it in `kv` as `no_session:eq:<date>` so it is never fetched again.
7. **Repair backfill, in this order:** (a) the 19 dates of B.3 Finding 2; (b) every calendar day in 2026-09-30 → yesterday; (c) every weekend day and every weekday missing from the stored sessions between 2010-01-01 and 2026-09-29 (about 1,900 quick requests, most 404; needed once to be certain no other special session is missing).
8. **Cross-check against the official NSE MCP server** for 20 random (symbol, date) pairs from the new rows of the last 5 years: connect with the `mcp` package (`mcp.client.streamable_http.streamablehttp_client("https://mcp.nseindia.in/bhavcopy/cm/mcp")`, `ClientSession`, `initialize()`, `list_tools()`), find the tool for a single-date price (`get_ltp_by_date` or `get_stock_history`), read its input schema from `list_tools()` to build the arguments, compare `close` within 0.01. If the server is unreachable, log a warning and continue (non-blocking); if reachable and any pair mismatches, fail.
**Edge cases → tests:** legacy sample parses with the trailing-comma column dropped; UDiFF sample parses and types are right; delivery file with a wrong `DATE1` is rejected with `no_session`; `-` delivery becomes null; a Saturday with a file is ingested; a day already present is skipped without any HTTP call (assert zero requests on the mock); 404 older than 3 days writes the `kv` marker and a second call makes no request; rerun of the same day leaves row count unchanged.
**Acceptance:** `stocksense ingest eq --from 2010-01-01 --to yesterday --repair` then `stocksense data-status`: `bhavcopy_eq` last date = last completed session, **zero** `missing_session_before` issues, sessions_missing = 0; `bhavcopy_delivery` last date the same. Paste the new session count (expected about 4,135 + 19 + ~29).

### T2.4 F&O bhavcopy ingester
**Provides:** `data.ingest_fo.ingest_fo_day(ctx, d)`, `ingest_fo_range`. **Parallel-safe:** yes (after T2.3's helpers exist).
**Spec:** URL in B.4; cache `cache/fo_bhavcopy/YYYY/YYYY-MM-DD.zip`. Map: `TckrSymb→symbol, FinInstrmTp→instrument_type, XpryDt→expiry_date, StrkPric→strike_price, OptnTp→option_type, TradDt→date, OpnPric/HghPric/LwPric/ClsPric→open/high/low/close, PrvsClsgPric→prev_close, SttlmPric→settle_price, OpnIntrst→oi, ChngInOpnIntrst→change_in_oi, UndrlygPric→underlying_price`. Probe: parse the cached 2026-09-18 file and compare with stored rows (must match). Backfill 2026-09-19 → yesterday, every calendar day, same 404 rules. Do not extend earlier than 2024-07-08.
**Tests:** sample parse; futures rows have null strike and option_type; idempotent upsert.
**Acceptance:** `data-status` shows `fo_bhavcopy` current.

### T2.5 Index and VIX ingester
**Provides:** `data.ingest_index.ingest_index_day(ctx, d)`, `ingest_index_range`, `INDEX_NAME_MAP`. **Parallel-safe:** yes.
**Spec:** URL in B.4, cache `cache/index/YYYY/YYYY-MM-DD.csv`. Parse numeric columns with `-` → null. Canonical name = upper-case, single spaces, then apply `INDEX_NAME_MAP`: `CNX NIFTY→NIFTY 50`, `CNX NIFTY JUNIOR→NIFTY NEXT 50`, `CNX 100→NIFTY 100`, `CNX 200→NIFTY 200`, `CNX 500→NIFTY 500`, `CNX MIDCAP→NIFTY MIDCAP 100`, `CNX SMALLCAP→NIFTY SMALLCAP 100`, `BANK NIFTY→NIFTY BANK`, `CNX IT→NIFTY IT`, `CNX FMCG→NIFTY FMCG`, `CNX PHARMA→NIFTY PHARMA`, `CNX AUTO→NIFTY AUTO`, `CNX METAL→NIFTY METAL`, `CNX REALTY→NIFTY REALTY`, `CNX ENERGY→NIFTY ENERGY`, `CNX INFRA→NIFTY INFRASTRUCTURE`, `CNX PSU BANK→NIFTY PSU BANK`, `CNX MEDIA→NIFTY MEDIA`, `CNX FINANCE→NIFTY FINANCIAL SERVICES`, `INDIA VIX→INDIA VIX`; unmapped names keep their upper-cased form (the executor prints the distinct names of one 2015 file and one 2026 file and extends the map so the 20 above are continuous across the rename). Backfill: walk backward from yesterday over every calendar day until 60 consecutive misses; record the true first date in `docs/DATA.md` (it lies between 2012-01-02 and 2015-01-05). Rows named `INDIA VIX` with date > the last `india_vix` date are appended to `india_vix`.
**Tests:** sample 2015 and 2026 files map `CNX Nifty` and `Nifty 50` to the same canonical name; `-` → null; VIX extension appends only new dates.
**Acceptance:** `data-status`: `index_daily` and `india_vix` current; a query printing NIFTY 50 close for the first and last stored dates.

### T2.6 Corporate-actions ingester and parser
**Provides:** `data.ingest_corpact.parse_subject(subject: str, face_val: float | None) -> ParsedAction`; `ingest_corpact_range(ctx, a, b)`; `reparse_unparsed(ctx)`.
**Spec:**
- API in B.4; date format in the response `02-Nov-2026`; `-` means null. Query in 31-day windows. Cache the raw JSON per window (`cache/corpact/<from>_<to>.json`).
- `ParsedAction(action_type, ratio_num, ratio_den, factor_price, dividend_amount, face_before, face_after, rights_price, parse_status)`. Rules, applied to the lower-cased subject with runs of spaces collapsed, **in this order** (a subject may match several; combine: factors multiply, dividends add; `action_type` joins with `+` in the order `bonus+split`):
  1. Split / sub-division: `(?:face value )?split|sub-?division` and `from\s+(?:rs\.?|re\.?)\s*([\d.]+)\s*/?-?\s*(?:per share\s*)?to\s+(?:rs\.?|re\.?)\s*([\d.]+)` → `face_before, face_after`, `factor = face_after/face_before`, `ratio_num = face_before/face_after, ratio_den = 1`. If the amounts are missing but `face_val` and a later row give the new face: leave `unparsed`.
  2. Consolidation: `consolidation` with the same `from … to …` → `factor = face_after/face_before` (> 1).
  3. Bonus: `bonus\s*(?:issue\s*)?(\d+)\s*:\s*(\d+)` → `A:B` means A new shares for every B held; `factor = B/(A+B)`.
  4. Rights: `rights?\s*(?:issue\s*)?(\d+)\s*:\s*(\d+)` → `ratio_num=A, ratio_den=B`; premium `@?\s*(?:a\s*)?premium\s*(?:of\s*)?(?:rs\.?|re\.?)\s*([\d.]+)`; `rights_price = premium + face_val` (face_val from the API field; if null, 10 is NOT assumed: leave `rights_price` null); `at par` → `rights_price = face_val`. `factor_price` stays null here (it depends on the market price; computed in T3.2). `parse_status='ok'` when ratio and price are found, else `unparsed`.
  5. Dividend / distribution: every `(?:rs\.?|re\.?)\s*([\d.]+)\s*(?:/-)?\s*per\s+(?:share|unit)` amount is summed into `dividend_amount`; `action_type='dividend'` (or `distribution` if the word appears and `dividend` does not). A subject containing `dividend` with no amount → `dividend`, `unparsed`.
  6. Keywords with no factor: `buy ?back→buyback`, `demerger→demerger`, `scheme of arrangement→scheme_of_arrangement`, `amalgamation|merger→amalgamation`, `capital reduction→capital_reduction` (all `parse_status='unparsed'`, meaning "price evidence decides", see T3.2).
  7. `annual general meeting|agm|egm|interest payment|redemption` alone → `noise`, `ok`.
- `reparse_unparsed`: re-run the parser over stored rows with `parse_status='unparsed'` or `action_type in ('rights','other')` and upsert the improved rows (key unchanged). Print before/after counts by type.
- Fetch range: 2026-09-01 → today + 45 days (future ex-dates are inputs to features and risk). Run daily thereafter with [today − 10, today + 45].
**Tests (exact strings; expected values in the test):** `Bonus 1:1` → 0.5; `Bonus 3:2` → 0.4; `Face Value Split (Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share` → 0.1, faces 10 and 1; `Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share` → 0.2; `Interim Dividend - Rs 2 Per Share` → 2.0; `Interim Dividend Rs  - Rs 2.50 Per Share` → 2.5; `Interim Dividend - Rs 5 Per Share/ Special Dividend - Rs 10 Per Share` → 15.0; `Third Distribution -Interest Payment - Rs 2.20 Per Unit/ Return Of Capital -Re 0.80 Per Unit` → distribution 3.0; `Rights 1:7 @ Premium Rs 30/-` with face 10 → 1, 7, price 40, ok; `Rights Issue 30:37@ Premium Rs 53/-` with face 10 → 30, 37, 63; `Rights 4:21 @ Premium Rs 1.80/-` face 1 → 2.80; `Buy Back` → buyback unparsed; `Scheme Of Arrangement` → unparsed; `Interim Dividend` → dividend unparsed; `Bonus 1:1 And Face Value Split From Rs 10/- To Rs 5/-` → `bonus+split`, factor 0.25; `Annual General Meeting` → noise.
**Acceptance:** `data-status` shows `corporate_actions` with max ex_date ≥ today; the reparse summary printed; unparsed dividend count reduced.

### T2.7 Reference data: symbol changes, listings, ETFs, sectors, price bands, holidays, F&O list
**Provides:** `data.ingest_reference.ingest_reference(ctx)` (runs all), individual functions per dataset. **Parallel-safe:** yes.
**Spec:**
- `symbol_changes`: header in B.4; date `%d-%b-%Y` case-insensitive; strip whitespace; replace the whole dataset each run.
- `listed_equities`: header names have leading spaces (strip); `DATE OF LISTING` `%d-%b-%Y`; append with `as_of = today` only if the content hash changed since the last snapshot.
- `etf_list`, `sectors` (both index list files; `source` = file name; a symbol in both keeps the Nifty 500 row), `price_bands` (from `sec_list.csv`, one snapshot per session, column `Band`), same append-on-change rule except `price_bands` which appends daily.
- `holidays`: API in B.4; take the `CM` list; `tradingDate` `%d-%b-%Y`; replace rows for the fetched year. Rebuild the calendar afterwards.
- F&O list: `fo_mktlots.csv` → kv `fno_symbols:<date>` as a JSON list of the `SYMBOL` column (strip), excluding index rows (`NIFTY`, `BANKNIFTY`, `FINNIFTY`, `MIDCPNIFTY`, `NIFTYNXT50`). Historical membership for 2024-07 onward comes from `fo_bhavcopy` (symbols with `instrument_type='STF'` on that date).
**Tests:** each parser on a 10-line sample; change-detection (same content twice → one snapshot).
**Acceptance:** `data-status` lists the six datasets; print counts (expect ≈ 2,000+ listed, ≈ 250+ ETFs, 500 + 750 sector rows, ≈ 1,000+ symbol changes).

### T2.8 Announcements, board meetings, results index
**Provides:** `data.ingest_announcements.backfill_announcements(ctx, start, end)`, `poll_announcements(ctx) -> int` (live; returns new rows), `ingest_board_meetings`, `ingest_results_index`.
**Spec:**
- Endpoint and fields in B.4. `ts` from `an_dt` (`%d-%b-%Y %H:%M:%S`, IST); `exch_ts` from `exchdisstime` (`-` → null); `seq_id` is the key; rows with empty `seq_id` get `seq_id = sha1(symbol|an_dt|attchmntText)[:16]`.
- **Truncation check first:** for the window 2026-11-02 → 2026-11-08 compare the 7-day call's row count with the sum of seven 1-day calls. If equal, use 7-day windows for all history; if the 7-day call returns fewer, use 1-day windows for years ≥ 2018 and 3-day windows before, and record it.
- Backfill backward from yesterday to 2010-01-01 or until 12 consecutive empty windows; cache each window's raw JSON gzipped (`cache/announcements/<from>_<to>.json.gz`); resumable (skip windows whose cache file exists). Expected size: 1–3 million rows; about 30–90 minutes.
- Live poll: `from_date = to_date = today`; new `seq_id`s get `first_seen_ts = clock.now()`. Backfilled rows have `first_seen_ts` null and decisions on historical data use `max(ts, exch_ts)`.
- Board meetings: `bm_symbol→symbol, bm_date→meeting_date (%d-%b-%Y), bm_purpose→purpose, bm_desc→description, bm_timestamp→announced_ts`. Backfill in 30-day windows as far back as the API returns data.
- Results index: fields in B.4; `filingDate` `%d-%b-%Y %H:%M`; the default call returns recent filings only: probe adding `&from_date=DD-MM-YYYY&to_date=DD-MM-YYYY` and `&period=Annual`; record which parameters the API honours; backfill as far back as possible.
**Tests:** parsing of a 5-row sample for each; `seq_id` fallback; live poll marks only new ids; resumable backfill skips cached windows (zero HTTP calls).
**Acceptance:** `data-status` shows `announcements` first date (expect 2011 or earlier), last date today, row count; `docs/DATA.md` gets a table of rows per year and the 30 most frequent `category` values with counts (this list is the input to the taxonomy in T8.2).

### T2.9 Brokers, read-only
**Provides:** `brokers.angel.AngelClient(settings, clock, transport=None)` with `login()`, `funds() -> Funds(available_cash, net, utilised)`, `holdings() -> list[Holding(symbol, qty, avg_price, ltp)]`, `positions()`, `quote_full(tokens: list[str]) -> list[Quote]`, `candles(token, interval, start_dt, end_dt) -> pl.DataFrame`; `brokers.upstox.UpstoxClient(settings, clock, transport=None)` with `candles(instrument_key, unit, interval, start, end)`, `quotes(instrument_keys) -> list[Quote]`; `brokers.instruments.InstrumentMap` (`angel_token(symbol)`, `upstox_key(symbol)`, refresh daily); dataclass `Quote(symbol, ts, ltp, open, high, low, prev_close, volume, bid, ask, bid_qty, ask_qty, upper_circuit, lower_circuit, source)`.
**Spec — Angel (direct REST with httpx; endpoints written from memory, so the first step is a probe):**
- Base `https://apiconnect.angelone.in`. Common headers: `Content-Type: application/json`, `Accept: application/json`, `X-UserType: USER`, `X-SourceID: WEB`, `X-ClientLocalIP: 127.0.0.1`, `X-ClientPublicIP: 127.0.0.1`, `X-MACAddress: 00:00:00:00:00:00`, `X-PrivateKey: <api_key>`; after login add `Authorization: Bearer <jwtToken>`.
- Login: `POST /rest/auth/angelbroking/user/v1/loginByPassword` body `{"clientcode","password": <pin>,"totp": pyotp.TOTP(secret).now()}` → `data.jwtToken, data.refreshToken, data.feedToken`. If `status` is false with a TOTP error, wait until the next 30-second boundary + 1 s and retry once. Session is cached in memory; any call that gets an auth error re-logs in once.
- Funds: `GET /rest/secure/angelbroking/user/v1/getRMS` → `data.availablecash`, `data.net`, `data.utiliseddebits` (strings → float).
- Holdings: `GET /rest/secure/angelbroking/portfolio/v1/getAllHolding` → `data.holdings[]` (`tradingsymbol, quantity, averageprice, ltp`). Positions: `GET /rest/secure/angelbroking/order/v1/getPosition`.
- Quotes: `POST /rest/secure/angelbroking/market/v1/quote/` body `{"mode":"FULL","exchangeTokens":{"NSE":[tokens…]}}`, max 50 tokens per call, 1 call per second; `data.fetched[]` fields `symbolToken, tradingSymbol, ltp, open, high, low, close (previous close), tradeVolume, exchFeedTime, upperCircuit, lowerCircuit, depth.buy[0].price/quantity, depth.sell[0].price/quantity`.
- Candles: `POST /rest/secure/angelbroking/historical/v1/getCandleData` body `{"exchange":"NSE","symboltoken","interval","fromdate":"YYYY-MM-DD HH:MM","todate":"YYYY-MM-DD HH:MM"}`; max span per request: ONE_MINUTE 30 days, FIVE_MINUTE 100 days, ONE_DAY 2000 days; rows `[ts, o, h, l, c, v]`. On error code `AB1021` or HTTP 403/429: sleep 60 s, retry; then 300 s, retry; then raise `BrokerError`.
- Instrument master: `https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json` (large JSON list; cache daily in `cache/instruments/angel-YYYY-MM-DD.json`); NSE equity rows have `exch_seg == "NSE"` and `symbol` ending `-EQ` (strip the suffix) → `token`.
- **Probe (first step):** a script that logs in and prints only: login ok true/false, the list of keys in the RMS response, the number of holdings, and the key names of one FULL quote for RELIANCE. If the login endpoint or shapes differ from the above, fix the client to what is observed and record it. **Fallback if direct REST cannot be made to work in two attempts:** `uv add smartapi-python logzero websocket-client pycryptodome` and wrap `SmartConnect` (`generateSession`, `rmsLimit`, `holding`, `getMarketData`, `getCandleData`) behind the same `AngelClient` interface; call `os.chdir` nowhere, and pass `disable_ssl=False`; the library creates a `logs/` folder in the working directory: add `logs/` to `.gitignore`.
**Spec — Upstox (documented, B.4):** token = `settings.secrets.upstox_analytics_token`; if missing, every method raises `BrokerError("upstox token not set")` and callers skip. Candles: URL in B.4; split a long range into ≤ 28-day requests for minute intervals; response `data.candles` is newest-first: sort ascending; timestamps are ISO with `+05:30`. Limit 2,000 requests per 30 minutes: space requests ≥ 1.0 s. Quotes: `GET https://api.upstox.com/v2/market-quote/quotes?instrument_key=k1,k2,…` (≤ 500 keys); response `data` is keyed by `NSE_EQ:SYMBOL` with `last_price, ohlc{open,high,low,close}, volume, depth{buy[],sell[]}, timestamp`; probe and record the exact shape. Instrument master: `https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz` → rows with `segment == "NSE_EQ"` and `instrument_type == "EQ"`: `trading_symbol → instrument_key` (`NSE_EQ|<ISIN>`).
**Tests:** all with recorded/synthetic JSON fixtures via MockTransport: login success and TOTP-retry path; funds parse strings to floats; quote chunking for 120 tokens = 3 calls spaced ≥ 1 s; AB1021 backoff sequence on `SteppingClock`; Upstox range splitting for 70 days of 1-minute = 3 requests; newest-first response is returned ascending; missing token raises `BrokerError`; no secret appears in captured logs.
**Acceptance:** the probe script output (non-secret) pasted; `stocksense doctor` shows `angel: login ok, available cash read` (the amount is printed only as "read", never the number, in doctor).

### T2.10 Intraday bars
**Provides:** `data.ingest_intraday.extend_5m(ctx, symbols)`, `backfill_1m(ctx, symbols, start)`, `load_bars(settings, symbols, start, end, interval="5m", basis="adjusted") -> pl.DataFrame`.
**Spec:**
- **Adjustment probe first:** KAMDHENU split 10-for-1 on 2025-01-08 (raw close 479.30 on 2025-01-07). Read the stored 5-minute bars for KAMDHENU on 2025-01-07: if prices are near 48, stored bars are back-adjusted; near 479, raw. Do the same for a fresh Angel `candles` call and a fresh Upstox call. Record all three answers in `docs/DATA.md`.
- **`load_bars(..., basis="adjusted")` makes the answer irrelevant:** for each (symbol, session) it rescales the session's bars by `k = panel.adj_open / first_bar_open` (prices × k, volume ÷ k), where `adj_open` comes from `panel/daily` (T3.2) and `first_bar_open` is the open of that session's 09:15 bar; if the 09:15 bar is missing use `k = panel.adj_close / last_bar_close`. (The bhavcopy open equals the 09:15 bar open at the median, measured in B.3.) Sessions with no panel row are dropped. `basis="raw_ratio"` returns bars untouched for within-session ratios only. Until T3.2 exists this function supports only `raw_ratio`; T3.2's card adds the adjusted path.
- `extend_5m`: for each of the existing 470 symbols, from its last stored bar to now, Angel `FIVE_MINUTE` in ≤ 90-day requests; `source='angel'`, `adj_asof=today`.
- `backfill_1m` (only if the Upstox token is set): symbol list = union over calendar years 2022…now of the 300 symbols with the highest median `turnover_inr` among series EQ in that year (computed directly from `bhavcopy_eq`); 1-minute candles from 2022-01-03 to yesterday, one request per symbol-month, into `bars_1m` (`source='upstox'`); resumable via `job_runs(job="bars_1m:<symbol>:<YYYY-MM>")`; about 400 symbols × 58 months ≈ 23,000 requests ≈ 7 hours at 1 request/s: run in the background and continue with other tasks. Expected ≈ 150–200M rows, 3–4 GB.
- Validation per symbol-session: 375 one-minute bars (09:15…15:29) or 75 five-minute bars for a full session; sum of bar volume within 5% of bhavcopy volume: violations are counted in the job summary, never dropped.
**Tests:** `load_bars` rescaling on a hand-made 2-session example with a 1:2 split between them (adjusted series is continuous); resumable month skipping; newest-first handling.
**Acceptance:** `data-status`: `intraday_bars` current; `bars_1m` present with its first date (or "skipped: no Upstox token").

### T2.11 Macro and market-wide extras
**Provides:** `data.ingest_macro.ingest_macro(ctx)`, `data.ingest_misc.ingest_participant_oi_range`, `ingest_fii_dii`, `ingest_deals_today`. **Parallel-safe:** yes. All non-blocking: any failure logs a warning and returns `failed` without stopping the pipeline.
**Spec:**
- Macro series and stooq symbols: `USDINR=usdinr`, `BRENT=cb.f`, `SPX=^spx`, `NDX=^ndx`, `US10Y=10usy.b`, `DXY=dx.f`, `GOLD=xauusd`; URL `https://stooq.com/q/d/l/?s=<sym>&i=d` (CSV `Date,Open,High,Low,Close,Volume`). If stooq returns "No data" or HTML for a symbol, skip that series and record it. `available_ts`: for US-session series (SPX, NDX, US10Y, DXY) = next calendar day 06:00 IST; for the rest = same day 23:30 IST.
- Participant OI: URL in B.4, first two lines are a title and the header; rows Client, DII, FII, Pro, TOTAL (drop TOTAL). Backfill backward from yesterday until 30 consecutive misses.
- FII/DII: API returns the latest day only; append daily. Seed history from the adopted `nse_snapshots/fii_dii` files.
- Deals: today's bulk and block CSVs appended daily (history starts now).
**Tests:** parsers on samples; `available_ts` rule.
**Acceptance:** `data-status` lists `macro_daily` (series count), `participant_oi` (first date), `fii_dii`, `deals`.

### T2.12 Live captures: pre-open auction and spreads
**Provides:** `data.ingest_misc.capture_preopen(ctx) -> int`, `capture_spreads(ctx, symbols, quotes_source) -> int`, `capture_quotes(ctx, quotes) -> int` (appends to `quotes_1m`).
**Spec:** Pre-open: `GET /api/market-data-pre-open?key=ALL` (referer `market-data/pre-open-market-cm-and-emerge-market`); each `data[i].metadata` has `symbol, previousClose, iep, finalPrice, lastPrice`; `detail.preOpenMarket` has `totalBuyQuantity, totalSellQuantity`; map `iep→indicative_price`, `finalPrice→final_price` (0 → null); `ts = clock.now()`. Probe the shape once and record; the adopted `nse_snapshots/pre_open` columns (`indicative_price, final_price, buy_quantity, sell_quantity`) confirm the fields exist. If `key=ALL` is refused, call `key=NIFTY`, `key=FO`, `key=OTHERS` and concatenate. Spreads: from `Quote` objects, rows with both bid and ask > 0.
**Tests:** parser on a recorded sample; zero `finalPrice` → null.

### T2.13 One command for everything
**Provides:** `data.ingest_all.catch_up(ctx) -> list[IngestResult]`; CLI `stocksense ingest --all --catch-up` and `stocksense ingest <dataset> --from --to`.
**Spec:** order: reference (incl. holidays) → eq+delivery → fo → index/vix → corporate actions → announcements (yesterday and today) → board meetings/results index → macro/misc → intraday 5m (and 1m if token). Each from its own last stored date to the last completed session. A failure in one dataset does not stop the others; the function returns all results and the CLI prints the `data-status` table and exits 1 if any required dataset (eq, delivery, corporate actions) is not fresh.
**Tests:** with fake ingesters: order respected; one failure does not stop the rest; exit code logic.

**Phase 2 done when:** `stocksense data-status` shows every required dataset fresh with zero issues; `docs/DATA.md` holds the before/after tables, true start dates, the announcements category table, and every probe outcome; suite green; pushed.

---

## PHASE 3 — Panel: stable identities, correct returns, the tradable universe

### T3.1 Security master
**Goal:** one permanent `sid` per company across symbol renames.
**Provides:** `panel.security_master.build_security_master(settings) -> pl.DataFrame` (writes `panel/security_master.parquet` and `panel/rename_autodetect.csv`); `sid_for(symbol, date)` lookup frame via `symbol_sid_map(settings) -> pl.DataFrame[symbol, valid_from, valid_to, sid]`.
**Spec:**
1. Nodes: every distinct `symbol` appearing in `bhavcopy_eq` with series in `EQ, BE, BZ`. For each: `first_date`, `last_date`, and the row values on those two dates (`first_prev_close`, `last_close`).
2. Edge A (official): for each `symbol_changes` row `(old, new, change_date)`: add edge `old → new` if `old.last_date` is within 15 calendar days before or 5 after `change_date`, and `new.first_date` is within 5 days before or 15 after it. Rows that fail this are ignored (many are mutual-fund or debt symbols not in the equity data).
3. Edge B (auto-detected): symbol `A` with `last_date = d1` not at the end of data (`d1 <` global last date − 30 days), symbol `B` with `first_date = d2`, `0 < sessions between d1 and d2 ≤ 5`, `|B.first_prev_close / A.last_close − 1| < 0.001`, `B` has no `listed_equities.listing_date` within 3 days of `d2`, and `A` has no outgoing official edge. If exactly one `B` matches `A` and exactly one `A` matches `B`: add the edge and append `(A, B, d1, d2, A.last_close, B.first_prev_close)` to `rename_autodetect.csv`.
4. A relisting of the **same** symbol after a long gap stays one node (same symbol = same sid). A new symbol appearing with a matching price after more than 5 sessions is NOT merged.
5. Chains (connected components, each node has at most one in and one out edge; if a node would get two outgoing edges keep the official one, else the closest in date) get one `sid`.
6. **Stable numbering:** if `security_master.parquet` already exists, every existing symbol keeps its `sid`; a chain containing an existing symbol takes that `sid` (if a chain contains symbols with two different existing sids, keep the smaller, log a warning to `adj_events`-style audit file `panel/sid_merges.csv`); new chains get `max_sid + 1…` in order of `first_date`, then symbol.
7. Output rows: one per (sid, symbol) with `valid_from = symbol.first_date`, `valid_to = symbol.last_date` (for the chain's last symbol: `2099-12-31`).
**Tests (synthetic market):** S07/S07N share a sid (official); S08/S08N share a sid (auto-detected) and appear in the CSV; S19 (new listing) is not merged with anything; a hand-made case where symbol X stops and unrelated Y starts next day at a different price → not merged; a hand-made case of the same symbol disappearing for 300 sessions and returning → one sid; running the build twice, then again after appending 5 new symbols, never changes an existing sid.
**Acceptance (realdata):** print number of sids, number of official and auto-detected merges, and the 10 most recent merges with names for eyeballing.

### T3.2 Adjustment factors and returns
**Goal:** a daily panel whose returns are correct across splits, bonuses, rights, demergers and renames, using the corporate-action table checked against price evidence (B.3 Finding 1).
**Depends on:** T3.1, T2.3, T2.6. **Provides:** `panel.adjust.compute_gap_factors(rows, ca) -> pl.DataFrame[sid, date, gap_factor, gap_src, r_obs, f_ca]`; `panel.returns.build_daily_panel(settings) -> None` (writes `panel/daily/`, `panel/adj_events.parquet`); the adjusted path of `load_bars` (T2.10).
**Spec — step by step:**
1. Rows: `bhavcopy_eq` with series in `EQ, BE, BZ`, joined to `sid`. If a sid has several rows on one date keep one by priority EQ > BE > BZ. Sort by `sid, date`. `lc` = previous row's `close` for the sid. `base = prev_close if prev_close > 0 else lc`. `r_obs = open / base`. `sessions_since_prev` from the calendar's session index.
2. **Factor events from the table.** Corporate actions are joined to `sid` through the symbol valid on `ex_date`. For each CA row with a numeric `factor_price` `f` (types split, bonus, bonus+split, consolidation; several on one date multiply): candidate rows = that sid's rows with date in `[ex_date − 5 calendar days, ex_date + 15 calendar days]`. Choose the candidate minimizing `|ln(r_obs / f)|`.
   - If that minimum ≤ 0.20 → assign `f` to that row; `gap_src = 'ca'` if its date equals `ex_date`, else `'ca_shifted'`.
   - Else, take the row at `ex_date` (or the first row after it within the window): if `|ln r_obs| > 0.15` and `sign(ln r_obs) == sign(ln f)` → assign `snap(r_obs)` with `gap_src='implied_at_ca'` (the table's ratio was wrong; the price is right).
   - Else → no factor; write an audit row with `gap_src='ca_ignored'` (factor 1.0).
3. **Rights** with `ratio_num=A`, `ratio_den=B`, `rights_price=S`: on the `ex_date` row (or first row after, within 5 sessions), `P = base`; if `S < P`: `f = (B·P + A·S) / ((A + B)·P)`, `gap_src='rights'`; else no factor.
4. **Events without a factor** (demerger, scheme_of_arrangement, capital_reduction, amalgamation, and split/bonus/rights rows that are `unparsed`): on the `ex_date` row (or first row after, within 5 sessions), if no factor was assigned by step 2–3 and `|ln r_obs| > 0.12` → `gap_factor = r_obs`, `gap_src='implied_event'` (the overnight move is treated as mechanical, i.e. zero overnight return that day).
5. **Suspected, no table row at all:** for rows still unassigned with `base ≥ 5`: if `ln r_obs < −0.28` and some `c` in `{1/2, 1/3, 1/4, 1/5, 1/6, 1/10, 1/20, 2/3, 3/4, 3/5, 2/5, 4/5, 5/6}` has `|r_obs/c − 1| ≤ 0.03` → `gap_factor = c`, `gap_src='suspected'`; if `ln r_obs > 0.60` and some `c` in `{2, 5, 10}` within 3% → `gap_factor = c`, `'suspected'`.
6. `snap(r)`: the nearest value in the step-5 sets if within 4%, else `r` itself.
7. All other rows: `gap_factor = 1.0`, `gap_src = ''`.
8. **Returns** (fractions): `ret = close / (base × gap_factor) − 1`; `ret_co = open / (base × gap_factor) − 1`; `ret_oc = close / open − 1`. First row of a sid: all three null (a listing-day move is not a return anyone could plan for).
9. **Dividends:** `div_ret = sum(dividend_amount on this ex_date) / base` (0 if none); `tret = ret + div_ret`.
10. **Adjusted prices:** per-row step `g = gap_factor × base / lc` (first row: 1). `adj[t] = product of g over all LATER rows of the sid` (latest row has `adj = 1`). `adj_open = open × adj`, likewise high, low, close; `adj_volume = volume / adj`. Property that must hold on every non-first row: `adj_close[t] / adj_close[t−1] − 1 == ret[t]` within 1e-9. Recomputed fully every night.
11. **Flags:** `flag_bigmove = |ret| > 0.40`; `flag_unexplained_gap = sessions_since_prev == 1 and |base/lc − 1| > 0.02 and gap_src == ''`; `flag_stale = sessions_since_prev > 5`; `flag_one_price = high == low`; `flag_locked_up = close == high and min over b in {0.02, 0.05, 0.10, 0.20} of |ret − b| ≤ 0.0025`; `flag_locked_dn` = mirror with `close == low` and `ret + b`.
12. Join `deliv_qty, deliv_pct` (on symbol, series, date) and `industry` (latest `sectors` snapshot; `'UNKNOWN'` if absent). Write by year. Write `adj_events.parquet` (every row with `gap_src != ''`, plus `ca_ignored`).
**Edge cases → tests (synthetic market unless stated):** S03 split: `adj_close` continuous, `ret` on session 200 equals truth within tick rounding (tolerance 0.003), `gap_src='ca'`; S05 bonus `'ca'`; S06 wrong-ratio bonus → `'implied_at_ca'` with factor 0.5; S14 unlisted split → `'suspected'`, factor 0.5; S16 demerger → `'implied_event'`, `ret_co == 0`; S25 rights → `'rights'` with `f = (4·P + 1·50)/(5·P)`; S13 dividend: `div_ret = 5/base`, `tret − ret == div_ret`; S07 rename: one continuous series; S09 suspension: `flag_stale` on the first row back; S15: `flag_locked_up` on the three sessions; a normal −9% overnight gap is NOT adjusted; a CA whose ex-date is 2 sessions before the price actually drops → `'ca_shifted'` on the right row (hand-made); the adj/ret identity on all rows (property test); for every stock, compounded `tret` over the whole synthetic history matches compounded `truth.true_ret` within 2%.
**Acceptance (realdata):** `stocksense build-panel` completes in < 5 minutes and prints: rows; counts by `gap_src`; and for KAMDHENU around 2025-01-08, JUBLFOOD around 2022-04-19, M&MFIN around 2020-07-22 (rights) and one demerger from `adj_events`, the five rows before/after with `close, gap_factor, ret, adj_close`. Expected: `ret` on each ex-date is a normal-sized daily return, not −50% or −80%. Also print the count of `flag_bigmove` rows within `u500`-eligible liquidity (expected: a few hundred over 16 years); paste the 20 largest for review into `docs/DATA.md`.

### T3.3 Universe
**Provides:** `panel.universe.build_universe(settings)` (writes `panel/universe/`). **Depends on:** T3.2.
**Spec (membership on date `t` uses only rows ≤ `t`):** eligible if: series on `t` is EQ; symbol not in `etf_list` and neither symbol nor name matches `etf_name_regex`; ≥ 252 prior rows; traded on ≥ 57 of the last 60 sessions (count rows in the 60-session calendar window ending at `t`); `close ≥ 20`; `med_turn_60` (median `turnover_inr` of the sid's last 60 rows ending at `t`) ≥ ₹1 crore; not `flag_stale` in the last 20 sessions. `turnover_rank` = rank of `med_turn_60` among eligible (1 = largest). Membership with hysteresis, computed forward in time per sid: enter `u500` when rank ≤ 500; stay while eligible and rank ≤ 600. `u200`: enter at rank ≤ 200 with `med_turn_60 ≥ ₹10 crore`; stay while rank ≤ 250 and the floor holds. `in_fno`: date ≥ 2024-07-08 → symbol has an `STF` row in `fo_bhavcopy` that date (carry forward the last known set over gaps of up to 5 sessions); earlier dates → `turnover_rank ≤ 150` (documented proxy).
Hysteresis must not use future data: implement as a per-date loop over sessions carrying the previous day's member set (4,200 iterations over small frames is fine), not as a vectorized trick.
**Tests:** S17 (ETF) never a member; S21 (price < 20) never; S19 not before 252 sessions after listing; S11 is a member before it is delisted; hysteresis: a stock oscillating between rank 480 and 560 stays in; truncation test: membership for dates ≤ T identical with and without later data (5 cut dates).
**Acceptance (realdata):** print average `u500` and `u200` size per year (expect ≈ 290–350 in 2011–2013 rising to 500 from 2017; `u200` smaller early).

### T3.4 Market frame
**Provides:** `panel.market.build_market(settings)` (writes `panel/market.parquet`).
**Spec:** for each date `t`: members = `u500` on the previous session; `ew500_ret/co/oc` = mean of members' `tret / ret_co / ret_oc` on `t` (ignoring nulls and `flag_bigmove` rows); `ew200_ret` likewise with `u200`; `nifty50_ret`, `nifty500_ret` from `index_daily` closes (null before data starts); `n_u500`; `breadth_200` = share of `u500` members with `adj_close >` their own 200-row mean; `xs_dispersion` = cross-sectional std of members' `ret`.
**Tests:** tiny market by hand; uses previous-day membership (a stock entering today does not count today).
**Acceptance (realdata):** print mean of `ew500_co` and `ew500_oc` in bps for 2012 → 2025-09: expect roughly +22 and −20 (B.3 Finding 3). This reproduces the planning measurement and proves the panel is wired correctly.

**Phase 3 done when:** the four acceptance outputs are in `docs/DATA.md`; suite green; pushed.

---

## PHASE 4 — Features and labels

### T4.1 Feature framework
**Provides:** `features.base.feature` decorator and `FeatureSpec(name, group, kind, fn, window, version, label)`; `features.registry.REGISTRY`; `features.build.build_features(settings, start=None, end=None, version=None)`; `features.build.load_features(settings, purpose, start=None, end=None, columns=None)`; helper expressions in `features/base.py`.
**Spec:**
- A feature function has signature `fn(ctx: FeatureContext) -> pl.LazyFrame` returning columns `date, sid, <name>` (several names allowed: a function may emit a family). `FeatureContext` gives lazy access to `panel` (daily panel, all sids, sorted by `sid, date`, with an integer `sx` = calendar session index), `market`, `universe`, raw datasets through `data.io.read`, and `as_of: date | None`. **When `as_of` is set every accessor filters its data to `date ≤ as_of` (and `ts ≤ as_of 20:00` for timestamped data) before returning it.** This single choke point is what the truncation test exercises.
- `kind='xs'` (cross-sectional): one value per (date, sid). `kind='mkt'`: one value per date, broadcast.
- Rolling helpers (all per sid, over the sid's rows): `roll_sum(col, w)`, `roll_mean`, `roll_std` (ddof=1), `roll_min`, `roll_max`, `roll_corr(a, b, w)`, `roll_slope_r2(col, w)`, each wrapping polars `rolling_*().over("sid")` with `min_samples = w`, and each returning null when the window's calendar span is too wide: `sx − sx.shift(w−1).over("sid") + 1 > ceil(1.25·w) + 2` (so a window that straddles a suspension is invalid).
- Build procedure: (1) for each registered function, compute over all sids, inner-join to the `(date, sid)` rows where `u500` is true, collect, hold as a frame keyed by (date, sid); (2) per date, for each `xs` feature: winsorize at the 1st and 99th percentiles of that date's non-null values, keep the winsorized value as `<name>__raw` (Float32), and compute `<name> = (rank_average − 1)/(n_valid − 1) − 0.5` as Float32, null stays null, all null if `n_valid < 50`; (3) `mkt` features are stored unranked under their own names; (4) write one parquet per year plus `manifest.json`.
- Version `N`: stored in `features/CURRENT` (a text file with the number). A build with a changed set of `(name, version)` pairs writes to `v<N+1>` and updates `CURRENT`. Old versions are kept until deleted by hand.
- `load_features(purpose="research")` truncates at `lockbox_start − 1 day` (T6.1 hooks in here); `purpose="production"` does not.
- Incremental nightly mode: `build_features(start=last_date − 400 calendar days)` recomputes only recent dates but still reads full history (windows need it), and rewrites only the affected year files, merging by key.
**Tests:** helper correctness on hand-made series (each helper: one test with numbers); the wide-span rule nulls a window containing S09's suspension; ranking: 5 values `[10, 20, 20, 40, null]` → `[−0.5, −0.125, −0.125, 0.5, null]`… with n_valid = 4: ranks 1, 2.5, 2.5, 4 → `(r−1)/3 − 0.5` = `−0.5, 0.0, 0.0, 0.5`; fewer than 50 valid → all null (use a threshold parameter in the test); manifest lists every feature; version bump on a changed feature version.

### T4.2 The truncation test (the look-ahead guard)
**Files:** `tests/unit/test_feature_truncation.py`.
**Spec:** parameterized over every `FeatureSpec` in the registry. Fixture: the synthetic market written to a temp root, panel/universe/market built. Compute the feature with `as_of=None`; for each of 5 cut dates (sessions 300, 380, 450, 520, 590) compute it with `as_of=cut`; assert that for all rows with `date ≤ cut` the two results are equal (`null == null`; floats within 1e-9), and that the truncated result has no row after `cut`. Additionally a meta-test registers a deliberately leaky feature (`next day's ret`) in a temporary registry and asserts the checker fails on it. The lists of cut dates and tolerance are constants at the top of the file.
**Acceptance:** this test file passes after every feature task; it is part of the default suite.

### T4.3 Feature library v1
Each bullet group is one task (`T4.3a` … `T4.3k`), **Parallel-safe: yes** (each touches only its own module and its own test file; the registry imports all modules). Notation: `la = ln(1 + ret)` (adjusted log return), `lco = ln(1 + ret_co)`, `loc = ln(1 + ret_oc)`; all rolling windows are over the sid's rows with the wide-span rule; `P = adj_close`. Each feature gets (i) a hand-computed unit test on 6–30 rows, (ii) a plain-English `label` for explanations, (iii) automatic coverage by T4.2. Signs are natural (no pre-flipping); the model learns direction.

**T4.3a `features/price.py` — momentum and trend:** `mom_5 = roll_sum(la, 5)`; `mom_21`; `mom_63`; `mom_126`; `mom_252_skip21 = roll_sum(la, 252) − roll_sum(la, 21)`; `mom_accel = mom_63 − (mom_126 − mom_63)`; `hi52_prox = P / roll_max(adj_high, 252)`; `lo52_prox = P / roll_min(adj_low, 252)`; `days_since_hi52 = (rows since the 252-row max of adj_high) / 252`; `ma_gap_20 = P/roll_mean(P, 20) − 1`, `ma_gap_50`, `ma_gap_200`; `trend_r2_63 = R² of ln P on row number over 63 rows × sign(slope)`; `resid_mom_126` = sum over 126 rows of `e = la − beta_63 × ln(1 + ew500_ret)` divided by `roll_std(e, 126)`, with `beta_63 = roll_cov(la, lm, 63)/roll_var(lm, 63)` shifted by one row (so the beta used on day `s` was estimated through `s − 1`).
**T4.3b `features/price.py` — reversal and price action:** `ret_1 = la`; `rsi_14` (Wilder smoothing: first average = simple mean of the first 14 gains/losses, then `avg = (prev × 13 + x)/14`; implement with `ewm_mean(alpha=1/14, adjust=False, min_samples=14)` and test against a hand-computed 20-row example); `bb_z_20 = (P − roll_mean(P, 20))/roll_std(P, 20)`; `max_ret_21 = roll_max(la, 21)`; `min_ret_21`; `close_loc = (close − low)/(high − low)` (null if `high == low`); `close_loc_5 = roll_mean(close_loc, 5)`; `range_pct = (high − low)/close`; `nr7 = 1 if range_pct == roll_min(range_pct, 7) else 0`; `inside_day = high ≤ prev high and low ≥ prev low` (adjusted); `up_streak` / `down_streak` = consecutive rows with `la > 0` / `< 0`, capped at 10; `dist_round` = `|close − round(close / u) × u| / close` with `u = 10^floor(log10(close))`.
**T4.3c `features/overnight.py` — the overnight/intraday split (B.3 Findings 3–5):** `co_1 = lco`; `co_sum_5 = roll_sum(lco, 5)`; `co_sum_21`; `oc_1 = loc`; `oc_sum_5`; `oc_sum_21`; `co_minus_oc_63 = roll_sum(lco, 63) − roll_sum(loc, 63)`; `co_vol_share_63 = roll_var(lco, 63) / (roll_var(lco, 63) + roll_var(loc, 63))`; `gap_abs_z = |lco| / roll_std(la, 21).shift(1)`; `gap_fill_5` = mean over 5 rows of `−sign(lco) × loc`; `co_hit_21` = share of the last 21 rows with `lco > 0`.
**T4.3d `features/risk.py`:** `vol_21 = roll_std(la, 21)`; `vol_63`; `vol_ratio = vol_21 / roll_std(la, 126)`; `parkinson_21 = sqrt(roll_mean(ln(high/low)², 21) / (4 ln 2))`; `gk_21 = sqrt(roll_mean(0.5 ln(high/low)² − (2 ln 2 − 1) ln(close/open)², 21))` (clip the inner mean at 0); `downside_vol_63 = sqrt(roll_mean(min(la, 0)², 63))`; `skew_63`, `kurt_63` (sample skewness and excess kurtosis over 63 rows); `atr_14_pct` = Wilder ATR(14) of adjusted prices ÷ `P`; `beta_126` = `roll_cov(la, lm, 126)/roll_var(lm, 126)`; `idio_vol_63 = roll_std(la − beta_126.shift(1) × lm, 63)`; `corr_mkt_63 = roll_corr(la, lm, 63)`.
**T4.3e `features/volume.py`:** `log_turn_21 = ln(roll_mean(turnover_inr, 21))`; `amihud_21 = roll_mean(|ret| / (turnover_inr / 1e7), 21)`; `turn_shock = turnover_inr / roll_mean(turnover_inr, 60).shift(1)`; `vol_shock_5 = roll_mean(adj_volume, 5) / roll_mean(adj_volume, 60).shift(1)`; `turn_trend = roll_mean(turnover_inr, 21)/roll_mean(turnover_inr, 126)`; `avg_trade_size_rel = (turnover_inr/n_trades) / roll_mean(turnover_inr/n_trades, 60).shift(1)` (null where `n_trades` null); `zero_ret_frac_63` = share of 63 rows with `ret == 0`; `size_proxy = ln(roll_mean(turnover_inr, 252))`; `pv_corr_21 = roll_corr(la, ln(turnover_inr), 21)`.
**T4.3f `features/delivery.py` (null before 2019-08-23):** `deliv_pct`; `deliv_z = (deliv_pct − roll_mean(deliv_pct, 60).shift(1)) / roll_std(deliv_pct, 60).shift(1)`; `deliv_val_shock = (deliv_qty × close) / roll_mean(deliv_qty × close, 60).shift(1)`; `deliv_up = deliv_z × sign(la)`; `deliv_trend = roll_mean(deliv_pct, 5) / roll_mean(deliv_pct, 60)`; `deliv_z_5 = roll_mean(deliv_z, 5)`.
**T4.3g `features/market.py` (kind `mkt` unless noted; all from `panel/market`, `india_vix`, `index_daily`, `macro_daily`, `fii_dii`, `participant_oi`):** `vix`, `vix_chg_5 = ln(vix/vix[−5])`, `vix_pctile_252` (rank of today's VIX in its last 252 values, 0–1); `mkt_ret_1/5/21` (sums of `ln(1+ew500_ret)`), `mkt_vol_21`, `mkt_co_5`, `mkt_oc_5`; `breadth_200`; `breadth_chg_10`; `adv_dec_10` = mean over 10 sessions of (advancers − decliners)/(advancers + decliners) in `u500`; `xs_dispersion`, `xs_disp_ratio = xs_dispersion / mean of its last 60`; `nifty_pe_pctile_756` (null before index data); macro: for each of USDINR, BRENT, SPX, NDX, US10Y, DXY, GOLD: `<s>_chg_1` and `<s>_chg_5` using only values with `available_ts ≤ decision cutoff of t` (**this means the US close of the night after session `t` is NOT in features for `t`**; it becomes available to the 09:07 auction decision as an auction feature, T4.4); `fii_net_5`, `dii_net_5` (sum of net ÷ 1e4; null before data); `fii_idx_fut_net` = FII (long − short)/(long + short) in index futures from `participant_oi`, and its 5-session change. Cross-sectional context (kind `xs`, same module): `ind_mom_21`, `ind_mom_126` = mean of the industry's members' `mom_*` excluding the stock; `rel_ind_mom_21 = mom_21 − ind_mom_21`; `ind_breadth_50` = share of the industry's `u500` members above their 50-row mean; industries with < 5 members or `UNKNOWN` → null.
**T4.3h `features/calendar_feats.py` (kind `mkt`):** `dow` (0–4, 5–6 for special sessions), `dom_bucket` (0 = first 3 sessions of the month, 2 = last 3, 1 = other), `days_to_month_end` (sessions), `is_expiry_week` (the week containing the monthly F&O expiry; derive the expiry date as the latest `expiry_date` in that month among `IDF`/`STF` rows of `fo_bhavcopy`; before 2024-07 use the last Thursday of the month, moved to the previous session if a holiday), `sessions_to_next_holiday` (cap 10), `is_budget_week` (the week containing 1 February, or the actual budget date for years listed in a small constant table).
**T4.3i `features/derivs.py` (null before 2024-07-08 and for non-F&O names):** per underlying and date, using the nearest expiry with ≥ 3 calendar days left (`near`) and the next one: `fut_basis = (near futures settle − close)/close × 365/days_to_expiry`; `fut_oi_chg_5 = (sum futures OI − its value 5 sessions ago)/its 20-session mean`; `oi_price_quad` = +1 if 5-day price up and OI up, −1 price down OI up, +0.5 price up OI down, −0.5 price down OI down; `pcr_oi` = put OI / call OI (near); `pcr_oi_z_20`; `atm_iv` = mean of the Black–Scholes implied vols of the call and the put at the strike nearest the underlying close (near expiry; r = 0.065; time = calendar days/365; solve with `scipy.optimize.brentq` on [0.01, 5.0]; null if `settle_price < 0.5`, if intrinsic value exceeds the price, or if no root); `iv_minus_rv = atm_iv − vol_21 × sqrt(252)`; `iv_chg_5`; `skew_put_call` = IV of the put ≈ 5% below spot − IV of the call ≈ 5% above; `max_pain_dist` = (strike minimizing total option-holder payoff − spot)/spot; `days_to_expiry`. Vectorize: compute IVs only for the ≤ 4 needed strikes per underlying-day (≈ 200 underlyings × 550 days × 4), not for the whole chain.
**T4.3j `features/events.py` (each null before its source starts):** `days_to_exdiv` (sessions to the next dividend ex-date known by `t`, cap 30, null if none), `exdiv_yield` (that dividend ÷ close); `days_to_board_meeting` (cap 30; only meetings with `announced_ts ≤ cutoff of t`); `days_since_result` (sessions since the last results event, cap 90), where a results event = an announcement with category matching `(?i)financial result|outcome of board meeting` plus a `results_index` filing within 2 days, timestamp = the earliest of them; `result_abret` = sum over the first 2 sessions on/after the results timestamp (the session of the timestamp if it is before 15:30, else the next) of `(tret − beta_126 × ew500_ret)`, available from the second of those sessions onward; `result_abvol` = mean turnover of those 2 sessions ÷ the 60-row mean before them; `result_gap` = `ret_co` of the first session after the results timestamp; `ann_count_5`, `ann_count_21` (announcements with decision timestamp ≤ cutoff); `corp_event_next_5` = 1 if a split/bonus/rights/demerger ex-date falls in the next 5 sessions and was already in the table by `t` (use the CA row's first-seen date; for historical rows assume known 7 sessions before ex-date). The taxonomy counts and `text_score_5` are added by T8.
**T4.3k `features/intraday.py` (null where no bars; uses `load_bars(basis="raw_ratio")`, within-session ratios only):** `last_hour_ret = ln(close_15:25 / close_14:25)`; `first_hour_ret = ln(close_10:10 / open_09:15)`; `first5_ret = ln(close_09:15bar / open_09:15)`; `vwap_dev = last close / VWAP − 1` (VWAP = Σ typical price × volume / Σ volume over the session's bars); `intraday_rv = sqrt(Σ squared 5-min log returns)`; `rv_ratio = intraday_rv / vol_21`; `vol_last_hour_share`; `vol_first5_share`; `open_drive = ln(close_09:40 / open_09:15) / atr_14_pct`.
The Kronos features (`features/kronos_feats.py`) are T7.5.

### T4.4 Auction features (for S2)
**Provides:** `features.build.build_auction_features(settings, start, end)` writing `features/auction/part-YYYY.parquet` keyed by `(trade_date, sid)`; `auction_features_live(ctx, preopen_frame, panel_tail) -> pl.DataFrame` for the 09:07 decision.
**Spec:** for trade date `d` (the session after decision date `t`) and `u200` members as of `t`: `gap = O(d) / (C(t) × gap_factor(d)) − 1` (in history `O(d)` is the official open; live it is the 09:07 indicative price, and `gap_factor(d)` comes from a corporate action known for `d`); `gap_z = gap / vol_21(t)`; `gap_rel_mkt = gap − median gap of u200`; `gap_rel_ind = gap − median gap of the industry (≥ 5 members)`; `mkt_gap = median gap`; `gap_rank` (cross-sectional rank); `preopen_imbalance = (buy_qty − sell_qty)/(buy_qty + sell_qty)` (null in history before captures exist); overnight macro moves that became available after the cutoff of `t`: `spx_overnight`, `ndx_overnight`, `usdinr_overnight`, `brent_overnight`.
**Honesty rule built in:** every stored historical row carries `gap_source='official_open'`; every live row `gap_source='indicative_0907'`; T10 records both the indicative and the final open so their difference is measured from day one.
**Tests:** hand-computed gap across a split day (must be near zero, not −50%); truncation: features for `d` use nothing after `O(d)`.

### T4.5 Labels
**Provides:** `labels.forward.build_labels(settings)`; `load_labels(settings, purpose, …)`.
**Spec:** for decision date `t` and sid `i` in `u500` on `t`; `d1 = next session after t`; `dh = shift(d1, h)`.
- `y_oc = close(d1)/open(d1) − 1`; `end_oc = d1`.
- `y_c2o_h = adj_open(x)/adj_close(d1) − 1 + Σ div_ret over the sid's rows in (d1, x]`, `h ∈ {5, 10, 20}`, where `x = dh` if the sid has a row on `dh`; `end_h = x`.
- If the sid has no row on `dh`: if it trades again later, `x` = its first row after `dh` (a real holder is stuck until then); if it never trades again and its last row is more than 20 sessions before the end of data: `y = adj_close(last)/adj_close(d1) − 1 − 0.20`, `label_delisted = true`, `end_h = shift(last, 1)`; if the data simply ends first: null.
- `y_c2c_1 = adj_close(d1+1)/adj_close(d1) − 1 + div_ret(d1+1)`; `y_o2o_5 = adj_open(d1+5)/adj_open(d1) − 1 + Σ div_ret over [d1+1, d1+5]` (reference only, to keep measuring the entry-time effect).
- Any label is null if the sid has no row on `d1`, or if `flag_unexplained_gap` occurs on any row in the holding window.
- `entry_blocked_close = flag_locked_up(d1)`; `entry_blocked_open = (open(d1) == high(d1)) and min_b |ret_co(d1) − b| ≤ 0.0025`.
- `z_<name>` = `(y − median)/(1.4826 × MAD)` within the date's `u500`, clipped to ±5 (MAD = 0 → null); `r_<name>` = rank scaled to [−0.5, 0.5].
**Tests (tiny market, numbers by hand):** each label on plain days; `y_c2o_5` across a split (≈ the true return, not −80%); across a dividend (dividend added once); at the end of data (null); S11 delisting (haircut applied, flag set); S09 suspension inside the window (exit at first row back); S15 lock → `entry_blocked_close`; `end_*` dates correct.
**Acceptance (realdata):** `stocksense build-features` then `build-labels` for 2011-01-03 → latest finish in < 20 minutes total; print for each label mean and count 2012 → 2025-09; expected means near `y_oc ≈ −0.0020`, `y_c2o_5 ≈ +0.0035`, `y_o2o_5 ≈ +0.0015` (B.3); print the manifest's feature count (expected 120–140 before Kronos/news) and each feature's `first_valid_date` and coverage into `docs/DATA.md`.

### T4.6 Reproduce the planning measurements (a wiring check, not research)
**Spec:** a script `stocksense research ic-table` (research purpose, so it stops at 2025-09-30) printing the mean daily rank IC and t-stat of each feature against `r_y_oc`, `r_y_c2o_5`, `r_y_c2o_20`. Compare with B.3 Finding 5: `vol_21` vs `y_oc` should be about −0.06 (the table there shows the negated signal), `co_sum_5` vs `y_oc` about −0.054, `mom_252_skip21` vs `y_c2o_20` about +0.04, `deliv_z` vs `y_oc` about +0.03. Signs and rough sizes must agree (within ±0.015). A large disagreement means a bug in the panel, features or labels: stop and find it before Phase 5.
**Acceptance:** the table pasted into `docs/DATA.md` with a one-line verdict per row.

**Phase 4 done when:** truncation test green for every feature; the IC table agrees with B.3; suite green; pushed.

---

## PHASE 5 — Costs and the backtest engine

### T5.1 Indian cost model
**Provides:** `costs.india.trade_cost(side: Literal["BUY","SELL"], product: Literal["delivery","intraday"], value_inr: float, sched: Costs, first_sell_of_scrip_today: bool = True) -> CostBreakdown(brokerage, stt, exchange, sebi, stamp, gst, dp, total)`; `round_trip_cost_frac(product, value_inr, sched) -> float`; `min_position_for_fixed_cost(product, sched, max_fixed_bps) -> float`.
**Spec:** `brokerage = max(brokerage_min, min(brokerage_cap, brokerage_rate × value))`; `stt = value × (stt_buy or stt_sell)`; `stamp = value × stamp_buy` on BUY only; `exchange = value × exchange`; `sebi = value × sebi`; `dp = dp_per_scrip_sell_inr` on a delivery SELL when `first_sell_of_scrip_today`, else 0; `gst = gst × (brokerage + exchange + sebi + dp)`; `total` = sum. `round_trip_cost_frac` = (buy cost + sell cost at the same value) / value. `min_position_for_fixed_cost`: the smallest value `v` (search ₹1,000 steps from ₹5,000 to ₹5,00,000) at which `(2 × brokerage(v) + dp) × (1 + gst) / v ≤ max_fixed_bps / 1e4`.
**First step:** open `https://www.angelone.in/pricing` and `https://www.angelone.in/angel-one-brokerage-charges` (browser tool or fetch); record the equity delivery and intraday brokerage, DP charge and the page date in `docs/DECISIONS.md`; update `config/costs.toml` and `verified_on` if they differ from D.7. If the pages disagree, keep the costlier figure.
**Tests (D.7 schedule; values to the paisa, tolerance 0.001):**
- Delivery BUY ₹1,00,000: brokerage 20.00, stt 100.00, stamp 15.00, exchange 2.97, sebi 0.10, gst 0.18 × 23.07 = 4.1526, dp 0 → total **142.2226**.
- Delivery SELL ₹1,05,000: brokerage 20.00, stt 105.00, stamp 0, exchange 3.1185, sebi 0.105, dp 20.00, gst 0.18 × 43.2235 = 7.78023 → total **156.00373**.
- Delivery BUY ₹4,000: brokerage = max(5, min(20, 4)) = 5.00.
- Intraday BUY ₹50,000: brokerage 20, stt 0, stamp 1.50, exchange 1.485, sebi 0.05, gst 0.18 × 21.535 = 3.8763 → **26.9113**. Intraday SELL ₹50,000: brokerage 20, stt 12.50, exchange 1.485, sebi 0.05, gst 3.8763 → **37.9113**. Round trip = 64.8226 = 12.96 bps.
- Second sell of the same scrip the same day: dp 0.
- `round_trip_cost_frac("delivery", 10_000)` > `round_trip_cost_frac("delivery", 100_000)`; print both in the test comment (≈ 0.0069 and ≈ 0.0030).
- `min_position_for_fixed_cost("delivery", sched, 15)` returns a value between ₹45,000 and ₹50,000 with the D.7 schedule (2 × 20 + 20 = 60, × 1.18 = 70.8; 70.8 / 0.0015 = ₹47,200).

### T5.2 Slippage model
**Provides:** `costs.slippage.slip_frac(turnover_rank: int, vol_21: float, order_value: float, med_turn_20: float, when: Literal["open","close","now"], table: SlippageTable) -> float`; `SlippageTable.load_latest(settings)`; `estimate_open_penalty(settings) -> dict` (run once here).
**Spec:** `half_spread` = the bps of the first bucket whose upper rank ≥ `turnover_rank` (rank > 500 → 20 bps) ÷ 1e4; `impact = impact_k × vol_21 × sqrt(order_value / med_turn_20)`; `penalty = open_penalty_bps/1e4` for `open`, `close_penalty_bps/1e4` for `close`, 0 for `now`; `slip = half_spread + impact + penalty`. `SlippageTable` rows are dated; the newest row with `date ≤ as-of` is used (the nightly job appends rows; T11.1).
`estimate_open_penalty`: from `bars` (research range only), per rank bucket the median of `|VWAP(09:15–09:30) / open_09:15 − 1|` and of `|VWAP(15:00–15:30) / last close − 1|`; write the two medians (in bps, averaged over buckets) to `docs/DECISIONS.md` and into `costs.toml` `open_penalty_bps` / `close_penalty_bps` if they exceed the defaults.
**Tests:** monotone in order value; rank 30 vs rank 400 spreads 1.5 vs 12 bps; zero order value → half-spread + penalty only; hand example: rank 120, `vol_21 = 0.02`, order ₹1,00,000, `med_turn_20 = ₹10 crore`, `open` → 0.0004 + 0.5 × 0.02 × sqrt(1e5/1e8) + 0.0003 = 0.0004 + 0.000316 + 0.0003 = **0.001016**.

### T5.3 Fill rules (shared by backtest and paper broker)
**Provides:** `backtest.fills.try_fill(order: Order, bar: DayBar, ctx: FillContext) -> Fill | Reject` where `DayBar` has `open, high, low, close, prev_base (= base × gap_factor), ret_co, ret, flags`, and `FillContext` has `turnover_rank, vol_21, med_turn_20, slippage table, costs, capital`.
**Spec, in order:**
1. No bar that session → `Reject("no_trade")`.
2. `flag_one_price` → `Reject("one_price")` for both sides.
3. `when='open'`: BUY rejected `"locked_up_open"` if `open == high` and `min_b |ret_co − b| ≤ 0.0025`; SELL rejected `"locked_dn_open"` if `open == low` and `min_b |ret_co + b| ≤ 0.0025`. `when='close'`: BUY rejected `"locked_up_close"` if `flag_locked_up`; SELL rejected `"locked_dn_close"` if `flag_locked_dn`.
4. Reference price = `open` or `close`. Participation: `max_value = max_adv_participation × med_turn_20`; quantity = `min(order.qty, floor(max_value / ref))`; if 0 → `Reject("too_small")`.
5. Fill price = `ref × (1 + slip)` for BUY, `ref × (1 − slip)` for SELL, rounded to the 0.05 tick away from the trader (up for BUY, down for SELL).
6. Costs from T5.1 on `qty × fill_price`.
A rejected SELL of a held position is retried by the caller at the next opportunity (next `open`, then next `close`, …); a rejected BUY is dropped.
**Tests:** each rejection reason on a hand-made bar; tick rounding direction; participation cap reduces quantity; S15's locked sessions block buys at open and close.

### T5.4 Backtest engine
**Provides:** `backtest.engine.run_backtest(panel_access, calendar, policy: Policy, start, end, capital_inr, costs, slippage, risk) -> BacktestResult(daily: pl.DataFrame, trades: pl.DataFrame, rejects: pl.DataFrame, summary: dict)`; `Policy` protocol: `decide(t: date, book: BookState, capital_inr: float) -> list[Order]` called after the close of each session `t` (orders are for later sessions and carry `trade_date` and `when`), plus optional `decide_auction(d: date, book, capital_inr, auction_view) -> list[Order]` called before the open of `d` for S2-style policies (the engine passes the day's open-derived auction features). `run_grid(...)` repeats over `costs.grid.capital_inr`.
**Spec — the daily loop for session `d`:**
1. **Corporate actions:** for each held position with a panel row on `d`: if `gap_factor ≠ 1`: `qty_new = qty / gap_factor`; keep `floor(qty_new)` shares and credit the fractional remainder × `base × gap_factor` to cash; `entry_price ×= gap_factor`. Credit `qty × div_ret × base` to cash as a dividend (record in `daily.dividends`).
2. **Open phase:** pending orders with `trade_date == d`, `when == 'open'`: all SELLs first (including retries of previously rejected sells), then BUYs in descending `priority`. A BUY is scaled down to available cash (after costs); short SELLs (intraday product, no existing position) require cash ≥ order value and reserve it.
3. **Close phase:** the same for `when == 'close'`; additionally every open intraday position is closed at the close (`when='close'`, reason `squareoff`), covering shorts.
4. **Mark:** equity = cash + Σ qty × close (a held sid with no row today is marked at its last close; after 20 sessions without a row and no later row in the data, it is written off at 80% of the last close, reason `delisted`).
5. Record `date, equity, nav (= equity/capital × 100), cash_frac, gross, n_pos, turnover (value traded / equity), costs_frac, day_ret`.
6. **Decide:** call `policy.decide(d, book, equity)`; store the returned orders.
Invariants asserted every session (raise `AssertionError` in debug mode): cash ≥ −0.01; no delivery position with negative quantity; no intraday position survives the close; equity equals cash + positions within ₹0.01.
Determinism: no randomness in the engine; same inputs give identical outputs.
Performance target: 16 years × 20 positions in < 60 s; hold the panel as per-date dicts of numpy arrays keyed by sid, built once.
**Reference policies for testing (in `backtest/engine.py`):** `BuyAndHold(sids)`, `EqualWeightUniverse(rebalance_every=21)`, `Clairvoyant(label_frame, k=20)` (buys the true top-k of `y_c2o_5` at the close, sells at the open 5 sessions later), `RandomPolicy(seed, k=20, h=5)`.
**Tests (tiny and synthetic markets; hand-computed):** buy-and-hold one stock for 10 sessions: final equity = initial − buy cost + price change − nothing else (to ₹0.01); a split mid-holding leaves equity unchanged apart from the day's real return; a ₹5 dividend on 100 shares adds ₹500 cash; a locked-limit sell is retried and fills on the first unlocked session; cash never negative with random orders (hypothesis, 200 examples); intraday short: sell at open 100 × ₹200, cover at close ₹198 → gross +₹200 minus both costs (numbers in the test); S11 delisting written off; the same run twice gives identical frames; results at ₹25,000 capital are worse than at ₹5,00,000 for a 20-name policy (fixed-fee effect).
**Integration test `test_engine_truth.py`:** on the synthetic market with `planted_ic=0.10`, `Clairvoyant` has a net Sharpe > 5 and `RandomPolicy` has a net return within ±1 Sharpe-standard-error of (universe return − its measured cost drag); this proves the engine neither leaks nor loses returns.
**Acceptance (realdata):** `stocksense backtest --policy equal_weight_u500 --capital 500000 --from 2014-01-01 --to 2025-09-30` prints CAGR, Sharpe, max drawdown, turnover and cost drag, and its gross daily return correlates > 0.98 with `ew500_ret`.

### T5.5 Metrics
**Provides:** `backtest.metrics.perf_stats(daily_ret: np.ndarray, benchmark: np.ndarray | None, periods=252) -> dict`; `ic_stats(pred_frame, label_col) -> dict`; `trade_stats(trades) -> dict`; `yearly_table(...)`.
**Spec:** `cagr = (Π(1+r))^(252/n) − 1`; `vol = std(r, ddof=1) × sqrt(252)`; `sharpe = mean(r)/std(r, ddof=1) × sqrt(252)` (0 if std = 0); `sortino` with downside deviation `sqrt(mean(min(r,0)²)) × sqrt(252)`; `max_drawdown` and its start/trough/recovery dates from the cumulative curve; `calmar`; `hit_rate_days`; `skew`, `kurtosis` (non-excess, as needed by T6.4); `excess_tstat` = Newey–West t-stat (5 lags, Bartlett weights) of `mean(r − benchmark)`; `frac_positive_years`. `ic_stats`: per date Spearman between `score` and the label (dates with < 50 pairs skipped) → `mean_ic`, `ic_std`, `ic_tstat = mean/std × sqrt(n_dates)`, `ic_by_year`, decile mean label and the monotonicity (Spearman of decile index vs decile mean). `trade_stats`: count, win rate, average win/loss, profit factor, average holding sessions, average cost fraction, reject counts by reason.
**Tests:** each statistic on a hand-computed 6–10 element series (numbers in comments); Newey–West against a direct formula on a 20-element series; drawdown dates on a constructed curve.

**Phase 5 done when:** truth integration test green; real-data acceptance pasted into `docs/DECISIONS.md`; pushed.

---

## PHASE 6 — Validation harness (finished before any model result is looked at)

### T6.1 Lockbox
**Provides:** `validation.lockbox.research_end(settings) -> date` (= `lockbox_start − 1 day`); `guard(frame, date_col, purpose)`; `open_lockbox(settings, clock, candidate_id, reason)` context manager.
**Spec:** `load_features`, `load_labels`, `load_auction_features`, panel accessors and `load_bars` all take `purpose ∈ {"research", "production", "lockbox"}`. `research` → rows after `research_end` are dropped. `lockbox` → allowed only inside `open_lockbox(...)`, which appends `{"ts", "candidate_id", "reason", "git_sha"}` to `state/lockbox_touches.jsonl` on entry and raises `LockboxError` if `candidate_id` already appears in the file. `production` → everything (used by fitting an already-selected spec and by the live system).
**Tests:** research load has no row ≥ lockbox start; lockbox load outside the context raises; second open with the same id raises; the file gains exactly one line per open.

### T6.2 Splits
**Provides:** `validation.splits.walk_forward_folds(settings, calendar) -> list[Fold(name, train_end, test_start, test_end)]`; `train_mask(label_end: pl.Series, fold, embargo_sessions=5)`; `inner_validation_split(dates, label_end, months=12, embargo_sessions=5)`; `cscv_blocks(n_rows, n_blocks=16)`.
**Spec:** folds are calendar years: test = `[Y-01-01, Y-12-31]` for `Y = 2014 … 2025`, the last one ending at `research_end` (12 folds). Training rows for a fold and a target: `date ≥ dev_start` and `end_<target> <` the session `embargo_sessions` before `test_start`. Sample weight `w = 0.5 ** (sessions_between(date, test_start) / 756)`. Inner validation: the last 12 months of the training rows by `date`; inner-training rows additionally need `end_<target> <` (inner-validation start − embargo).
**Tests:** for random targets with horizons 1–20 (hypothesis): no training row has `end ≥ test_start − embargo`; folds tile 2014-01-01 → research end with no gap or overlap; weights are 1.0 at the boundary and 0.5 three years back.

### T6.3 Trial registry
**Provides:** `validation.trials.register_trial(state, family, target, spec) -> trial_id` (idempotent on `spec_hash`: re-registering the same spec returns the existing id), `finish_trial(trial_id, metrics, returns_path)`, `n_trials(family=None)`, `sharpe_variance_across_trials(family=None)`, `seed_planning_trials(state)`.
**Spec:** migration 2 (D.6). `register_trial` is called **before** any test-fold evaluation of a spec. There is no delete. `seed_planning_trials` inserts 30 rows with `family='planning-exploration'`, `status='done'`, `note` = "signals examined on development data while writing the plan, 2026-11-11" and null metrics, exactly once (guarded by a `kv` flag); it runs inside `State.migrate()`.
**Tests:** idempotent registration; count includes the 30 seeds; no public function removes rows (assert the module has no `DELETE`).

### T6.4 Deflated Sharpe ratio
**Provides:** `validation.dsr.deflated_sharpe(sr: float, T: int, skew: float, kurt: float, n_trials: int, var_sr_trials: float) -> float` where `sr` is the **per-period (daily)** Sharpe (mean/std, not annualized) and `kurt` is non-excess kurtosis (normal = 3).
**Spec:** `γ = 0.5772156649`; `N = max(n_trials, 2)`; `V = var_sr_trials if finite and > 0 else 1/T`; `SR0 = sqrt(V) × ((1 − γ) × Φ⁻¹(1 − 1/N) + γ × Φ⁻¹(1 − 1/(N·e)))`; `DSR = Φ( (sr − SR0) × sqrt(T − 1) / sqrt(1 − skew × sr + ((kurt − 1)/4) × sr²) )`.
**Tests:** hand example in the test: `sr = 0.08, T = 2500, skew = −0.3, kurt = 5, N = 50, V = 0.0004`: `Φ⁻¹(0.98) = 2.0537`, `Φ⁻¹(1 − 1/(50e)) = Φ⁻¹(0.99264) = 2.4398`, `SR0 = 0.02 × (0.4228 × 2.0537 + 0.5772 × 2.4398) = 0.02 × 2.2766 = 0.045532`; denominator `sqrt(1 + 0.024 + 0.0064) = 1.01509`; `z = (0.08 − 0.045532) × 49.99 / 1.01509 = 1.6975`; `DSR = Φ(1.6975) ≈ 0.9552` (assert within 0.002). DSR decreases when `N` rises; `N = 1` behaves as `N = 2`.

### T6.5 Probability of backtest overfitting
**Provides:** `validation.pbo.pbo_cscv(returns: np.ndarray[T, n_trials], n_blocks=16) -> dict(pbo, logits)`.
**Spec:** cut rows into 16 contiguous blocks; precompute per block and trial the sum and sum of squares of returns and the count; for each of the C(16, 8) = 12,870 choices of in-sample blocks: in-sample and out-of-sample Sharpe per trial from the block sums; `n* = argmax` in-sample; `ω = (rank of trial n* among out-of-sample Sharpes, 1 = worst) / (n_trials + 1)`; `λ = ln(ω / (1 − ω))`; `pbo = share of λ ≤ 0`. Needs ≥ 2 trials, else returns `pbo = nan`.
**Tests:** 50 pure-noise trials (seeded) → pbo in [0.35, 0.65]; one trial with a real mean edge of 0.1 daily Sharpe among 49 noise trials → pbo < 0.10; runtime < 20 s for 3,000 × 50.

### T6.6 Gate
**Provides:** `validation.gate.evaluate_research_gate(candidate: CandidateResult, settings, state) -> GateReport`; `evaluate_lockbox_gate(...)`; CLI `stocksense validate <candidate_id> [--open-lockbox]`.
**Spec — the nine research lines, each PASS/FAIL with the number shown:**
1. `mean_rank_ic ≥ 0.02` (out-of-fold, on the candidate's primary target).
2. `ic_tstat ≥ 3.0`.
3. `net_sharpe ≥ 1.0` (walk-forward backtest, 2014 → research end, capital ₹5,00,000, all costs and slippage).
4. `net_sharpe_2x_slip ≥ 0.5` (same with every slippage component doubled).
5. `excess_tstat ≥ 2.0` (net daily return minus `ew500_ret`; for S2 the benchmark is zero).
6. `frac_positive_years ≥ 0.70`.
7. `dsr ≥ 0.95` using `n_trials()` over all families and the variance of per-period Sharpes of finished trials.
8. `pbo ≤ 0.30` over the daily net return columns of all finished trials of the same family (needs ≥ 10 trials, else the line reads `FAIL (too few trials to estimate)`; the fixed grids of Phase 7 and 9 produce more than 10).
9. `max_drawdown ≤ 0.25`.
`GateReport` is written to `models/<candidate>/validation.json` and printed as a table. `--open-lockbox` is refused unless all nine pass; then it runs the frozen spec's production-style walk-forward on `[lockbox_start, last date]` inside `open_lockbox` and checks `net_sharpe ≥ 0.5` and `net_return ≥ 0`.
**Tests:** a `CandidateResult` built from synthetic numbers passes/fails each line as constructed; `--open-lockbox` refused on a research fail.

### T6.7 Walk-forward runner
**Provides:** `validation.walkforward.run_walk_forward(spec: ModelSpec, settings, state, purpose="research", seeds=(1,2,3)) -> WFResult(preds: pl.DataFrame[date, sid, score, fold], fold_metrics, trial_id)`.
**Spec:** registers the trial; for each fold: build train/inner/test matrices (`load_features` + `load_labels`, drop rows with null target, drop `entry_blocked_*` rows for the relevant entry), fit by the model's two-step recipe (T7.x: fit on inner-train with early stopping on inner-validation → refit on all training rows with the found complexity), predict the test year, average seeds after ranking each seed's scores per date; writes `preds/<model_id>/` with `mode='walkforward'`; returns IC stats per fold. Resumable: a fold whose prediction file exists is skipped. Feature matrix dtype Float32; target `r_<name>` unless the model says otherwise.
**Tests (synthetic market, `planted_ic = 0.06`, a registered test feature equal to `alpha`):** a ridge model recovers out-of-fold mean IC between 0.04 and 0.08; with `planted_ic = 0` mean IC within ±0.01; resumability (second call fits nothing: assert via a counter).

### T6.8 Harness self-tests (integration, must pass before Phase 7 results are read)
`tests/integration/test_harness_selftest.py`: (a) a random-score "model" on the synthetic market **fails** the research gate; (b) the planted signal (`planted_ic = 0.08`) yields IC lines PASS; (c) a model given the leaky feature would pass, and the leaky feature is rejected by the truncation checker first (assert the checker raises); (d) the lockbox cannot be opened twice.

**Phase 6 done when:** T6.8 green; pushed.

---

## PHASE 7 — Models

Common contract (`models/base.py`): `ModelSpec(family: str, target: str, universe: str, features: list[str] | "all", params: dict, entry: str)`; class `Model` with `fit(X, y, w, dates, X_val=None, y_val=None, dates_val=None, seed=0) -> FitInfo`, `refit_full(X, y, w, dates, fit_info, seed)`, `predict(X) -> np.ndarray`, `save(dir)`, `load(dir)`, `explain(X, feature_names) -> np.ndarray | None` (per-row, per-feature contributions). `X` is a Float32 numpy array with NaN for missing.

Time budget rule for every training task: time one fit first, multiply out, and if the total exceeds the card's budget, reduce in this order: number of seeds (min 2), then `n_estimators` cap / epochs, then retrain every 2 years instead of every year (the model from fold `Y` also predicts year `Y+1`). Record the choice.

### T7.1 Floors: single factors and ridge
**Provides:** `models.linear.SingleFactor(feature, sign)`, `models.linear.RidgeModel`.
**Spec:** `SingleFactor.predict` returns `sign × feature`. `RidgeModel`: NaN → 0.0 (the cross-sectional median after ranking), `mkt` features excluded, closed-form ridge with sample weights (`solve((XᵀWX + αI), XᵀWy)`), `α ∈ {1, 10, 100, 1000}` chosen by inner-validation mean daily rank IC. `explain` = `coef × x`.
**Trials to run (each registered):** single factors `−vol_21`, `−co_sum_5`, `deliv_z`, `−ret_1`, `−mom_5`, `mom_252_skip21`, `mom_126` against the three S1 targets and `y_oc`; ridge on all `xs` features for each of the four targets.
**Acceptance:** an IC table (mean IC, t-stat, by year) for all of the above in `docs/RESULTS-G1.md` section 1. These are the bars everything else must clear.

### T7.2 LightGBM
**Provides:** `models.gbm.GbmModel` (objective `regression` or `lambdarank`).
**Spec:**
- Fixed parameters: `boosting_type="gbdt", learning_rate=0.05, n_estimators=1500, subsample=0.7, subsample_freq=1, reg_lambda=10.0, max_bin=127, min_child_samples` from the grid, `colsample_bytree` from the grid, `num_leaves` from the grid, `deterministic=True, force_row_wise=True, n_jobs=8, verbose=−1, random_state=seed`.
- Regression: label `r_<target>`. Lambdarank: label = decile index 0–9 of the target within the date; `group` = row counts per date in date order (rows sorted by date); `lambdarank_truncation_level=50`; `label_gain = [2**i − 1 for i in range(10)]`.
- Early stopping: custom `feval` returning `("mean_ic", value, True)` where value = mean over inner-validation dates of the Spearman correlation between prediction and `r_<target>` (pre-compute per-date index slices once; use `scipy.stats.rankdata`); `lgb.early_stopping(stopping_rounds=100, first_metric_only=True)`; `metric="None"`.
- Two-step fit: (1) inner-train/inner-validation → `best_iteration`; (2) `refit_full` on all training rows with `n_estimators = ceil(1.1 × best_iteration)`, no early stopping.
- Grid (12 configurations): `num_leaves ∈ {15, 31, 63}` × `min_child_samples ∈ {200, 1000}` × `colsample_bytree ∈ {0.4, 0.7}`.
- **Hyperparameter selection uses inner validation only, at three anchor folds** (test years 2016, 2019, 2022), on a 30% random subsample of dates (seeded): pick the configuration with the best mean inner-validation IC across the three anchors, per (target, objective). Selection touches no test fold and registers no trial.
- Then run the full 12-fold walk-forward (T6.7) once per selected spec with 3 seeds: targets `y_c2o_5`, `y_c2o_10`, `y_c2o_20` (universe `u500`, all features) with `regression`; `y_oc` (universe `u200`, all features **plus** the auction features of T4.4) with `regression`; and `lambdarank` for the one S1 target with the best regression out-of-fold IC t-stat. Five walk-forward trials in total.
- `explain`: `booster.predict(X, pred_contrib=True)`; drop the last column (the bias); contributions + bias sum to the prediction within 1e-5.
- Diagnostics written with each trial: gain importance by feature group; out-of-fold IC with and without each late-starting group (delivery, derivatives, events, intraday), measured only on dates where the group exists.
- Budget: 10 hours. Run as a background job with a log file; the orchestrator moves on to T7.3–T7.5 preparation meanwhile.
**Tests:** on the synthetic market (`planted_ic = 0.08`): out-of-fold IC > 0.04; `explain` additivity; determinism (same seed → identical predictions); the custom metric equals a direct per-date Spearman on a toy example; lambdarank group construction (sum of groups = rows).
**Acceptance:** `docs/RESULTS-G1.md` section 2: for each of the five trials the IC table by year, decile spreads, and the selected hyperparameters.

### T7.3 The project's own sequence network (GPU)
**Provides:** `models.seqnet.SeqNet` (torch module), `SeqNetModel` (the `Model` wrapper), `features.build.build_sequence_tensors(settings, version)`.
**Spec:**
- Sequence input per (date `t`, sid): the last 60 rows of the sid ending at `t` (invalid if the wide-span rule fails → sample dropped), 12 channels: `la, lco, loc, ln(high/low), close_loc, ln(turn_shock), deliv_z (0 if null), ln(1+ew500_ret), mkt_vol_21, vix_chg_1, la − ln(1+ew500_ret), ln(adj_volume / its 60-row mean)`. Standardize the per-stock return-like channels (1, 2, 3, 4, 11) by the stock's own `vol_63` as of `t − 60` (causal), clip everything to ±10.
- Stored as float16 memory-mapped arrays per year: `features/seq/v<N>/X_YYYY.npy` `[n, 60, 12]` and an index parquet `(row, date, sid)`.
- Network: sequence branch `GRU(input=12, hidden=H, num_layers=2, dropout=p, batch_first=True)` → last hidden state (H); tabular branch: input `[x, isnan(x)]` with NaN→0 (2 × n_features) → `Linear(→256) → LayerNorm → GELU → Dropout(0.3) → Linear(→64) → GELU`; head: `concat (H + 64) → Linear(→64) → GELU → Dropout(p) → Linear(→1)`.
- Training: one batch = all samples of one date (shuffle the order of dates each epoch); loss = `1 − Pearson(pred, z_<target>)` within the batch (skip batches with < 50 samples); `AdamW(lr=1e-3, weight_decay=1e-4)`, cosine schedule over the epochs, ≤ 30 epochs, early stop on inner-validation mean daily rank IC with patience 5; `torch.autocast("cuda", dtype=torch.float16)` + `GradScaler`; gradient clip 1.0; `cudnn.benchmark=False`.
- Checkpoint after every epoch to `models/_work/seqnet/<spec_hash>/<fold>/seed<k>/epoch_<n>.pt` (`model, optimizer, scheduler, scaler, epoch, best_ic, best_state`) and resume from the newest on restart. Out-of-memory handling: catch `torch.cuda.OutOfMemoryError`, `torch.cuda.empty_cache()`, split the date batch in two halves (loss computed per half), continue.
- Two-step fit like T7.2: inner split to find the best epoch count, then refit on all training rows for `ceil(1.1 × best_epoch)` epochs.
- Grid (inner validation at anchors 2019 and 2022 only): `H ∈ {32, 64}` × `p ∈ {0.2, 0.4}`.
- Run the walk-forward for **one** target: the S1 target with the best GBM IC t-stat; retrain every 2 years; 2 seeds. Budget 8 hours, background.
- `explain` returns `None` (explanations come from the GBM).
**Tests (CPU, tiny):** forward shape; the IC loss is 0 for perfectly correlated outputs and 2 for perfectly anti-correlated; on the synthetic market with a planted signal placed in a tabular feature, 3 epochs reach inner IC > 0.03; checkpoint-resume produces the same final weights as an uninterrupted run with the same seed on CPU; NaN inputs produce finite outputs.
**Acceptance:** `docs/RESULTS-G1.md` section 3: IC by year for the seqnet trial and its correlation with the GBM scores.

### T7.4 Model store
**Provides:** `models.store.save_model(settings, state, model, spec, train_end, meta) -> model_id`, `load_model(model_id)`, `set_status(model_id, status, note)`, `current(family_prefix, status)`, `append_preds(model_id, frame, mode)`.
**Spec:** migration 3 (D.6). Directory `models/<model_id>/`: `spec.json`, `weights.*`, `feature_manifest.json` (ordered feature names + feature version), `fit_info.json`, `train_range.json`, `git_sha.txt`, later `validation.json`. Status changes append a `model_events` row. At most one `champion` per strategy family prefix (`s1`, `s2`, `s3`): promoting one demotes the other to `shadow` in the same transaction.
**Tests:** save/load round trip gives identical predictions; champion uniqueness; events recorded; `append_preds` upsert on (date, sid).

### T7.5 Kronos
Five sub-tasks, in order. GPU work runs under `state/gpu.lock`.

**T7.5a Vendor and verify.**
```powershell
git clone --depth 1 https://github.com/shiyu-coder/Kronos "$env:TEMP\kronos_src"
git -C "$env:TEMP\kronos_src" rev-parse HEAD
```
Copy `model/` → `third_party/kronos/model/`, `LICENSE` → `third_party/kronos/LICENSE`, `tests/test_kronos_regression.py` and `tests/data/regression_*.csv` → `third_party/kronos/tests/` (keep the relative layout the test expects, or adjust only its path constants). Write the SHA to `third_party/kronos/COMMIT`. Import path: `from kronos.model import Kronos, KronosTokenizer, KronosPredictor` (`third_party/kronos/__init__.py` empty; the hatch `packages` entry from T1.1 makes it importable after `uv sync`; if not, add `third_party` to `[tool.pytest.ini_options] pythonpath` and to `sys.path` in `models/kronos_model.py`). If the vendored `model/__init__.py` uses absolute imports like `from model.module import …`, change them to relative (`from .module import …`) and mark each changed line `# STOCKSENSE PATCH: relative import`; list patches in `docs/DECISIONS.md`. Do not install the upstream `requirements.txt`.
**Acceptance:** the upstream regression test passes here (`.venv\Scripts\python.exe -m pytest third_party/kronos/tests -q -m "" -p no:cacheprovider`) on CPU. If it fails only because of pandas-3 behaviour inside `KronosPredictor` (e.g. `.dt` accessors on an index, chained assignment), patch minimally with the same marker and re-run. If outputs differ numerically beyond the test's tolerance on this torch version, record the maximum difference and continue only if the relative difference of close prices is < 1e-3.

**T7.5b Wrapper with sample paths (`models/kronos_model.py`).**
- `load_kronos(size="small", device="cuda")` → `(tokenizer, model)` in `eval()`: tokenizer `NeoQuasar/Kronos-Tokenizer-base` for `small`/`base`, `NeoQuasar/Kronos-Tokenizer-2k` for `mini`; `max_context` 512 (2048 for `mini`). Weights are cached by `huggingface_hub` under the user's HF cache.
- `sample_paths(tokenizer, model, x, x_stamp, y_stamp, max_context, pred_len, n_paths, T=1.0, top_k=0, top_p=0.9, clip=5.0, seed=0) -> torch.Tensor[n_series, n_paths, pred_len, 6]` (normalized units): **copy the body of upstream `auto_regressive_inference` verbatim** (B.4 describes it) with `sample_count = n_paths`, call `torch.manual_seed(seed)` first, and replace its last three statements (`preds = z.cpu().numpy(); preds = np.mean(preds, axis=1); return preds`) with `return z[:, :, -pred_len:, :]` after the reshape to `[n_series, n_paths, seq, 6]`.
- `forecast_panel(frames: list[KronosInput], pred_len, n_paths, …) -> np.ndarray[n_series, n_paths, pred_len, 6]` in price units: per series reproduce upstream pre-processing exactly: columns in the order `open, high, low, close, volume, amount`; `mean, std` over the lookback rows; `(x − mean)/(std + 1e-5)`; clip ±5; time features in the order `minute, hour, weekday, day, month` from the session dates (daily bars: minute 0, hour 0) for both history and the `pred_len` future sessions (from the trading calendar); after sampling, de-normalize `× (std + 1e-5) + mean`. This is the only place pandas is used (to call `calc_time_stamps` if convenient; otherwise compute the five integers directly and skip pandas).
- Mixed precision: try `torch.autocast("cuda", dtype=torch.float16)`; keep it only if the mean path differs from the float32 mean path by < 0.02 normalized units on a 16-series test batch with the same seed; otherwise run float32.
- Batch sizing: start at 64 series × 20 paths; on `torch.cuda.OutOfMemoryError` halve the series count and retry; log the batch size that worked.
**Tests:** (CPU, `mini` or a 2-layer randomly initialised model with the tokenizer mocked for speed where possible; one `gpu`-marked test with real weights) with a fixed seed, `sample_paths(...).mean(dim=1)` equals upstream `auto_regressive_inference` output within 1e-4; output shapes; de-normalization inverts normalization on a constant-plus-noise series; series with a zero-variance column (volume all equal) do not produce NaN.

**T7.5c Kronos features (`features/kronos_feats.py`).**
- For decision date `t` and each `u500` sid with ≥ 256 valid rows (wide-span rule over 256): input = last 256 rows of `adj_open, adj_high, adj_low, adj_close, adj_volume, turnover_inr`; `pred_len = 5`, `n_paths = 20`, `T = 1.0`, `top_p = 0.9`, `seed = int(t.strftime("%Y%m%d"))`.
- From paths (index 0 = first future session): `kr_ret5_mean` = mean over paths of `close[4]/last_close − 1`; `kr_ret5_std` = std over paths; `kr_up_prob` = share of paths with `close[4] > last_close`; `kr_oc1_mean` = mean of `close[0]/open[0] − 1`; `kr_co1_mean` = mean of `open[0]/last_close − 1`; `kr_range5` = mean over paths and steps of `ln(max(high, open, close)/min(low, open, close))`; `kr_volchg5` = mean forecast volume ÷ mean of the last 20 actual volumes.
- Stored one file per date: `features/kronos/<tag>/YYYY-MM-DD.parquet` (`tag` = `small-zs` for zero-shot); a date whose file exists is skipped (resumable). Registered as `xs` features that read these files (null where missing); they pass the truncation test trivially because each file is computed from data ≤ its date (the truncation test runs them on the synthetic market with a tiny random-weight model via a fixture flag `STOCKSENSE_KRONOS_TEST_MODEL=1`).
**Run:** time three dates; then compute every 5th session of 2014-01-01 → research end if the estimate is ≤ 10 hours, else every 10th. Background job.

**T7.5d Zero-shot evaluation.** On the computed dates: standalone IC table of the seven features vs `y_oc`, `y_c2o_5`, `y_c2o_20`; correlation with `mom_21`, `ret_1`, `vol_21`; incremental test: refit the selected GBM spec for `y_c2o_5` and `y_oc` on those dates only, with and without the Kronos features (same folds) → difference in out-of-fold IC with a paired t-test across dates. **Admission rule (fixed now):** incremental mean IC ≥ 0.002 with paired t ≥ 2 on either target → compute all dates and add the features to the feature version. Otherwise write "Kronos zero-shot: no incremental value" with the tables. Either way continue to T7.5e. Registered as trial family `kronos-zs`.

**T7.5e Fine-tune on NSE (`models/kronos_finetune.py`).** Single-GPU loop reproducing the upstream recipe (B.4):
- Data: windows of `lookback + pred + 1 = 256 + 10 + 1 = 267` consecutive valid rows of one sid, six channels as in T7.5c, sampled uniformly at random from all `u500`-member rows whose **last** row date is before the anchor's train end minus 25 sessions; validation windows from the 12 months before the train end. Normalize each window by the mean/std of its first 256 rows, clip ±5.
- Step: `with torch.no_grad(): s1, s2 = tokenizer.encode(batch_x, half=True)`; `logits = model(s1[:, :-1], s2[:, :-1], batch_stamp[:, :-1, :])`; `loss, s1_loss, s2_loss = model.head.compute_loss(logits[0], logits[1], s1[:, 1:], s2[:, 1:])`; backward; `clip_grad_norm_(model.parameters(), 3.0)`.
- Optimizer `AdamW(lr=4e-5, betas=(0.9, 0.95), weight_decay=0.1)`; `OneCycleLR(max_lr=4e-5, total_steps=N, pct_start=0.03, div_factor=10)`; batch 16 with gradient accumulation 3 (effective 48 ≈ upstream 50); `N = 20,000` optimizer steps or 6 hours, whichever first; validate every 1,000 steps on 200 fixed validation batches; keep the best-validation-loss weights.
- **Checkpoint every 500 optimizer steps** to `models/_work/kronos_ft/<anchor>/step_<n>.pt` (`model, optimizer, scheduler, step, best_val, rng states`), keep the last 3; on start, resume from the newest. A crash or out-of-memory error must never cost more than 500 steps. On OOM: halve the batch, double the accumulation, continue from the last checkpoint.
- Tokenizer frozen (stage A). Stage B (tokenizer fine-tuned first with lr 2e-4, per upstream `train_tokenizer.py`) is a separate later trial and is not run in this task.
- Anchors: train ends 2017-12-31, 2020-12-31, 2023-12-31. Feature tag `small-ft`: a decision date uses the newest anchor model whose train end precedes it; dates before 2018-01-01 use zero-shot weights.
- Evaluate exactly as T7.5d (trial family `kronos-ft`), same admission rule.
- Budget: 3 anchors × ≤ 6 h. Background.
**Tests:** one optimization step on CPU with a tiny random model reduces the loss on a repeated batch after 20 steps; checkpoint-resume continues the step counter and the LR schedule; window sampler never returns a window whose last date ≥ train end − 25 sessions (hypothesis).
**Acceptance for T7.5:** `docs/RESULTS-G1.md` section 4 with both evaluations and a one-line decision each. Recorded caveats: the public weights were pre-trained on data to roughly 2024–2025, possibly including NSE symbols, so pre-2025 zero-shot scores may be flattered and the lockbox/forward results are what count; the larger test of Kronos is on 5-minute bars in Phase 13.

### T7.6 Ensemble
**Provides:** `models.ensemble.blend_equal(pred_frames)`, `blend_online(pred_frames, label_frame, target, eta=20.0, halflife=60, floor=0.05) -> (blend, weights_by_date)`, `expected_return_map(blend, label_frame, target, window=504)`.
**Spec:** inputs are per-model scores ranked per date to [−0.5, 0.5]. Online weights for date `t`: `S_i(t) = Σ over dates s with end_target(s) < t of 0.5^((t − s in sessions)/halflife) × IC_i(s)`; `w_i ∝ exp(eta × S_i(t) / Σ_s decay weights)`; floor each at `floor`, renormalize. `expected_return_map`: for each date `t`, slope `b(t)` of the regression (through the origin) of realized raw label on blended score over the trailing 504 dates whose labels were known before `t`; `E[ret] = b(t) × score`, with `b` floored at 0.
**Trials:** equal blend and online blend of the surviving models per strategy family (2 trials per family).
**Tests:** weights sum to 1 and respect the floor; with one model given a planted IC and another noise, the weight of the good model exceeds 0.8 after 250 dates; no label with `end ≥ t` influences weights for `t` (construct a case where using it would change the answer).

**Phase 7 done when:** `docs/RESULTS-G1.md` sections 1–4 exist; every model family has out-of-fold predictions stored; suite green; pushed. No gate decision yet (that is T9.4).

---

## PHASE 9 — Portfolio construction, risk, and the Gate G1 decision

(Phase 9 comes before Phase 8 on purpose: the system goes live on price-based models first; news is added to a running system.)

### T9.1 S1 policy: the multi-day delivery book
**Provides:** `portfolio.construct.S1Policy(spec, settings, score_source, panel_access, costs, slippage)` implementing `Policy.decide`; pure helper `build_s1_orders(t, scores_t, er_t, book, equity_inr, panel_t, cfg) -> list[Order]`. The same class is used by the backtest and by the live daemon.
**Spec — everything happens after the close of decision date `t`; `d1` = next session; `h` = the model target's horizon; `K` defined below:**
1. **Inputs for `t`:** blended score and expected `h`-session return `er` per `u500` sid; `vol_21`, `turnover_rank`, `med_turn_20`, `industry`, flags; upcoming corporate events.
2. **Capital-aware breadth:** `min_pos = max(risk.min_position_inr_floor, min_position_for_fixed_cost("delivery", costs, risk.max_fixed_cost_bps))`; `K = clamp(floor(equity × gross_target / min_pos), 0, s1.k_max)`. `K == 0` → no delivery entries (the report says "capital below one minimum position").
3. **Exposure target:** `gross_target = min(risk.max_gross, risk.target_vol_annual / port_vol)` where `port_vol = sqrt(wᵀ Σ w × 252)` for the equal-weight candidate top-`K`, `Σ` = Ledoit–Wolf covariance (`sklearn.covariance.LedoitWolf`) of the candidates' last 126 `ret` values (rows with any null dropped; fewer than 60 common rows → use `mean(vol_21)² × (0.3 + 0.7·I)` as a fallback covariance). If the optional overlay is on: multiply by 0.5 when `breadth_200 < 0.30 and mkt_ret_21 < 0`. Book drawdown rule: multiply by 0.5 if the book NAV is ≥ `book_dd_half` below its peak; set to 0 if ≥ `book_dd_stop` below (stays 0 until `stocksense resume --book <name>` writes a `kv` flag).
4. **Exits (orders: SELL, `when='open'`, `trade_date=d1`):** for each holding with `held = sessions since entry`: (a) `stop`: `close(t) ≤ entry_price × (1 − max(risk.stop_min, risk.stop_vol_mult × vol_21_at_entry × sqrt(h)))`; (b) `horizon`: `held ≥ h − 1` (so it leaves at `O(entry + h)`) **unless** it is in today's top `K` by score and eligible, in which case its entry date is reset to `d1` and nothing trades (reason `extended`); (c) `rank_decay`: `held ≥ 2` and score rank > `3K`; (d) `universe_exit`: no longer in `u500` or `flag_stale`.
5. **Eligibility for entry:** in `u500`; `er − round_trip_cost_frac("delivery", target_value) − 2 × slip_frac(…, when="close") > 0`; no split/bonus/rights/demerger ex-date within `no_entry_sessions_before_split` sessions; no board meeting for results within `no_entry_sessions_before_results` sessions; not `flag_locked_up` today; not already held; not sold today.
6. **Selection:** slots = `K −` (holdings remaining after exits). Take the highest-scored eligible names until slots are filled, skipping a name if adding it would put its industry above `max_industry_weight` (industry `UNKNOWN` capped at the same limit).
7. **Weights for new entries:** `raw_i = (0.5 + score_i) / vol_21_i` (score in [−0.5, 0.5] so the term is in [0, 1]); scale so that new + existing weights sum to `gross_target`, cap each name at `min(max_name_weight, 1.5 × gross_target / K)`, redistribute excess pro-rata once; drop any name whose weight × equity < `min_pos` (and give its weight to cash, not to others). Existing holdings are not resized (no trading for small weight drift).
8. **Orders:** BUY, `when='close'`, `trade_date=d1`, `qty = floor(weight × equity / close(t))`, `priority = score`. If no name is eligible: no entries; the decision record says "no stock clears costs today".
**Edge cases → tests:** equity ₹30,000 with D.7 costs → `K = 0`; equity ₹1,00,000 → `K = 2`; a holding at `held = h − 1` in the top `K` is extended, not sold; a holding outside `3K` after 2 sessions is sold; industry cap blocks the 4th name of one industry at K = 10 (30%); a name with results tomorrow is not entered; `er` below costs → not entered; weights never exceed caps and sum ≤ `gross_target` (hypothesis over random scores/vols); drawdown 12% halves gross; 16% zeroes it; `qty` integer and BUY value ≤ cash expected after tomorrow's open sells.

### T9.2 S2 policy: the auction book
**Provides:** `portfolio.construct.S2Policy` implementing `decide_auction(d, book, equity_inr, auction_view)`; helper `build_s2_orders(...)`.
**Spec (called before the open of `d`; live at 09:07):**
1. Universe: `u200` as of the previous session, with a pre-open price present.
2. Score = S2 model on `[features(t), auction features(d)]`; `er_oc` from the expected-return map of T7.6.
3. Cost per round trip `c = round_trip_cost_frac("intraday", pos_value) + 2 × slip_frac(…, when="open")`.
4. Longs: top names with `er_oc − c > 0`. Shorts: bottom names with `−er_oc − c > 0` and `in_fno` (when `shorts_require_fno`), not within 2% of an upper price band (live: from `Quote.upper_circuit`; backtest: skip names with `gap ≥ 0.18`).
5. Budget = `free cash × risk.s2_free_cash_frac` (S1 holdings are not touched); `min_pos = min_position_for_fixed_cost("intraday", …)` (≈ ₹31,500 with D.7); `n = min(s2.k_max, floor(budget / min_pos))`; split `n` between the two sides in proportion to how many names qualify, at most `ceil(n/2)` per side unless the other side is empty; equal weights.
6. Orders: `when='open'`, `trade_date=d`, product `intraday`; exits are the engine's/daemon's 15:18 square-off. Hard intraday stop (live only): cover/sell `when='now'` if the position is 3% adverse.
**Tests:** no short for a non-F&O name; nothing when no name clears costs; budget smaller than one `min_pos` → no orders; side split rule.

### T9.3 Pre-trade risk checks
**Provides:** `portfolio.risk.pre_trade_checks(ctx) -> list[Check(name, ok, detail)]`; `entries_allowed(checks) -> bool`.
**Checks:** `data_fresh` (panel has the last completed session); `features_fresh` (features exist for it, null share of each feature group within 2× its 60-day mean); `model_fresh` (champion/shadow fitted within 7 days); `universe_size` (300 ≤ n_u500 ≤ 500); `score_sane` (score cross-sectional std within [0.5×, 2×] of its 60-day mean, and < 2% nulls); `capital_read_today`; `quotes_alive` (a quote newer than 3 minutes, market hours only); `no_missing_session` (validator reports none in the last 10 sessions). Any failure → no new entries today (exits still run); all results are stored in the decision record.
**Tests:** each check with a passing and a failing fixture.

### T9.4 Meta-labeling (optional layer)
**Provides:** `models.meta.MetaModel`. **Spec:** samples = walk-forward S1 entries that the policy would have taken (from a backtest run with meta disabled); label = 1 if the trade's net return > 0; features = score, score rank, model disagreement (std across models), `vol_21`, `turnover_rank`, market-state features, `days_since_result`, `gross_target`; LightGBM binary (`num_leaves=7, min_child_samples=200, learning_rate=0.05`, early stopping on inner validation log-loss), fitted walk-forward with the same folds (training trades must have closed before the fold's test start). Use: `size multiplier = clip(2·p − 1, 0, 1)` applied to `raw_i`, and veto when `p < 0.5`. Registered as one more trial per S1 target; kept only if net walk-forward Sharpe improves.
**Tests:** no training trade closes after the test start; multiplier bounds.

### T9.5 The Gate G1 research run and decision
**Goal:** one honest verdict per strategy family.
**Spec, in order:**
1. For each S1 target (`y_c2o_5`, `y_c2o_10`, `y_c2o_20`): out-of-fold scores = online blend of that target's models. Portfolio trials (each registered, each a full walk-forward backtest 2014-01-01 → research end over the capital grid, plus the 2×-slippage run): `k_max ∈ {10, 20}` × overlay `{off, on}` = 4 per target → 12 S1 trials; plus the meta-label variant of the best one → 13.
2. S2: `k_max ∈ {5, 10}` → 2 trials (fills at the official open; flagged `auction-assumption`). **Supplementary strict replay:** for S2 trades falling inside the 5-minute-bar history (research range), recompute each trade's return from the close of the 09:15 bar (≈ 09:20) to the close of the 15:10 bar and report `strict_capture = strict net return / auction net return`. This number is reported next to every S2 result.
3. Reference rows in the same table: `ew500`, NIFTY 500 (where data exists), single-factor portfolios (`mom_252_skip21`, `−vol_21`) through the same S1 policy, and the same S1 policy with `entry='open'` (to quantify the entry-time effect of B.3 Finding 3).
4. Candidate per family = the trial with the highest net Sharpe at the reference capital. Run `stocksense validate` on each candidate (T6.6).
5. Outcomes: **all nine research lines pass** → run the lockbox step once → pass: status `champion`, `validated=1`; lockbox fail: status `shadow`. **Research fail** → status `shadow`, `validated=0`. S2 can be at most `shadow` here regardless (D.7 `[auction]`).
6. Whatever the outcome, the best S1 candidate and the best S2 candidate are fitted in production mode (all data, T11.1's routine) and saved, so Phase 10 has something to run.
7. Write `docs/RESULTS-G1.md` section 5: the full trial table (every trial, not just winners), the gate tables, the per-year table of the candidates, capital-grid results, cost drag, turnover, the strict-capture number, the total trial count used for DSR, and a plain-English verdict.
**Rule:** no threshold is changed and no extra grid point is added to rescue a fail. Improvements come later through new registered trials (Phase 8, 11, 13).
**Acceptance:** `docs/RESULTS-G1.md` complete; `stocksense doctor` lists the model ids and their status.

**Phase 9 done when:** T9.5 written up; at least one S1 model and one S2 model stored with status `champion` or `shadow`; pushed.

---

## PHASE 10 — The paper broker and the all-day daemon

### T10.1 Ledger
**Provides:** `paper.ledger.Ledger(state)` with `record_decision`, `add_orders` (INSERT OR IGNORE on `order_id`), `fill_order`, `cancel_order`, `positions(book) -> list[Position]` (derived from fills and `corp_adjust`), `cash_frac(book)`, `mark(book, date, prices)`, `nav_series(book)`, `apply_corp_action(book, sid, date, factor, div_frac)`, `add_correction(fill_id, new_price, reason)`.
**Spec:** migration 4 (D.6). **Append-only:** the module contains no `DELETE`, and the only `UPDATE`s are `orders.status/cancel_reason` and `fills.reconciled/price/official_ref` through `add_correction` (which also inserts a `corrections` row). A guard test greps the module. Books: `main` (S1 champion, or best S1 shadow flagged unvalidated), `s1-shadow-<n>` (other S1 models, max 3), `auction` and `auction_strict` (S2, same orders, different fill rule), `events` (Phase 8). Every book starts at NAV 100 on its first decision date.
**Book arithmetic is in weights, so changes in the real account's capital never distort performance:**
- At a fill on day `d`: `w_fill = qty × fill_price / capital_ref_inr(d)` (the actual weight after integer rounding, signed negative for a short); `cost_frac = total fees / capital_ref_inr(d)`. `cash_events` gets `−w_fill − cost_frac` for a BUY, `+w_fill − cost_frac` for a SELL (kind `trade`).
- Each position carries a current weight `w_i` (fraction of current book equity) and a last mark price `p_i`.
- `mark(book, d, prices)`: for each position `r_i = price_i / p_i − 1` (for a position opened today `p_i` is its fill price); book return `R = Σ w_i × r_i − Σ cost_frac of today's fills + Σ dividend fractions`; then `w_i ← w_i × (1 + r_i) / (1 + R)`, `p_i ← price_i`, `nav ← nav × (1 + R)`; stores `day_ret = R`, `cash_frac = 1 − Σ w_i`, `gross = Σ |w_i|`.
- A SELL of the whole position removes `w_i`; its realized return is already inside the marks.
- When the real capital changes between days, existing weights are left as they are (the book behaves as if existing positions were scaled with capital); only new orders are sized from the new capital. Rupee figures shown anywhere are `fraction × capital_ref_inr` of that day.
**Tests:** a buy then two daily marks then a sell reproduce a hand-computed NAV path (numbers in the test); duplicate `add_orders` is ignored; split adjustment keeps position value; correction changes price and logs; the no-DELETE guard.

### T10.2 Quote sources
**Provides:** `brokers.quotes.QuoteHub(sources: list[QuoteSource], clock)` with `get(symbols) -> dict[str, Quote]`; sources in order: `AngelQuotes` (FULL mode), `UpstoxQuotes`, `NseMcpQuotes` (tool `cm_get_stock_quote` on `https://mcp.nseindia.in/cmmkt/mcp`, marked `source='nse_mcp_delayed'`, one symbol per call, used only for held positions when both brokers fail).
**Spec:** try each source for the symbols still missing; a source that raises is skipped for 5 minutes (circuit breaker); the hub returns whatever it has and reports `degraded=True` if the first source did not serve. Quotes older than 3 minutes (by `Quote.ts` vs clock) are dropped as stale.
**Tests:** fallback order; circuit breaker timing on `SteppingClock`; stale quote dropped.

### T10.3 Paper broker
**Provides:** `paper.broker.PaperBroker(ledger, hub, costs, slippage, clock, calendar)` with `process_open(d)`, `process_now()`, `process_close(d)`, `squareoff_intraday(d)`, `reconcile(d, panel_rows)`.
**Spec:**
- **Provisional fills (live):** `process_open` at 09:15:10+: for pending `when='open'` orders of today: books `main`, `s1-shadow-*`, `auction`: provisional price = `Quote.open` (the day's opening price; if 0/None, wait and retry each minute until 09:25, then use `ltp`); book `auction_strict` and `events`: provisional price = `ltp` of the first quote with `ts ≥ 09:16:00`. `process_close` at 15:22: pending `when='close'` orders: provisional price = `ltp`. `squareoff_intraday` at 15:18: closes all intraday positions at `ltp`. `process_now`: `when='now'` orders at `ltp`.
- Apply the shared fill rules live: BUY blocked if `ask_qty == 0` or `ltp ≥ upper_circuit − 0.05`; SELL blocked if `bid_qty == 0` or `ltp ≤ lower_circuit + 0.05`; blocked orders stay pending and are retried every minute until 15:20 (`open`/`now`) or 15:28 (`close`), then cancelled with the reason. Slippage and costs as in the backtest. Fill price never better than the touch when bid/ask exist.
- **Reconciliation (evening, after the official bhavcopy):** for books `main`, `s1-shadow-*`, `auction`: every `open` fill is re-priced to `official open × (1 ± slip)` and every `close` fill to `official close × (1 ± slip)` through `add_correction` (reason `official_reprice`), and the backtest fill rules (T5.3) are re-applied on the official bar: a fill that would have been rejected is reversed by a correction to quantity 0 and the order marked cancelled (`locked_*`). This makes the paper books follow exactly the rules the backtests used. For `auction_strict` and `events`: keep the live price but check it lies within the official `[low, high]` (violations are flagged in the report and clamped). The live-vs-official difference of every fill is stored (`quote_ltp`, `official_ref`) for the slippage learning in T11.1.
- **Corporate actions:** before `process_open`, positions with a `gap_factor ≠ 1` or dividend for today (known from the CA table the previous evening) are adjusted via `apply_corp_action`; reconciliation corrects any that were only detected from prices.
**Tests (scripted quotes, `SteppingClock`):** open order fills at `Quote.open` for `auction` and at the 09:16 `ltp` for `auction_strict`; blocked buy at upper circuit retried then cancelled; square-off closes shorts; reconciliation re-prices to official values and reverses a fill that the official bar rejects; idempotent re-run of each process function (no double fills).

### T10.4 Schedule
**Files:** `config/schedule.toml`, `live/schedule.py`. **Provides:** `load_schedule(settings) -> list[JobDef(name, at, every_s, until, days, max_retries, retry_backoff_s, needs_market)]`.
`config/schedule.toml` (IST; `days="session"` = tradable session days; `"any"` = every day; `"weekend"`):

| Job | Time | Days | What it does |
|---|---|---|---|
| `preflight` | at start and 07:30 | any | single-instance lock; rebuild calendar; `catch_up` ingestion; rebuild panel/features/predictions for the last completed session if missing; resume unfinished evening jobs |
| `broker_login` | 07:40 | session | Angel login; instrument maps refresh |
| `overnight_data` | 07:45 | session | macro, corporate actions look-ahead, announcements since last poll, RSS (Phase 8) |
| `capital_am` | 08:45, retry every 2 min until 09:05 | session | `funds()` + holdings → `capital_ref_inr`; on final failure use yesterday's value with `capital_stale=1` |
| `plan_open` | 08:50 | session | re-validate last night's S1 exit orders against overnight corporate actions; mark them ready |
| `preopen_capture` | 09:00–09:08 every 60 s | session | `capture_preopen` |
| `s2_decide` | 09:07:00 | session | live auction features from the newest pre-open capture → S2 model → orders for `auction` and `auction_strict` + decision records |
| `open_fills` | 09:15:10, retry each minute until 09:25 | session | `PaperBroker.process_open` |
| `market_loop` | 09:16–15:30 every 60 s | session | quotes for the watch list (holdings + today's orders) → `quotes_1m`; pending fills; S2 hard stops; announcement poll |
| `spread_capture` | 09:20, 11:00, 13:00, 15:00 | session | FULL quotes for `u500` → `spreads` |
| `capital_pm` | 15:05 | session | re-read funds; compute free cash per book |
| `s1_entries` | 15:10 | session | finalize tonight's planned S1 entries with current cash, risk checks and today's events → orders `when='close'` |
| `s2_squareoff` | 15:18 | session | `squareoff_intraday` |
| `close_fills` | 15:22, retry each minute until 15:28 | session | `process_close` |
| `eod_mark` | 15:35 | session | provisional marks from last quotes |
| `eod_ingest` | 18:30, retry every 15 min until 23:30 | session | `catch_up`; done when eq, delivery, F&O, index and corporate actions are fresh |
| `eod_pipeline` | after `eod_ingest` | session | `reconcile` → `build_panel` → `build_features` (incremental) → Kronos features for today (if admitted) → `predict` for all active models → `s1_plan` (tomorrow's exits and intended entries + decision records) → `nightly_learn` (T11.1) → `drift` → `promote` → `report` |
| `housekeeping` | 23:45 | any | reconfigure log file; weekly SQLite `VACUUM`; delete cache files older than 400 days except bhavcopy |
| `weekly_research` | Saturday 09:00 | weekend | T11.5 |
| `monthly_kronos` | first Saturday 14:00 | weekend | Kronos refresh fine-tune if admitted |
| `collect_only` | 08:00–20:00 every 10 min | non-session weekdays and weekends | announcements and RSS polling |

Special sessions with `tradable=false` (Muhurat) run only `eod_ingest` and `eod_pipeline`.
**Tests:** parse; a job's due times for a given date; `days` filtering with a calendar fixture.

### T10.5 Daemon
**Provides:** `live.daemon.run(settings, clock=SystemClock(), once=False)`; `live.jobs.JOBS: dict[str, Callable[[JobContext, date], str]]`; `live.preflight`, `live.keepawake`, `live.notify`.
**Spec:**
- Start: acquire the single-instance lock (`msvcrt.locking` on `state/daemon.lock`, non-blocking; if held, print "already running" and exit 0). Configure logging. `State.migrate()`. `keepawake.on()` = `ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)`; released in `finally`.
- Loop every 5 s: `now = clock.now()`; `today = now.date()`; for each `JobDef` due (`at ≤ now`, within its window, right kind of day) whose latest `job_runs(job, today)` status is not `done`/`skipped`, and whose attempts < `max_retries + 1` and whose backoff has elapsed: run it. Repeating jobs (`every_s`) use a `kv` timestamp of the last run instead of `job_runs` per tick, and one `job_runs` row per day summarizing counts.
- A job runs in-process. Exceptions are caught at this boundary: log with traceback, `job_finish(..., 'failed', detail=repr(e)[:500])`, notify, continue. `DataNotAvailable` → status stays retryable without counting as an error in the report.
- Late start rules: `s2_decide` is skipped (`skipped: late`) if now > 09:08; `open_fills` if now > 09:25 (pending open orders are cancelled `late_start`; pending S1 exits are moved to the next session's open); `s1_entries` if now > 15:20; everything from `eod_ingest` on always runs. Starting on a later day: `preflight` detects sessions with no `eod_pipeline` and runs the evening pipeline for each missed session in order (data, panel, features, predictions, marks for held positions, report marked `catch-up`), with no orders for missed days.
- Idempotency: every decision and order id is deterministic (D.3), inserted with `INSERT OR IGNORE`; every job first checks what already exists for the date.
- Heartbeat: write `state/heartbeat.json` (`ts, pid, current_job, last_ok_job, degraded`) every 30 s via temp file + replace.
- GPU jobs take `state/gpu.lock` (file lock) so only one runs at a time.
- `notify`: if Telegram secrets are set: messages at `s1_entries` (summary), after `report` (summary + NAV), and on any `failed` job; otherwise no-op. Never raises.
- `stocksense job <name> --date YYYY-MM-DD` runs one job by hand through the same wrapper; `stocksense run --once` processes everything currently due and exits (used by tests).
**Integration tests (`tests/integration/test_full_day.py`; synthetic market written to a temp root, fake brokers serving quotes derived from the synthetic bars, `SteppingClock`):** (1) a full session from 07:30 to 23:59 runs every job in order and ends with a report, marks for every book, and reconciled fills; (2) kill the process state after `s1_entries` inserted orders (simulate by constructing a new daemon on the same state) → no duplicate orders, day completes; (3) a holiday → only `preflight`, `collect_only`, `housekeeping`; (4) the daemon starts at 10:40 → `s2_decide` and `open_fills` skipped with the stated reasons, S1 exits rolled to the next open; (5) three sessions are skipped entirely, then the daemon starts → three catch-up evening pipelines in order, then a normal day; (6) the first quote source fails → fallback used, `degraded` recorded; (7) second instance exits 0 with "already running".

### T10.6 Install as a Windows task and write the runbook
**Files:** `scripts/install-task.ps1`, `scripts/uninstall-task.ps1`, `docs/RUNBOOK.md`.
**`install-task.ps1`:**
```powershell
$repo = Split-Path -Parent $PSScriptRoot
$action   = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$repo\scripts\run-daemon.ps1`"" -WorkingDirectory $repo
$triggers = @((New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME), (New-ScheduledTaskTrigger -Daily -At 07:25))
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -WakeToRun `
            -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName 'StockSense' -Action $action -Trigger $triggers -Settings $settings -Description 'StockSense paper-trading daemon' -Force
Start-ScheduledTask -TaskName 'StockSense'
```
`uninstall-task.ps1`: `Stop-ScheduledTask` then `Unregister-ScheduledTask -TaskName 'StockSense' -Confirm:$false`.
**`docs/RUNBOOK.md`:** how to check it is alive (`stocksense doctor`: heartbeat age < 60 s); where reports and logs are; how to stop/start (`Stop-ScheduledTask` / `Start-ScheduledTask`); what happens after absence (nothing to do: it catches up); how to resume a book stopped by the drawdown rule; the user's standing requirements (laptop on, logged in, lid action "Do nothing", no sleep on AC); how to rotate the Upstox token yearly; what each `failed` job means and its fix (table copied from Part F).
**Acceptance — the "runs itself" proof:** the task is installed and the system runs **five consecutive sessions** with no manual action. For each day: a report exists; `job_runs` has no unresolved `failed`; every paper fill is reconciled; `data-status` is all fresh. The orchestrator checks once per day and records the five dates in Part H.

**Phase 10 done when:** integration tests green and the five-session proof is recorded.

---

## PHASE 11 — Learning every day

### T11.1 Nightly learning job
**Provides:** `learn.nightly.run_nightly(ctx, d) -> dict` (called inside `eod_pipeline`), `fit_production(spec, settings, state, as_of) -> model_id`.
**Spec, in order:**
1. **Score what was predicted.** For every active model and each of its targets: for decision dates whose label matured today (`end_<target> == d`), compute the daily rank IC between the **stored** live prediction (`preds/<model_id>`, `mode='live'`) and the realized label → `live_ic`. Never recompute predictions for this.
2. **Update the ensemble weights and the expected-return slope** (T7.6) with the newly matured dates; persist to `kv` (`ens_weights:<family>`), with history in `model_events`.
3. **Refit.** `fit_production(spec, as_of=d)`: same recipe as a walk-forward fold whose "test start" is tomorrow: training rows are all rows with `end_<target> <` (`d` − 5 sessions), inner validation = last 12 months, then refit on all with the found complexity; 3 seeds. Cadence: GBM specs every night (expected 5–15 minutes each; if the measured time is more than 40 minutes in total, switch to every Friday night and record it); seqnet Saturdays; Kronos monthly. The new `model_id` inherits the old one's status; the old one becomes `retired` and its files are deleted after 30 days. The fitted model is used from tomorrow's predictions.
4. **Slippage learning.** From `spreads`: per rank bucket, the median of `(ask − bid) / (ask + bid)` (= half-spread as a fraction of mid) over the last 20 sessions → if ≥ 20 sessions of data exist, append a dated row to the slippage table. From reconciled fills: mean of `(live provisional price − official price)/official price` by `when` and side → stored in `kv` and shown in the report (this is how real the open/close fill assumption is).
5. **Auction honesty measure.** From `preopen`: for `u200`, the distribution of `final_price / indicative_price(09:07) − 1` (median absolute value, 95th percentile) → report and `kv`. From the two S2 books: cumulative `strict_capture`.
6. **Event table refresh** (Phase 8 onward).
7. **Walk-forward extension:** when the calendar year rolls over and its labels have matured, append that year as a new out-of-sample fold for every registered spec (weekly job).
**Tests:** live IC uses stored predictions (a changed model does not change past `live_ic`); refit trains on nothing with `end ≥ d − 5 sessions` (assert on the mask); slippage row appended only with ≥ 20 sessions; idempotent for the same `d`.

### T11.2 Drift and decay monitors
**Provides:** `learn.drift.run_drift(ctx, d) -> list[Alert]`; migration 5.
**Spec:**
- Feature drift: for each `xs` feature's `__raw` values over the last 60 sessions vs the previous 756: population stability index over 10 quantile bins of the reference (`PSI = Σ (a − e) × ln(a/e)`, shares floored at 1e-4); `> 0.25` → alert (report only). A feature that is entirely null today but was < 20% null over the last 60 sessions → `features_fresh` check fails (no new entries).
- Model decay, per active model: `m60` = mean of the last 60 `live_ic` values, `t60 = m60 / std × sqrt(60)`. `m60 < 0 and t60 < −1` → that book's `gross_target × 0.5`; `m60 < 0 and t60 < −2` → `gross_target = 0` ("switched off by decay rule"); automatically lifted when the 20-session mean IC is > 0. Stored in `kv` as `decay_scale:<book>`, read by the policies.
- Calibration: realized mean label by score decile over the last 120 matured dates vs the walk-forward decile means (report only).
**Tests:** PSI = 0 for identical distributions and > 0.25 for a shifted one (numbers by hand for 3 bins); decay thresholds; automatic lift.

### T11.3 Champion and challenger
**Provides:** `learn.promote.evaluate_promotions(ctx, d) -> list[Action]`.
**Spec:** for each strategy family with a champion `C` and each shadow `S` that has `validated=1` (passed research and lockbox gates) and ≥ `min_shadow_sessions` common book days: paired one-sided t-test on daily net book returns (`S − C`); `p ≤ max_p_value` → promote `S`, demote `C` to `shadow`, the `main` book switches to follow `S` from the next decision (existing positions are kept and managed by the new model's policy). Rollback: a champion promoted fewer than 40 sessions ago that underperforms its predecessor's shadow book with `p ≤ 0.10` over the days since promotion (≥ 20) is demoted and the predecessor restored. If a family has no champion: `main` follows the shadow with the best trailing-60-session net book return (ties → higher walk-forward Sharpe), flagged `validated=0`, re-evaluated every 20 sessions. Every action writes `model_events` and a report line.
**Tests:** an injected random-score shadow is never promoted over 200 simulated sessions (seeded; assert zero promotions); a shadow with a real edge of +0.1 daily Sharpe over the champion is promoted within 120 sessions in > 80% of 50 seeded simulations; rollback path.

### T11.4 Controlled alpha search
**Provides:** `learn.alpha_search.run_search(ctx, budget_hours=3, seed=None) -> list[Candidate]`; expression engine `learn.alpha_search.Expr`.
**Spec:**
- Inputs (all adjusted, per sid): `open, high, low, close, volume, turnover, deliv_pct, ret, ret_co, ret_oc, vwap = turnover/volume × adj`. Operators (each causal by construction, implemented with the T4.1 helpers): `ts_mean, ts_std, ts_min, ts_max, ts_rank (rank of the latest value within the window), ts_delta, ts_corr, ts_argmax, decay_linear` with windows `{3, 5, 10, 21, 63}`; binary `+ − × ÷`; unary `abs, sign, log (of |x|+1e-9), neg`; `xs_rank`; `ind_neutral` (subtract the industry mean). Depth ≤ 4. An expression is serialized as a nested list; `spec_hash` identifies it.
- Search: genetic programming, population 200, 10 generations, tournament size 5, crossover 0.6, subtree mutation 0.3, reproduction 0.1. Fitness on **search data only = decision dates 2011-01-03 → 2019-12-31** (sampled every 3rd session for speed): `|mean rank IC vs z_y_c2o_5| − 0.5 × max |correlation| with any existing feature's ranked values − 0.002 × number of nodes`.
- Admission test on **held-out data 2020-01-01 → research end**, never seen by the search: take the top 10 distinct expressions; each is registered as a trial (family `alpha-search`) before its held-out test; admit if: IC sign matches the search period; `|IC| ≥ 0.01`; t-stat ≥ `3.48` (two-sided 5% with Bonferroni for 10 tests is `Φ⁻¹(1 − 0.025/10) = 2.81`; 3.48 additionally allows for one search every week for a year); and adding it to the champion spec improves walk-forward net Sharpe on inner validation of the last 3 folds. At most 3 admissions per run.
- An admitted expression becomes a registered feature (code generated into `features/mined.py` with its serialized form and a truncation test entry), bumps the feature version, and enters a new `shadow` model. It reaches the champion only through T11.3.
**Tests:** every operator on hand data; a generated expression passes the truncation checker; on the synthetic market with a planted nonlinear signal (`alpha` hidden in `ts_rank(volume, 5) × sign(ret)`), the search's best fitness exceeds that of 200 random expressions; the search never reads a date after 2019-12-31 (assert via a guarded loader).

### T11.5 Weekly research job
**Spec (Saturday):** (1) extend walk-forward folds if a year matured; (2) alpha search (3 h budget); (3) seqnet refit; (4) re-run hyperparameter selection for GBM on inner validation at the three most recent anchors and, if the selected configuration changed, register the new spec as a new trial and a new `shadow`; (5) regenerate `docs/RESULTS-latest.md` (same tables as RESULTS-G1 section 5 with current data and the trial count); (6) disk check: warn below 20 GB free. GPU steps one at a time under the GPU lock, each with out-of-memory handling (halve batch, retry twice).

**Phase 11 done when:** after 10 further sessions `live_ic` has rows for every active model and day, ensemble weights have changed, nightly refits are registered, one weekly cycle has completed, and T11.3's never-promote test is green.

---

## PHASE 12 — Why this stock: decision records, daily report, read API

### T12.1 Decision records
**Provides:** `explain.decision.build_decision_payload(...) -> dict`; `explain.labels_text.describe(feature, raw_value, pct) -> str`.
**Spec — payload JSON stored in `decisions.payload_json` (one per book/strategy/date) with an entry per order:**
```json
{"sid": 123, "symbol": "XYZ", "side": "BUY", "when": "close", "product": "delivery",
 "models": [{"model_id": "...", "score": 0.41, "rank": 4}],
 "blend_score": 0.43, "universe_rank": 4, "universe_n": 500,
 "expected_return": 0.0131, "horizon_sessions": 5,
 "cost_frac": 0.0031, "slip_frac": 0.0010, "net_edge": 0.0090,
 "target_weight": 0.094, "target_value_inr": 47000, "qty": 38, "ref_price": 1234.5,
 "reasons_for":     [{"feature": "mom_252_skip21", "text": "...", "raw": 0.62, "pctile": 0.95, "contribution": 0.031}],
 "reasons_against": [{"feature": "...", "text": "...", "raw": 0.0, "pctile": 0.0, "contribution": -0.008}],
 "events": [{"ts": "...", "category": "...", "summary": "...", "text_score": 0.0}],
 "risk_checks": [{"name": "data_fresh", "ok": true, "detail": ""}],
 "exit_reason": null, "validated_model": false}
```
`reasons_for` = the 8 largest positive LightGBM contributions (`pred_contrib`) among `xs` features, `reasons_against` = the 4 most negative; `pctile = rank value + 0.5`. `describe` turns each into one sentence from the feature's manifest label, e.g. label "12-month price trend, ignoring the last month" → "12-month price trend, ignoring the last month: +62%, stronger than 95% of stocks today". Exits carry `exit_reason` (`horizon`, `rank_decay`, `stop`, `universe_exit`, `squareoff`, `hard_stop`). A plain-English `summary` string is generated per order: "Ranked 4 of 500. Main reasons: … Expected 5-session return +1.3%, costs 0.4%, net +0.9%."
**Tests:** contributions listed sum (with the remaining features and bias) to the model score within 1e-5; text for a known feature; exits have a reason; payload validates against a pydantic model.

### T12.2 Daily report
**Provides:** `explain.report.build_report(ctx, d) -> (markdown, dict)`; CLI `stocksense report --date`.
**Sections:** (1) Health: jobs with status, data freshness, quote source and degraded minutes, alerts. (2) Capital read (₹, stale or not). (3) Today's trades per book with the one-line summary of each. (4) Open positions with weight, entry date, return since entry, planned exit. (5) Book performance: day, 5-session, 21-session, since start, in % and in ₹ at today's capital, against `ew500` and NIFTY 500; drawdown; costs paid. (6) S2: `auction` vs `auction_strict`, capture ratio, pre-open indicative-vs-final error. (7) Models: status, validated or not, walk-forward IC vs live 20/60-session IC, ensemble weights, decay scale. (8) What was learned today: refits, slippage table changes, live-vs-official fill differences, promotions, alpha-search admissions. (9) Tomorrow's plan: intended exits at the open and intended entries at the close. Written to `reports/YYYY-MM-DD.md` and `.json`. The first line of the report is always one sentence: e.g. "System healthy; main book −0.21% today, +3.4% since start; model not yet validated."
**Tests:** renders from a fixture state; every number in the markdown equals the JSON; a failed job appears in section 1.

### T12.3 Read-only API
**Provides:** `api.app.create_app(settings)`; CLI `stocksense serve` (uvicorn on `127.0.0.1:8765`).
**Endpoints (GET, JSON, all read-only; shapes documented in `docs/API.md` with one example each):** `/health` (heartbeat age, degraded), `/status` (job table for today), `/books`, `/books/{book}/nav`, `/books/{book}/positions`, `/books/{book}/trades?from=&to=`, `/decisions?date=&book=`, `/decisions/{id}`, `/models`, `/models/{id}`, `/models/{id}/validation`, `/reports/{date}`, `/reports/latest`, `/universe?date=`, `/signals/today` (planned exits and entries), `/quotes/{symbol}` (last cached), `/data-status`.
**Tests:** `fastapi.testclient` on a fixture state for every endpoint; no endpoint accepts POST/PUT/DELETE (assert 405).

**Phase 12 done when:** every order has a decision record; reports render for the five proof sessions; API tests green.

---

## PHASE 8 — News and events (no LLM), added to the running system

### T8.1 News collectors
**Provides:** `news.sources.poll_rss(ctx) -> int`; `RSS_FEEDS` config in `config/default.toml` under `[news]`.
**Spec:** `feedparser` over a list of feeds; initial candidates (verify each returns entries; drop dead ones and record): Economic Times Markets `https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms`, ET Stocks `https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms`, Moneycontrol business `https://www.moneycontrol.com/rss/business.xml`, Moneycontrol markets `https://www.moneycontrol.com/rss/marketreports.xml`, LiveMint markets `https://www.livemint.com/rss/markets`, Business Standard markets `https://www.business-standard.com/rss/markets-106.rss`. Raw XML saved to `data_store/news/YYYY-MM-DD/<source>/<HHMMSS>.xml`; rows upserted to `news_items` with `raw_id = entry.id or link`, `ts_seen = clock.now()` on first sight only. A feed failing three polls in a row is skipped for the day with one warning. Re-read the 17 days of old captures in `data_store/news/` if they are RSS XML or JSON with obvious `title/link/published` fields; otherwise skip.
**Tests:** parse a saved feed sample; first-seen timestamp is not overwritten by a later poll.

### T8.2 Entity linking and taxonomy
**Provides:** `news.entity.link(text) -> list[(sid, confidence)]`; `news.taxonomy.classify(category: str, text: str) -> str`.
**Spec:**
- Aliases per sid: current and past symbols; company name from `listed_equities` normalized (lower-case; remove `limited, ltd, ltd., corporation, corp, industries, company, co., (india), india, the`; collapse spaces); plus a hand list for the 100 largest (`ril→RELIANCE, sbi→SBIN, hul→HINDUNILVR, l&t→LT, m&m→M&M, tcs→TCS, hdfc bank→HDFCBANK, icici bank→ICICIBANK, …` written out in full in the module). Matching: (1) exact word-boundary match of a normalized alias of ≥ 4 characters (symbols must match case-sensitively in upper case in the original text); (2) else `rapidfuzz.fuzz.token_set_ratio ≥ 92` between each capitalized 1–4-word n-gram of the title and the alias list; if the top two candidates are within 3 points → no link.
- Taxonomy classes: `results, board_meeting, dividend, bonus_split, buyback, order_win, acquisition_merger, fund_raise, rating_up, rating_down, mgmt_exit, mgmt_appoint, litigation_regulatory, pledge, insider_trade, capacity_expansion, investor_meet, agm_egm, clarification, press_release, other`. Rules: first by the NSE `category` string (the executor writes the mapping for the 30 most frequent categories listed in `docs/DATA.md` by T2.8, e.g. `Financial Result Updates→results`, `Outcome of Board Meeting→board_meeting` unless the text contains `financial results` → `results`, `Analysts/Institutional Investor Meet/Con. Call Updates→investor_meet`, `Resignation→mgmt_exit`, `Credit Rating→` up/down by keywords `upgrad|reaffirm` vs `downgrad|default|withdraw`, `Bagging/Receiving of orders/contracts→order_win`, `Acquisition→acquisition_merger`, `Trading Window→other`), then keyword regexes on the text for the rest (`order|contract|loi|letter of intent → order_win`; `sebi|penalty|fraud|search|raid|show cause|insolvency|nclt|default → litigation_regulatory`; `pledge → pledge`; `qip|preferential|rights issue|fund rais → fund_raise`; `resign|cessation → mgmt_exit`).
**Tests:** 40 real `(category, text)` pairs sampled from the stored announcements with expected classes (the executor samples them and writes the expectations by reading them); entity linking on 20 headlines including ambiguous ones.

### T8.3 Text model trained on NSE market reactions
**Provides:** `news.textmodel.embed(texts) -> np.ndarray`, `TextReactionModel` (fit/predict), feature `text_score_5`.
**Spec:** encoder `BAAI/bge-small-en-v1.5` through `sentence-transformers` (384 dims, max 256 tokens, on GPU, batch 128, normalized embeddings); input string `"<class> | <category> | <summary truncated to 600 characters>"`; embeddings cached in `features/text_emb/part-YYYY.parquet` by `seq_id` (resumable; about 1–3 million rows; a few hours on the GPU, background). Head: LightGBM regression predicting the event's 2-session abnormal return (T8.4 definition) from `[32 PCA components of the embedding (PCA fitted on training rows only), class one-hot, hour bucket (pre-open, 09:15–12:00, 12:00–15:30, after close), turnover_rank, vol_21, mom_5, market-state features]`; walk-forward with the T6.2 folds keyed on the event date and purged by the event window end. Output `text_score` per announcement (out-of-fold in history, live model going forward). Stock-day feature `text_score_5(t) = Σ over the sid's announcements with decision timestamp ≤ cutoff of t of text_score × 0.5^(sessions_since/2)`, limited to the last 5 sessions; plus class counts over 21 sessions for the 8 most frequent classes. These are registered features (truncation-tested) and bump the feature version.
**Tests:** PCA fitted on training rows only (assert by construction); embedding cache hit avoids re-encoding; the stock-day aggregation on a hand example.

### T8.4 Event study
**Provides:** `news.eventstudy.build_event_table(settings, purpose) -> pl.DataFrame` (writes `panel/event_table.parquet`), `event_returns(events) -> pl.DataFrame`.
**Spec:** event timestamp `τ = max(ts, exch_ts)` (history) or `first_seen_ts` (live). Entry: if `τ` is within 09:15–15:10 on a session and 1-minute or 5-minute bars exist for the sid → the open of the first bar starting ≥ `τ + 60 s`; otherwise the next session's open. Abnormal return over window `W ∈ {to_close, +1, +2, +5, +10 sessions}` = stock return from entry to the window end (close) − `beta_126 ×` `ew500` return over the same span (for intraday entry use `ew500_oc` scaled by the remaining fraction of the session for the first day). Cells: class × sign bucket of `text_score` (negative / neutral / positive terciles) × liquidity tercile. Per cell and window: `n`, mean, std, t-stat, hit rate, mean net of round-trip cost (delivery for multi-day windows, intraday for `to_close`), share of calendar years with a positive mean, and the shrunk mean `n/(n+50) × cell + 50/(n+50) × class mean`.
**Tests:** synthetic events with a planted +1% two-session drift in one cell are recovered (t > 4) and other cells are flat; an event at 15:20 enters at the next open; an event on a holiday enters at the next session's open.

### T8.5 Event sleeve (S3)
**Provides:** `news.signals.EventPolicy`; live hook in `market_loop`.
**Spec:** a cell is tradable when, on research data: `n ≥ 200`, shrunk net mean > 0, t ≥ 3, positive in ≥ 70% of years. A new live announcement in a tradable cell produces an order in book `events`: long only for multi-day windows (BUY `when='now'` if during market hours and before 15:00, else `when='open'` next session; exit at the open after the cell's best window); position size = `min_pos`-based equal slices, sleeve capped at `s3.max_capital_frac`. The sleeve is a candidate strategy that goes through `stocksense validate` (trial family `s3-events`) with its own walk-forward backtest (the cell table at each fold is built from training-period events only). `s3.enabled` flips to true only after its validation report is written; before that the `events` book runs as an unvalidated shadow.
**Tests:** cell eligibility rule; fold-wise table uses only earlier events; sleeve cap.

### T8.6 Fundamentals from results filings (later milestone)
**Spec:** download the XBRL files in `results_index.xbrl_url` (cache `cache/xbrl/`), parse with `lxml` the tags for revenue from operations, profit before tax, profit after tax, basic EPS (the executor inspects three files to fix the tag names per taxonomy version: `in-bse-fin:RevenueFromOperations`, `ProfitLossForPeriod`, `BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations` or their Ind-AS equivalents; record them) into `fundamentals(symbol, period_to, filing_ts, consolidated, revenue, pbt, pat, eps)`; point-in-time features keyed on `filing_ts`: `rev_yoy`, `pat_yoy`, `margin_chg_yoy`, `earnings_yield = 4-quarter EPS / close`, `sue = (eps − eps_4q_ago) / std of that difference over the last 8 quarters`. Registered features, truncation-tested, new feature version, new shadow model.

**Phase 8 done when:** event features are live in a shadow model; the event table and the S3 validation report exist; suite green.

---

## PHASE 13 — Minute-bar intraday (after 20 clean sessions of Phase 10)

- **T13.1 Minute data in place:** `bars_1m` backfill complete (T2.10); live `quotes_1m` aggregated nightly into `bars_1m` for the watch list; `load_bars(interval="1m"|"5m", basis="adjusted")`.
- **T13.2 Same-day S1 variant ("decide at 15:05, buy at the close today"):** features for `t` built from the day's bar-so-far at 15:00 (open, high, low, last, volume to 15:00 scaled by the stock's median share of volume traded by 15:00) instead of the full bar; label `y_c2o_h` starting at `C(t)` instead of `C(t+1)`. Backtest on the minute-bar history only; compare with the standard S1 on the same dates. This recovers one overnight return per trade (about +16 to +30 bps by B.3) if the 15:00 features are nearly as good; registered trials; shadow book `s1-sameday`.
- **T13.3 Kronos on 5-minute bars:** fine-tune `Kronos-small` (context 512 bars ≈ 7 sessions) and `Kronos-mini` (context 2048) on the NSE 5-minute history with the T7.5e loop (window 512 + 12); forecast the next 12 bars with 20 paths at fixed times (09:20, 10:00, 11:00, 13:00, 14:00); features `kr5_ret60m_mean, kr5_up_prob, kr5_range`; evaluate IC against the realized next-60-minute return; admission rule as T7.5d.
- **T13.4 Minute-bar model:** LightGBM on bar-level features (time of day, VWAP deviation, volume surprise vs the stock's own intraday volume curve, 30-minute momentum/reversal, market and industry move over the same bars, opening-range position, Kronos features if admitted) predicting the next 30-minute return; traded only when the prediction exceeds twice the intraday round-trip cost; all positions closed by 15:18.
- **T13.5 Strict S2:** an S2 variant trained to predict the return from 09:20 to 15:15 (not from the open), using the first 5-minute bar as an input. This is the version that does not depend on auction fills.
- Validation honesty: with roughly 2–4 years of minute data the folds are 3 months long and the statistical power is low; the gate numbers do not change; anything that cannot clear them stays a shadow book.

## PHASE 14 — UI (last)

Needed from the user then: the URL of the existing UI repository. Approach: keep its components, replace its data layer with a client for `docs/API.md`, map US-market concepts to NSE ones (tickers, 09:15–15:30 IST session, ₹, lakh/crore formatting). Hosting choice to settle then: local against `127.0.0.1:8765` (simplest), or deployed on Vercel reading a nightly read-only JSON snapshot the daemon pushes to private storage (no inbound connection to the laptop).

## PHASE 15 — Optional hardware

An ESP32 with an RGB LED and buzzer polling `/health` over Wi-Fi: green = heartbeat fresh and data fresh; amber = degraded quotes or stale data; red = heartbeat older than 2 minutes in market hours. No effect on results. Arduino and FPGA parts have no role.

## PHASE 16 — Real money (out of scope; checklist only)

Preconditions before anyone writes order code: `[paper]` gate passed on ≥ 60 sessions by a validated champion; for S2 additionally `[auction]`; a static IP registered with the broker and an Algo-ID under the SEBI framework; a reviewed order module with hard limits and a kill switch; an explicit decision by the user. Rule 2 and its test stay until then.

---

# PART F — Known failure modes and the fix built into the plan

| Problem | Where it bites | Built-in fix |
|---|---|---|
| NSE `prev_close` is not split/bonus adjusted (B.3 F1) | returns, labels, backtest | adjustment from the CA table cross-checked against price evidence; implied/suspected factors; audit table (T3.2) |
| 19 special sessions missing from stored data (B.3 F2) | calendar, returns around them | every calendar day is tried, repair backfill, missing-session detector (T2.2, T2.3) |
| `sec_bhavdata_full` returns the previous session's file on a holiday | delivery ingest | reject when `DATE1` ≠ requested date (T2.3) |
| NSE API 401/403 or HTML served with 200 | all `www.nseindia.com/api` calls | cookie warm-up and retry, HTML-body-as-404 rule, 0.6 s spacing (T2.1) |
| Bhavcopy not yet published at 18:30 | `eod_ingest` | 404 = not published; retry every 15 min to 23:30; catch-up at next start (T10.4) |
| Header names with spaces; trailing comma columns in legacy files | parsers | strip; drop empty columns; probe against stored rows before any backfill (T2.3) |
| Symbol renames break history | every time series | stable `sid`, official + auto-detected merges, append-only numbering (T3.1) |
| Survivorship bias | universe, backtest | universe from raw data including dead names; delisting haircut in labels; write-off in the engine (T3.3, T4.5, T5.4) |
| Look-ahead in a feature | all results | every accessor filtered by `as_of`; automatic truncation test (T4.1, T4.2) |
| Train/test leakage through overlapping labels | all results | purge by `end_<target>` + embargo (T6.2) |
| Trying many things until one looks good | all results | trial registry seeded with planning trials; deflated Sharpe; PBO; frozen gate; one-look lockbox (T6) |
| A session "improving" results by editing thresholds | credibility | SHA test on `gate.toml`; lockbox touch log; rule 4 |
| Buying at the open when the edge is in buying at the close (B.3 F3) | S1 | S1 buys at the close, sells at the open (T9.1) |
| Opening-gap fade not capturable after 09:20 (B.3 F4) | S2 | two books, `auction` and `auction_strict`; `strict_capture` reported; gate `[auction]` (T9.2, T9.5, T10.3) |
| Flat fees destroy small accounts | portfolio | capital-aware `K` and minimum position from the cost table (T9.1, T9.2) |
| Locked circuits | backtest and paper | shared fill rules; blocked sells retried (T5.3) |
| Angel `AB1021` false rate limits; TOTP rejection | broker calls | backoff 60 s then 300 s; second TOTP window; Upstox and NSE-MCP quote fallbacks (T2.9, T10.2) |
| Upstox token missing or expired | intraday backfill, quotes | optional; skip with one warning; Analytics Token lasts a year (A.5) |
| Laptop asleep, off, or rebooted | everything | Task Scheduler restart/wake, keep-awake call, idempotent date-keyed jobs, catch-up evening pipelines (T10.5, T10.6) |
| Duplicate orders after a crash | paper ledger | deterministic ids, `INSERT OR IGNORE` (D.3, T10.1, T10.5) |
| Two processes writing state | SQLite, parquet | single-instance lock; WAL; atomic parquet replace (T1.5, T1.6, T10.5) |
| GPU out of memory on 4 GB | seqnet, Kronos, embeddings | float16, small batches, one GPU job at a time, OOM→halve→retry, checkpoints with resume (T7.3, T7.5, T11.5) |
| Kronos is not a pip package; code written for pandas 2.2 | Kronos | vendored at a pinned commit; upstream regression test must pass first; never install its requirements (T7.5a) |
| Kronos averages sample paths internally | Kronos | our `sample_paths` returns all paths (T7.5b) |
| Windows code page breaks on `₹` or company names | logs, CSV | `PYTHONUTF8=1`, explicit `encoding="utf-8"` (D.1, D.2) |
| Mixed parquet schemas across months | readers | one reader that casts to the catalog schema (T1.6) |
| Naive vs aware datetimes | calendar, quotes | single `Clock`; `date` columns are IST session dates; naked `now()` forbidden by test (T1.3, D.3) |
| torch wheel missing for a Python version | environment | Python 3.12 pinned; CUDA index fallback cu128 → cu126 (T1.1) |
| Model silently stale or broken | live | pre-trade checks, drift monitors, decay rule that cuts exposure to zero (T9.3, T11.2) |
| Disk fills (1-minute bars 3–4 GB, caches) | long runs | cache pruning, weekly disk check, warning below 20 GB (T10.4, T11.5) |

---

# PART G — Verification (how anyone can tell it works)

1. **Reset:** `git log --all --oneline` → 1 line; GitHub branch list → `main` only; no scheduled task except `StockSense`; Claude memory folder holds only the two new notes.
2. **Data:** `stocksense data-status` → every dataset fresh and zero issues; the 19 historical missing sessions gone; 20-point cross-check against the NSE MCP server passed or logged as unreachable.
3. **Correct returns:** KAMDHENU (2025-01-08), JUBLFOOD (2022-04-19) and M&MFIN (2020-07-22) show normal-sized returns on their ex-dates; `ew500` overnight/intraday means reproduce +22 / −20 bps.
4. **No leakage:** `pytest tests/unit/test_feature_truncation.py tests/integration/test_harness_selftest.py` green; the IC table of T4.6 agrees with B.3.
5. **Engine:** clairvoyant and random policy integration test green; hand-computed cost tests green.
6. **Models:** `docs/RESULTS-G1.md` exists with the full trial list, total trial count, per-year tables, DSR, PBO and a PASS/FAIL per gate line for each family, and the Kronos decisions.
7. **Runs itself:** five consecutive sessions with reports, complete `job_runs`, reconciled fills, and no manual action; `test_full_day.py` covers crash-resume, holiday, late start, and multi-day catch-up.
8. **Learns:** `live_ic` grows daily; nightly refits registered; ensemble weights change; a weekly research cycle completes; an injected bad shadow is never promoted.
9. **Explains:** each order has a decision record whose contributions sum to the model score; the daily report opens with a one-sentence status.
10. **Paper only:** `pytest tests/unit/test_no_orders.py` green.
11. Whole suite: `.venv\Scripts\python.exe -m pytest tests -q` green before every commit.

---

# PART H — Status checklist (the orchestrator updates this in `docs/PLAN.md` after each phase)

- [x] P0 Reset — done 2026-11-11 (fresh-start commit `a696fa5`; GitHub has only `main`; memory reset)
- [x] P1 Foundation (T1.1–T1.9) — done 2026-11-11; 131 tests; `doctor` matches B.2. Next: T2.1
- [ ] P2 Data (T2.1–T2.13)
- [ ] P3 Panel (T3.1–T3.4)
- [ ] P4 Features and labels (T4.1–T4.6)
- [ ] P5 Costs and backtest engine (T5.1–T5.5)
- [ ] P6 Validation harness (T6.1–T6.8)
- [ ] P7 Models (T7.1–T7.6), including Kronos decisions
- [ ] P9 Portfolio, risk, Gate G1 verdict (T9.1–T9.5)
- [ ] P10 Paper broker and daemon (T10.1–T10.6); five unattended sessions: ____ ____ ____ ____ ____
- [ ] P11 Daily learning (T11.1–T11.5)
- [ ] P12 Decision records, report, API (T12.1–T12.3)
- [ ] P8 News and events (T8.1–T8.6)
- [ ] P13 Minute-bar intraday
- [ ] P14 UI
- [ ] P15 Optional hardware

Execution order is the order of this list. Phase 9 deliberately precedes Phase 8.

---

# PART I — Sources (consulted 2026-11-11)

- Kronos: https://github.com/shiyu-coder/Kronos · paper https://arxiv.org/abs/2508.02739 · weights https://huggingface.co/NeoQuasar
- Upstox: https://upstox.com/developer/api-documentation/v3/get-historical-candle-data · https://upstox.com/developer/api-documentation/rate-limiting · https://upstox.com/developer/api-documentation/analytics-token
- NSE MCP: https://www.nseindia.com/nse-mcp
- Angel One SmartAPI forum (rate limits): https://smartapi.angelone.in/smartapi/forum/post/16438 · https://smartapi.angelone.in/smartapi/forum/post/19227
- Charges: https://upstox.com/help-center/t-257195/ · https://groww.in/blog/what-is-stt · https://www.angelone.in/pricing
- SEBI retail algo framework: https://fyers.in/notice-board/new-sebi-framework-for-retail-algo-trading-from-april-01-2026/
- Indian anomaly literature (momentum, short-term reversal, low volatility): https://eprints.nottingham.ac.uk/66413
