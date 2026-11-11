# Data

Status of the datasets under `data_store/parquet`, as reported by `stocksense data-status`.

## Baseline (adopted data, 2026-11-11, before ingestion)

Output of `.venv\Scripts\python.exe -m stocksense data-status` (run time about 8 s). Nothing is fresh: ingestion (Phase 2) has not run yet. `sessions_missing` is measured against the calendar derived from `bhavcopy_eq`.

```
| dataset           | first_date | last_date  | rows    | sessions_missing | fresh | issues |
|-------------------|------------|------------|---------|------------------|-------|--------|
| bhavcopy_eq       | 2010-01-04 | 2026-09-29 | 8270877 | 0                | false | 31     |
| bhavcopy_delivery | 2019-08-23 | 2026-09-29 | 4235252 | 25               | false | 0      |
| corporate_actions | 2010-01-04 | 2026-09-18 | 35319   | null             | false | 0      |
| fo_bhavcopy       | 2024-07-08 | 2026-09-18 | 8372243 | 0                | false | 2      |
| india_vix         | 2010-01-04 | 2026-09-17 | 4103    | 24               | false | 0      |
| intraday_bars     | 2024-09-13 | 2026-09-17 | 9434028 | 0                | false | 0      |
| nse_snapshots     | 2026-09-18 | 2026-10-21 | 2498056 | null             | null  | 0      |
```

### Missing-session issues in `bhavcopy_eq` (19, PLAN B.3 Finding 2)

Each is attached to the first stored session after the missing special session (the missing dates are the day before or earlier, see PLAN B.3). Run `stocksense data-status --issues` for the full list.

```
missing_session_before:2010-02-08  96.2% of 1136 EQ symbols have prev_close != previous stored close
missing_session_before:2012-01-09  86.4% of 1423 EQ symbols have prev_close != previous stored close
missing_session_before:2012-03-05  84.5% of 1436 EQ symbols have prev_close != previous stored close
missing_session_before:2012-04-30  82.5% of 1404 EQ symbols have prev_close != previous stored close
missing_session_before:2012-09-10  84.2% of 1445 EQ symbols have prev_close != previous stored close
missing_session_before:2013-05-13  82.7% of 1297 EQ symbols have prev_close != previous stored close
missing_session_before:2013-11-05  90.1% of 1199 EQ symbols have prev_close != previous stored close
missing_session_before:2014-03-24  86.9% of 1368 EQ symbols have prev_close != previous stored close
missing_session_before:2015-03-02  93.4% of 1417 EQ symbols have prev_close != previous stored close
missing_session_before:2016-11-01  94.3% of 1534 EQ symbols have prev_close != previous stored close
missing_session_before:2019-10-29  95.1% of 1481 EQ symbols have prev_close != previous stored close
missing_session_before:2020-02-03  96.3% of 1469 EQ symbols have prev_close != previous stored close
missing_session_before:2020-11-17  93.9% of 1616 EQ symbols have prev_close != previous stored close
missing_session_before:2023-11-13  96.3% of 1797 EQ symbols have prev_close != previous stored close
missing_session_before:2024-01-23  96.3% of 1782 EQ symbols have prev_close != previous stored close
missing_session_before:2024-03-04  95.2% of 1779 EQ symbols have prev_close != previous stored close
missing_session_before:2024-05-21  95.3% of 1883 EQ symbols have prev_close != previous stored close
missing_session_before:2025-02-03  96.3% of 1996 EQ symbols have prev_close != previous stored close
missing_session_before:2026-02-02  97.4% of 2404 EQ symbols have prev_close != previous stored close
```

### Other `bhavcopy_eq` and `fo_bhavcopy` issues

- `bhavcopy_eq`: 91 rows with `high < max(open, close)`, 126 rows with `low > min(open, close)`.
- `bhavcopy_eq`: 10 `session_row_count_off` sessions, 2014-08-06 to 2014-08-20 (about 1,420-1,434 EQ rows against a trailing median near 1,189; a genuine step up in the EQ count).
- `fo_bhavcopy`: 26 rows with `high < max(open, close)`, 74 rows with `low > min(open, close)`.


## After T2.3 (2026-11-11)

Equity bhavcopy + delivery ingester (`stocksense ingest eq`) and the repair backfill were run: `stocksense ingest eq --from 2010-01-01 --to yesterday --repair` (log: `data_store/logs/t23_repair.log`; a pre-write copy of `bhavcopy_eq` and `bhavcopy_delivery` is in `data_store/_backup_t23/`).

### Probe (before any backfill)

The parsers were run on the cached files and compared with the stored rows (join on symbol, series; floats within 1e-6, volume exact, `turnover_inr` relative 1e-6):

```
2026-09-29 (UDiFF)  parsed 3688  stored 3688  rows only on one side 0
  mismatches: open 0, high 0, low 0, close 0, prev_close 0, volume 0, turnover_inr 0   -> MATCH
2020-02-03 (legacy) parsed 1977  stored 1977  rows only on one side 0
  mismatches: open 0, high 0, low 0, close 0, prev_close 0, volume 0, turnover_inr 0   -> MATCH
2026-09-29 (delivery) parsed 3522 stored 3522 one-sided 0, deliv_qty mismatches 0, deliv_pct mismatches 0
```

No mapping fix was needed.

### Sessions added: 47 (4,135 -> 4,182)

The 19 special sessions of PLAN B.3 Finding 2 (all had files; no other special session with more than 500 rows exists between 2010-01-01 and 2026-09-29):
`2010-02-06, 2012-01-07, 2012-03-03, 2012-04-28, 2012-09-08, 2013-05-11, 2013-11-03, 2014-03-22, 2015-02-28, 2016-10-30, 2019-10-27, 2020-02-01, 2020-11-14, 2023-11-12, 2024-01-20, 2024-03-02, 2024-05-18, 2025-02-01, 2026-02-01`

New sessions 2026-09-30 to 2026-11-09 (28 days): `2026-09-30, 2026-10-01, 2026-10-05, 2026-10-06, 2026-10-07, 2026-10-08, 2026-10-09, 2026-10-12, 2026-10-13, 2026-10-14, 2026-10-15, 2026-10-16, 2026-10-19, 2026-10-21, 2026-10-22, 2026-10-23, 2026-10-26, 2026-10-27, 2026-10-28, 2026-10-29, 2026-10-30, 2026-11-02, 2026-11-03, 2026-11-04, 2026-11-05, 2026-11-06, 2026-11-08, 2026-11-09`. This includes the Sunday 2026-11-08 Muhurat session (3,550 rows). 2026-11-10 (holiday) was not published; 2026-11-07 (Saturday) is not a session.

Two gold-ETF-only Sunday files exist (2010-05-16: 7 rows, 2012-11-11: 14 rows). They are not equity sessions and were not stored (see `docs/DECISIONS.md`).

Repair log counts (dataset, status): bhavcopy_eq ok 47, no_session 1,510, not_published 1, skipped 4,600; bhavcopy_delivery ok 38, no_session 23, failed 1, skipped 1,728.

### `stocksense data-status --issues` after the repair

```
| dataset           | first_date | last_date  | rows    | sessions_missing | fresh | issues |
|-------------------|------------|------------|---------|------------------|-------|--------|
| bhavcopy_eq       | 2010-01-04 | 2026-11-09 | 8409721 | 0                | false | 12     |
| bhavcopy_delivery | 2019-08-23 | 2026-11-09 | 4357401 | 24               | false | 0      |
| corporate_actions | 2010-01-04 | 2026-09-18 | 35319   | null             | false | 0      |
| fo_bhavcopy       | 2024-07-08 | 2026-09-18 | 8372243 | 2                | false | 2      |
| india_vix         | 2010-01-04 | 2026-09-17 | 4103    | 43               | false | 0      |
| intraday_bars     | 2024-09-13 | 2026-09-17 | 9434028 | 0                | false | 0      |
| nse_snapshots     | 2026-09-18 | 2026-10-21 | 2498056 | null             | null  | 0      |
```

- `bhavcopy_eq`: zero `missing_session_before` issues, `sessions_missing` 0. The 12 remaining issues are the old ones (91 + 126 OHLC-consistency rows, 10 `session_row_count_off` in 2014-08).
- `fresh` is `false` for `bhavcopy_eq` and `bhavcopy_delivery` only because the `holidays` dataset does not exist yet (T2.7): the calendar projects 2026-11-10 (a holiday) as a session, so it expects data through 2026-11-10; the stores' last date 2026-11-09 is the true last completed session.
- `bhavcopy_delivery` `sessions_missing` 24: 2019-08-26 .. 2019-09-27 (23 sessions, no file at NSE) and 2022-08-08 (file is an xlsx workbook).

### MCP cross-check (`cross_check_mcp`, `https://mcp.nseindia.in/bhavcopy/cm/mcp`, tool `get_ltp_by_date`)

- 20 random weekday (symbol, date) pairs from the 27 weekday sessions ingested here (seed 11): 20 checked, 0 mismatches within 0.01.
- Weekend special sessions cannot be checked this way: the server forward-fills non-weekday dates with the previous close (MOM30IETF 2026-11-07 and 2026-11-08 are both 30.4 while the stored Muhurat close is 30.73). The first CLI run sampled weekends and flagged 3 of 18 such pairs; `sample_pairs` now draws weekdays only.

## After T2.4 / T2.5 / T2.7 (2026-11-11)

- `fo_bhavcopy`: rebuilt from the cached files (the adopted table had no expiry and ~40% of the rows): 21,540,909 rows, 2024-07-08 -> 2026-11-09, no missing sessions.
- `index_daily`: true first date 2012-02-21 (NIFTY 50 close 5607.15); last 2026-11-11 (25255.05); 296,413 rows, 3,575 days.
- `india_vix`: 4,140 rows, last 2026-11-11.
- Reference datasets (2026-11-11): holidays 302 (2011-2026), symbol_changes 1,080, listed_equities 2,633, etf_list 360, sectors 754, price_bands 3,612, F&O list 218 symbols.


## Panel acceptance (real data)

Run 2026-11-11 on the real store (`bhavcopy_eq` through 2026-11-09). `stocksense build-panel` (security master, daily panel, universe, market frame; log `data_store/logs/t3_build_panel.log`) completes in **about 15 s** wall time (target < 5 min).

### T3.2 rows and gap sources

```
panel_daily rows: 7,354,058       security_master rows: 4,435     panel_universe rows: 2,726,689
panel_market rows: 3,930          adj_events rows: 1,647
gap_src counts:
  (none)               7,352,429
  ca                       1,104
  rights                     292
  implied_event              138
  suspected                   72
  ca_shifted                  17
  implied_at_ca                5
  implied_event_close          1     (added during acceptance, see below)
```

### T3.2 the ex-date rows (close, gap_factor, ret, adj_close)

```
KAMDHENU (10-for-1 split)
2025-01-06  close 469.90  gap 1.0                 ret -0.0704  adj_close  46.99
2025-01-07  close 479.30  gap 1.0                 ret +0.0200  adj_close  47.93
2025-01-08  close  44.50  gap 0.1  (ca)           ret -0.0716  adj_close  44.50   <- ex-date
2025-01-09  close  51.00  gap 1.0                 ret +0.1461  adj_close  51.00
2025-01-10  close  46.40  gap 1.0                 ret -0.0902  adj_close  46.40

JUBLFOOD (5-for-1 split)
2022-04-13  close 2780.80 gap 1.0                 ret +0.0017  adj_close 556.16
2022-04-18  close 2844.70 gap 1.0                 ret +0.0230  adj_close 568.94
2022-04-19  close  546.20 gap 0.2  (ca)           ret -0.0400  adj_close 546.20   <- ex-date
2022-04-20  close  551.70 gap 1.0                 ret +0.0101  adj_close 551.70
2022-04-21  close  594.75 gap 1.0                 ret +0.0780  adj_close 594.75

M&MFIN (rights)
2020-07-20  close 229.60  gap 1.0                 ret +0.1044  adj_close 136.21
2020-07-21  close 227.90  gap 1.0                 ret -0.0074  adj_close 135.20
2020-07-22  close 153.85  gap 0.6097 (rights)     ret +0.1072  adj_close 149.69   <- ex-date
2020-07-23  close 148.60  gap 1.0                 ret -0.0341  adj_close 144.59
2020-07-24  close 141.55  gap 1.0                 ret -0.0474  adj_close 137.73

VEDL (demerger, implied_event)
2026-04-28  close 739.30  gap 1.0                 ret -0.0043  adj_close 276.66
2026-04-29  close 773.60  gap 1.0                 ret +0.0464  adj_close 289.50
2026-04-30  close 271.55  gap 0.3742 (implied)    ret -0.0620  adj_close 271.55   <- ex-date, ret_co 0
2026-05-04  close 294.65  gap 1.0                 ret +0.0851  adj_close 294.65
2026-05-05  close 303.90  gap 1.0                 ret +0.0314  adj_close 303.90
```

Every ex-date return is a normal-sized daily return. HEG 2026-09-07 (factor 0.357, ret +4.7%) and KMSUGAR 2026-10-01 (factor 0.805, ret +2.4%) behave the same way.

### Defect found and fixed during acceptance: event inside the session (ADANIENT 2015-06-03)

The first run listed ADANIENT 2015-06-03 as the largest `flag_bigmove` row: close 637.00 -> 109.75, `ret` -82.8%, `gap_src` empty. The corporate-actions row is a `scheme_of_arrangement` (a demerger) with no factor. Root cause: T3.2 step 4 decides from the OPEN ratio and needs `|ln r_obs| > 0.12`, but on that day the open was 573.30 (r_obs 0.90, ln -0.105): the stock opened at the old (cum) price and the price adjustment happened during the session (open 573.30, low 106.35, close 109.75), so the open carried no evidence. Fix (regression tests `test_intraday_demerger_gets_close_based_factor`, `test_event_with_mild_close_move_and_small_open_gap_is_untouched`, `test_big_close_crash_without_a_table_event_is_not_adjusted`, `test_intraday_event_day_has_zero_ret_and_null_open_split`): when a factorless table event row passes the table-event filter but its open test fails, and `|ln(close/base)| > 0.5`, then `gap_factor = close/base`, `gap_src = 'implied_event_close'`, so `ret = 0` for that day, `ret_co` and `ret_oc` are null (the open is cum-event and the close ex-event, so neither split of the day is defined), and the adjusted history is continuous. Only one row in the whole history met the rule; no other row changed.

### 16 `flag_bigmove` rows among u500-eligible rows (ret magnitude > 40%)

`flag_bigmove` count within the rows of `panel/universe` (rows eligible for the u500 on that date): **16** (after the fix; 17 before). The plan's guess of "a few hundred" does not apply; only 16 over 15 years. They are all the rows there are (the "20 largest" list is the whole list). Eyeballed: INFIBEAM 2018-09-28, FINANTECH 2013-08-01, COREEDUTEC 2013-02, YESBANK 2020-03-16/17, JETAIRWAYS 2019-06, DHFL 2018-09-21, PNB 2017-10-25, INDUSINDBK 2020-03-26 are genuine price events (crashes, rallies); SINTEX 2017-05-25 and TRIVENI 2011-05-03 are demergers (`implied_event`, the overnight gap removed, the intraday move remains, which is why their `ret` is still large).

```
| symbol     | date       | close  | prev_close | gap_src       | ret    | turnover_rank |
|------------|------------|--------|------------|---------------|--------|---------------|
| JETAIRWAYS | 2019-06-20 |  62.85 |  33.10     | -             | +0.899 | 27  |
| INFIBEAM   | 2018-09-28 |  58.45 | 200.35     | -             | -0.708 | 83  |
| FINANTECH  | 2013-08-01 | 191.65 | 539.85     | -             | -0.645 | 94  |
| COREEDUTEC | 2013-02-25 | 110.50 | 293.85     | -             | -0.624 | 311 |
| YESBANK    | 2020-03-17 |  58.65 |  37.10     | -             | +0.581 | 12  |
| AMTEKAUTO  | 2015-09-11 |  46.95 |  30.50     | -             | +0.539 | 109 |
| PNB        | 2017-10-25 | 202.05 | 138.20     | -             | +0.462 | 63  |
| COREEDUTEC | 2013-02-27 |  60.50 | 111.65     | -             | -0.458 | 310 |
| YESBANK    | 2020-03-16 |  37.10 |  25.55     | -             | +0.452 | 12  |
| INDUSINDBK | 2020-03-26 | 435.90 | 301.30     | -             | +0.447 | 6   |
| SINTEX     | 2017-05-25 |  25.85 | 104.10     | implied_event | +0.444 | 85  |
| PCJEWELLER | 2018-05-04 | 174.65 | 121.45     | -             | +0.438 | 22  |
| TRIVENI    | 2011-05-03 |  39.90 |  99.65     | implied_event | -0.429 | 246 |
| DHFL       | 2018-09-21 | 350.55 | 610.55     | -             | -0.426 | 17  |
| JETAIRWAYS | 2019-06-18 |  40.50 |  68.40     | -             | -0.408 | 26  |
| IBREALEST  | 2017-04-17 | 148.20 | 105.85     | -             | +0.400 | 88  |
```

### T3.1 security master

- 4,435 symbols -> **3,891 sids** (544 merged edges): **524 official** (`symbol_changes`) and **20 auto-detected** (`panel/rename_autodetect.csv`; all but a few are the 2022-2024 ETF renames, e.g. ICICIGOLD -> GOLDIETF 2023-12-20, KOTAKNV20 -> NV20 2024-03-20, PHILIPCARB -> PCBL 2022-01-13, BURGERKING -> RBA 2022-02-17).
- Ten most recent merges (all official): NOVARTIND -> EVONILE (2026-11-04), ABCOTS -> ABCIND (2026-11-02), then eight SBI ETF renames on 2026-10-30 (SETF10GILT -> SBI10YGILT, SBIETFQLTY -> SBIQUALITY, SBIETFIT -> SBIIT, SBIETFPB -> SBIPVTBNK, SBIETFCON -> SBICONS, SBINEQWETF -> SBIEQLWT50, SBIBPB -> SBIPSUBNK, SBINMID150 -> SBIMID150).

### T3.3 universe size (average members per session)

```
year   u500   u200        year   u500   u200
2011  357.1  145.3        2019  486.1  212.6
2012  329.4  134.9        2020  522.9  216.8
2013  296.0  130.0        2021  552.3  223.1
2014  443.6  164.5        2022  555.8  226.5
2015  496.0  192.7        2023  556.0  227.8
2016  506.3  194.0        2024  553.1  227.6
2017  550.3  216.6        2025  552.2  227.5
2018  554.2  224.7        2026  560.2  229.0
```

Matches the expectation (about 290-350 in 2011-2013, rising). From 2017 the u500 holds about 550 because of the exit rank 600 hysteresis; u200 holds about 225 for the same reason (exit rank 250).

### T3.4 market frame, 2012-01-01 -> 2025-09-30 (3,409 sessions, `n_u500` 249 to 568)

```
mean ew500_co  (arithmetic mean of members' ret_co)  +23.5 bps
mean ew500_oc  (arithmetic mean of members' ret_oc)           -16.0 bps
mean ew500_ret (tret)                                          +7.5 bps
like-for-like with PLAN B.3 (mean of LOG returns, same members, bigmoves excluded):
  overnight ln(1+ret_co)  +22.7 bps     intraday ln(1+ret_oc)  -19.2 bps
```

PLAN expects about +22 and -20, measured as mean LOG returns. The log-return means above match (+22.7, -19.2). The panel stores fractions, so `ew500_co/oc` are arithmetic means; those exceed log means by about sigma^2/2 per day (about 1.7 bps for the intraday leg, volatile at 1.8% a day). Per year (arithmetic) `ew500_co` / `ew500_oc` in bps: 2012 21.5/-7.4, 2013 12.0/-15.5, 2014 21.5/-2.0, 2015 27.5/-22.3, 2016 24.9/-22.8, 2017 31.4/-13.1, 2018 15.0/-24.6, 2019 17.7/-20.5, 2020 31.1/-19.2, 2021 36.6/-18.9, 2022 19.9/-17.5, 2023 27.5/-10.5, 2024 27.9/-17.5, 2025 11.3/-10.7: overnight positive and intraday negative in every year, as in Finding 3.

## After T2.8 (2026-11-12): announcements, board meetings, results index

**Truncation check (before the backfill).** 2026-11-02 -> 2026-11-08: the 7-day call returned 6,699 rows; the seven 1-day calls returned 904 + 1,142 + 1,419 + 1,530 + 1,239 + 426 + 39 = 6,699 with identical seq_id sets. Four other weeks (2011-03, 2015-03, 2020-03, 2024-03) also matched exactly. So 7-day windows are used for all history. As a safety net a window that returns 9,000 rows or more is re-read day by day (never triggered).

**Backfill.** `stocksense ingest announcements` walks 7-day windows back from yesterday to 2010-01-01 (the API serves 2010-01-01 onward; the plan expected 2011). Log `data_store/logs/t28_announcements.log`. 863 of 880 windows succeeded on the first pass; the 17 that failed (2012-09-05..10-02 and 2015-01-21..04-21) were transient server errors: they succeeded on retry (14,948 rows). Raw windows are cached gzipped under `cache/announcements/`. Stored: **1,559,164 rows**, 2010-01-01 09:25 -> 2026-11-11 23:59 IST, 858 distinct categories. 15 seq_ids occur in two month partitions (the same filing stamped at a month boundary); key is seq_id per partition file. `exch_ts` is null for 36.5% of rows (NSE's `exchdisstime` is `-`); `first_seen_ts` is null for every backfilled row (live poll only).

Rows per year (by `ts`):

| year | rows |
rows 1559164 unique 1559149 first 2010-01-01 09:25:00+05:30 last 2026-11-11 23:59:36+05:30
results_index 195678 2010-01-04 15:20:00+05:30 2026-11-03 10:29:00+05:30
board_meetings 181993 2010-01-04 2026-12-05

30 most frequent `category` values (input to the T8.2 taxonomy; note the case variants "General Updates" / "General updates" and "Loss of Share Certificates" / "Loss of share certificate"):

rows 1559164 unique 1559149 first 2010-01-01 09:25:00+05:30 last 2026-11-11 23:59:36+05:30
| year | rows |
|---|---|
| 2010 | 26,100 |
| 2011 | 31,562 |
| 2012 | 35,739 |
| 2013 | 37,911 |
| 2014 | 41,617 |
| 2015 | 48,007 |
| 2016 | 57,359 |
| 2017 | 66,145 |
| 2018 | 77,172 |
| 2019 | 93,981 |
| 2020 | 101,469 |
| 2021 | 117,100 |
| 2022 | 136,895 |
| 2023 | 163,701 |
| 2024 | 183,403 |
| 2025 | 175,936 |
| 2026 | 165,067 |
categories 858 null 0
| # | category | rows |
|---|---|---|
| 1 | Updates | 188,537 |
| 2 | Analysts/Institutional Investor Meet/Con. Call Updates | 144,853 |
| 3 | Outcome of Board Meeting | 95,966 |
| 4 | Financial Result Updates | 93,701 |
| 5 | Press Release | 83,017 |
| 6 | Shareholders meeting | 74,778 |
| 7 | Copy of Newspaper Publication | 73,512 |
| 8 | Trading Window | 70,803 |
| 9 | Loss of Share Certificates | 63,048 |
| 10 | Disclosure under SEBI Takeover Regulations | 53,371 |
| 11 | Certificate under SEBI (Depositories and Participants) Regulations, 2018 | 37,642 |
| 12 | General Updates | 36,390 |
| 13 | Investor Presentation | 29,088 |
| 14 | Dividend | 20,860 |
| 15 | Appointment | 20,821 |
| 16 | General updates | 17,888 |
| 17 | Record Date | 17,075 |
| 18 | Disc. under Reg.30 of SEBI (SAST) Reg.2011 | 15,863 |
| 19 | Allotment of Securities | 14,329 |
| 20 | Credit Rating | 14,288 |
| 21 | Change in Director(s) | 13,645 |
| 22 | Resignation | 12,320 |
| 23 | Loss of share certificate | 12,008 |
| 24 | Related Party Transactions | 11,619 |
| 25 | ESOP/ESOS/ESPS | 11,323 |
| 26 | Schedule of Analysts/Institutional Investor Meet/Con. Call | 10,465 |
| 27 | AGM/Book Closure | 10,251 |
| 28 | Annual General Meeting Updates | 10,249 |
| 29 | Book Closure | 9,509 |
| 30 | Disclosure under Insider Trading Regulation | 9,062 |

**Board meetings** (`ingest board-meetings`, 30-day windows by meeting date): 181,993 rows, meeting dates 2010-01-04 -> 2026-12-05 (the API returns announced future meetings); the earliest window is 2010-01 (data exists back to at least 2010-01-01).

**Results index** (`ingest results`): probe 2026-11-11: the API honours `from_date`/`to_date` (a filter on the FILING date) and `period=Quarterly` or `Annual` (`Half-Yearly` returns nothing); without dates it returns only recent filings (3,816 quarterly). Backfilled in 31-day windows for both periods: 195,678 rows, filing dates 2010-01-04 -> 2026-11-03. Fields: `seqNumber` (key), `fromDate`/`toDate` (period), `filingDate` (`%d-%b-%Y %H:%M`, no seconds), `exchdisstime`, `audited`, `consolidated`, `relatingTo`, `xbrl`.