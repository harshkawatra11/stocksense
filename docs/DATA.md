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
