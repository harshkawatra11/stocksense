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
