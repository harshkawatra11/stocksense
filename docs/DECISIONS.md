# Decisions and deviations

Each entry: date, task id, what was observed or decided, and why. Newest at the bottom of each section.

## Defaults chosen at planning time (2026-11-11)

| Decision | Default | Why |
|---|---|---|
| Backup before wipe | `C:\Users\harsh\Desktop\HARSH\_stocksense_archive_2026-11-11\` (outside the repo; never referenced) | the wipe is irreversible |
| GitHub | same repo, new orphan `main`, every other branch deleted | user request |
| Trading mode | paper only, enforced by a test | user request; SEBI 2026 algo rules need a static IP and Algo-ID for real API orders |
| Strategy families | S1 delivery (buy at close, sell at open), S2 auction open→close, S3 events, S4 minute bars later | measured on the data (PLAN B.3) |
| LLM | none in the decision path | user request |
| Runs on | this laptop via Windows Task Scheduler | GPU and data are here |
| Commits | author Harsh Kawatra, no attribution lines | user preference |

## Phase 0

- 2026-11-11 T0.6: `data_store/pytest-temp-20260929` (old test scratch with a permission-locked subfolder) could not be deleted; it was moved into `data_store/_archive_v1/`.
- 2026-11-11 T0.7: `.env` uses `STOCKSENSE_`-prefixed key names; all eleven keys are set, including the Upstox Analytics Token. Mapping recorded in PLAN T1.2.

## Phase 1

- T1.1 (2026-11-11): torch resolved from the cu128 index as written; uv picked torch 2.11.0+cu128 (not 2.9.x; the `>=2.9` constraint allows it). CUDA available, RTX 3050 detected. cu126 fallback not needed. hatch wheel target with `third_party/kronos` resolved and built without issue.
- T1.1 (2026-11-11): `cli.py` has a `@app.callback()` so the placeholder `version` stays a subcommand (a single-command typer app otherwise collapses it); T1.x will add real commands.
- T1.2 (2026-11-11): `config/gate.toml` is not created here (T1.7 freezes it). `Settings.gate` is `Gate | None`; `load_settings` loads it only if the file exists, otherwise None. T1.7 should make it required. `python-dotenv` added to pyproject explicitly (it is a pydantic-settings dependency, used to read `.env` so that empty values can be skipped for "first non-empty wins").
- T1.2 (2026-11-11): `Settings` also carries a `schedule` model (only `special_sessions` for now, extra keys allowed) so the calendar needs no separate config read; relative `data_root` is resolved against the parent of the config directory.
- T1.2 (2026-11-11): conftest `tmp_data_root` now returns a `Settings`; the raw `Path` is the new fixture `tmp_data_dir`.
- T1.4 (2026-11-11): old `bhavcopy_eq` partitions mix timestamp units (ms and us), so `scan_parquet` over a glob fails with a SchemaError; the calendar reads the `date` column file by file (private helper, TODO T1.6 to move to `data/io.py`). The holidays dataset does not exist yet; `from_store` logs one warning and treats future weekdays as sessions. `config/schedule.toml` is a stub with `special_sessions = []` (T10.4 completes it).
- T1.5 (2026-11-11): `State(settings, clock=None, migrations=None)` takes an optional Clock (default SystemClock) because timestamps and the 6-hour stale-`running` rule need "now"; `migrations` lets tests inject a list. `Secrets.secret_values()` returns the non-empty raw values for log scrubbing only. Logging is a thin `StructLogger` (`log.info("event", **fields)`) over stdlib logging; `configure_logging` installs handlers on the root logger and returns the file handler.
- T1.6 (2026-11-11): PLAN/task fact FALSE: 3,688 is the TOTAL `bhavcopy_eq` row count for 2026-09-29, not the EQ-series count. Series EQ alone is 2,671. The realdata test asserts both.
- T1.6 (2026-11-11): stored dtypes observed: `bhavcopy_eq`/`bhavcopy_delivery` `date` is naive Datetime (ms in most files, us in 2026-09); `bhavcopy_eq.n_trades` is Float64 in 2010-2011-06 files and Int64 after (catalog Float64, cast on read); `fo_bhavcopy` `date` AND `expiry_date` are naive Datetime(us) (catalog: both Date); `intraday_bars.ts` is stored as `Datetime(us, UTC)` (not IST) and `read` converts to `Datetime(us, Asia/Kolkata)` (same instants; 03:45 UTC = 09:15 IST); `corporate_actions` has no `record_date/series/isin/source/rights_price` yet (read as typed nulls).
- T1.6 (2026-11-11): every catalog `Datetime` column is `Datetime("us", "Asia/Kolkata")`; a naive source is assumed to be IST. Partition names used: month, year, all, day. `nse_snapshots` IS in the catalog (partition `day`, `nested=True`, `keep_extra=True`, no key): `read(settings, "nse_snapshots", sub="gainers")`. The same `sub` argument serves `features` / `labels` (`sub="v1"`) and `preds` (`sub=<model_id>`); these three keep extra columns (`keep_extra`). Derived dataset names: `security_master`, `panel_daily`, `panel_universe`, `panel_market`, `adj_events`, `features`, `labels`, `preds`; single-file ones carry `file=` (`panel/security_master.parquet`, `panel/market.parquet`, `panel/adj_events.parquet`). `adj_events` key is not given in D.5; (sid, date, symbol) chosen.
- T1.6 (2026-11-11): `fo_bhavcopy` key contains `strike_price` and `option_type`, which are null for futures, so `DatasetSpec.nullable_key` exempts them from the null-key check (polars `unique` treats nulls as equal, so upsert still de-duplicates). `write_partition`/`upsert` take an optional `clock` (default SystemClock) used for the PermissionError retry sleep (a naked `time.sleep` is banned by the T1.3 guard); `upsert` conforms old partitions to the catalog schema when it rewrites them (so a rewritten month gets `Date` and the new nullable columns). `core/calendar.py` now reads dates via `data.io.read(..., columns=["date"])`.
- T1.7 (2026-11-11): `config/gate.toml` copied from PLAN D.7 programmatically (LF endings); SHA-256 after CRLF->LF normalization `e26a13769b707cba98b409f19880f905d012abaa5d0205d2ab84f062492f270e`, pinned in `tests/unit/test_gate_frozen.py`. Both measures taken for stable hashes: the test normalizes CRLF, and a `.gitattributes` (`eol=lf` for *.py, *.toml, *.md, *.lock) was added. `Settings.gate` is now required (`Gate`, not `Gate | None`); `load_settings` raises `ConfigError` if `gate.toml` is missing; T1.2's "gate none" test replaced by two tests (loaded values, required).
- T1.7 (2026-11-11): the T1.5 `Paths` class originally had an `archive` property naming the old-system folder; `test_no_archive_refs` correctly rejected it, so it was removed (nothing may reference that folder from src/).
- T1.8 (2026-11-11): `doctor` takes its Settings from `ctx.obj` when a Settings is injected (tests) and otherwise loads `config/` (`--config-dir` option on the app callback). Last dates: newest partition file only, via `data.io.read` with a start bound (dataset with `date_col`); `nse_snapshots` (no date column) takes the newest `YYYY-MM-DD.parquet` file name across kinds (real store: 2026-10-21) and sums metadata row counts over all kinds. `doctor` runs `State.migrate` (creates `state/stocksense.sqlite` if missing).
- T1.9 (2026-11-11): the synthetic market has 601 sessions (600 weekdays + the Saturday at index 120); all scripted "session N" indices are positions in that 601-long list. S16 demerger and S25 rights rows use `factor_price=1.0`, `parse_status='unparsed'` (the convention in the real `corporate_actions` table: 1.0 = no factor), and the demerger's `true_ret` is the underlying return (the holder keeps the value in the demerged shares), only the raw price drops 30%. S09 price is frozen (return 0) during the suspension so the path is continuous; the first listed row of every stock has `true_ret` null; S19 lists at a base price of 350.

## Phase 2

- 2026-11-11 T2.1: live smoke check passed (EQUITY_L.csv 185,986 bytes starting `SYMBOL,NAME OF COMPANY`; `/api/holiday-master` returned a dict with `CM`). NSE trading holidays 2026-11: 08-Nov (Diwali Laxmi Pujan, Sunday Muhurat), 10-Nov (Diwali-Balipratipada), 24-Nov. `NseClient` also raises `DataNotAvailable` on an API 404 and after 5 exhausted attempts on 429/5xx (archive and API); persistent transport errors re-raise the httpx error.
- 2026-11-11 T2.2: `validate_dataset` has a keyword-only `min_session_rows` (default 500, the plan's value) because the synthetic market has only 40 stocks per session; tests pass 10.
- 2026-11-11 T2.2: OHLC and positive-price checks skip rows with open = high = low = 0 (untraded instruments). Without this, `fo_bhavcopy` showed 4.9M false violations (no-trade contracts carry only close/settle). Remaining real-data findings: bhavcopy_eq 91 `high_below_max_open_close` and 126 `low_above_min_open_close` rows; fo_bhavcopy 26 and 74. Price issues are aggregated into one Issue per check with a row count.
- 2026-11-11 T2.2: besides the 19 `missing_session_before` issues, `bhavcopy_eq` has 10 `session_row_count_off` issues (2014-08-06..2014-08-20: 1,419-1,434 EQ rows vs trailing median ~1,189, a genuine step up in the EQ count that the trailing-20 median needs two weeks to absorb); reported, not suppressed.
- 2026-11-11 T2.2: `fresh` = newest date >= last completed session for per-session datasets (catalog freshness <= 5 days), else (today - newest date) <= catalog freshness days; `None` where the catalog has no freshness rule (`nse_snapshots`). `sessions_missing` is computed for per-session datasets only (calendar sessions between first and last date with no rows; the calendar is derived from bhavcopy_eq, so that dataset is 0 by construction). Status lists every catalog dataset that has files on disk (7 today). `data-status --issues` also prints each issue.
- 2026-11-11 T2.3: plan fact FALSE (mcp package): `mcp.client.streamable_http.streamablehttp_client` no longer exists in the installed `mcp`; the function is `streamable_http_client(url)` and it yields TWO streams (not three), and `Tool.inputSchema` is `Tool.input_schema`. `cross_check_mcp` uses the new names (reads either spelling of the schema attribute). `get_ltp_by_date(symbol, date)` returns a bare number as text (e.g. `1182.0`); `get_stock_history` needs `months` and `endDate`, so only `get_ltp_by_date` is used.
- 2026-11-11 T2.3: `ingest eq` is a sub-command of an `ingest` group; bare `stocksense ingest` still prints "not implemented yet (phase 2)" and exits 2 (the group callback), so later ingesters add their own sub-commands.
- 2026-11-11 T2.3: skip rule is per dataset. Equity: skip a day with >= 500 stored rows. Delivery (days >= 2019-08-23 only, history is never extended earlier): fetched for any day that has equity rows but no delivery rows, which also fills the 25 stored sessions that lacked delivery. A delivery file whose DATE1 differs from the requested date is rejected and NOT written to the cache (the cache holds only validated files). Markers are `no_session:eq:<date>` and `no_session:delivery:<date>`, written only for 404/rejections more than 3 days old.
- 2026-11-11 T2.3: AVG_PRICE is merged into `bhavcopy_eq.avg_price` only for days whose delivery file is ingested in this run (join on symbol, series; every other eq column is kept); old rows (and old days that already had delivery) keep avg_price null. `isin` and `name` are likewise null for rows of days not re-ingested.
- 2026-11-11 T2.3: `holidays` dataset does not exist until T2.7, so the calendar projects every future weekday as a session. 2026-11-10 (a holiday per NSE's holiday API) therefore answers 404 and is reported `not_published`, never marked; `data-status` freshness for that day cannot be judged until T2.7 lands.
- 2026-11-11 T2.3: plan assumption incomplete: not every Saturday/Sunday file with data is an equity session. 2010-05-16 (7 rows) and 2012-11-11 (14 rows) are gold-ETF-only Sunday trading days (GOLDBEES, KOTAKGOLD ...). Stored, they would have become calendar sessions (the calendar is derived from stored dates) and broken every per-session count. Rule added: a parsed day with fewer than `MIN_SESSION_ROWS` (500) rows is `no_session` and is not stored (cache kept, marker written). The repair found NO special session beyond the 19 of B.3 with more than 500 rows. The first repair attempt had stored those two days; it was killed, `bhavcopy_eq` and `bhavcopy_delivery` were restored from `data_store/_backup_t23/`, and the repair was rerun (kv markers and the file cache made the rerun cheap). The aborted log is `data_store/logs/t23_repair_attempt1_aborted.log`.
- 2026-11-11 T2.3: result: 19 B.3 specials + 28 new sessions 2026-09-30..2026-11-09 (including the Sunday 2026-11-08 Muhurat session, 3,550 EQ-file rows) were added: 47 sessions, 4,135 -> 4,182. NSE did not publish 2026-11-10 (holiday), reported `not_published`.
- 2026-11-11 T2.3: delivery gaps that remain (`sessions_missing` 24): the 23 sessions 2019-08-26 .. 2019-09-27 (NSE has no sec_bhavdata_full for them; markers written) and 2022-08-08, whose file in the existing cache is an Excel workbook (zip, `PK` header) saved under a `.csv` name, not CSV; it is reported `failed` on every run (no HTTP is made) and left unparsed, since reading xlsx would need a new dependency.
- 2026-11-11 T2.3: MCP cross-check finding: `get_ltp_by_date` answers a non-weekday date with the previous close (MOM30IETF 2026-11-07 and 2026-11-08 both 30.4; BEARDSELL 2025-01-31, 02-01 and 02-02 all 36.67) so weekend special sessions cannot be verified against it (the first CLI run flagged 3 of 18 such pairs). `sample_pairs` therefore draws weekday rows only; 20 weekday pairs from the 27 ingested weekday sessions matched within 0.01 (20 of 20).
- 2026-11-11 T2.3: the realdata tests `test_real_store_calendar` (now 4,154 sessions through 2026-09-29, index 4153) and `test_real_store_missing_sessions...` (now expects zero `missing_session_before`) were updated to the post-repair store; their "before" facts remain in `docs/DATA.md` Baseline.

## Phase 3

## Phase 4

## Phase 5

## Phase 6

## Phase 7

## Phase 8

## Phase 9

## Phase 10

## Phase 11

## Phase 12

## Phase 13
