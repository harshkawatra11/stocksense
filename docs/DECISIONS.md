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

## Phase 2

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
