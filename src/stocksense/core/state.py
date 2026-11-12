from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import date, datetime, timedelta

from stocksense.core.clock import Clock, SystemClock
from stocksense.core.config import Settings
from stocksense.core.paths import Paths

# (version, sql). Later tasks append theirs; see PLAN D.6 for which task owns which.
MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
CREATE TABLE job_runs(job TEXT, trade_date TEXT, attempt INTEGER,
  status TEXT CHECK(status IN('running','done','failed','skipped')),
  started_at TEXT, finished_at TEXT, detail TEXT, PRIMARY KEY(job, trade_date, attempt));
CREATE TABLE kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);
""",
    ),
    (
        2,
        """
CREATE TABLE trials(trial_id TEXT PRIMARY KEY, created_at TEXT, family TEXT, target TEXT, spec_json TEXT,
  spec_hash TEXT, status TEXT, metrics_json TEXT, returns_path TEXT, git_sha TEXT, note TEXT);
""",
    ),
    (
        3,
        """
CREATE TABLE models(model_id TEXT PRIMARY KEY, family TEXT, target TEXT, spec_hash TEXT, train_end TEXT,
  created_at TEXT, status TEXT CHECK(status IN('candidate','shadow','champion','retired')),
  validated INTEGER DEFAULT 0, path TEXT, note TEXT);
CREATE TABLE model_events(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, model_id TEXT, event TEXT,
  detail TEXT);
CREATE TABLE live_ic(model_id TEXT, decision_date TEXT, target TEXT, ic REAL, n INTEGER,
  PRIMARY KEY(model_id, decision_date, target));
""",
    ),
    (
        4,
        """
CREATE TABLE decisions(decision_id TEXT PRIMARY KEY, book TEXT, strategy TEXT, trade_date TEXT,
  created_ts TEXT,
  capital_ref_inr REAL, capital_stale INTEGER, risk_json TEXT, payload_json TEXT);
CREATE TABLE orders(order_id TEXT PRIMARY KEY, decision_id TEXT, book TEXT, sid INTEGER, symbol TEXT,
  side TEXT CHECK(side IN('BUY','SELL')), product TEXT CHECK(product IN('delivery','intraday')),
  when_kind TEXT CHECK(when_kind IN('open','close','now')), trade_date TEXT, qty INTEGER, ref_price REAL,
  created_ts TEXT, status TEXT CHECK(status IN('pending','filled','cancelled')), reason TEXT,
  cancel_reason TEXT);
CREATE TABLE fills(fill_id TEXT PRIMARY KEY, order_id TEXT, ts TEXT, price REAL, qty INTEGER,
  slip_assumed REAL,
  quote_bid REAL, quote_ask REAL, quote_ltp REAL, source TEXT, official_ref REAL, reconciled
  INTEGER DEFAULT 0);
CREATE TABLE fees(fill_id TEXT PRIMARY KEY, brokerage REAL, stt REAL, exchange REAL, sebi REAL, stamp REAL,
  gst REAL, dp REAL, total REAL);
CREATE TABLE cash_events(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, book TEXT, kind TEXT,
  amount_frac REAL,
  ref TEXT);
CREATE TABLE corp_adjust(id INTEGER PRIMARY KEY AUTOINCREMENT, book TEXT, sid INTEGER, date TEXT, factor REAL,
  note TEXT);
CREATE TABLE marks(book TEXT, date TEXT, nav REAL, cash_frac REAL, gross REAL, n_pos INTEGER,
  capital_ref_inr REAL, day_ret REAL, PRIMARY KEY(book, date));
CREATE TABLE corrections(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, fill_id TEXT, old_price REAL,
  new_price REAL, reason TEXT);
CREATE TABLE mark_prices(book TEXT, date TEXT, sid INTEGER, price REAL, PRIMARY KEY(book, date, sid));
""",
    ),
]

_STALE_RUNNING = timedelta(hours=6)
_SEED_FLAG = "planning_trials_seeded"
_FINAL_STATUSES = ("done", "failed", "skipped")


class State:
    """SQLite state store (WAL) with numbered migrations, a job table and a key/value table."""

    def __init__(
        self,
        settings: Settings,
        clock: Clock | None = None,
        migrations: Sequence[tuple[int, str]] | None = None,
    ) -> None:
        self._path = Paths(settings).sqlite
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._migrations = list(MIGRATIONS if migrations is None else migrations)

    def connect(self) -> sqlite3.Connection:
        """A new autocommit connection (WAL, busy_timeout 30 s, foreign keys on)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self._path, timeout=30, isolation_level=None)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA busy_timeout=30000")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def migrate(self) -> None:
        """Apply missing migrations, each in its own transaction; idempotent."""
        con = self.connect()
        try:
            con.execute("CREATE TABLE IF NOT EXISTS schema_version(version INTEGER PRIMARY KEY)")
            done = {r[0] for r in con.execute("SELECT version FROM schema_version")}
            for version, sql in sorted(self._migrations):
                if version in done:
                    continue
                try:
                    con.executescript(
                        f"BEGIN;\n{sql}\nINSERT INTO schema_version(version) VALUES({int(version)});\nCOMMIT;"
                    )
                except sqlite3.Error:
                    if con.in_transaction:
                        con.execute("ROLLBACK")
                    raise
            has_trials = (
                con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='trials'").fetchone()
                is not None
            )
        finally:
            con.close()
        if has_trials and self.kv_get(_SEED_FLAG) is None:
            from stocksense.validation.trials import seed_planning_trials

            seed_planning_trials(self)
            self.kv_set(_SEED_FLAG, "1")

    # ---- helpers ----------------------------------------------------------------------------

    def _now_iso(self) -> str:
        return self._clock.now().isoformat(timespec="seconds")

    @staticmethod
    def _d(trade_date: date | str) -> str:
        return trade_date if isinstance(trade_date, str) else trade_date.isoformat()

    # ---- jobs -------------------------------------------------------------------------------

    def job_start(self, job: str, trade_date: date | str) -> int:
        """Record a new `running` attempt for (job, trade_date); returns its attempt number."""
        td = self._d(trade_date)
        con = self.connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT COALESCE(MAX(attempt), 0) FROM job_runs WHERE job=? AND trade_date=?", (job, td)
            ).fetchone()
            attempt = int(row[0]) + 1
            con.execute(
                "INSERT INTO job_runs(job, trade_date, attempt, status, started_at) VALUES(?,?,?,?,?)",
                (job, td, attempt, "running", self._now_iso()),
            )
            con.execute("COMMIT")
            return attempt
        except BaseException:
            if con.in_transaction:
                con.execute("ROLLBACK")
            raise
        finally:
            con.close()

    def job_finish(
        self, job: str, trade_date: date | str, attempt: int, status: str, detail: str = ""
    ) -> None:
        """Close an attempt with status done / failed / skipped."""
        if status not in _FINAL_STATUSES:
            raise ValueError(f"status must be one of {_FINAL_STATUSES}, got {status!r}")
        con = self.connect()
        try:
            con.execute(
                "UPDATE job_runs SET status=?, finished_at=?, detail=? "
                "WHERE job=? AND trade_date=? AND attempt=?",
                (status, self._now_iso(), detail, job, self._d(trade_date), attempt),
            )
        finally:
            con.close()

    def job_status(self, job: str, trade_date: date | str) -> str | None:
        """Status of the latest attempt; a `running` one older than 6 h counts as `failed`."""
        con = self.connect()
        try:
            row = con.execute(
                "SELECT status, started_at FROM job_runs WHERE job=? AND trade_date=? "
                "ORDER BY attempt DESC LIMIT 1",
                (job, self._d(trade_date)),
            ).fetchone()
        finally:
            con.close()
        if row is None:
            return None
        status, started = row
        age = self._clock.now() - datetime.fromisoformat(started) if started else timedelta(0)
        if status == "running" and age > _STALE_RUNNING:
            return "failed"
        return status

    # ---- key/value --------------------------------------------------------------------------

    def kv_get(self, key: str, default: str | None = None) -> str | None:
        con = self.connect()
        try:
            row = con.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        finally:
            con.close()
        return default if row is None else row[0]

    def kv_set(self, key: str, value: str) -> None:
        con = self.connect()
        try:
            con.execute(
                "INSERT INTO kv(key, value, updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, value, self._now_iso()),
            )
        finally:
            con.close()
