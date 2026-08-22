"""
Job management (Phase F1). The infrastructure every control-center panel
triggers work through: a closed allowlist of runnable CLI commands, a
typed-parameter spawn path (never arbitrary string execution from the
renderer -- same "closed set, no hallucinated/injected commands"
discipline `foreman/tools/base.py`'s ToolRegistry already uses for agent
tools), live stdout capture, and a stop path.

A hard constraint this module is built around, verified directly twice
in this project's own session history (the daily bhavcopy backfill and
the intraday backfill both hit it): DuckDB's embedded engine gives the
process holding a read-write connection an EXCLUSIVE lock on the file --
no other process, not even a read-only connection, can open it at the
same time. Every long-running CLI command this registry spawns
(backfills, foreman run, train-candidate, statement-ingest) opens its
own `Store` and holds that lock for its ENTIRE run. Two consequences,
both handled deliberately rather than discovered by a confusing error
later:

1. Only one write-holding job may run at a time (`JobRegistry.start`
   raises `JobAlreadyRunningError` otherwise) -- this isn't a UI
   limitation, it's the database's own constraint made explicit and
   enforced in one place instead of failing unpredictably at whichever
   command happens to open its Store connection second.
2. `ui_jobs` (the durability table) is written to ONLY at job start and
   finish -- never polled/updated mid-run, since a mid-run write from
   THIS process would itself block on the lock the subprocess is
   holding. Live status while a job runs comes entirely from this
   module's in-memory registry (PID, ring buffer of output), which needs
   no database access at all.
"""

from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import structlog

from stocksense.core.config import DATA_STORE, REPO_ROOT

log = structlog.get_logger(__name__)

JOB_LOG_DIR = DATA_STORE / "ui_job_logs"
RING_BUFFER_LINES = 500


class UnknownCommandError(Exception):
    pass


class MissingParamError(Exception):
    pass


class JobAlreadyRunningError(Exception):
    """Raised when a second write-holding job is requested while one is
    already running -- see module docstring for why this is a database
    constraint, not an arbitrary UI restriction."""


@dataclass(frozen=True)
class JobCommandSpec:
    """One allowlisted CLI command the UI may trigger. `param_flags`
    is a closed map of accepted parameter names -> CLI flags; a request
    naming any other parameter is rejected, not silently ignored."""
    name: str
    cli_args: tuple[str, ...]
    param_flags: dict[str, str] = field(default_factory=dict)
    required_params: tuple[str, ...] = ()


COMMANDS: dict[str, JobCommandSpec] = {
    "backfill-nse-archive": JobCommandSpec(
        name="backfill-nse-archive",
        cli_args=("backfill-nse-archive",),
        param_flags={"start": "--start", "end": "--end", "kind": "--kind"},
        required_params=("start", "end"),
    ),
    "backfill-corporate-actions": JobCommandSpec(
        name="backfill-corporate-actions",
        cli_args=("backfill-corporate-actions",),
        param_flags={"start": "--start", "end": "--end"},
        required_params=("start", "end"),
    ),
    "backfill-intraday": JobCommandSpec(
        name="backfill-intraday",
        cli_args=("backfill-intraday",),
        param_flags={"start": "--start", "end": "--end", "top_n": "--top-n"},
        required_params=("end",),
    ),
    "index-corpus": JobCommandSpec(
        name="index-corpus",
        cli_args=("index-corpus",),
    ),
    "statement-ingest": JobCommandSpec(
        name="statement-ingest",
        cli_args=("statement-ingest",),
        param_flags={"file": ""},  # positional -- see build_command
        required_params=("file",),
    ),
    "kundli": JobCommandSpec(
        name="kundli",
        cli_args=("kundli",),
        param_flags={"broker": "--broker"},
    ),
    "train-candidate": JobCommandSpec(
        # AUDIT FIX (Phase G/B1): param_flags was empty, so the UI could
        # only ever run this at the hardcoded defaults (horizon=20,
        # top_n=20, cost_bps=25.0) -- cli/main.py's train_candidate takes
        # all three as real options. Gate criteria are deliberately NOT
        # exposed here, matching the command's own docstring: allowing
        # ad-hoc gate overrides from a form would recreate the exact
        # evaluator-overfitting failure pre-registration exists to
        # prevent.
        name="train-candidate",
        cli_args=("train-candidate",),
        param_flags={"horizon": "--horizon", "top_n": "--top-n", "cost_bps": "--cost-bps"},
    ),
    "reconcile": JobCommandSpec(
        # AUDIT FIX (Phase G/B1): same gap -- horizon/lifecycle were
        # real cli/main.py options with no way to reach them.
        name="reconcile",
        cli_args=("reconcile",),
        param_flags={"horizon": "--horizon", "lifecycle": "--lifecycle"},
    ),
    "predict": JobCommandSpec(
        # Phase G/B2: previously unreachable from the UI entirely.
        name="predict",
        cli_args=("predict",),
        param_flags={"horizon": "--horizon", "lifecycle": "--lifecycle"},
    ),
    "tax-summary": JobCommandSpec(
        name="tax-summary",
        cli_args=("tax-summary",),
        param_flags={"fy_start": "--fy-start", "fy_end": "--fy-end", "ltcg_exemption_used": "--ltcg-exemption-used"},
        required_params=("fy_start", "fy_end"),
    ),
    "universe-as-of": JobCommandSpec(
        name="universe-as-of",
        cli_args=("universe-as-of",),
        param_flags={"as_of": "--as-of"},
        required_params=("as_of",),
    ),
    "foreman-run": JobCommandSpec(
        name="foreman-run",
        cli_args=("foreman", "run"),
        param_flags={"goal": "", "max_attempts": "--max-attempts"},  # goal is positional
        required_params=("goal",),
    ),
    "foreman-assess": JobCommandSpec(
        name="foreman-assess",
        cli_args=("foreman", "assess"),
    ),
}


def build_command(spec: JobCommandSpec, params: dict) -> list[str]:
    """Renders a validated params dict into CLI argv. Any param not in
    spec.param_flags is rejected outright -- the closed-set property
    that makes this safe to expose to a renderer."""
    unknown = set(params) - set(spec.param_flags)
    if unknown:
        raise MissingParamError(f"unrecognized parameter(s) for '{spec.name}': {sorted(unknown)}")
    missing = [p for p in spec.required_params if p not in params or params[p] in (None, "")]
    if missing:
        raise MissingParamError(f"missing required parameter(s) for '{spec.name}': {missing}")

    args = list(spec.cli_args)
    for key, flag in spec.param_flags.items():
        if key not in params or params[key] in (None, ""):
            continue
        if flag == "":  # positional argument, e.g. statement-ingest <file>, foreman run <goal>
            args.append(str(params[key]))
        else:
            args += [flag, str(params[key])]
    return args


@dataclass
class _RunningJob:
    job_id: str
    command: str
    proc: subprocess.Popen
    buffer: collections.deque = field(default_factory=lambda: collections.deque(maxlen=RING_BUFFER_LINES))
    log_path: Path | None = None
    status: str = "running"


class JobRegistry:
    """In-memory tracker for currently-running jobs, backed by `ui_jobs`
    for durability across a server restart (start/finish writes only --
    see module docstring). One instance lives for the lifetime of the
    server process; `server/app.py` holds a single module-level instance."""

    def __init__(self, duckdb_path: Path):
        self._duckdb_path = duckdb_path
        self._jobs: dict[str, _RunningJob] = {}
        self._lock = threading.Lock()

    def _store(self):
        from stocksense.data.store import Store
        return Store(self._duckdb_path)

    def _any_running(self) -> bool:
        return any(j.status == "running" for j in self._jobs.values())

    def start(self, command: str, params: dict | None = None) -> str:
        params = params or {}
        spec = COMMANDS.get(command)
        if spec is None:
            raise UnknownCommandError(f"unknown command: {command!r} (not in the allowlist)")

        with self._lock:
            if self._any_running():
                raise JobAlreadyRunningError(
                    "a job is already running -- DuckDB allows only one write-holding "
                    "process at a time; stop it or wait for it to finish first"
                )

            cli_args = build_command(spec, params)
            job_id = str(uuid.uuid4())[:12]
            cmd = [sys.executable, "-m", "stocksense.cli.main"] + cli_args

            JOB_LOG_DIR.mkdir(parents=True, exist_ok=True)
            log_path = JOB_LOG_DIR / f"{job_id}.log"

            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                bufsize=1, cwd=str(REPO_ROOT),
            )
            job = _RunningJob(job_id=job_id, command=command, proc=proc, log_path=log_path)
            self._jobs[job_id] = job

        store = self._store()
        try:
            store.insert_ui_job({
                "job_id": job_id, "command": command, "args_json": json.dumps(params),
                "pid": proc.pid, "status": "running",
                "started_at": datetime.now(timezone.utc), "finished_at": None,
                "log_path": str(log_path),
            })
        finally:
            store.close()

        threading.Thread(target=self._pump_output, args=(job,), daemon=True).start()
        return job_id

    def _pump_output(self, job: _RunningJob) -> None:
        try:
            with open(job.log_path, "w", encoding="utf-8") as f:
                for line in job.proc.stdout:
                    f.write(line)
                    f.flush()
                    job.buffer.append(line.rstrip("\n"))
            job.proc.wait()
        except Exception as e:  # noqa: BLE001 -- must still finalize job status on any pump failure
            job.buffer.append(f"[job runner error: {e}]")
        finally:
            job.status = "completed" if job.proc.returncode == 0 else (
                "stopped" if job.status == "stopping" else "failed"
            )
            store = self._store()
            try:
                store.finish_ui_job(job.job_id, job.status, datetime.now(timezone.utc))
            finally:
                store.close()
            log.info("ui_job_finished", job_id=job.job_id, command=job.command, status=job.status)

    def stop(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.status != "running":
            return False
        job.status = "stopping"
        if os.name == "nt":
            subprocess.run(["taskkill", "/pid", str(job.proc.pid), "/f", "/t"], capture_output=True)
        else:
            job.proc.terminate()
        return True

    def tail(self, job_id: str) -> list[str]:
        job = self._jobs.get(job_id)
        return list(job.buffer) if job else []

    def status(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return {"job_id": job.job_id, "command": job.command, "status": job.status, "pid": job.proc.pid}

    def list_active(self) -> list[dict]:
        return [self.status(jid) for jid in self._jobs]
