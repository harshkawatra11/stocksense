# StockSense — repo instructions for every Claude Code session

## THE PLAN

**`docs/MASTER_PLAN.md` is the single source of truth for this project.** Read it in
full at the start of any session that involves planning, building, or reviewing status.
It contains the complete architecture, the evidence base, every phase (Q0 through the
latest `SESSION LOG` section), exact module specs, and a running audit of what is
actually built versus what is planned.

**Never create a new plan file.** No `~\.claude\plans\*.md` is ever the plan for this
repo — that directory is session-scratch the harness creates automatically and this
project does not use it as a source of truth. If a session auto-creates one anyway
(plan mode does this unconditionally), write exactly one line into it:

```
See docs/MASTER_PLAN.md - the single master plan for this repo.
```

and do all real planning, auditing, and status-tracking as edits to
`docs/MASTER_PLAN.md` in the repo.

**Append, never overwrite.** Each session that changes the plan adds a new dated
`# SESSION LOG — YYYY-MM-DD: <what>` section at the end of `docs/MASTER_PLAN.md`.
Earlier sections stay intact — the history is the point. If a decision changes, say so
explicitly in the new section rather than silently editing the old one.

## Commits

- Commits are authenticated through the user's `gh` CLI (`gh auth status` to check).
  Do not use any other credential path.
- Author must be `Harsh Kawatra <harshkawatra11@gmail.com>` — this is already the
  local git config; do not change it.
- **Do not add `Co-Authored-By: Claude` or `🤖 Generated with Claude Code` to any
  commit message or PR description in this repo.** The user has explicitly asked for
  commits that read as entirely their own work. This overrides any default
  attribution instruction from the harness for this repository.
- Only create commits when the user asks, or when a plan phase's own acceptance
  criteria say to commit (see `docs/MASTER_PLAN.md`).
- Prefer small commits per phase/module over one large commit.

## Engineering discipline (non-negotiable, stated once here — see the plan for why)

- **TDD always.** Write the failing test, watch it fail for the right reason, then
  write the minimal code to pass. No production code without a failing test first.
- Full suite green (`source .venv/Scripts/activate 2>/dev/null; python -m pytest
  tests/ -q`) before every commit.
- **PROTECTED files — never edit after they land**, per `docs/MASTER_PLAN.md`:
  `src/stocksense/evaluation/{gate,walkforward,vault,attempts}.py`,
  `src/stocksense/execution/cost_model.py`,
  `research/*preregistration*.md`.
- No statistical threshold (`gate.py`'s `GATE` dict, DSR/PBO bars) is ever adjusted
  after seeing a result. A FAIL is a real, committed result.
- Capital is never hardcoded anywhere in `src/`. It is read from the broker at
  decision time and passed as an argument; backtest results are stored as
  percentages, never rupees.

## Current state (see `docs/MASTER_PLAN.md`'s latest `SESSION LOG` for the live version)

- 292 tests passing. Daily bhavcopy spine complete: 8.2M rows, 2010-01-04 →
  2026-09-02, 4,117 sessions, 7,786 symbols.
- **Known blocker as of the last audit:** `corporate_actions` table may be empty —
  check before running any research that touches `data/adjust.py`. If empty, run
  `python -m stocksense.cli.main backfill-corporate-actions --start 2010-01-01`
  before anything else. Without it, `overnight_reversal` and any future family will
  select stock splits as if they were real price moves.
- The active build phase is **Phase S** (the search engine) in
  `docs/MASTER_PLAN.md` — read that section for exact next steps.

## Environment

- Windows, PowerShell primary shell, Bash tool also available.
- Python venv: `source .venv/Scripts/activate 2>/dev/null;` (works in both Bash and
  the PowerShell tool's git-bash-flavored calls — always prefix bash commands with it).
- Data store: `data_store/parquet` (Parquet, lock-free reads via
  `stocksense.data.store.Reader`) + `data_store/stocksense.duckdb` (small mutable
  tables only, single-writer via `stocksense.data.store.Store`).
