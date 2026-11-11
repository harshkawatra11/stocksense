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
