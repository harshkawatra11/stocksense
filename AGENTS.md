# StockSense

An autonomous NSE quant research and paper-trading system. Built from zero on 2026-11-11.

## The ten rules

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

## Where things are

- Plan and status: `docs/PLAN.md` (status checklist in Part H; conventions and data contracts in Part D; task cards in Part E).
- Decisions and deviations: `docs/DECISIONS.md`.
- Run tests: `.venv\Scripts\python.exe -m pytest tests -q`
- Commits: author Harsh Kawatra <harshkawatra11@gmail.com>; add no attribution lines.
- "Kronos" in this project means `github.com/shiyu-coder/Kronos`, never Amazon Chronos.
