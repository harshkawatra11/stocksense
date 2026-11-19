# StockSense Program G2 — audit verdict and the execution plan to win

Written 2026-11-19 by the orchestrator after five read-only audits (operations; data sources; ML/quant pipeline; two literature reviews) and three interface extractions. In Phase G2-A this file is copied into the repo as `docs/PLAN-G2.md`. `docs/PLAN.md` stays the master rulebook (Parts C, D, F remain binding); this file adds Program G2 on top of it. Executors: Sonnet-class subagents, one task card at a time.

How to read: Part 0 is for everyone (rules). Part 1 is the target and the user's decisions. Part 2 is the audit verdict. Part 3 is the verified facts every executor relies on (exact interfaces). Part 4 is the shared conventions for G2. Part 5 is the task cards. Parts 6–9 are ordering, failure table, verification and the status checklist.

---

# PART 0 — Rules for this program (binding, in addition to the ten rules in CLAUDE.md)

1. **Read before acting.** Every executor reads Part 0, Part 3, Part 4 and its own card completely, plus `docs/PLAN.md` Part D. Do not read other cards except their **Provides** lines.
2. **Worktree.** The built-in worktree isolation is broken on this machine (drive-letter case). The orchestrator creates the worktree before dispatch:
   `git -C C:\Users\harsh\Desktop\HARSH\WEB-DEV-PROJECTS\stocksense worktree add ..\stocksense-wt-<card> -b g2/<card> main`
   then inside it creates junctions exactly like the existing worktrees: `cmd /c mklink /J .venv ..\stocksense\.venv` and `cmd /c mklink /J data_store ..\stocksense\data_store`, and copies `.env` (`copy ..\stocksense\.env .env`). `.venv`, `data_store`, `.env` are git-ignored.
3. **Test first.** Write the card's tests, run them, see them fail for the stated reason, then implement. Whole suite (`.venv\Scripts\python.exe -m pytest tests -q`) and `ruff check src tests` green before every commit. Run `ruff format` **only on files you created or changed** (four older files are unformatted; leave them).
4. **Commits.** Author is configured (Harsh Kawatra). Message `G2-<card>: <title>`. **No attribution or Co-Authored-By lines** (CLAUDE.md overrides any harness reminder). One commit per card unless the card says otherwise.
5. **Merge.** The executor merges its branch into `main` in the main repo (`git -C ..\stocksense merge --no-ff g2/<card>`), runs the full suite on main, then `git -C ..\stocksense push origin main`. If the merge conflicts in `docs/DECISIONS.md`, `docs/DATA.md` or `docs/RESULTS-*.md`, keep both sides.
6. **Long jobs.** A job expected to exceed 10 minutes runs as ONE self-terminating command with a wall cap, launched from the Bash tool: `timeout <seconds> .venv/Scripts/python.exe -m stocksense ... > data_store/logs/g2_<card>_<name>.log 2>&1`. Use `run_in_background` for it and wait for the completion notification. **Never write a supervisor, retry loop, watchdog, or `for i in ...; do ... done` relauncher script.** Every long job is resumable by design; if it dies, the orchestrator decides whether to relaunch.
7. **GPU.** Only one GPU process at a time, holding `data_store/state/gpu.lock` (existing helper used by `scripts/t75d_compute.py`: msvcrt non-blocking lock; reuse it, see card D0.6). After any GPU job: `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader` must show ~0% and no python of yours in `Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' }`. Kill leftovers with `taskkill /F /T /PID <pid>`. Float32 by default; float16 autocast only where a card says so.
8. **Trials.** Every model or portfolio variant evaluated on a walk-forward test fold is registered (`validation.trials.register_trial`) **before** evaluation. Nothing is deleted from the registry. Hyperparameter selection uses inner-validation only (no test fold) and registers nothing.
9. **Gates.** `config/gate.toml` (v1) is never edited. Gate v2 is created once in card B1 and then frozen (SHA test). No threshold is ever changed to rescue a result. A fail is a result: record it.
10. **Paper only.** No code that places, modifies or cancels a broker order. `tests/unit/test_no_orders.py` stays green. Broker clients are read-only.
11. **Secrets.** Never print, log or commit a value from `.env`. When probing a broker, print only status codes, key names, shapes and counts.
12. **Probes.** When a card says "probe", write the observed fact (status, field names, shapes, limits) into `docs/DECISIONS.md` under `## Program G2` → `### <card>` before writing code that depends on it. If reality contradicts this plan, adapt the smallest piece and record it (rule 9 of CLAUDE.md).
13. **Report format** (last message of every executor): what was done; acceptance command and its real output; last line of the full test run; `git log --oneline -3` on main; anything not done and why; confirmation that no process of yours is running.
14. **Stop rule.** If the same step fails twice after two different fixes, stop and report exactly what fails and what was tried. Do not improvise a redesign.

---

# PART 1 — Target and decisions

## 1.1 What "winning" means here

The user wants a system that, trained on Indian stocks, makes a steady small profit — **₹2,000–3,000 per month** — first proven on paper. Honest translation, net of all costs:

| Account capital | ₹2,500/month as a yearly return | Verdict |
|---|---|---|
| ₹5,00,000 | 6% | realistic: within S1's current backtest |
| ₹2,00,000 | 15% | a stretch |
| ₹1,00,000 | 30% | not realistic; flat fees cost more at small sizes |

No system profits every day; good ones are positive on ~55% of days and most months. The program's job is to raise the **net-of-cost, out-of-sample** Sharpe and then prove it forward on paper.

## 1.2 Decisions the user made on 2026-11-19

1. **Gate v2 for new trials** (card B1): DSR variance within the strategy family; excess-t against an exposure-matched benchmark; every other threshold unchanged; v1 verdicts stay recorded; lockbox and 60-session paper gate remain final judges.
2. **Start paper trading now** (Phase G2-G) with the best current S1 model labelled unvalidated; install the Windows task once tests pass.
3. **Broad model sweep beyond boosting** (Phase G2-D), chosen by evidence, each family on a fixed budget, judged by incremental net value.
4. **Probability and stochastic-process track** (Phase G2-E).
5. **Use all four sources to full capacity** (Phase G2-H): Angel One, Upstox Analytics Token, NSE MCP `cm-market`, NSE MCP `nse-bhavcopy`.

---

# PART 2 — Audit verdict: what went wrong, why, and which card fixes it

| # | What went wrong | Why (evidence) | Fixed by |
|---|---|---|---|
| W1 | S1 loses most of its edge to costs: best trial `g1-s1-y_c2o_10-cce12fb757` gross Sharpe 2.51 → net 1.17; 2× slippage 0.22 | 92% of exits are `rank_decay` after ~5 sessions. Exit threshold `3*k` (`portfolio/construct.py:362`) with `k ≈ 7` at ₹5 lakh ⇒ exit below rank ~21 of 500. Extension requires `_s1_entry_ok` (`:365`), charging a full round-trip hurdle although nothing trades. Trades held ≤3 sessions earn +0.75% net, longer ones +0.05% | C1–C5 |
| W2 | All 13 original G1 S1 trials flat 2021–2025 | backtest drawdown stop never cleared | done (T9.6, `auto_resume_sessions=20`) |
| W3 | Gate line 7 (DSR) unreachable | `sharpe_variance_across_trials(state)` pools all families incl. S2 (+3.7) and y_oc floors (−7) (`validation/gate.py:171`, `validation/trials.py:88`) | B1 |
| W4 | Gate line 5 (excess t) unreachable | net of a ~50%-gross vol-targeted book minus a fully invested cost-free ew500 (`gate.py` line 5, `backtest/metrics.py:92`) | B1 |
| W5 | S2 looked like net Sharpe 3.73, strict capture −4.10/−5.77 | the gap feature and `y_oc` share the official open print (`features/auction.py:4-7`, `labels/forward.py:180`); print noise manufactures skill; the edge is gone by 09:20 | F1 |
| W6 | Ensemble, meta-label, seqnet never truly ran | one model per target → `blend_online` no-op; meta features `days_since_result` hard-coded 90 (`construct.py:811`), disagreement 0; `run_walk_forward` never calls `models.seqnet.add_keys` | C3, D0, D11 |
| W7 | Kronos zero-shot added nothing (Δ IC −0.0007, t −0.48 on y_oc) | its forecasts ≈ short-term reversal (corr −0.39 with `mom_5`); independent literature (Rahimikia et al. 2025) agrees | A7 removes `kronos_feats` from default registry; D9 tests fine-tuning once |
| W8 | Daemon makes no predictions or orders | `predict`, `s1_plan`, `nightly_learn`, `drift`, `promote` in `STUB_STAGES` (`live/jobs.py:610`); `build_context` passes no hooks (`live/daemon.py:225`) | G1–G5 |
| W9 | Data pipeline defects | `backfill_1m` marks the current partial month `done` (`data/ingest_intraday.py:279-307`); 1-minute history only 2026-03→11; `extend_5m` has no failure cap and stores unfinished last bars; `cross_check_mcp` returns `ok` with 0 checks (`data/ingest_eq.py:470`); `InstrumentMap` silently empty on Upstox failure and maps only `-EQ`; labels never rebuilt nightly; derived data stale since 2026-11-11 | A2–A6, A9 |
| W10 | Most source capability unused | Angel: depth 2–5, `opnInterest`, `totBuyQuan/totSellQuan`, `feedToken`, ONE_MINUTE unused (`brokers/angel.py:52-73,173`). Upstox quotes were wired against a dead token (now valid). ~15 MCP tools unused | H1–H6 |
| W11 | Operations failures | leftover retry-loop script relaunched a GPU job; agents died on API 403 mid-job; wrong GitHub credential; built-in worktree isolation broken | Part 0 rules 2, 6, 7 |
| W12 | Smaller biases | GBM hyperparameters chosen on anchors 2016/2019/2022 and applied to 2014–15 folds (`models/gbm.py:206`); S1 universe exit uses `rank ≤ 500` not the u500 hysteresis flag (`construct.py:808`); `estimate_open_penalty` stub; delivery brokerage ₹0 vs ₹20 unresolved; `cost_drag_annual` divides by starting capital | C3, C4, D0.4 |

What did NOT go wrong: the data panel, labels, purged walk-forward and the trial registry. The S1 GBM ranks stocks with mean daily rank IC 0.079 (t 27.6), positive in 12/12 years. The model is not the bottleneck; turnover, timing and gate specification are.

---

# PART 3 — Verified facts executors rely on (2026-11-19)

## 3.1 Repository and machine

- Repo `C:\Users\harsh\Desktop\HARSH\WEB-DEV-PROJECTS\stocksense`, `main` = `origin/main` (HEAD `94edff5` or later). ~1,050 tests, ruff clean. Python 3.12 `.venv`, torch ≥2.9 cu128, polars, lightgbm, scipy, scikit-learn, einops. **No `torch_geometric`** (graph models are written in pure torch with dense matrices). Windows 11, RTX 3050 Laptop 4 GB, 16 GB RAM, 12 logical CPUs.
- Credentials verified live 2026-11-19 (names in `.env`): `STOCKSENSE_ANGEL_API_KEY, _CLIENT_CODE, _PASSWORD, _TOTP_SECRET` (login OK); `STOCKSENSE_UPSTOX_ANALYTICS_TOKEN` (LTP quote and v3 daily candles OK); `STOCKSENSE_TELEGRAM_BOT_TOKEN, _ALLOWED_CHAT_ID, _ALLOWED_USER_ID` (bot OK). Both NSE MCP servers answered (`cm_get_stock_quote`, `cm_get_data_status`, `get_ltp_by_date`, `nse_lookup_symbol`).
- `.mcp.json`: `nse-bhavcopy` → `https://mcp.nseindia.in/bhavcopy/cm/mcp`; `cm-market` → `https://mcp.nseindia.in/cmmkt/mcp`.

## 3.2 Data state

- Raw data current to 2026-11-18 (eq, delivery, fo, index, VIX, participant OI, reference, announcements, macro). 10/10 closes for 12–18 Nov matched `nse-bhavcopy` `get_ltp_by_date`.
- Derived data (`panel_daily`, `panel_universe`, `panel_market`, features v1, labels v1, preds) end **2026-11-11**.
- Features: `data_store/features/CURRENT` = 1; v1 manifest has **138 features** (99 `xs`, 39 `mkt`), no `kr_*`. `features/seq/` does not exist. `features/kronos/small-zs-r150/` holds 242 dates.
- `intraday_bars` (5-minute, Angel, back-adjusted at download) 2024-09-13 → 2026-11-11, ~470 symbols. `bars_1m` (Upstox) only `part-2026-03`…`part-2026-11`.
- Research end `2025-09-30` (`lockbox_start` 2025-10-01). Walk-forward folds: calendar years 2014…2025 (last ends 2025-09-30), embargo 5 sessions, label-end purge.

## 3.3 Results so far (from the trial registry and `data_store/reports/g1/`)

- Trial registry: 102+ trials (30 planning seeds, floors, 4 GBM walk-forwards, G1 portfolio trials, references, auto-resume trials, 2 Kronos).
- GBM walk-forward mean IC: y_c2o_5 0.0763 (t 29.9), y_c2o_10 0.0786 (t 27.6), y_c2o_20 0.0828 (t 27.8), y_oc 0.1645 (t 74.0).
- Best S1 (auto-resume) `g1-s1-y_c2o_10-cce12fb757`: net Sharpe 1.1655, 2× slip 0.2216, excess t −1.70, positive years 0.75, DSR 0.00, PBO 0.0805, max DD 0.172. Turnover 18.6%/day, mean gross 0.51. y_c2o_20 k10 variant `…-4092717971`: net 1.08, 2× slip 0.60.
- Production models (status `shadow`, `validated=0`): `s1-gbm-y_c2o_10-120055de-20250930`, `s2-gbm-y_oc-4fd8ff9b-20250930` (directories under `data_store/models/`; loaded with `learn.nightly.load_bag`).

## 3.4 Interfaces (verbatim signatures; file:line on main)

**Models** (`src/stocksense/models/base.py`)
- `ModelSpec(family: str, target: str, universe: str = "u500", features: list[str] | str = "all", params: dict = {}, entry: str = "close")` (frozen; `to_dict`, `from_dict`).
- `FitInfo(best_iteration: int | None = None, val_ic: float | None = None, params: dict = {})`.
- `class Model(ABC)`: class attr `target_prefix: str = "r_"`; `configure(feature_names, kinds)`; `fit(X, y, w, dates, X_val=None, y_val=None, dates_val=None, seed=0) -> FitInfo`; `refit_full(X, y, w, dates, fit_info, seed)`; `predict(X) -> np.ndarray`; `save(dir)`; `load(dir)`; `explain(X, feature_names) -> np.ndarray | None`.
- Registry: `register_model(key, factory)`; `build_model(spec)` imports `gbm`, `linear` (and `seqnet` only for key `seqnet`) and dispatches on `key = spec.family.rsplit("-", 1)[-1]`. **A new family must be imported inside `build_model`** (card D0.1 generalises this).
- `mean_daily_rank_ic(pred, y, dates, min_pairs=10) -> float` (L130).

**Walk-forward** (`validation/walkforward.py:116`)
`run_walk_forward(spec, settings, state, purpose="research", seeds=(1,2,3), registry=None, folds=None, *, model_factory=None, calendar=None, clock=None, inner_months=12, embargo_sessions=5) -> WFResult(preds, fold_metrics, trial_id, ic)`. Registers the trial first (`model_id = trial_id`); label column `factory(spec).target_prefix + target`; loads labels/features (`spec.params["auction"]` adds `auc_*`); X float32, y float64; per fold: `train_mask` → `inner_validation_split` → `sample_weights` → per seed `configure/fit/refit_full/predict` → per-date rank average → `io.upsert(settings, "preds", …, sub=model_id)` (mode `walkforward`) → `preds/<id>/folds_done.json`. Resumable per fold. `backtest_walk_forward(result, spec, settings, state, *, calendar=None, clock=None, k=10, capital_inr=500_000.0, finish=True)` (L319).

**Splits** (`validation/splits.py`): `Fold(name, train_end, test_start, test_end)`; `walk_forward_folds(settings, calendar)`; `train_mask(label_end, fold, embargo_sessions=5, *, calendar, dates=None, dev_start=None)`; `sample_weights(dates, fold, calendar)`; `inner_validation_split(dates, label_end, months=12, embargo_sessions=5, *, calendar)`.

**GBM** (`models/gbm.py`): `GbmModel(objective="regression", num_leaves=31, min_child_samples=200, colsample_bytree=0.7, n_estimators=1500, n_jobs=8)`; `select_hyperparams(target, objective, settings, state, anchors=(2016, 2019, 2022), date_frac=0.3, seed=0, *, folds=None, calendar=None, inner_months=12, embargo_sessions=5, grid=None, base_params=None, universe="u500", entry=None, purpose="research", auction=False) -> {"best","mean_ic","table"}`; `make_mean_ic_feval(dates_val, y_val)`.

**SeqNet** (`models/seqnet.py`): `add_keys(X, sids, dates) -> np.ndarray` appends `[sid, days_since_epoch]` float32; `SeqProvider.get(days, sids) -> (seq[n,60,12], valid_mask)`; `MemmapSeqProvider(root)`; `SeqNet(n_tab, hidden=64, dropout=0.2, n_ch=12)`; `ic_loss(pred, target, eps=1e-8)` (= 1 − Pearson); `SeqNetModel(provider=None, hidden=64, dropout=0.2, max_epochs=30, patience=5, lr=1e-3, weight_decay=1e-4, min_batch=50, device=None, work_dir=None, pred_chunk=4096)`, `target_prefix="z_"`, one batch per date, AdamW+cosine, AMP on CUDA, OOM halves the batch, checkpoints `work_dir/{fit|refit}/epoch_N.pt`.

**Store / production** (`models/store.py`, `learn/nightly.py`): `save_model(settings, state, model, spec, train_end, meta) -> str`; `load_model(settings, model_id)`; `set_status(state, model_id, status, note="")` (champion uniqueness per prefix); `current(state, prefix, status) -> str | None`; `append_preds(settings, model_id, frame, mode, clock=None)` (upsert `date, sid, score, made_at, mode`). `ModelBag(members=None, factory=None)` (predict = mean of member per-row ranks; call with ONE date's rows); `load_bag(settings, model_id)`; `fit_production(spec, settings, state, as_of, *, seeds=(1,2,3), calendar=None, clock=None, inner_months=12, embargo_sessions=5, model_factory=None, allow_beyond_research=False, note=None) -> str` (idempotent; raises `LockboxError` past research end unless `allow_beyond_research=True`).

**Features / labels / io**: `features.base.feature(name, *, group, kind="xs", window=1, version=1, label="", outputs=None, registry=None)`; `FeatureContext(settings, as_of=None)` with `panel()`, `market()`, `universe()`, `raw(dataset, columns=None, sub=None)`, `sessions()`; rolling helpers `roll_sum, roll_mean, roll_std, roll_var, roll_min, roll_max, roll_corr, roll_cov, roll_slope_r2, log_ret`. `features/registry.py` `FEATURE_MODULES` tuple (contains `"kronos_feats"` at L27). `build_features(settings, start=None, end=None, version=None, *, min_valid=50, registry=None, clock=None) -> int` (auto version bump); `load_features(settings, purpose, start=None, end=None, columns=None)`. `features.seq.build_sequence_tensors(settings, version=1)`. `labels.forward.build_labels(settings) -> int`; `load_labels(settings, purpose, …)`; label columns `y_oc, y_c2o_5, y_c2o_10, y_c2o_20, y_c2c_1, y_o2o_5` plus `z_*`, `r_*`, ends `end_oc, end_5, end_10, end_20, end_c2c_1, end_o2o_5`, flags `entry_blocked_open, entry_blocked_close, label_delisted`. `data.io.read(settings, dataset, start=None, end=None, columns=None, sub=None)`, `upsert(settings, dataset, df, sub=None, clock=None)`.

**Portfolio** (`portfolio/construct.py`)
- Constants L32-44: `S1_RANK_DECAY_MULT=3`, `S1_RANK_DECAY_MIN_HELD=2`, `CASH_HAIRCUT=0.005`, `BUY_CUSHION_INR=30.0`.
- `DayScores(score, er, disagreement={})`; `ScoreSource.scores(d) -> DayScores | None`; `NameRow(close, vol_21, turnover_rank, med_turn_20, industry="UNKNOWN", in_universe=True, stale=False, locked_up=False, in_fno=False, days_to_split=None, days_to_results=None, days_since_result=90, rets=None, upper_circuit=None)`; `PanelDay(date, next_date, rows, breadth_200=None, mkt_ret_21=None)`; `PanelAccess.day(t, next_date) -> PanelDay`.
- `S1Config(h, k_max, risk, costs, slippage, overlay=False, hysteresis=True)` + derived `min_pos`; `S1Config.from_settings(settings, h, slippage, *, k_max=None, overlay=False, hysteresis=True)`.
- `Holding(sid, qty, entry_price, held, vol_at_entry)`; `S1Book(holdings, cash, dd=0.0, stopped=False)`; `S1Plan(orders, k_raw, k, gross_target, port_vol, min_pos, cap, exits, extended, weights, entries, note="")`.
- `_s1_entry_ok(sid, row, er, value, cfg) -> bool` (L279).
- `plan_s1(t, scores_t, er_t, book, equity_inr, panel_t, cfg, *, disagreement=None, meta=None, allow_entries=True) -> S1Plan` (L298). Exit order per holding (sorted sid, first match): `universe_exit` → `stop` → (hysteresis) `rank_decay` if `held >= 2 and r > 3*k`; else if `held >= h-1`: extend when `k>0 and r<=k and _s1_entry_ok(...)` else `horizon`. Entries: `raw=(0.5+score)/max(vol_21,1e-4)*mult`, `allocate_weights`, drop `< min_pos`, BUY `when="close"` on d1.
- `S1Policy(spec, settings, score_source, panel_access, costs, slippage, *, k_max=None, overlay=False, hysteresis=True, meta=None, state=None, book_name="s1", entries_allowed=None, auto_resume_sessions=None)` (L481), `decide(t, book, capital_inr)` (L534).
- `S2Config`, `AuctionRow(open, gap, turnover_rank, med_turn_20, vol_21, in_fno=False, upper_circuit=None)`, `plan_s2(d, scores_d, er_d, auction, free_cash, equity_inr, cfg) -> S2Plan` (L641), `S2Policy(settings, score_source, panel_access, slippage, *, k_max=None)`.
- `EnginePanelAccess(panel, calendar, *, industry=None, split_dates=None, result_dates=None, fno=None, market=None, stale=None, universe_rank=500)` (L751); `in_universe` = `rank <= 500`; `days_since_result` = 90 (hard-coded).
- `portfolio/risk.py`: `RiskContext(...)`, `pre_trade_checks(ctx) -> list[Check]`, `entries_allowed(checks)`.

**G1 research machinery** (`validation/g1.py`, `g1_run.py`, `g1_report.py`, `cli_research.py`)
- `G1Config(start=None, end=None, capitals=None, reference_capital=None, slippage_stress=2.0, folds=None, seeds=(1,2,3), inner_months=12, embargo_sessions=5, min_survivor_ic=0.0, model_factory=None, meta_factory=None, s1_targets=S1_TARGETS, s1_auto_resume=None)`; `resolved(settings)`.
- `_spec_s1(cfg, target, k_max, overlay, models, **extra) -> dict` (keys define the trial hash; optional keys added only when set so old hashes never change).
- `plan_grid(cfg, s1_models, s2_models) -> list[TrialPlan]`; `TrialPlan(kind, family, target, spec, label, trial_id=None)`, kinds `s1, s1-meta, s2, ref-factor, ref-open`.
- `G1Runner(settings, state, calendar, cfg=None, *, clock=None, world=None)`; `run_trial(plan) -> TrialResult` (backtests each capital in the grid + reference + 2× slippage; writes `reports/g1/trials/<id>/{result.json, daily.parquet, trades.parquet, entries.parquet}`; `_ensure_finished` writes `models/_trials/<id>/returns.parquet` and calls `finish_trial`). `load_result(settings, trial_id)`.
- `StoredScoreSource(frame, *, calendar=None, trade_date=False)` (frame columns `date, sid, score, exp_ret, disagreement`).
- `build_world(settings, calendar, *, start, end, slippage=None, cut_events=True) -> G1World` (`s1_access()`, `s2_access()`, `benchmark` with `ew500_ret`, `nifty500_ret`).
- `blend_models(frames, labels, target) -> Blended(frame, weights, ic)`; `surviving_models(state, strategy, target, min_ic=0.0)`.
- `run_g1(settings, state, calendar, cfg=None, *, specs=None, production=True, runner=None, fit=None, factors=SINGLE_FACTORS)`; `make_candidate`, `run_gate(runner, settings, state, out)`; `write_outcome(settings, state, out, auto_resume=None)`.
- CLI: `research g1 [--production/--no-production] [--s1-auto-resume N]` (auto-resume path does not rewrite RESULTS-G1.md); `research finish-trials`; `validate CANDIDATE [--open-lockbox]`; `research select`, `research ic-table`, `research wf-status`, `research diagnostics`, `research trial-report`.
- Stats keys in `result.json` (`_stats`, g1.py:620): `capital, n_days, sharpe, cagr, vol, max_drawdown, frac_positive_years, net_return, gross_sharpe, mean_turnover, mean_gross, cost_drag_total, cost_drag_annual, n_trades, avg_trade_cost_frac, trade_stats`. `daily.parquet`: `date, net, ret_gross, n_pos, turnover, costs_frac, gross, cash_frac, net_2x, net_<cap>…, benchmark`.

**Gate** (`validation/gate.py`): `CandidateResult(candidate_id, family, target, daily, ic=None, preds=None, label_col=None, trial_ids=())`; `evaluate_research_gate(candidate, settings, state) -> GateReport` (L153); lines as listed in Part 2 W3/W4; `validate_candidate(candidate_id, settings, state, open_lockbox=False, *, clock=None, lockbox_runner=None)` (L271). `trials.py`: `register_trial(state, family, target, spec) -> str` (id `f"{family}-{target}-{sha[:10]}"`, idempotent on spec hash), `finish_trial(state, trial_id, metrics, returns_path)`, `n_trials(state, family=None)` (counts all statuses), `sharpe_variance_across_trials(state, family=None)`. `tests/unit/test_gate_frozen.py` pins SHA-256 of `config/gate.toml` with CRLF→LF (`GATE_SHA256 = "e26a1376…270e"`).

**Engine / metrics / costs**: `run_backtest(panel_access, calendar, policy, start, end, capital_inr, costs, slippage, risk, *, debug=True) -> BacktestResult(daily, trades, rejects, summary, fills)`; `perf_stats(daily_ret, benchmark, periods=252, dates=None)`; `newey_west_tstat(x, lags=5)`; `ic_stats(pred_frame, label_col)`. `trade_cost(side, product, value_inr, sched, first_sell_of_scrip_today=True) -> CostBreakdown`; `round_trip_cost_frac(product, value_inr, sched)`; `min_position_for_fixed_cost(product, sched, max_fixed_bps)`; `SlippageTable.load_latest(settings)`, `.scaled(f)`; `slip_frac(turnover_rank, vol_21, order_value, med_turn_20, when, table, as_of=None)`. `models/ensemble.py`: `blend_online(pred_frames, label_frame, target, eta=20.0, halflife=60, floor=0.05) -> (blend, weights)`; `expected_return_map(blend, label_frame, target, window=504)` (`exp_ret = max(0, slope)*score`).

**Live** (`live/jobs.py`, `live/daemon.py`)
- `JobContext(settings, clock, state, calendar, *, hub=None, broker=None, instruments=None, nse=None, notifier=None, steps=None, hooks=None, required_fresh=DEFAULT_REQUIRED_FRESH, log_console=None)`; attrs incl. `ledger`; `ctx.hook(name)`; `ctx.paper()`; `ctx.catch_up(only=None)`.
- Hook keys and signatures: `s2_orders(ctx, d, cap)` and `s1_orders(ctx, d, cap)` → iterable of `(book, list[NewOrder], payload)`; `s1_plan(ctx, d) -> Any`; `capture_preopen(ctx) -> int`; `corp_actions(ctx, d) -> Iterable[CorpAction]`; `u500_symbols(ctx) -> list[str]`; `stage:<name>(ctx, d) -> str` overrides an evening stage.
- `_decide_with_hook`: no hook → `"skipped: not implemented (...)"`; otherwise `capital_ref(ctx)` (None → `DataNotAvailable`), `ledger.record_decision(book, strategy, d, cap, stale, {}, payload)`, `ledger.add_orders(dec, book, orders)`.
- `STAGES` order: `build_panel, reconcile, marks, build_features, kronos_features(stub), predict(stub), s1_plan, nightly_learn(stub), drift(stub), promote(stub), report`. `STUB_STAGES` (L610) = `("kronos_features","predict","s1_plan","nightly_learn","drift","promote")`. `run_eod_pipeline(ctx, d, catchup=False)` resumes per stage via `job_runs` key `eod_pipeline.<stage>`.
- `build_context(settings, clock, *, log_console=True, **overrides) -> JobContext` (L225); `run(settings, clock=None, once=False, *, ctx=None, poll_s=POLL_S, stop_at=None)` calls `build_context(settings, clock)` without hooks (L257).
- `capital_am/capital_pm`: capital = `broker.funds().available_cash + Σ holdings qty*ltp`; kv `capital_ref_inr`, `capital_stale`, `capital_date`; `capital_pm` sets kv `free_cash_frac:{book}`.
- Tests: `tests/integration/test_full_day.py` builds `JobContext(..., hooks=self.hooks, ...)`; `order_hooks(env)` returns `{"s2_orders": s2, "s1_orders": s1}`; L278 asserts the stub stages/jobs record `skipped: not implemented` (update when wiring).

**Ledger** (`paper/ledger.py`): `BOOKS = ("main","auction","auction_strict","events")`; `NewOrder(sid, symbol, side, product, when_kind, trade_date, qty, ref_price, reason="")`; `Position(book, sid, symbol, product, qty, weight, last_price)`; `Ledger(state, clock=None)` with `record_decision(book, strategy, trade_date, capital_ref_inr, capital_stale, risk, payload) -> str` (idempotent), `add_orders(decision_id, book, orders) -> int` (INSERT OR IGNORE), `pending_orders(trade_date, when_kind, books=None)`, `fills_for_date(...)`, `positions(book)`, `cash_frac(book)`, `mark(...)` (first mark of a day wins), `nav_series(book)`.

**Brokers** (`brokers/`): `AngelClient(settings, clock, transport=None)` with `login()`, `funds()`, `holdings()`, `positions()`, `quote_full(tokens)` (50/call, 1 s spacing), `candles(token, interval, start_dt, end_dt)` (`ONE_MINUTE` 30-day span, `FIVE_MINUTE` 100, `ONE_DAY` 2000); `parse_quote(q, now)` (L52) parses only `tradingSymbol, exchFeedTime, ltp, open, high, low, close, tradeVolume, depth.buy[0], depth.sell[0], upperCircuit, lowerCircuit`; login keeps only `jwtToken` (L173-174). `UpstoxClient(settings, clock, transport=None)` with `candles(instrument_key, unit, interval, start, end)` (`/v3/historical-candle/...`, newest-first handled, `oi` column) and `quotes(instrument_keys)` (`/v2/market-quote/quotes`). `QuoteHub(sources, clock).get(symbols, held=None)`; sources `AngelQuotes`, `UpstoxQuotes`, `NseMcpQuotes` (held_only; one call per symbol). `Quote` (`brokers/types.py:34`): `symbol, ts, ltp, open, high, low, prev_close, volume, bid, ask, bid_qty, ask_qty, upper_circuit, lower_circuit, source`. `InstrumentMap(settings, clock, transport=None)`: `refresh()`, `angel_token(symbol)`, `upstox_key(symbol)`.

**Ingest**: `extend_5m(ctx, symbols=None, *, angel=None, instruments=None, on_result=None)` (L122); `backfill_1m(ctx, symbols, start=date(2022,1,3), *, end=None, upstox=None, instruments=None, on_result=None)` (L258; job name `bars_1m:{sym}:{YYYY-MM}`, trade_date = first of month); `load_bars(settings, symbols, start, end, interval="5m", basis="adjusted")`; `cross_check_mcp(settings, pairs, closes_fn=None) -> McpCheck(status, checked, mismatches)` (L434); `catch_up(ctx, steps=None)` order `reference, eq, fo, index, corporate_actions, announcements, board_results, macro_misc, intraday`; `capture_preopen(ctx)`, `capture_spreads(ctx, symbols, quotes_source)`, `capture_quotes(ctx, quotes)` (`ingest_misc.py`). Catalog datasets `spreads` (month), `preopen` (month), `quotes_1m` (day).

---

# PART 4 — Shared conventions for Program G2

## 4.1 Trial families and naming

| Family | Meaning |
|---|---|
| `g1-s1`, `g1-s2`, `g1-ref` | existing G1 trials (unchanged) |
| `g2-s1cost` | S1 portfolio-construction variants on fixed GBM scores (Phase C) |
| `g2-<model>` | model walk-forward trials, e.g. `g2-master`, `g2-hist`, `g2-gats`, `g2-tra`, `g2-dadapt`, `g2-tabm`, `g2-ipca`, `g2-cae`, `g2-fvae`, `g2-ltr`, `g2-kronosft`, `g2-dens`, `g2-gbmv2` |
| `g2-feat` | GBM retrained on the v2 feature set (Phase E incremental tests) |
| `g2-ens` | ensembles of admitted models |
| `g2-s1` | final S1 portfolio trials on G2 ensembles |
| `g2-s2strict` | S2 rebuilt from 09:20 (Phase F) |

`ModelSpec.family` for a model trial is `s1-<key>` where `<key>` is the registry key (so `build_model` dispatches): e.g. `ModelSpec(family="s1-master", target="y_c2o_10", ...)`. The trial family string passed to `register_trial` is `g2-<key>`. (Card D0.1 makes `run_walk_forward` accept `trial_family=` so these differ cleanly.)

## 4.2 The G2 evaluation protocol (EP) — fixed now, before any G2 result

- **Evaluation window:** test folds 2016…2025 (10 folds; 2025 ends 2025-09-30). Folds 2014–2015 are excluded for all G2 comparisons because hyperparameters may only use data before 2016 (anchor fold 2016's inner validation). The GBM baseline is re-scored on the same window from its stored preds.
- **Hyperparameter selection:** only on the inner-validation split of fold `2016` (training data ≤ 2015-12-31 minus embargo). At most 6 configurations per family. The chosen config is fixed for all folds. Selection writes `data_store/models/_selection/<family>.json` and registers no trial.
- **Retraining cadence for neural families:** refit every 2 years (fold 2016 model predicts 2016–2017, 2018 model 2018–2019, …) to fit the GPU budget; GBM-type families every year.
- **Seeds:** neural families 3 seeds (rank-averaged per date); report mean ± std of per-seed IC.
- **Primary target:** `y_c2o_10` (neural families train on `z_y_c2o_10`, prefix `z_`). A family admitted on `y_c2o_10` is then run on `y_c2o_20` as a second trial.
- **Metrics (per model trial), written to `data_store/reports/g2/models/<trial_id>/eval.json`:**
  1. `mean_ic`, `ic_tstat`, `ic_by_year` (daily Spearman vs `r_<target>`, dates with ≥ 50 names).
  2. `corr_with_base`: mean daily Spearman between the model's score and the baseline GBM score.
  3. `orth_ic`: per date, regress the model's rank score on the baseline's rank score (OLS with intercept), take residuals, Spearman of residuals with the label; mean and Newey–West t (lags = horizon h).
  4. `blend_gain`: daily IC of `0.5*rank(model)+0.5*rank(base)` minus base IC; mean and Newey–West t (lags h).
  5. `net_sharpe_s1`: backtest of the C5-winning S1 policy using this model's scores alone (via the G1 runner, capital ₹5 lakh, auto-resume 20) — `sharpe`, `net_2x`, `mean_turnover`.
- **Admission rule (pre-registered):** a family is admitted to the ensemble if (`orth_ic ≥ 0.005` and its t ≥ 2.0) **or** (`blend_gain ≥ 0.003` and its t ≥ 2.0), and its own `mean_ic ≥ 0.04`. Otherwise it is recorded as "no incremental value established".
- **Budgets (hard caps):** per family: selection ≤ 2 GPU-hours; walk-forward ≤ 8 GPU-hours (stop cleanly, record progress, resumable); total program GPU time ≤ 80 hours. CPU families ≤ 6 hours each.

## 4.3 Gate v2 (specified here; implemented in B1)

`config/gate_v2.toml` — identical numbers to v1 plus scope keys:
```toml
version = 2
effective_from = "2026-11-20T00:00:00+05:30"   # trials registered before this are judged by v1
[research]
min_mean_rank_ic        = 0.02
min_ic_tstat            = 3.0
min_net_sharpe          = 1.0
min_net_sharpe_2x_slip  = 0.5
min_excess_tstat        = 2.0
min_frac_positive_years = 0.70
min_dsr                 = 0.95
max_pbo                 = 0.30
max_drawdown            = 0.25
reference_capital_inr   = 500000
dsr_variance_scope      = "family"     # v1 behaviour was "all"
excess_benchmark        = "exposure_matched"   # v1 behaviour was "raw"
[lockbox]
min_net_sharpe          = 0.5
min_net_return          = 0.0
[paper]
min_sessions            = 60
min_net_sharpe          = 0.5
max_drawdown            = 0.10
max_ic_shortfall        = 0.5
[promotion]
min_shadow_sessions     = 20
max_p_value             = 0.10
[auction]
min_strict_capture      = 0.5
```
- Line 7 under v2: `deflated_sharpe(sr, T, skew, kurt, n_trials(state), sharpe_variance_across_trials(state, family=candidate.family))` — N stays the count of **all** trials.
- Line 5 under v2: benchmark `b_t = gross_{t-1} × ew500_ret_t` where `gross_{t-1}` is the candidate book's previous-day gross exposure from its daily frame (first day uses 0). Excess = `net − b`; Newey–West t, lags 5. S2 keeps benchmark 0.
- A candidate is judged by v2 only if **every** trial id in `candidate.trial_ids` has `created_at ≥ effective_from`; otherwise v1, and the report says which.

---

# PART 5 — Task cards

Card format: Goal · Depends on · Provides · Files · Spec · Tests (each with concrete numbers) · Acceptance (one command + expected output) · Budget · Commit.

## PHASE G2-A — Finish, repair, bring data current (CPU; first)

### A0 Copy this plan into the repo
**Goal:** the program lives in the repo. **Files:** `docs/PLAN-G2.md` (this file, unchanged), `docs/PLAN.md` Part H (add one line: `- [ ] Program G2 — see docs/PLAN-G2.md Part 9`), `CLAUDE.md` and `AGENTS.md` (append one line under "Where things are": `- Program G2 (audit + plan to win): docs/PLAN-G2.md`; keep both files identical), `docs/DECISIONS.md` (add heading `## Program G2`). **Acceptance:** `git show --stat HEAD` lists the four files. **Commit:** `G2-A0: add Program G2 plan`.

### A1 Finish T9.6 (auto-resume verdict write-up)
**Goal:** record the auto-resume S1 results honestly. **Depends:** none (the run finished; `data_store/reports/g1/outcome_s1_auto_resume.json` exists).
**Spec:**
1. `.venv\Scripts\python.exe -m stocksense validate g1-s1-y_c2o_10-cce12fb757` → paste the 9-line table.
2. From `data_store/reports/g1/trials/<id>/result.json` of every trial listed in `outcome_s1_auto_resume.json` `plans`, build tables: (a) all auto-resume trials: label, target, k_max, overlay, `by_capital["500000"].sharpe`, `stress_2x.sharpe`, `excess_tstat`, `frac_positive_years`, `max_drawdown`, `mean_turnover`, `mean_gross`; (b) capital grid for the candidate (`by_capital` keys 25000/100000/500000/2500000: sharpe, cagr, cost_drag_annual, n_trades); (c) per-year net return of the candidate from `daily.parquet` (`net` grouped by year, compounded).
3. Append section `## 5.7 S1 with backtest auto-resume (T9.6)` to `docs/RESULTS-G1.md` **by hand** (do NOT call `write_results`; it would overwrite section 4). Content: the three tables, the gate table, the trial count used for DSR, one paragraph verdict: "fails v1 on lines 4, 5, 7; lines 5 and 7 are addressed by gate v2 (PLAN-G2 4.3); line 4 is the turnover problem addressed by Phase C".
4. Update `docs/PLAN.md` Part H line P9 with "T9.6 done (auto-resume), still fails gate v1".
**Acceptance:** `Select-String -Path docs\RESULTS-G1.md -Pattern '5.7 S1 with backtest auto-resume'` prints one match. **Commit:** `G2-A1: T9.6 results and verdict`.

### A2 Evening data catch-up and nightly labels
**Goal:** derived data current; labels rebuilt every evening. **Depends:** A1 (no research process may be reading the panel during the rebuild).
**Spec:**
1. Wait until after 18:30 IST on a session day (check with `.venv\Scripts\python.exe -c "from stocksense.core.clock import SystemClock; print(SystemClock().now())"`).
2. Run in order (each is a single foreground command; each must finish): `stocksense ingest --all --catch-up` (this is safe: the 1-minute step only extends recent months; it does not start the 2022 backfill), then `stocksense build-panel`, then `.venv\Scripts\python.exe -c "from stocksense.core.config import load_settings; from stocksense.features.build import build_features; s=load_settings(); print(build_features(s))"`, then the same for `from stocksense.labels.forward import build_labels; print(build_labels(s))`. **Note:** before running `build_features`, card A7 must have removed `kronos_feats` from `FEATURE_MODULES`, otherwise a v2 feature version with null `kr_*` is created. If A7 is not merged yet, do A7 first.
3. Add `build_labels` to the evening pipeline: in `live/jobs.py`, add stage function `_stage_build_labels(ctx, d) -> str` calling `labels.forward.build_labels(ctx.settings)` and returning `f"rows={n}"`; insert `("build_labels", _stage_build_labels)` into `STAGES` immediately after `build_features`. Update `tests/integration/test_full_day.py` expectations (stage list) accordingly.
4. Wire the CLI commands `build-features` and `build-labels` (currently "not implemented", exit 2) to these functions (print the returned count; exit 0).
5. Verify: `stocksense data-status`; then 10 random (symbol, date) pairs from the new rows checked through `data.ingest_eq.cross_check_mcp` (status must be `ok` and `checked == 10`).
**Tests:** `test_full_day` passes with the new stage; `tests/unit/test_cli.py` asserts `build-features` and `build-labels` are no longer in the not-implemented list (use `CliRunner` with monkeypatched build functions returning 7 → output contains `7`, exit 0).
**Acceptance:** `stocksense data-status` shows `bhavcopy_eq`, `bhavcopy_delivery`, `fo_bhavcopy`, `index_daily`, `panel_daily`, `panel_universe`, `panel_market` with last_date = last completed session and `fresh=true`. **Commit:** `G2-A2: nightly labels stage, build CLI commands, data current`.

### A3 Fix the 1-minute current-month bug
**Files:** `src/stocksense/data/ingest_intraday.py`, `tests/unit/test_ingest_intraday.py` (or the existing intraday test file).
**Spec:** in `backfill_1m` (L258-307): a symbol-month is marked `done` only if `month_end(m0) < today` (calendar month fully in the past, using `ctx.clock.today()`); for the current month record status `running`-then-`skipped` with detail `partial:<last_day>` via `state.job_finish(..., "skipped", f"partial:{last_day}")` so the next run re-fetches from the stored last bar. The skip check at L287 skips only `done`. Also: when a month is partial, request range = (last stored ts for that symbol-month + 1 minute, or month start) → `last_day`.
**Tests:** (1) clock 2026-11-19, symbol X, months Oct and Nov: Oct marked `done`, Nov marked `skipped` with `partial:2026-11-18`; (2) second run on 2026-11-20 re-requests Nov from 2026-11-19 only (assert the fake Upstox client received start date 2026-11-19); (3) Oct not requested again.
**Acceptance:** the new tests pass. **Commit:** `G2-A3: 1-minute backfill re-fetches the current month`.

### A4 Make `extend_5m` robust
**Files:** `data/ingest_intraday.py`, tests.
**Spec:** (1) stop after 10 consecutive symbol failures (`BrokerError`/`httpx.HTTPError`): return results so far plus one `IngestResult(dataset="intraday_bars", status="failed", ...)` with detail `aborted after 10 consecutive failures`; (2) request end = the last **completed** 5-minute bar: `floor(now to 5 min) − 5 min` during market hours, or 15:25 of the last completed session otherwise; never store a bar whose `ts + 5 min > now`; (3) per-call timeout already exists in the client; add a per-run wall budget parameter `max_seconds=1800` (stop cleanly with a `skipped: wall budget` result).
**Tests:** a fake Angel raising for every symbol → exactly 10 calls then abort; a fake clock at 10:07 → requested end 10:00 (last complete bar starts 10:00, ends 10:05 ≤ 10:07: allowed; bar 10:05 excluded); budget exhaustion with a stepping clock.
**Acceptance:** tests pass. **Commit:** `G2-A4: extend_5m failure cap, complete bars only, wall budget`.

### A5 MCP cross-check must check something
**Files:** `data/ingest_eq.py` (`cross_check_mcp`, L434-470), tests.
**Spec:** empty `pairs` → return `McpCheck("empty", 0, [])` without indexing; result `ok` only if `checked == len(pairs) > 0`; `checked < len(pairs)` with no mismatches → status `partial`. CLI `ingest eq --repair --mcp-check` prints status and counts.
**Tests:** empty → `empty`; 3 pairs, closes_fn returns 2 → `partial`; 3 matching → `ok`; one mismatch → `mismatch`.
**Commit:** `G2-A5: MCP cross-check statuses empty and partial`.

### A6 InstrumentMap reliability
**Files:** `brokers/instruments.py`, tests.
**Spec:** (1) atomic cache write (write `*.tmp` then `os.replace`); (2) on Upstox master download/parse failure log `log.warning("instruments_upstox_failed", error=type(e).__name__)` and keep the previous day's cache if present; (3) map Angel rows with `symbol` ending `-EQ`, `-BE`, `-BZ` (strip suffix; prefer EQ when a symbol has several), Upstox rows with `segment == "NSE_EQ"` and `instrument_type` in `("EQ","BE","BZ")`; (4) expose `series_of(symbol) -> str | None`.
**Tests:** a fixture master with `ABC-EQ` and `XYZ-BE` maps both; a corrupt Upstox payload keeps yesterday's map and logs the warning; crash between write and replace leaves the old cache intact (monkeypatch `os.replace` to raise).
**Commit:** `G2-A6: InstrumentMap atomic cache, fallback, BE/BZ series`.

### A7 Housekeeping: Kronos out of default features; docs truth
**Files:** `features/registry.py` (remove `"kronos_feats"` from `FEATURE_MODULES`; keep the module and its tests importing it explicitly), `.env.example` (key names only: the seven `STOCKSENSE_*` names currently in `.env` + `STOCKSENSE_UPSTOX_ANALYTICS_TOKEN`; remove Upstox API key/secret/access token names), `docs/DECISIONS.md` (Program G2 → A7: "Upstox Analytics Token re-issued and verified 2026-11-19: `/v2/market-quote/ltp` and v3 daily candles return success; earlier UDAPI100050 entries refer to the expired token"; ".env now holds 7 keys"), `brokers/upstox.py` docstring of `quotes` (remove the "refuses the analytics token" statement; card H4 re-probes the full-quote endpoint).
**Tests:** `test_feature_truncation` still passes (Kronos test registry used explicitly); a test asserts `"kronos_feats" not in FEATURE_MODULES`.
**Commit:** `G2-A7: drop Kronos from default features; env example and docs`.

### A9 Upstox 1-minute history back to 2022 (overnight, after A3)
**Spec:** one command: `timeout 36000 .venv/Scripts/python.exe -m stocksense ingest intraday --backfill-1m --from 2022-01-03 --top 300 > data_store/logs/g2_A9_bars1m.log 2>&1` (Bash, `run_in_background`). Resumable (symbol-month jobs). Run only when no GPU job and no other network-heavy job runs. If it ends unfinished, the orchestrator relaunches the same command on a later night. Record coverage (symbol-months done / expected) in `docs/DATA.md`.
**Acceptance:** `data-status` shows `bars_1m` first_date 2022-01-03. **Commit (docs only):** `G2-A9: 1-minute history coverage`.

---

## PHASE G2-B — Gate v2 (before any G2 trial is registered)

### B1 Gate v2
**Goal:** the two mis-specified lines computed correctly for new trials; everything else identical. **Depends:** A0.
**Files:** `config/gate_v2.toml` (exact content of Part 4.3), `src/stocksense/core/config.py` (load optional `gate_v2` into `Settings.gate_v2: Gate2 | None`; `Gate2` = the v1 `Gate` model plus `version: int`, `effective_from: datetime`, `research.dsr_variance_scope: Literal["all","family"]`, `research.excess_benchmark: Literal["raw","exposure_matched"]`), `src/stocksense/validation/gate.py`, `src/stocksense/validation/g1_run.py` (`make_candidate` adds a `gross` column to the candidate daily frame from the trial's `daily.parquet`), `tests/unit/test_gate_frozen.py` (second pinned SHA for `gate_v2.toml`), `tests/unit/test_gate_v2.py`, `docs/DECISIONS.md`.
**Spec:**
1. `evaluate_research_gate(candidate, settings, state, version: int | None = None)`: if `version is None`: use 2 when `settings.gate_v2` exists and every `candidate.trial_ids` trial has `created_at >= effective_from` (query `SELECT created_at FROM trials WHERE trial_id IN (...)`); else 1. Store `report.kind = "research-v1"` / `"research-v2"`.
2. v2 line 7: `var = sharpe_variance_across_trials(state, family=candidate.family)`; `N = n_trials(state)` (all). If fewer than 2 finished trials in the family, `var = 1/T` (existing fallback).
3. v2 line 5: requires column `gross` in `candidate.daily` (else raise `ValueError("gate v2 needs gross")`); `b = gross.shift(1).fill_null(0) * benchmark`; `excess_tstat = newey_west_tstat(net - b, lags=5)`. For family prefixes `g1-s2`, `g2-s2strict`: benchmark 0 (unchanged).
4. `validate_candidate(..., gate_version: int | None = None)` and CLI `validate CANDIDATE [--gate-version 1|2]` pass it through. `GateReport.table()` prints the version in its header.
5. Compute and pin the SHA-256 of `config/gate_v2.toml` (CRLF→LF) in `test_gate_frozen.py` as `GATE_V2_SHA256`.
6. DECISIONS: record the decision text of Part 1.2 item 1, the date, and that it was taken after v1 results were seen.
**Tests (hand numbers):** (a) family variance: trials A,B in family `f1` with `sharpe_daily` 0.05 and 0.07, trial C in family `f2` with 0.30 → v2 variance = var([0.05,0.07], ddof=1) = 0.0002; v1 = var([0.05,0.07,0.30], ddof=1) = 0.019233…; (b) exposure-matched: `net=[0.01,0.0,0.02]`, `gross=[0.5,0.5,0.5]`, `benchmark=[0.02,0.02,0.02]` → `b=[0,0.01,0.01]`, excess `[0.01,−0.01,0.01]`; (c) version auto-selection: a trial created before `effective_from` forces v1; (d) gate_v2.toml SHA pinned; v1 SHA unchanged; (e) every v1 test still passes unchanged.
**Acceptance:** `stocksense validate g1-s1-y_c2o_10-cce12fb757 --gate-version 2` prints a 9-line table headed `research-v2` (informational only: this candidate's trials predate v2, so its official verdict stays v1; record the v2 numbers in DECISIONS as "for information"). **Commit:** `G2-B1: gate v2 (family DSR variance, exposure-matched excess)`.

---

## PHASE G2-C — Cost and turnover program (CPU; highest expected gain)

### C1 S1 policy options (all default to current behaviour)
**Files:** `src/stocksense/portfolio/construct.py`, `tests/unit/test_s1_policy_v2.py`.
**Spec — new fields on `S1Config` (frozen dataclass; defaults reproduce current behaviour exactly):**
- `exit_rank: int | None = None` — when set, `rank_decay` fires if `held >= min_hold and r > exit_rank` (replaces `3*k`).
- `min_hold: int = S1_RANK_DECAY_MIN_HELD` (2).
- `extend_rank: int | None = None` — rank limit for extension at horizon; default `k`.
- `free_extension: bool = False` — when True, extension requires only `er > 0` and the row passes the non-cost parts of `_s1_entry_ok` (in universe, not stale, not locked, no split/results within the windows); no round-trip cost hurdle.
- `rebuy_block: int = 0` — a sid sold within the last `rebuy_block` sessions is not eligible for entry. `S1Book` gains `recent_sells: Mapping[int, int] = {}` (sid → sessions since sale); `S1Policy.decide` maintains it from the engine's trades (sells observed via positions disappearing between calls).
- `switch_hurdle: bool = False` — when True and `slots == 0`, consider replacing holdings: sort candidates (eligible, not held) by `er` descending and holdings with `held >= min_hold` by `er_held` ascending; while `er_cand − er_held > round_trip_cost_frac("delivery", target_value) + 2*slip(close)` for the best remaining pair: exit the holding with reason `switch` (SELL open d1) and add the candidate as an entry. At most `max(1, k // 4)` switches per day.
- `S1Config.from_settings(..., **overrides)` accepts these by name.
- `S1Policy.__init__` gains `policy_opts: Mapping[str, Any] | None = None` passed to `S1Config`.
- `plan_s1` reads `cfg.exit_rank`, `cfg.min_hold`, `cfg.extend_rank`, `cfg.free_extension`, `cfg.switch_hurdle`; book's `recent_sells` for `rebuy_block`.
**Tests (hand-built `PanelDay` with 30 names, `k=5`):** (1) defaults: identical `S1Plan` to the current code on a fixed scenario (golden: assert exits/entries equal those produced before the change — compute the golden before editing and paste it in the test); (2) `exit_rank=10`: a holding at rank 8 with held 3 is kept, at rank 12 exits `rank_decay`; (3) `min_hold=5`: rank 25, held 3 → kept; held 5 → exits; (4) `free_extension`: holding at horizon with `er=0.001` and round-trip cost 0.003 → extended; with `er=−0.001` → `horizon`; (5) `rebuy_block=5`: a sid sold 3 sessions ago at rank 1 is skipped; (6) `switch_hurdle`: slots 0, holding er 0.002, candidate er 0.012, cost+slip 0.004 → one `switch` exit + one entry; candidate er 0.005 → none; (7) never more than `k//4` switches.
**Commit:** `G2-C1: S1 exit buffer, min hold, free extension, re-buy block, switch hurdle`.

### C2 Score smoothing and a convex expected-return map
**Files:** `src/stocksense/models/ensemble.py`, `src/stocksense/validation/g1.py`, tests.
**Spec:**
1. `smooth_scores(frame: pl.DataFrame, span: int) -> pl.DataFrame`: frame `date, sid, score`; per sid in date order `ewm_mean(span=span, adjust=False)` over consecutive session rows of that sid (a gap > 5 sessions restarts the EMA); then re-rank per date to [−0.5, 0.5] (average ranks, `(r−1)/(n−1) − 0.5`). Causal: the value on date t uses only rows ≤ t.
2. `expected_return_map_decile(blend, label_frame, target, window=504, n_bins=10, shrink_k=200) -> pl.DataFrame` (`date, sid, score, exp_ret`): for each date t, use the trailing `window` dates s with `end_<target>(s) < t`; bin scores into deciles per date; mean raw label per decile over the window (`m_j`, count `n_j`); shrink `m_j* = n_j/(n_j+shrink_k) * m_j + shrink_k/(n_j+shrink_k) * mean(all)`; enforce non-decreasing across deciles with `sklearn.isotonic.IsotonicRegression(increasing=True)` fit on (decile index, m_j*) weighted by n_j; a name's `exp_ret` = the isotonic value of its decile. No flooring.
3. `G1Config` gains `score_ema: int | None = None` and `er_map: Literal["linear","decile"] = "linear"`; `G1Runner.blended` applies `smooth_scores` when `score_ema` is set (before the ER map) and calls the decile map when `er_map == "decile"`. `_spec_s1` adds `score_ema` / `er_map` keys only when non-default (old hashes unchanged).
**Tests:** smoothing on a 3-date toy (sid A scores −0.5, 0.5, 0.5 with span 3: EMA α=0.5 → −0.5, 0.0, 0.25 before re-rank) and causality (truncating later dates does not change earlier values); decile map: synthetic labels where decile j has mean 0.001·j → exp_ret increases by ~0.001 per decile (within shrink); isotonic fixes a single inversion; no label with `end ≥ t` used (construct a case where using it changes the answer).
**Commit:** `G2-C2: score EMA and convex decile expected-return map`.

### C3 Universe flag, real days-since-result, meta features
**Files:** `portfolio/construct.py` (`EnginePanelAccess`), `validation/g1.py` (`build_world` passes the u500 flag frame and result dates), tests.
**Spec:** `EnginePanelAccess(..., u500: Mapping[date, set[int]] | None = None, universe_flag: bool = False)`: when `universe_flag` is True, `in_universe = sid in u500[t]` (hysteresis flag from `panel_universe.u500`), else the current `rank <= universe_rank`. `days_since_result` = sessions since the latest result date ≤ t for that sid (cap 90; 90 when none), from `result_dates` already in the world. `G1Config.universe_flag: bool = False`, added to `_spec_s1` only when True.
**Tests:** a stock with rank 540 but `u500` True stays in universe with the flag, leaves without it; `days_since_result` = 3 when a result date is 3 sessions before t.
**Commit:** `G2-C3: u500 hysteresis flag and real days-since-result`.

### C4 Cost verification (Angel brokerage and DP)
**Spec:** probe the Angel One charges for equity delivery: (1) fetch `https://www.angelone.in/pricing` and `https://www.angelone.in/knowledge-center/demat-account/angel-one-brokerage-charges` with the web-fetch agent or `httpx` + `lxml` (print only the brokerage/DP sentences); (2) if Angel exposes a brokerage calculator API in SmartAPI (`/rest/secure/angelbroking/brokerage/v1/estimateCharges`, verify existence; read-only), call it for a 1-share RELIANCE delivery BUY and SELL and print the returned charge components (no secrets). Record findings in DECISIONS. If delivery brokerage is ₹0 per the authoritative source, set `config/costs.toml [delivery] brokerage_rate = 0.0`, `brokerage_cap_inr = 0.0`, `brokerage_min_inr = 0.0` and update `verified_on`; otherwise leave the costlier values. Update the cost unit tests' expected numbers only if the config changed (recompute by hand in comments).
**Acceptance:** DECISIONS entry with source URL and date. **Commit:** `G2-C4: delivery cost verification`.

### C5 The `g2-s1cost` grid and verdict
**Depends:** B1, C1, C2, C3, C4. **Files:** `validation/g1.py` (`G1Config.s1_policy: Mapping[str, Any] | None = None` → passed to `S1Policy(policy_opts=...)` and added to `_spec_s1` under key `policy` only when set; `G1Config.family_s1: str = FAMILY_S1` so this grid registers under `g2-s1cost`), `cli_research.py` (command `research s1-cost [--targets y_c2o_10,y_c2o_20]`), `docs/RESULTS-G2.md` (new file).
**Spec:**
- Base: the spec of `g1-s1-y_c2o_10-cce12fb757` read from its `result.json` (`k_max`, `overlay`, models, meta setting) with `auto_resume_sessions=20`, `universe_flag=True`. Targets `y_c2o_10` and `y_c2o_20` (base k_max/overlay same for both). With `h` = target horizon:

| Variant | policy opts | score_ema | er_map | other |
|---|---|---|---|---|
| V1 | `exit_rank=50, min_hold=h//2` | – | linear | – |
| V2 | V1 + `free_extension=True, rebuy_block=5` | – | linear | – |
| V3 | V2 | 3 | linear | – |
| V4 | V2 | 5 | linear | – |
| V5 | V4 + `switch_hurdle=True` | 5 | linear | – |
| V6 | V5 | 5 | decile | – |
| V7 | V6 with `exit_rank=75` | 5 | decile | – |
| V8 | V6 | 5 | decile | `risk.max_fixed_cost_bps=10` (fewer, larger positions; pass via policy opts `max_fixed_cost_bps`) |

  16 trials (8 × 2 targets), each a full G1 trial (capital grid + 2× slippage), registered in family `g2-s1cost` before running.
- Run: `timeout 43200 .venv/Scripts/python.exe -m stocksense research s1-cost > data_store/logs/g2_C5.log 2>&1` (resumable: finished trials are skipped by `load_result`).
- Candidate = best `by_capital["500000"].sharpe`; then `stocksense validate <id>` (auto-selects v2).
- `docs/RESULTS-G2.md` section 1: full table (variant, target, net, 2× slip, excess t v2, positive years, max DD, mean turnover, average hold sessions from `trade_stats`, cost drag), the gate table, and a verdict. Record the winning policy opts in `config/strategies.toml` under `[s1.policy]` (new table; S1Config reads it in live mode) **only after** the verdict is written.
**Acceptance:** RESULTS-G2.md section 1 exists with 16 rows; `research wf-status --family g2-s1cost` shows 16 done. **Commit:** `G2-C5: S1 cost/turnover grid and verdict`.

---

## PHASE G2-G — Paper trading now (CPU; in parallel with Phase C after A2)

### G1 Live score source and the `predict` stage
**Files:** `src/stocksense/live/wiring.py` (new), `src/stocksense/live/jobs.py`, tests `tests/unit/test_live_wiring.py`.
**Spec:**
1. kv keys: `live_model:s1` (model id; initial value `s1-gbm-y_c2o_10-120055de-20250930`), `live_wf_trial:s1` (the walk-forward trial id whose preds calibrate the ER map; initial: the GBM y_c2o_10 trial id found via `surviving_models(state, "s1", "y_c2o_10")[0].trial_id`), `live_model:s2` (`s2-gbm-y_oc-4fd8ff9b-20250930`). A helper `wiring.live_ids(state) -> dict` initialises missing keys with these defaults.
2. `_stage_predict(ctx, d) -> str`: for each strategy in (`s1`): `bag = load_bag(settings, model_id)`; feature names from `models/<id>/feature_manifest.json`; `X` = `load_features(settings, "production", start=d, end=d, columns=names)` for `u500` members of d (join `panel_universe`), float32 in manifest order; `score = bag.predict(X)` (one date); `append_preds(settings, model_id, frame(date=d, sid, score), mode="live", clock=ctx.clock)`; return `f"s1:{n} names"`. Missing features for d → raise `DataNotAvailable`.
3. `LiveScoreSource(settings, state, strategy="s1", target="y_c2o_10")` implementing `ScoreSource.scores(d) -> DayScores | None`: reads live preds of `live_model:s1` for d, ranks to [−0.5, 0.5]; ER: `expected_return_map(blend, labels, target)` computed on the union of walk-forward preds of `live_wf_trial:s1` (ranked per date) and live preds, using `load_labels(settings, "production")`; take the latest slope available for d (dates with labels known before d) → `er = max(0, b)*score`. If `[s1.policy].er_map == "decile"` is configured (after C5), use `expected_return_map_decile` the same way. Cache per evening.
4. Replace the `predict` stub in `STAGES`; remove `"predict"` from `STUB_STAGES`.
**Tests:** on the synthetic market (`tests/fixtures/synth.py`, `planted_ic=0.08`): train a tiny `ModelBag` of 2 `RidgeModel`s via `fit_production(..., allow_beyond_research=True)` into a temp root; run `_stage_predict` for the last date → preds file with `mode == "live"` and n = u500 count; `LiveScoreSource.scores(d)` returns scores in [−0.5, 0.5] and `er ≥ 0`; missing features → `DataNotAvailable`.
**Commit:** `G2-G1: live predict stage and live score source`.

### G2 S1 plan (evening) and entries (15:10) hooks
**Files:** `live/wiring.py`, `live/jobs.py`, `live/daemon.py`, tests.
**Spec:**
1. `live_panel_access(settings, calendar, d) -> EnginePanelAccess`: `build_world(settings, calendar, start=calendar.shift(d, -260), end=d, cut_events=False)` then `world.s1_access()` with `universe_flag=True` (after C3). Cached per date.
2. `ledger_book(ctx, book="main", d) -> S1Book`: holdings from `ctx.ledger.positions(book)` (delivery product only); entry price/date from the BUY fills of each sid (`fills` joined to `orders`); `held = calendar.session_index(d) − session_index(entry_date)`; `vol_at_entry` = `vol_21` on entry date from the panel (0.02 if missing); `cash = capital_ref × cash_frac(book)`; `dd` = 1 − nav/peak from `ledger.nav_series(book)`; `stopped` = kv `book_stop:main == "1"`; `recent_sells` from SELL fills in the last 10 sessions.
3. Hook `s1_plan(ctx, d)`: if `pre_trade_checks` (built from a `RiskContext` assembled by `wiring.risk_context(ctx, d)`) fail → still plan exits, `allow_entries=False`. Call `plan_s1(d, scores, er, book, equity_inr=capital_ref, panel_t, cfg)` with `cfg = S1Config.from_settings(settings, h=10, slippage, k_max=..., overlay=..., **[s1.policy])` (k_max/overlay from the live candidate spec stored in kv `live_policy:s1` as JSON; default the cce12fb757 spec values). Persist `S1Plan` (orders, exits, entries, weights, note) as JSON in kv `s1_plan:{d}`. Immediately record exits: `record_decision("main", "s1-exit", d1, cap, stale, risk_dict, payload)` + `add_orders` with the SELL `NewOrder`s (`when_kind="open"`, `trade_date=d1`, `ref_price=close(d)`). Return `f"exits={n_exits} planned_entries={n_entries}"`.
4. Hook `s1_orders(ctx, d, cap)` (15:10 job `s1_entries`): load kv `s1_plan:{prev_session(d)}`; re-check each planned entry: still in universe, not locked (quote `ltp < upper_circuit − 0.05` from `ctx.hub.get`), re-size `qty = floor(weight × cap / ltp)` capped by available cash (`cap × free_cash_frac:main` kv set by `capital_pm`) minus `BUY_CUSHION_INR`; drop qty 0. Return `[("main", buy_orders(when_kind="close", trade_date=d, ref_price=ltp), payload)]` where payload holds, per order: sid, symbol, score, rank, er, cost_frac, weight and the plan note (decision records, Phase 12 extends this).
5. `live/daemon.py` `build_context`: if `"hooks"` not in overrides, set `hooks = wiring.production_hooks(settings)` which returns `{"s1_plan": ..., "s1_orders": ..., "s2_orders": ...(G3), "corp_actions": wiring.corp_actions_hook, "u500_symbols": wiring.u500_symbols}`; tests keep injecting their own hooks.
6. Remove `"s1_plan"` from `STUB_STAGES`; update the assertion in `tests/integration/test_full_day.py` (L278) to the new stub set.
**Tests:** unit: `ledger_book` reconstructs holdings (2 fills → 2 holdings, held counts by hand); plan exits recorded as SELL open orders for d1 with deterministic ids (second call inserts 0); `s1_orders` resizes with a changed capital and drops a name at the upper circuit. Integration (`test_full_day.py`, new scenario 8): synthetic market + production-like hooks built from a tiny model → day 1 evening produces a plan, day 2 15:10 produces BUY close orders, 15:22 fills them, evening marks; day 2 evening plan exits a name past horizon.
**Commit:** `G2-G2: S1 evening plan and 15:10 entries wired into the daemon`.

### G3 S2 measurement books
**Files:** `live/wiring.py`, tests.
**Spec:** hook `s2_orders(ctx, d, cap)`: newest `preopen` capture of d (09:07) → `auction_features_live(ctx.ingest_ctx(), preopen_frame, panel_tail)` → S2 production bag `live_model:s2` predicts → `plan_s2(d, scores, er (linear map from its WF trial), auction_rows, free_cash=cap × free_cash_frac:auction, equity_inr=cap, S2Config.from_settings(settings, slippage))` → return `[("auction", orders, payload), ("auction_strict", same orders, payload)]`. Payload notes `"measurement only: S2 failed strict capture (RESULTS-G1)"`. Never book `main`.
**Tests:** with a fake preopen frame of 12 names and a fake model → orders for both books, identical sids/sides/qty; no orders when preopen is empty.
**Commit:** `G2-G3: S2 auction and auction_strict measurement books`.

### G4 Evening learning minimum: live IC, weekly refit, report
**Files:** `live/wiring.py`, `live/jobs.py` (replace `nightly_learn`, `drift` stubs), `explain/report.py` (new, minimal), tests.
**Spec:**
1. `drift` stage: for `live_model:s1`, decision dates s with `end_10(s) == d` (labels `production`): Spearman between live preds of s and `r_y_c2o_10` → insert into `live_ic(model_id, decision_date, target, ic, n)`; return counts. Also compute `m60`, `t60` of the last 60 live ICs and set kv `decay_scale:main` per PLAN T11.2 (0.5 if `m60<0 and t60<−1`, 0 if `t60<−2`, 1 otherwise); `plan_s1` multiplies `gross_target` by this scale (new optional arg `gross_scale: float = 1.0`; default unchanged).
2. `nightly_learn` stage: on Fridays (or if the live model is ≥ 6 days old by `models.created_at`): `fit_production(spec_of(live_model:s1), settings, state, as_of=d, allow_beyond_research=True, note="weekly refit")` (spec from `models/<id>/spec.json`); `set_status(new_id, "shadow")`; kv `live_model:s1 = new_id`; old id `retired`. Return the new id. Time it; if > 40 minutes, record in DECISIONS and switch to fortnightly.
3. Report (`report` stage): extend the minimal report: first line one sentence (`"System healthy; main book −0.21% today, +3.4% since start; model not yet validated."` pattern), sections: health (job statuses), capital (₹ read, stale flag — amount shown only in the report file, never logged), trades today per book with sid/symbol/side/qty/price/reason, open positions with weight and days held, NAV of each book (day, 5-session, since start) and ₹ at today's capital, live IC last 20/60, decay scale, tomorrow's plan (exits/entries from kv `s1_plan:{d}`).
4. Remove `"nightly_learn"`, `"drift"` from `STUB_STAGES` (keep `promote`, `kronos_features` stubs).
**Tests:** live IC row inserted from a hand-made preds/labels pair (IC computed by hand for 5 names: perfect order → 1.0); decay scale thresholds; report renders from a fixture ledger with the first-line sentence and every number equal to the JSON.
**Commit:** `G2-G4: live IC, decay scale, weekly refit, daily report`.

### G5 Go live and the five-session proof
**Depends:** G1–G4 merged; suite green; A2 done.
**Spec:**
1. Before install: `stocksense job eod_pipeline --date <last completed session>` runs every stage (stubs excepted) and writes `reports/<date>.md`; `stocksense job s1_entries --date <today>` is only run by the daemon.
2. Fit a fresh production model up to the last completed session: `.venv\Scripts\python.exe -c "..."` calling `fit_production(spec, settings, state, as_of=<last session>, allow_beyond_research=True, note='go-live')`; set kv `live_model:s1` to it.
3. Install: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install-task.ps1`. Verify: `Get-ScheduledTask -TaskName StockSense | Select-Object State`; `stocksense doctor` heartbeat < 60 s.
4. Tell the user (in the report): laptop on, plugged in, lid "Do nothing", sleep never on AC; deposit determines K (₹ table in Part 1).
5. The orchestrator checks once per session for five consecutive sessions: `reports/<d>.md` exists; `job_runs` has no unresolved `failed`; fills reconciled (`fills.reconciled = 1` for that date); `data-status` fresh. Record the five dates in `docs/PLAN.md` Part H P10 and in Part 9 here.
**Commit:** `G2-G5: go live on paper` (docs only).

---

## PHASE G2-H — Use all four sources to full capacity (CPU + network)

### H1 Angel full quote: depth, OI, buy/sell totals
**Files:** `brokers/types.py` (`Quote` gains optional fields with defaults `None`: `oi: float | None`, `total_buy_qty: float | None`, `total_sell_qty: float | None`, `avg_price: float | None`, `depth_bid: tuple[tuple[float, float], ...] = ()`, `depth_ask: tuple[...] = ()`, `week52_high`, `week52_low`), `brokers/angel.py` (`parse_quote`: `opnInterest`, `totBuyQuan`, `totSellQuan`, `avgPrice`, `depth.buy[0..4]`, `depth.sell[0..4]` as (price, quantity), `52WeekHigh`, `52WeekLow`), `data/catalog.py` (`spreads` gains nullable `depth_imbalance_5`, `total_buy_qty`, `total_sell_qty`, `oi`), `data/ingest_misc.py` (`capture_spreads` computes `depth_imbalance_5 = (Σbid_qty − Σask_qty)/(Σbid_qty + Σask_qty)` over 5 levels), tests with the recorded quote key list from `docs/DATA.md:332`.
**Spec also:** store `feedToken` from login in memory (`AngelClient.feed_token` property, never logged) for H3.
**Tests:** a FULL-quote fixture with 5 depth levels parses all fields; imbalance by hand: bids (100, 200, 0, 0, 0) asks (50, 50, 0, 0, 0) → (300−100)/400 = 0.5; old spreads partitions still read (new columns null).
**Commit:** `G2-H1: Angel full depth, OI, totals into spreads`.

### H2 Upstox: full quotes, intraday minute bars, market status, option chain probe
**Files:** `brokers/upstox.py`, `data/ingest_intraday.py`, `live/jobs.py` (preflight market-status check), `cli.py` (`doctor`), tests.
**Spec:**
1. Probe (print status codes and key names only): `GET /v2/market-quote/quotes?instrument_key=NSE_EQ|INE002A01018`, `GET /v2/market-quote/ohlc?instrument_key=...&interval=1d`, `GET /v3/historical-candle/intraday/NSE_EQ|INE002A01018/minutes/1`, `GET /v2/market/status/NSE` (or the current documented market-status path; record which works), `GET /v2/option/chain?instrument_key=NSE_INDEX|Nifty 50&expiry_date=<next Thursday/Tuesday expiry from fo_bhavcopy>`. Record all in DECISIONS.
2. If full quotes work: keep `UpstoxQuotes` as source 2; `doctor` checks `quotes` (one symbol) in addition to candles.
3. `UpstoxClient.intraday_candles(instrument_key, unit="minutes", interval=1) -> pl.DataFrame` (same columns as `candles`); `ingest_intraday.capture_today_1m(ctx, symbols)` after 15:35 for the watch list + u200 → `bars_1m` (`source="upstox_intraday"`); add to the `eod_ingest` intraday step.
4. If market status works: `preflight` reads it; if NSE reports closed on a calendar session day, record `skipped: exchange closed` for trading jobs that day (calendar still the source of truth for history).
5. Option chain: if it works for NIFTY and a stock, add dataset `option_chain_snap` (key `underlying, expiry, strike, option_type, ts`; columns `ltp, oi, iv, delta, volume, bid, ask`) captured at 15:00 for NIFTY, BANKNIFTY and the u200 F&O stocks, forward-only (history starts now). If it fails, record and skip.
**Tests:** MockTransport fixtures for each endpoint shape recorded in the probe; market closed → trading jobs skipped.
**Commit:** `G2-H2: Upstox quotes, intraday 1m, market status, option chain capture`.

### H3 Angel WebSocket 2.0 (probe-first, optional)
**Spec:** probe `wss://smartapisocket.angelone.in/smart-stream` with headers `Authorization: <jwt>`, `x-api-key`, `x-client-code`, `x-feed-token` (values from the client; never printed); subscribe mode 3 (snap quote) for 5 tokens; decode the binary packet per the official SmartAPI WebSocket 2.0 spec (fetch the docs or the official `smartapi-python` `SmartWebSocketV2` source to read the byte layout; do not install it as a runtime dependency unless direct decoding fails twice). If it works: `brokers/angel_stream.py` `AngelStream(client, tokens, on_tick)` used by `market_loop` to keep a 1-second latest-quote cache; `QuoteHub` consults the cache first. If it fails twice: record and drop (polling stays). Budget 4 hours.
**Commit:** `G2-H3: Angel streaming quotes` or `G2-H3: Angel streaming probe (not adopted)`.

### H4 NSE MCP `cm-market` to full use
**Spec:** probe with the `mcp` client (`mcp.client.streamable_http.streamablehttp_client`, `ClientSession.initialize()`, `list_tools()`): print each tool's name and input schema; call `cm_get_equity_stocks` (record row count and fields), `cm_get_live_indices` (group `broad`), `cm_get_index_quote` ("NIFTY 50", "INDIA VIX"), `cm_get_data_status`, `cm_get_call_auction_stocks`. Then:
1. If `cm_get_equity_stocks` returns ≥ 1,500 symbols with LTP in one call: `brokers/quotes.py` `NseMcpBulkQuotes` (held_only=False, `source="nse_mcp_bulk_delayed"`, one call per hub request, cached 60 s) inserted before `NseMcpQuotes`.
2. `cm_get_data_status` → both MCP sources raise `DataNotAvailable` if the last crawl is older than 15 minutes during market hours.
3. Live NIFTY 50 and INDIA VIX at 09:07 and 15:05 → kv `live_index:{d}:{hhmm}` used by the report and as S2 auction-view inputs.
4. `cm_get_call_auction_stocks` daily at 08:50 → kv `call_auction:{d}` (JSON list); `_s1_entry_ok` live path and `plan_s2` exclude these symbols.
**Tests:** recorded tool outputs as fixtures; staleness gate; exclusion.
**Commit:** `G2-H4: cm-market bulk quotes, indices, staleness gate, call-auction exclusion`.

### H5 NSE MCP `nse-bhavcopy` nightly verification
**Spec:** probe `list_tools()` schemas for `get_bulk_quote`, `get_corporate_actions`, `get_index_history`, `nse_lookup_symbol`. New evening stage `mcp_verify` (after `build_panel`): (1) 20 random (symbol, d) closes via `get_bulk_quote` if it accepts a date (else `get_ltp_by_date` ×20) → must match `bhavcopy_eq` within 0.01; (2) `get_corporate_actions` for symbols with an ex-date in the last 30 days → compare ex-dates and type with `corporate_actions`; (3) `get_index_history("NIFTY 50", last 5 days)` vs `index_daily`. Mismatches → report section "Data checks" and a Telegram alert; never block trading (status `done` with detail `mismatch=n`). Unreachable → `skipped: unreachable`.
**Tests:** fake MCP responses: all match → `mismatch=0`; one wrong close → `mismatch=1`; unreachable → skipped.
**Commit:** `G2-H5: nightly NSE MCP verification stage`.

---

## PHASE G2-D — Models beyond boosting (GPU, one family at a time)

### D0 Shared infrastructure (CPU; must land before D1–D11)
**D0.1 Registry and trial family.** `models/base.py`: `build_model` imports every module listed in `MODEL_MODULES = ("gbm","linear","seqnet","master","hist","gats","tra","tabm","ipca","cae","factorvae","dens")` lazily by key (import only the module named by the key; unknown key → `KeyError`). `Model` gains class attrs `needs_keys: bool = False` and `needs_cube: bool = False`. `run_walk_forward(..., trial_family: str | None = None, fold_names: Sequence[str] | None = None, refit_every: int = 1)`: `trial_family` overrides the registered family (default unchanged = `spec.family`); `fold_names` restricts folds (EP uses `2016..2025`); `refit_every=2` reuses the model fitted at fold Y for fold Y+1 (still writing preds per fold). When `factory(spec).needs_keys`: `X_all = add_keys(X_all, sids, dates)` (also in `fit_production` and `live.wiring._stage_predict`).
**D0.2 Feature cube.** `features/cube.py`: `FeatureCube(settings, version, columns, purpose)` loads the feature panel once into `np.float16` array `[n_dates, n_sids_max, F]` with a `sid_index` per date and NaN for missing; `window(date_idx, sids, T) -> np.ndarray[n, T, F]` (zero-filled + mask where history is missing). Memory check: 3,950 dates × 620 sids × 138 × 2 bytes ≈ 0.68 GB. `days` ↔ `date_idx` mapping. Used by D1, D2, D3, D7.
**D0.3 Sequence tensors.** Build the existing 12-channel tensors once: `timeout 7200 .venv/Scripts/python.exe -c "from stocksense.core.config import load_settings; from stocksense.features.seq import build_sequence_tensors; print(build_sequence_tensors(load_settings(), 1))"`. Expected ~2.65M windows × 60 × 12 float16 ≈ 3.8 GB under `features/seq/v1/`.
**D0.4 Torch cross-section base.** `models/torch_common.py`:
- `DateBatches(dates: np.ndarray, min_names=50)` → list of index arrays, one per date, shuffled per epoch with a seeded RNG.
- Losses: `ic_loss` (re-export), `mse_loss`, `listnet_loss(pred, y, tau=1.0)` = cross-entropy between `softmax(y/tau)` and `log_softmax(pred)` per date.
- `class TorchCSModel(Model)`: `target_prefix="z_"`; constructor params `lr, weight_decay, max_epochs, patience, loss, seed_offset, device, work_dir, amp`; abstract `build_net(n_feat, n_mkt) -> nn.Module` and `batch_inputs(idx) -> dict[str, Tensor]`; implements `fit` (early stopping on inner-validation mean daily rank IC computed with `mean_daily_rank_ic`, patience), `refit_full` (`ceil(1.1 × best_epoch)` epochs), `predict` (per-date chunks), checkpoint every epoch to `work_dir/{fit|refit}/epoch_N.pt` with resume, OOM → `torch.cuda.empty_cache()` + split the date batch in halves (loss averaged), gradient clip 1.0, AdamW, cosine schedule, `torch.backends.cudnn.benchmark=False`, deterministic seeds (`torch.manual_seed`, `np.random.default_rng`).
- Wall-time guard: `max_minutes` per fit (stop at the best epoch so far, record `FitInfo.params["stopped"]="wall"`).
**D0.5 Evaluation command.** `validation/g2_eval.py` + CLI `research model-eval --trial <id> [--base <gbm trial id>]` computing every EP metric of Part 4.2 and writing `reports/g2/models/<trial>/eval.json`; the S1 backtest uses the C5 winner policy via `G1Runner` with a single-model score frame (registered as a `g2-s1` trial with spec key `model_eval_of=<trial>`). `research g2-table` prints all families' eval rows.
**D0.6 GPU lock helper.** `core/gpu.py`: `gpu_lock(settings) -> contextmanager` (msvcrt non-blocking lock on `state/gpu.lock`, raises `RuntimeError("gpu busy")`), reused by every GPU entry point; `cli_research` `research wf-run --spec-json <path> --trial-family g2-<key> --folds 2016..2025 --refit-every 2 --max-hours 8` takes the lock and enforces the wall cap (stops after the current fold).
**D0.7 Selection command.** `research select-nn --family <key> --target y_c2o_10 --configs <path.json>`: for each config (≤ 6), fit on fold `2016` inner-train, score inner-validation mean IC, write `models/_selection/g2-<key>.json` with the table and the best config.
**Tests (CPU):** cube windows equal hand-built arrays on the synthetic market; `needs_keys` path appends keys; `refit_every=2` fits 5 times for 10 folds (counter); `TorchCSModel` with a 2-layer MLP on the synthetic market (planted_ic 0.08, 3 epochs, CPU) reaches inner IC > 0.03; listnet loss equals a hand computation for 3 names; checkpoint resume reproduces final weights; eval metrics on toy frames (orth IC of a copy of the base score = 0 ± 1e-9; blend gain of the base with itself = 0).
**Commit:** `G2-D0: model infrastructure for the G2 sweep` (may be split into D0.1–D0.7 commits).

### Family cards D1–D10 (common shape)

For every family card: **Files** `src/stocksense/models/<key>.py`, `tests/unit/test_models_<key>.py`, `configs/g2/<key>.json` (the ≤ 6 selection configs); **Spec** below; **Tests** (CPU, synthetic market): forward shape, finite outputs with NaN inputs, 3-epoch learning on `planted_ic=0.08` reaches inner IC > 0.03, save/load round trip gives identical predictions; one `gpu`-marked smoke test. **Run:** (1) `research select-nn --family <key>` (≤ 2 GPU-h); (2) `research wf-run` with the selected config, target `y_c2o_10`, 3 seeds, folds 2016..2025, refit every 2 (≤ 8 GPU-h); (3) `research model-eval --trial <id>`; (4) if admitted (EP rule), run `y_c2o_20` too; (5) append the family's row and a 3-line verdict to `docs/RESULTS-G2.md` section 2. **Reference code:** where a card names an open-source reference, read it (GitHub raw files via the web-fetch agent), reimplement in our `TorchCSModel` contract, and record the repo URL, commit and licence in `docs/DECISIONS.md`; vendor code only if MIT/Apache/BSD, under `third_party/<name>/` with its LICENSE. **Commit:** `G2-D<n>: <family> model and walk-forward result`.

**D1 `master` — market-guided stock transformer** (Li et al., AAAI 2024; reference `github.com/SJTU-DMTai/MASTER`). Inputs: cube window T=20 of the 99 `xs` features (rank-scaled) per stock + the 39 `mkt` features at t as the market vector. Architecture: market-guided gate `α = F·softmax(W_g m_t / β)` multiplying stock features; linear to `d_model`; positional encoding; intra-stock temporal self-attention (heads 4) per stock; inter-stock attention across all stocks of the date at each time step (heads 2); temporal attention aggregation (query = last step) → linear → score. Loss MSE on `z_` label. Selection grid (6): `d_model ∈ {128, 256}` × `dropout ∈ {0.3, 0.5}` × `lr ∈ {1e-4}` plus `(256, 0.5, 3e-5)`, `(128, 0.3, 3e-5)`; `β = 5`; batch = one date (~500 stocks); AMP float16 allowed.

**D2 `hist` — concept-oriented graph model** (Xu et al. 2021; reference Qlib `pytorch_hist.py`, MIT). Backbone: 2-layer GRU (hidden 64) over the 12-channel 60-day sequence (`needs_keys=True`, `MemmapSeqProvider`). Predefined concepts per date (causal): industry one-hot (from `sectors`), F&O membership, and 20 return-correlation clusters (k-means on the 252-day correlation of residual returns, recomputed each January using data up to the previous Dec 31; stored `features/concepts/part-YYYY.parquet`). Modules: explicit shared info (concept embedding = weighted mean of member hidden states; stock receives cosine-attention-weighted concept info; backcast/forecast split), implicit shared info (top-k=10 cosine neighbours among stocks of the date), individual info; output = sum of three forecasts. Pure torch dense `[n, C]` and `[n, n]` matrices. Grid (4): hidden {64, 128} × dropout {0.0, 0.2}; lr 2e-4.

**D3 `gats` — graph attention over stocks** (Qlib `pytorch_gats.py`). GRU backbone as D2; graph = kNN (k=10) on 120-day return correlation computed at t (causal); one GAT layer (dense masked attention) + residual; linear head. Grid (4): hidden {64, 128} × dropout {0.0, 0.3}.

**D4 `tra` — temporal routing adaptor** (Lin et al. KDD 2021; Qlib `pytorch_tra.py`). Backbone ALSTM (hidden 64) over the 12-channel sequence; K=3 linear predictors; router input = backbone latent + each predictor's recent 20-day loss history per stock (memory updated only with labels whose end date < current date); training assignment by Sinkhorn optimal transport with λ=2.0 (anneal per the reference), inference by router softmax. Grid (4): K {3, 5} × λ {1.0, 2.0}.

**D5 `dadapt` — DoubleAdapt wrapper** (Zhao et al. KDD 2023; `github.com/SJTU-DMTai/DoubleAdapt`). Wrap the best of D1–D4 by `orth_ic` (if none admitted, wrap D2). Incremental tasks every 20 sessions; data adapter (feature + label adaptation layers) and model adapter (MAML-style first-order) meta-trained on 2011…fold train end; inside a test year, each 20-session update uses only labels with `end_10 < update_date − 5 sessions`. Budget 8 GPU-h. Trial family `g2-dadapt`.

**D6 `tabm` — parameter-efficient MLP ensemble** (Gorishniy et al. ICLR 2025; pip package `tabm` — add to `pyproject.toml` if its licence is permissive, else reimplement BatchEnsemble MLP). Inputs: 99 `xs` ranks + 39 `mkt` + `isnan` flags (NaN → 0). k=32 members, BatchEnsemble linear layers; loss: ListNet per date vs MSE (both in the grid). Grid (6): d_block {256, 512} × n_blocks {2, 3} with listnet, plus (512, 3, mse), (256, 2, mse). CPU possible; GPU faster.

**D7 `ipca` and `cae` — characteristic-driven latent factors** (Kelly-Pruitt-Su 2019; Gu-Kelly-Xiu 2021). `ipca`: pip `ipca` (bkelly-lab, check licence) or implement ALS directly in numpy: managed portfolios `x_t = Z_tᵀ r_{t+1}/N_t` with Z = 99 `xs` ranks + constant; K ∈ {3, 5, 8}; predicted score = `Z_t Γ λ̂` where λ̂ = mean factor realisation over the training window; target raw `y_c2o_10` (prefix `""`, a new `target_prefix`). `cae`: beta network MLP(99→32→16→K) on characteristics, factor network linear on managed portfolios, loss MSE of `y_c2o_10`; predicted = `β(Z_t)·λ̂`. Grid: ipca K {3,5,8}; cae K {3,5} × L2 {1e-4, 1e-3}. Two trial families `g2-ipca`, `g2-cae`.

**D8 `factorvae`** (Duan et al. AAAI 2022). Feature extractor GRU (hidden 64) over the cube window T=20; factor encoder (portfolio layer with M=64 portfolios + mapping to K factor posteriors using the realised label — train only); factor decoder `ŷ = α + β·z` (α, β per stock from latent); factor predictor (multi-head global attention, H=4) giving the prior; loss = Gaussian NLL of y + γ·KL(posterior‖prior), γ=1; inference uses the prior mean. Grid (4): K {8, 16} × hidden {64, 128}.

**D9 `ltr` — listwise loss on the best neural family.** Re-run the best admitted D1–D8 family (by `orth_ic`) with `loss="listnet"` (if it was MSE/IC) — one extra trial `g2-ltr`.

**D10 `kronosft` — Kronos fine-tune** (PLAN T7.5e exactly: anchors 2017-12-31, 2020-12-31, 2023-12-31; 20,000 steps or 6 hours each; checkpoints every 500 steps; tokenizer frozen; features tag `small-ft`). Evaluate as T7.5d (pre-registered rule unchanged) and via EP `orth_ic` of a GBM with `kr_*` added. One trial family `g2-kronosft`. Run last among GPU cards (largest budget, weakest prior).

**D11 `dens` — DoubleEnsemble upgraded baseline (CPU)** (Zhang et al. 2020; Qlib `DEnsembleModel`). K=6 LightGBM sub-models with sample reweighting by loss-curve bins (10 bins, decay 0.5) and shuffle-based feature selection (ratios 0.8, 0.7, 0.6, 0.5, 0.4). Not a "beyond boosting" model: it is the stronger baseline every family must also beat in the final table.

### D12 Ensemble of survivors and the G2 S1 portfolio
**Spec:** for target `y_c2o_10` (and `y_c2o_20` if ≥ 1 family admitted there): members = baseline GBM + `dens` + every admitted family. Two trials: equal-rank blend and `blend_online` (registered family `g2-ens`). Then the G2 S1 portfolio trials (family `g2-s1`): C5-winning policy × {equal, online} × capital grid + 2× slippage. Candidate = best net Sharpe at ₹5 lakh. `stocksense validate <candidate>` (v2). If all nine lines pass: `stocksense validate <candidate> --open-lockbox` (once). Write `docs/RESULTS-G2.md` section 4 (ensemble), section 5 (gate + lockbox). If it beats the live candidate (higher net Sharpe and passes at least as many gate lines), set kv `live_model:s1`/`live_policy:s1` to it after a production fit (`fit_production` per member, `allow_beyond_research=True`).
**Commit:** `G2-D12: ensemble, G2 S1 portfolio and gate verdict`.

---

## PHASE G2-E — Probability and stochastic-process track (CPU unless stated)

All new features: registered with `@feature(...)` in new modules, pass the truncation test, bump the feature version (v2) once at the end of E-phase (single `build_features` run). Evaluation: `g2-feat` GBM trial (same GBM spec as the baseline, features v2) vs the baseline on EP metrics (`orth_ic`, `blend_gain`) — and the C5 policy backtest.

### E1 Volatility forecasting (HAR-RV, GARCH)
**Files:** `features/volforecast.py`. `rv_d` = sqrt(Σ squared 5-minute log returns) where 5-minute bars exist (2024-09+), else Garman–Klass daily estimate (full history), scaled to daily units. HAR (Corsi 2009), pooled panel regression fitted each January on data ≤ previous Dec 31: `RV_{t+1} = b0 + b1 RV_t + b2 mean(RV_{t−4..t}) + b3 mean(RV_{t−21..t})` (log RV); features `har_vol_1d`, `har_vol_10d` (sum of iterated forecasts). GARCH(1,1) per stock (add `arch` to pyproject), refit each month-end on the last 1,000 returns, forecasts held fixed within the month: `garch_vol_10d`. Portfolio option `vol_model ∈ {"vol21","har","garch"}` in `S1Config` used for raw weights and the stop (`vol_at_entry`).
**Tests:** HAR coefficients on a simulated HAR process recovered within 0.05; GARCH forecast on a simulated GARCH(1,1) (ω=1e-6, α=0.08, β=0.9) within 15% of the true conditional vol; truncation.

### E2 Bayesian shrinkage of expected returns
`expected_return_map_decile(..., shrink="james_stein")`: per date, decile means `m_j`, sampling variances `s_j²/n_j`; `τ² = max(0, var(m_j) − mean(s_j²/n_j))`; `m_j* = μ + (1 − (s_j²/n_j)/(s_j²/n_j + τ²))(m_j − μ)`; then isotonic. Option `er_map="decile_js"`.
**Tests:** when `τ² = 0` all `m_j* = μ`; when sampling variance → 0, `m_j* = m_j`.

### E3 Fractional Kelly sizing
`S1Config.sizing ∈ {"score_over_vol", "kelly"}` (default the current). Kelly: `raw_i = c × er_i / σ_i²` with `σ_i` = the selected vol model forecast over h days, `c ∈ {0.25, 0.5}`, then the existing caps, gross target and `allocate_weights`; names with `er_i ≤ 0` get 0.
**Tests:** two names with er 0.01/0.02 and equal σ → weights 1:2 before caps.

### E4 Regime detection (filtered, causal)
`features/regime.py`: (1) Gaussian HMM, 3 states, on daily `[ew500_ret, 5-day ew500 return, mkt_vol_21, breadth_200, xs_dispersion, vix_chg_5]` standardised with expanding statistics; parameters re-estimated each January on data ≤ previous Dec 31 (add `hmmlearn` to pyproject, or implement Baum–Welch in numpy); **state probabilities computed with the forward (filtering) recursion only** — never `predict_proba` (it smooths with future data). States ordered by mean return (bear, neutral, bull). Features (kind `mkt`): `regime_p_bear`, `regime_p_bull`. (2) Bayesian online change-point detection (Adams & MacKay 2007) on `ew500_ret` with a constant hazard 1/250 and Normal-Gamma prior: `cp_prob` (probability run length < 5), `run_length_mean`. Overlay option in `S1Config`: `regime_overlay: bool` → gross ×0.5 when `regime_p_bear > 0.6`.
**Tests:** forward filter equals a hand computation on a 2-state 4-observation example; truncation test proves no future dependence; BOCPD detects a mean shift in a simulated series within 10 observations.

### E5 Mean reversion as an Ornstein–Uhlenbeck process
`features/ou.py`: residual return `e = la − β_63·lm − industry mean(la − β·lm)` (β shifted one row); cumulative residual X over 60 rows; AR(1) fit `X_{k+1} = a + b X_k + ε` → `κ = −ln b × 252`, `m = a/(1−b)`, `σ_eq = sqrt(var(ε)/(1−b²))`; features `ou_sscore = (X_t − m)/σ_eq` (Avellaneda & Lee 2010) when `0 < b < 1` and half-life `ln2/κ×252 ≤ 30` sessions, else null; `ou_halflife`. Pair feature: within each industry, Engle–Granger on 252-day log prices for the top-50 liquid members each January (data ≤ previous Dec 31), keep pairs with ADF p < 0.05 and half-life 5–30; `pair_z` = z-score of the spread vs its 60-day mean for the stock's best pair (null if none). Residual reversal raw `resid_rev_5 = −Σ_{5} e` (Blitz et al. 2013).
**Tests:** a simulated OU path with κ=20 (annual) recovers κ within 25%; truncation.

### E6 Extreme-value stops
`portfolio/evt.py`: per stock, losses of vol-scaled daily returns beyond the pooled 95th percentile over the last 750 rows; fit `scipy.stats.genpareto` (pooled across the u500 by liquidity tercile, refit monthly); `ES_99` per stock = vol × pooled ES; option `stop_model ∈ {"fixed","evt"}`: stop distance = `max(stop_min, ES_99 × sqrt(h))`.
**Tests:** GPD fit on simulated GPD(ξ=0.2) recovers ξ within 0.1.

### E7 Event intensity and drift
`features/events_v2.py`: Hawkes-style exponential-kernel intensity of announcements per sid using decision timestamps ≤ cutoff: `ann_intensity_2 = Σ exp(−Δ/2)`, `ann_intensity_10 = Σ exp(−Δ/10)` (Δ in sessions); category counts over 21 sessions for the 8 most frequent categories in `docs/DATA.md`; `result_drift_proxy` = `result_abret` × exp(−sessions_since_result/20).
**Tests:** intensity by hand for events at Δ = 0, 2 → 1 + e^{−1} for kernel 2.

### E8 Forward-only information capture
No backtest possible yet: depth imbalance (H1), option-chain PCR/IV (H2), live indices (H4). Card: a nightly aggregation `features/live_micro.py` writing per (date, sid) `depth_imb_mean`, `spread_bps_mean`, `oi_chg` from `spreads`/`quotes_1m` to a dataset `micro_daily` — to become features once ≥ 250 sessions exist (Program G3).

### E9 Feature v2 build and evaluation
Run one `build_features` (v2 = v1 − nothing + E1/E4/E5/E7 features), `build_labels` unchanged. Register and run `g2-feat` GBM walk-forward on v2 (target `y_c2o_10`, same GBM spec), then `research model-eval --trial <id>` and the C5 policy with options E1 (`vol_model`), E2 (`er_map="decile_js"`), E3 (`sizing="kelly"`, c 0.25), E4 (`regime_overlay`), E6 (`stop_model="evt"`) as 6 registered `g2-s1cost` trials. Results to `docs/RESULTS-G2.md` section 3.
**Commit per card:** `G2-E<n>: <title>`.

---

## PHASE G2-F — S2 rebuilt honestly

### F1 Strict S2 from the 09:20 bar
**Depends:** A9 (1-minute history to 2022).
**Spec:** labels `y_920_1510 = close(15:10 bar)/close(09:15 5-min bar) − 1` from `load_bars(..., interval="5m" or aggregated 1m, basis="raw_ratio")`; features: `gap_920 = close(09:15–09:20 bar)/(prev close × gap_factor) − 1`, `first5_ret`, `first5_vol_share`, `first5_range`, plus the existing auction-day `xs` features of t; universe u200; folds quarterly from 2023-01 to 2025-09 (12 folds; training expanding from 2022-01), embargo 1 session. Model GBM (CPU) and the best neural family if admitted (input-compatible). Portfolio: `plan_s2` with entry `when="now"` at 09:20 and exit 15:10, intraday costs. Family `g2-s2strict`; gate v2 (S2 benchmark 0). Expect low statistical power; whatever the result, record it.
**Commit:** `G2-F1: strict S2 from 09:20`.

---

# PART 6 — Order, parallelism, and schedule

1. **Day 1 (sequential, CPU):** A0 → A7 → A1 → B1 → A3 → A4 → A5 → A6 → A2 (after 18:30).
2. **Then two lanes in parallel** (at most two executors at once):
   - **Lane CPU-1:** C1 → C2 → C3 → C4 → C5 → G1 → G2 → G3 → G4 → G5 (go live) → H1 → H2 → H4 → H5 → H3 (optional) → E1 → E2 → E3 → E4 → E5 → E6 → E7 → E8 → E9 → F1.
   - **Lane GPU:** D0 (CPU parts first, then D0.3 build) → D11 (CPU, can run in Lane CPU-1 gaps) → D1 → D2 → D3 → D4 → D6 → D7 → D8 → D5 → D9 → D10 → D12.
   - G5 (go live) must not wait for Phase D: it uses the best model available, and D12 upgrades it later.
3. **Nights:** A9 (1-minute backfill) only on nights when the GPU lane is idle (network + disk heavy). Never two long jobs on the GPU.
4. **The orchestrator after every card:** re-runs the acceptance command and the full suite on main itself; checks `nvidia-smi` and stray processes; ticks Part 9; relays a 3-line summary to the user.

---

# PART 7 — Failure table (G2-specific)

| Problem | Where | Built-in response |
|---|---|---|
| Agent dies mid-job (API 403, usage limit) | any card | every long job is resumable; the orchestrator resumes the agent with SendMessage; never relaunch a job that is still running (check processes first) |
| GPU out of memory | D-cards | `TorchCSModel` halves the date batch; AMP float16 for D1/D2; reduce `d_model`/hidden one step and record |
| A family exceeds its budget | D-cards | `--max-hours` stops after the current fold; record partial result as "incomplete"; do not extend the budget |
| Gate v2 still fails | C5, D12 | record; the system keeps paper-trading the best candidate labelled unvalidated; the 60-session paper gate decides |
| Live model older than 7 days blocks entries (`model_fresh`) | G4 | weekly refit; if the refit fails, entries stop (exits continue) and the report says why |
| Capital below one minimum position | G2 | K=0; report "capital below one minimum position (₹…)"; no entries |
| Angel login/TOTP failure | live | existing retry; capital falls back to yesterday's value with `capital_stale=1` after 09:03 |
| MCP unreachable | H4, H5 | sources skipped; verification stage `skipped: unreachable`; never blocks trading |
| Upstox token expires (yearly) | live | `doctor` shows it; RUNBOOK rotation steps; quotes fall back to Angel |
| A feature leaks | E-cards | truncation test fails the suite; fix before merge |
| HMM smoothing leak | E4 | only the forward recursion is allowed; a test asserts filtered probabilities at t are unchanged when later observations are appended |
| Laptop sleeps | live | Task Scheduler wake + keep-awake; catch-up on next start; user power settings |

---

# PART 8 — Verification (how anyone can tell Program G2 worked)

1. `.venv\Scripts\python.exe -m pytest tests -q` green; `test_no_orders`, `test_gate_frozen` (v1 and v2 SHAs), `test_feature_truncation` green.
2. `stocksense data-status` fresh to the last completed session; nightly `mcp_verify` reports `mismatch=0`.
3. `docs/RESULTS-G1.md` 5.7 and `docs/RESULTS-G2.md` sections 1–5 exist, every number traceable to `data_store/reports/g1|g2/` files and the trial registry; the total trial count is stated.
4. The S1 cost grid shows the turnover reduction (average hold, mean turnover) and its effect on net and 2×-slippage Sharpe.
5. Every model family has a row in the G2 table (admitted or not), with orthogonalised IC and blend gain.
6. The daemon has run five consecutive sessions unattended with daily reports, reconciled fills, live IC rows, and no unresolved failed jobs.
7. After every GPU card: GPU idle, no stray processes.

---

# PART 9 — Status checklist (the orchestrator ticks this in `docs/PLAN-G2.md`)

- [ ] A0 plan in repo · [ ] A1 T9.6 write-up · [ ] A2 data current + labels stage · [ ] A3 1m month bug · [ ] A4 extend_5m · [ ] A5 MCP check · [ ] A6 InstrumentMap · [ ] A7 housekeeping · [ ] A9 1m history to 2022
- [ ] B1 gate v2
- [ ] C1 policy options · [ ] C2 smoothing + decile ER · [ ] C3 universe flag + days since result · [ ] C4 cost verification · [ ] C5 cost grid verdict
- [ ] G1 predict · [ ] G2 S1 hooks · [ ] G3 S2 books · [ ] G4 live IC/refit/report · [ ] G5 live; five sessions: ____ ____ ____ ____ ____
- [ ] H1 Angel depth/OI · [ ] H2 Upstox · [ ] H3 Angel stream (optional) · [ ] H4 cm-market · [ ] H5 nse-bhavcopy verify
- [ ] D0 infra · [ ] D1 master · [ ] D2 hist · [ ] D3 gats · [ ] D4 tra · [ ] D5 dadapt · [ ] D6 tabm · [ ] D7 ipca/cae · [ ] D8 factorvae · [ ] D9 ltr · [ ] D10 kronos-ft · [ ] D11 dens · [ ] D12 ensemble + verdict
- [ ] E1 vol forecasts · [ ] E2 JS shrinkage · [ ] E3 Kelly · [ ] E4 regimes · [ ] E5 OU · [ ] E6 EVT · [ ] E7 events · [ ] E8 micro capture · [ ] E9 feature v2 eval
- [ ] F1 strict S2

---

# PART 10 — Sources

Cost and turnover: Gârleanu & Pedersen (2013) J. Finance 68(6); Novy-Marx & Velikov (2016) RFS 29(1); DeMiguel, Martín-Utrera, Nogales & Uppal (2020) RFS 33(5); Jensen, Kelly, Malamud & Pedersen, "Machine Learning and the Implementable Efficient Frontier" (RFS, in press); Liang, Zohren & Cucuringu (2026) arXiv 2610.15144; Frazzini, Israel & Moskowitz (2018) SSRN 3229719.
ML for returns: Gu, Kelly & Xiu (2020) RFS 33(5) and (2021) J. Econometrics 222(1); Kelly, Pruitt & Su (2019) JFE; Chen, Pelger & Zhu (2024) Mgmt Sci 70(2); Kelly, Malamud & Zhou (2024) JF 79(1) and Nagel (2025) rebuttal; Avramov, Cheng & Metzker (2023) Mgmt Sci 69(5); Leippold, Wang & Zhou (2022) JFE 145(2); Jensen, Kelly & Pedersen (2023) JF 78(5); Bailey & López de Prado (2014) JPM 40(5); Bailey, Borwein, López de Prado & Zhu (2017) J. Comp. Finance.
Deep cross-sectional models: Microsoft Qlib benchmark README (`github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md`); Li et al. MASTER (AAAI 2024, arXiv 2312.15235); Xu et al. HIST (arXiv 2110.13716); Lin et al. TRA (KDD 2021); Zhao et al. DoubleAdapt (KDD 2023, arXiv 2306.09862); Duan et al. FactorVAE (AAAI 2022); Gorishniy et al. TabM (ICLR 2025, arXiv 2410.24210); Poh, Lim, Zohren & Roberts learning-to-rank (arXiv 2012.07149).
Foundation models: Shi et al. Kronos (arXiv 2508.02739); Rahimikia, Ni & Wang (2025) arXiv 2511.18578; Wan et al. (2026) arXiv 2608.14106.
Anomalies and microstructure: Lou, Polk & Skouras (2019) JFE 134(1); Berkman, Koch, Tuttle & Zhang (2012) JFQA 47(4); Blitz, Huij, Lansdorp & Verbeek (2013) J. Financial Markets 16(3); Avellaneda & Lee (2010) Quant. Finance 10(7); Corsi (2009) J. Financial Econometrics 7(2); Adams & MacKay (2007) arXiv 0710.3742.
India: Joshipura & Joshipura (2020) IMFI; Acharya, Anshuman & Kumar (NYU/IIMB working paper on FII flows); NSE working paper 81 (F&O open interest); Lalwani & Meshram (2022) Applied Economics 54(16).
