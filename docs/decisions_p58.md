# Decisions, track p58 (T4.3g-k, T4.4) - to be folded into DECISIONS.md

## T4.3g market.py
- Macro feature names are lower case (`usdinr_chg_1`, `spx_chg_5`, ...). `US10Y` change is a difference in percentage points, all other series use ln ratio. Value for session t = newest observation with `available_ts <= 20:00 IST on t` (as-of join); a 1/5-change compares the as-known value of t with that of the session 1/5 rows earlier.
- `vix_pctile_252` / `nifty_pe_pctile_756` = (#other window values below + 0.5 #equal)/(w-1), null until w values exist.
- `xs_disp_ratio` divides by the 60-session mean INCLUDING today. `fii_idx_fut_chg_5` is the name for the 5-session change of `fii_idx_fut_net`.
- `fii_net_5`/`dii_net_5` need 5 non-null daily values; categories matched by substring FII / DII.
- Industry features (`ind_mom_21`, `ind_mom_126`, `rel_ind_mom_21`, `ind_breadth_50`) compute `mom_*` locally (sum of ln(1+ret) over 21/126 rows) so they do not depend on T4.3a; members are the date's u500 rows; null if the industry has fewer than 5 u500 members that day or is `UNKNOWN`.
- `mkt_ret_1` removed from `seed.py`.
- Shared weekday arithmetic lives in `features/_calutil.py`; the hand-made test environment is `tests/fixtures/feat58.py`.

## T4.3h calendar_feats.py
- Forward-looking counts (`days_to_month_end`, last-3 `dom_bucket`, `sessions_to_next_holiday`) cannot use the stored session list under truncation, so they use a calendar = past sessions + projected Monday-Friday dates (75 days) minus holidays. Holidays = `holidays` dataset (read WITHOUT the as_of cut; NSE publishes the list in advance) plus every weekday inside the stored history that was not a session. Weekend special sessions are not projected.
- `sessions_to_next_holiday` = sessions strictly between the date and the next holiday (0 = holiday tomorrow), capped 10, also 10 when none is known.
- `is_expiry_week`: per month the latest `expiry_date` among IDF/STF rows visible by as_of, else last Thursday moved back past holidays; week = ISO Monday-Sunday.
- `is_budget_week`: `BUDGET_DATES` table lists years whose budget (incl. interim) was not 1 Feb, written from memory and marked as best knowledge; other years use 1 Feb.

## T4.3i derivs.py
- Eleven features (`fut_basis`, `fut_oi_chg_5`, `oi_price_quad`, `pcr_oi`, `pcr_oi_z_20`, `atm_iv`, `iv_minus_rv`, `iv_chg_5`, `skew_put_call`, `max_pain_dist`, `days_to_expiry`) are separate specs sharing one computation memoised on the FeatureContext instance (`ctx._derivs_cache`), so a build computes implied vols once.
- Only STF and STO rows (stock futures/options) are used; index contracts are not mapped to sids. `near` futures expiry and `near` option expiry are chosen independently (nearest expiry with >= 3 calendar days left). Spot for Black-Scholes and basis is the panel's raw `close`.
- `atm_iv` = mean of the call and put IV at the strike nearest the close; if only one of them solves, that one is used. `skew_put_call` needs both sides (put at the strike nearest 0.95 S, call nearest 1.05 S). IVs are solved for at most four options per underlying-day (ATM call/put, skew put, skew call).
- `fut_oi_chg_5` divides by the 20-session mean INCLUDING today; `pcr_oi_z_20` uses the previous 20 values (shifted by one) with ddof=1 and is null if their std is 0. `iv_chg_5` is a difference in volatility points (not log). `iv_minus_rv` uses annualised vol_21 computed locally. `oi_price_quad` uses the 5-session sum of ln(1+ret) and the change in total futures OI over all expiries; zero changes give null. `days_to_expiry` is Float64.
- `max_pain_dist`: payoff minimised over candidate strikes of the near expiry chain, lowest strike wins ties.
- Truncation fixture: 3 stocks (S00-S02) get synthetic futures and a 7-strike chain from session 200; test runtime of the whole truncation file is ~38 s.
