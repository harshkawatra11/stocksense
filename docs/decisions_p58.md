# Decisions, track p58 (T4.3g-k, T4.4) - to be folded into DECISIONS.md

## T4.3g market.py
- Macro feature names are lower case (`usdinr_chg_1`, `spx_chg_5`, ...). `US10Y` change is a difference in percentage points, all other series use ln ratio. Value for session t = newest observation with `available_ts <= 20:00 IST on t` (as-of join); a 1/5-change compares the as-known value of t with that of the session 1/5 rows earlier.
- `vix_pctile_252` / `nifty_pe_pctile_756` = (#other window values below + 0.5 #equal)/(w-1), null until w values exist.
- `xs_disp_ratio` divides by the 60-session mean INCLUDING today. `fii_idx_fut_chg_5` is the name for the 5-session change of `fii_idx_fut_net`.
- `fii_net_5`/`dii_net_5` need 5 non-null daily values; categories matched by substring FII / DII.
- Industry features (`ind_mom_21`, `ind_mom_126`, `rel_ind_mom_21`, `ind_breadth_50`) compute `mom_*` locally (sum of ln(1+ret) over 21/126 rows) so they do not depend on T4.3a; members are the date's u500 rows; null if the industry has fewer than 5 u500 members that day or is `UNKNOWN`.
- `mkt_ret_1` removed from `seed.py`.
- Shared weekday arithmetic lives in `features/_calutil.py`; the hand-made test environment is `tests/fixtures/feat58.py`.
