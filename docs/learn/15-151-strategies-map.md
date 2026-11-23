# 151 Trading Strategies: what StockSense implements

Source: Kakushadze, Z. and Serur, J.A. (2018), *151 Trading Strategies*, Palgrave Macmillan; SSRN 3247865; arXiv 1912.04492. The paper is a survey: it gives formulas but no backtests.
This table has one row per strategy section in the paper's contents (174 headings) plus the two Appendix A strategies. Chapter 20 (infrastructure) has no strategy heading and is not tradable on our data.
Status words: implemented, adapted (with the change stated), informational (shown, not gated), await_data, deferred, infeasible (with the reason). Book page numbers are one less than the PDF page numbers.

## What it is

A map from every strategy section of the book to a StockSense strategy key, or to the reason it cannot be built. The keys live in `config/strategy_library.toml` and `strategies/k151.py`.

## How it is computed

Every `implemented` or `adapted` row is a registered strategy with the formula in the third column, evaluated by the pre-registered walk-forward in `validation/strategy_eval.py` and judged by gate v2. Paper equation numbers are quoted so each rule can be checked against the book.

## What it says about flow and positioning

Only a few sections touch flow directly: 3.5 (call versus put implied-volatility changes), 6.2 (basis), 9.1 and 9.2 (roll yield and positioning, mapped to stock-futures and participant open interest) and 10.3.1 (volume and open-interest filters). The rest are price-based.

## What NSE data allows and how StockSense computes it

Stocks and sector indices have long history (2010 or 2012 onward). Options and futures strategies use the F&O bhavcopy from 2024-07, about 14 monthly cycles, so those trials have low statistical power; RESULTS-M says so beside every verdict.

## Evidence

The authors state that they give no backtests or profitability claims. All evidence here comes from our own gated runs. The paper's reference list (about 1,740 entries) is the place to look for published evidence per strategy. India-specific references in it: Birari and Rode (2014) on Donchian channels for the Nifty, Gunasekarage and Power (2001) on moving-average rules in South Asian markets, and Vipul (2009) on box-spread arbitrage in Nifty options.

## How traders misuse it

Treating a formula in a survey as a proven edge; running dozens of variants and reporting the best (the gate's deflated Sharpe uses the full trial count for this reason); ignoring costs and the fact that short legs are not available in delivery equity.

## Where it appears in the dashboard

The Strategies page lists each key with its gate verdict, and Daily Picks uses only gate-passing keys.

## Sources

- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3247865
- https://arxiv.org/abs/1912.04492

### Full map of sections (one row per section)

| § | name | one-line formula (eq. numbers) | asset class | status and reason | StockSense key |
|---|---|---|---|---|---|
| 2.2 | Covered call | option payoff by legs, eq. 1 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_covered_call |
| 2.3 | Covered put | option payoff by legs, eq. 5 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_covered_put |
| 2.4 | Protective put | option payoff by legs, eq. 9 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_protective_put |
| 2.5 | Protective call | option payoff by legs, eq. 13 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_protective_call |
| 2.6 | Bull call spread | option payoff by legs, eq. 17 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_bull_call_spread |
| 2.7 | Bull put spread | option payoff by legs, eq. 21 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_bull_put_spread |
| 2.8 | Bear call spread | option payoff by legs, eq. 25 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_bear_call_spread |
| 2.9 | Bear put spread | option payoff by legs, eq. 29 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_bear_put_spread |
| 2.10 | Long synthetic forward | option payoff by legs, eq. 33 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_long_synth_fwd |
| 2.11 | Short synthetic forward | option payoff by legs, eq. 37 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_short_synth_fwd |
| 2.12 | Long combo | option payoff by legs, eq. 41 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_long_combo |
| 2.13 | Short combo | option payoff by legs, eq. 47 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_short_combo |
| 2.14 | Bull call ladder | option payoff by legs, eq. 53 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_bull_call_ladder |
| 2.15 | Bull put ladder | option payoff by legs, eq. 58 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_bull_put_ladder |
| 2.16 | Bear call ladder | option payoff by legs, eq. 63 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_bear_call_ladder |
| 2.17 | Bear put ladder | option payoff by legs, eq. 68 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_bear_put_ladder |
| 2.18 | Calendar call spread | option payoff by legs, eq. 73-74 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_calendar_call |
| 2.19 | Calendar put spread | option payoff by legs, eq. 75-76 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_calendar_put |
| 2.20 | Diagonal call spread | option payoff by legs, eq. 77-78 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_diagonal_call |
| 2.21 | Diagonal put spread | option payoff by legs, eq. 79-80 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_diagonal_put |
| 2.22 | Long straddle | option payoff by legs, eq. 81 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_long_straddle |
| 2.23 | Long strangle | option payoff by legs, eq. 86 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_long_strangle |
| 2.24 | Long guts | option payoff by legs, eq. 91 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_long_guts |
| 2.25 | Short straddle | option payoff by legs, eq. 96 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_short_straddle |
| 2.26 | Short strangle | option payoff by legs, eq. 101 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_short_strangle |
| 2.27 | Short guts | option payoff by legs, eq. 106 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_short_guts |
| 2.28 | Long call synthetic straddle | option payoff by legs, eq. 111 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_long_call_synth_straddle |
| 2.29 | Long put synthetic straddle | option payoff by legs, eq. 116 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_long_put_synth_straddle |
| 2.30 | Short call synthetic straddle | option payoff by legs, eq. 121 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_short_call_synth_straddle |
| 2.31 | Short put synthetic straddle | option payoff by legs, eq. 126 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_short_put_synth_straddle |
| 2.32 | Covered short straddle | option payoff by legs, eq. 131 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_covered_short_straddle |
| 2.33 | Covered short strangle | option payoff by legs, eq. 135 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_covered_short_strangle |
| 2.34 | Strap | option payoff by legs, eq. 138 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_strap |
| 2.35 | Strip | option payoff by legs, eq. 143 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_strip |
| 2.36 | Call ratio backspread | option payoff by legs, eq. 148 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_call_ratio_backspread |
| 2.37 | Put ratio backspread | option payoff by legs, eq. 153 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_put_ratio_backspread |
| 2.38 | Ratio call spread | option payoff by legs, eq. 158 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_ratio_call_spread |
| 2.39 | Ratio put spread | option payoff by legs, eq. 163 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_ratio_put_spread |
| 2.40 | Long call butterfly | option payoff by legs, eq. 168 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_long_call_butterfly |
| 2.40.1 | Modified call butterfly | option payoff by legs, eq. 173 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_mod_call_butterfly |
| 2.41 | Long put butterfly | option payoff by legs, eq. 177 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_long_put_butterfly |
| 2.41.1 | Modified put butterfly | option payoff by legs, eq. 182 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_mod_put_butterfly |
| 2.42 | Short call butterfly | option payoff by legs, eq. 186 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_short_call_butterfly |
| 2.43 | Short put butterfly | option payoff by legs, eq. 191 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_short_put_butterfly |
| 2.44 | "Long" iron butterfly | option payoff by legs, eq. 196 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_long_iron_butterfly |
| 2.45 | "Short" iron butterfly | option payoff by legs, eq. 201 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_short_iron_butterfly |
| 2.46 | Long call condor | option payoff by legs, eq. 206 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_long_call_condor |
| 2.47 | Long put condor | option payoff by legs, eq. 211 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_long_put_condor |
| 2.48 | Short call condor | option payoff by legs, eq. 216 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_short_call_condor |
| 2.49 | Short put condor | option payoff by legs, eq. 221 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_short_put_condor |
| 2.50 | Long iron condor | option payoff by legs, eq. 226 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_long_iron_condor |
| 2.51 | Short iron condor | option payoff by legs, eq. 231 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_short_iron_condor |
| 2.52 | Long box | f_T = K1 - K2 - D, eq. 236 | Options | informational: box-spread arbitrage check on NIFTY options, no trial (see Vipul 2009) | none |
| 2.53 | Collar | option payoff by legs, eq. 238 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_collar |
| 2.54 | Bullish short seagull spread | option payoff by legs, eq. 242 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_bull_short_seagull |
| 2.55 | Bearish long seagull spread | option payoff by legs, eq. 248 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_bear_long_seagull |
| 2.56 | Bearish short seagull spread | option payoff by legs, eq. 254 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_bear_short_seagull |
| 2.57 | Bullish long seagull spread | option payoff by legs, eq. 260 | Options (NIFTY, BANKNIFTY) | implemented | k151_opt_<und>_bull_long_seagull |
| 3.1 | Price-momentum | R_cum = P(t-21)/P(t-252) - 1; risk-adjusted mean/sigma, eqs. 267-270 | Stocks | implemented | k151_mom_cum, k151_mom_riskadj |
| 3.2 | Earnings-momentum | SUE = (E - E')/sigma, eq. 274 | Stocks | infeasible: EPS values are not parsed (results_index has no numbers); possible after T8.6 | none |
| 3.3 | Value | B/P ratio | Stocks | infeasible: no book value in our data | none |
| 3.4 | Low-volatility anomaly | -sigma(r, 252) or 126 | Stocks | implemented | k151_low_vol, k151_low_vol_126 |
| 3.5 | Implied volatility | delta IV call minus delta IV put over 1 month | Stocks | adapted: F&O stocks only, history from 2024-07 | k151_implied_vol |
| 3.6 | Multifactor portfolio | mean demeaned rank over factors, eqs. 276-277 | Stocks | adapted: value unavailable, residual momentum replaces it | k151_multifactor |
| 3.7 | Residual momentum | residuals of FF3 regression, eqs. 278-282 | Stocks | adapted: no Fama-French factors for India (market and size proxies) | k151_resid_mom |
| 3.8 | Pairs trading | -(R_A - mean(R_A, R_B)), eqs. 285-289 | Stocks | adapted: partner = most correlated stock in the same industry | k151_pairs |
| 3.9 | Mean-reversion - single cluster | -(R_i - mean R), eqs. 292-297 | Stocks | implemented | k151_mr_single |
| 3.9.1 | Mean-reversion - multiple clusters | -(R_i - cluster mean), eq. 309 | Stocks | implemented | k151_mr_multi |
| 3.10 | Mean-reversion - weighted regression | eps = R - Omega Q^-1 Omega^T Z R, eqs. 314-317 | Stocks | implemented | k151_mr_weighted |
| 3.11 | Single moving average | long if P > MA(T), eq. 321 | Stocks | implemented | k151_ma1_sma, k151_ma1_ema |
| 3.12 | Two moving averages | long if MA(10) > MA(30), eq. 322; stop variant eq. 323 | Stocks | implemented | k151_ma2, k151_ma2_stop |
| 3.13 | Three moving averages | long if MA(3) > MA(10) > MA(21), eq. 324 | Stocks | implemented | k151_ma3 |
| 3.14 | Support and resistance | C = (H+L+C)/3, R = 2C - L, S = 2C - H, eqs. 325-328 | Stocks | implemented | k151_pivots |
| 3.15 | Channel | Donchian B_up and B_down, eqs. 329-331 | Stocks | implemented: floor-bounce base rule plus the paper's breakout alternative | k151_channel_mr, k151_channel_bo |
| 3.16 | Event-driven - M&A | long target, short acquirer | Stocks | infeasible: no M&A deal-terms data | none |
| 3.17 | Machine learning - single-stock KNN | KNN on moving-average features, eqs. 332-341 | Stocks | adapted: scale-free features, purged training sample | k151_knn, k151_knn_xs |
| 3.18 | Statistical arbitrage - optimization | w = (1/lambda)[C^-1 E - ...], eq. 358 | Stocks | adapted: diagonal and PCA covariance variants | k151_statarb_opt, k151_statarb_opt_pca |
| 3.19 | Market-making | buy at bid, sell at ask, eq. 359 | Stocks | infeasible: needs resting orders and queue position; the paper-only rule forbids order code | none |
| 3.20 | Alpha combos | alpha-combo regression residuals, steps 1-11 | Stocks | deferred: run over the finished m-xs trials after M8.1, with the 8.4 min-variance combo | none |
| 4.1 | Sector momentum rotation | R_cum = P(t)/P(t+T) - 1 over sector indices, eq. 361 | Sector indices | implemented | k151_sector_mom, k151_sector_mom_6m |
| 4.1.1 | Sector momentum rotation with MA filter | hold only if P > MA(T'), eq. 362 | Sector indices | implemented | k151_sector_mom_ma |
| 4.1.2 | Dual-momentum sector rotation | else hold a diversifier when the index is below its MA, eq. 363 | Sector indices | adapted: diversifier = gold in INR from macro_daily | k151_sector_dual_mom |
| 4.2 | Alpha rotation | alpha of a regression on market factors, eq. 364 | Sector indices | adapted: market-only regression | k151_sector_alpha |
| 4.3 | R-squared | selectivity 1 - R^2 and alpha, eqs. 365-369 | Sector indices | adapted: median split on R^2 then top alpha | k151_sector_r2 |
| 4.4 | Mean-reversion | IBS = (C - L)/(H - L), eq. 370 | Sector indices | implemented | k151_sector_ibs |
| 4.5 | Leveraged ETFs (LETFs) | short LETF and inverse LETF | ETFs | infeasible: no leveraged or inverse ETFs on NSE | none |
| 4.6 | Multi-asset trend following | w_i = gamma R_cum/sigma^2 among positive-trend assets, eqs. 371-373 | Multi-asset | adapted: NIFTY 50, midcap, G-Sec index, gold in INR | k151_multi_asset_trend |
| 5.2 | Bullets | all bonds one maturity | Bonds | informational: shown on G-Sec maturity-bucket indices only if those exist in index_daily; no trial | none |
| 5.3 | Barbells | two maturities, eqs. 390-394 | Bonds | informational: shown on G-Sec maturity-bucket indices only if those exist in index_daily; no trial | none |
| 5.4 | Ladders | equal capital across maturities, eq. 395 | Bonds | informational: shown on G-Sec maturity-bucket indices only if those exist in index_daily; no trial | none |
| 5.5 | Bond immunization | bond-level duration, rating or swap formulas (chapter 5) | Bonds | infeasible: needs bond prices, credit ratings, CDS or swap curves | none |
| 5.6 | Dollar-duration-neutral butterfly | bond-level duration, rating or swap formulas (chapter 5) | Bonds | infeasible: needs bond prices, credit ratings, CDS or swap curves | none |
| 5.7 | Fifty-fifty butterfly | bond-level duration, rating or swap formulas (chapter 5) | Bonds | infeasible: needs bond prices, credit ratings, CDS or swap curves | none |
| 5.8 | Regression-weighted butterfly | bond-level duration, rating or swap formulas (chapter 5) | Bonds | infeasible: needs bond prices, credit ratings, CDS or swap curves | none |
| 5.8.1 | Maturity-weighted butterfly | bond-level duration, rating or swap formulas (chapter 5) | Bonds | infeasible: needs bond prices, credit ratings, CDS or swap curves | none |
| 5.9 | Low-risk factor | bond-level duration, rating or swap formulas (chapter 5) | Bonds | infeasible: needs bond prices, credit ratings, CDS or swap curves | none |
| 5.10 | Value factor | bond-level duration, rating or swap formulas (chapter 5) | Bonds | infeasible: needs bond prices, credit ratings, CDS or swap curves | none |
| 5.11 | Carry factor | bond-level duration, rating or swap formulas (chapter 5) | Bonds | infeasible: needs bond prices, credit ratings, CDS or swap curves | none |
| 5.12 | Rolling down the yield curve | bond-level duration, rating or swap formulas (chapter 5) | Bonds | infeasible: needs bond prices, credit ratings, CDS or swap curves | none |
| 5.13 | Yield curve spread (flatteners & steepeners) | short or long yield-curve spread, eq. 416 | Bonds | informational: shown on G-Sec maturity-bucket indices only if those exist in index_daily; no trial | none |
| 5.14 | CDS basis arbitrage | bond-level duration, rating or swap formulas (chapter 5) | Bonds | infeasible: needs bond prices, credit ratings, CDS or swap curves | none |
| 5.15 | Swap-spread arbitrage | bond-level duration, rating or swap formulas (chapter 5) | Bonds | infeasible: needs bond prices, credit ratings, CDS or swap curves | none |
| 6.2 | Cash-and-carry arbitrage | F* = (S - D) exp(r (T - t)), basis B, eqs. 421-422 | Index | informational: basis shown on /flow, no trial | k151_cash_carry |
| 6.3 | Dispersion trading in equity indexes | long constituent straddles, short index straddle, eqs. 423-424 | Index options | adapted: monthly NIFTY straddles, about 14 cycles, equal weights | k151_dispersion |
| 6.3.1 | Dispersion trading - subset portfolio | statistical risk model psi, eqs. 425-427 | Index options | adapted: same trial, constituents chosen by lowest specific variance | k151_dispersion |
| 6.4 | Intraday arbitrage between index ETFs | buy ETF2 short ETF1 if bid1 >= ask2 x kappa, eq. 428 | Index | infeasible: needs tick bid/ask of two ETFs on one index | none |
| 6.5 | Index volatility targeting with risk-free asset | w = sigma*/sigma | Index | implemented: sigma from INDIA VIX | k151_vol_target |
| 7.2 | VIX futures basis trading | D = (UX1 - VIX)/T, eqs. 429-431 | Volatility | infeasible: no VIX futures on NSE | none |
| 7.3 | Volatility carry with two ETNs | short VXX, long VXZ | Volatility | infeasible: no VIX ETNs on NSE | none |
| 7.3.1 | Hedging short VXX with VIX futures | hedge VXX with a futures basket, eq. 432 | Volatility | infeasible: no VIX futures on NSE | none |
| 7.4 | Volatility risk premium | sell straddle when VIX minus realised vol > 0 | Index options | adapted: realised vol over the previous 21 sessions | k151_vrp_straddle |
| 7.4.1 | Volatility risk premium with Gamma hedging | VRP straddle with gamma hedging | Index options | implemented | k151_vrp_straddle_dh |
| 7.5 | Volatility skew - long risk reversal | long risk reversal captures put skew | Index options | implemented through the 2.12 combo | k151_opt_<und>_long_combo |
| 7.6 | Volatility trading with variance swaps | P = N (v - K), eqs. 434-436 | Volatility | infeasible: no variance swaps | none |
| 8.1 | Moving averages with HP filter | HP filter then MA(T1) vs MA(T2), eqs. 437-439 | Stocks (from FX) | adapted: one-sided HP on monthly closes of each stock | k151_hp_ma |
| 8.2 | Carry trade | forward discount, carry and triangular arbitrage (FX) | FX | infeasible: no multi-currency forward or interest-rate data | none |
| 8.2.1 | High-minus-low carry | forward discount, carry and triangular arbitrage (FX) | FX | infeasible: no multi-currency forward or interest-rate data | none |
| 8.3 | Dollar carry trade | forward discount, carry and triangular arbitrage (FX) | FX | infeasible: no multi-currency forward or interest-rate data | none |
| 8.4 | Momentum & carry combo | w1 = (s2^2 - s1 s2 rho)/(...), eqs. 445-452 | FX | deferred: min-variance combo of finished m-xs trials, registered after M8.1 | none |
| 8.5 | FX triangular arbitrage | forward discount, carry and triangular arbitrage (FX) | FX | infeasible: no multi-currency forward or interest-rate data | none |
| 9.1 | Roll yields | phi = P1/P2, eq. 454 | Stock futures | adapted: stock-futures near/next settle, 2024-07 onward | k151_roll_yield |
| 9.2 | Trading based on hedging pressure | hedging pressure = long/(long+short) | Index futures | adapted: NSE participant OI (FII as speculators, CLIENT as hedgers) | k151_hedging_pressure |
| 9.3 | Portfolio diversification with commodities | passive or active commodity allocation | Allocation | informational: 80/20 NIFTY and gold shown, no trial | k151_gold_diversify |
| 9.4 | Value | v = P(5y ago)/P(now), eq. 455 | Stocks | adapted: long-term reversal on stocks | k151_lt_value |
| 9.5 | Skewness premium | skewness, eqs. 456-458 | Stocks | adapted: stocks instead of commodity futures | k151_skew |
| 9.6 | Trading with pricing models | OU fit, eqs. 459-462 | Stocks | adapted: AR(1) fit of ln P per stock | k151_ou |
| 10.1 | Hedging risk with futures | hedge ratios, eqs. 463-468 | Futures | infeasible: hedging recipes without an alpha rule | none |
| 10.1.1 | Cross-hedging | hedge ratios, eqs. 463-468 | Futures | infeasible: hedging recipes without an alpha rule | none |
| 10.1.2 | Interest rate risk hedging | hedge ratios, eqs. 463-468 | Futures | infeasible: hedging recipes without an alpha rule | none |
| 10.2 | Calendar spread | buy near, sell deferred futures | Futures | informational: NIFTY near/next spread on /flow, no rule in the paper | none |
| 10.3 | Contrarian trading (mean-reversion) | w = -gamma (R - R_m), eqs. 469-471 | Stock futures | adapted: F&O stock returns, weekly | k151_fut_contrarian |
| 10.3.1 | Contrarian trading - market activity | volume and open-interest filters, eqs. 472-473 | Stock futures | adapted: OI from flow_oi_daily | k151_fut_contrarian_act |
| 10.4 | Trend following (momentum) | w = gamma eta/sigma, eta = sign(R), eqs. 474-480 | Stock futures | adapted: sign, tanh and de-skewed variants | k151_fut_trend, k151_fut_trend_tanh, k151_fut_trend_rel |
| 11.2 | Carry, equity tranche - index hedging | CDO and MBS hedge ratios, eqs. 481-491 | Structured assets | infeasible: no CDO or MBS market in our data | none |
| 11.3 | Carry, senior/mezzanine - index hedging | CDO and MBS hedge ratios, eqs. 481-491 | Structured assets | infeasible: no CDO or MBS market in our data | none |
| 11.4 | Carry - tranche hedging | CDO and MBS hedge ratios, eqs. 481-491 | Structured assets | infeasible: no CDO or MBS market in our data | none |
| 11.5 | Carry - CDS hedging | CDO and MBS hedge ratios, eqs. 481-491 | Structured assets | infeasible: no CDO or MBS market in our data | none |
| 11.6 | CDOs - curve trades | CDO and MBS hedge ratios, eqs. 481-491 | Structured assets | infeasible: no CDO or MBS market in our data | none |
| 11.7 | Mortgage-backed security (MBS) trading | CDO and MBS hedge ratios, eqs. 481-491 | Structured assets | infeasible: no CDO or MBS market in our data | none |
| 12.1 | Convertible arbitrage | h = delta x C; OAS bisection, eqs. 492-494 | Convertibles | infeasible: no convertible bond data | none |
| 12.2 | Convertible option-adjusted spread | h = delta x C; OAS bisection, eqs. 492-494 | Convertibles | infeasible: no convertible bond data | none |
| 13.1 | Municipal bond tax arbitrage | tax-arbitrage payoffs, eqs. 495-499 | Tax | infeasible: no muni bonds and no dividend imputation in India | none |
| 13.2 | Cross-border tax arbitrage | tax-arbitrage payoffs, eqs. 495-499 | Tax | infeasible: no muni bonds and no dividend imputation in India | none |
| 13.2.1 | Cross-border tax arbitrage with options | tax-arbitrage payoffs, eqs. 495-499 | Tax | infeasible: no muni bonds and no dividend imputation in India | none |
| 14.1 | Inflation hedging - inflation swaps | inflation swaps, TIPS, weather, spark spread, eqs. 500-518 | Miscellaneous | infeasible: no inflation swaps, TIPS, weather or power futures data | none |
| 14.2 | TIPS-Treasury arbitrage | inflation swaps, TIPS, weather, spark spread, eqs. 500-518 | Miscellaneous | infeasible: no inflation swaps, TIPS, weather or power futures data | none |
| 14.3 | Weather risk - demand hedging | inflation swaps, TIPS, weather, spark spread, eqs. 500-518 | Miscellaneous | infeasible: no inflation swaps, TIPS, weather or power futures data | none |
| 14.4 | Energy - spark spread | inflation swaps, TIPS, weather, spark spread, eqs. 500-518 | Miscellaneous | infeasible: no inflation swaps, TIPS, weather or power futures data | none |
| 15.1 | Buying and holding distressed debt | distressed debt (qualitative) | Distressed assets | infeasible: not exchange-traded | none |
| 15.2 | Active distressed investing | distressed debt (qualitative) | Distressed assets | infeasible: not exchange-traded | none |
| 15.2.1 | Planning a reorganization | distressed debt (qualitative) | Distressed assets | infeasible: not exchange-traded | none |
| 15.2.2 | Buying outstanding debt | distressed debt (qualitative) | Distressed assets | infeasible: not exchange-traded | none |
| 15.2.3 | Loan-to-own | distressed debt (qualitative) | Distressed assets | infeasible: not exchange-traded | none |
| 15.3 | Distress risk puzzle | buy lowest bankruptcy probability, sell highest | Stocks | adapted: market-only distress proxy, not a bankruptcy probability | k151_distress |
| 15.3.1 | Distress risk puzzle - risk management | HMD* = sigma_target/sigma_hat x HMD, eq. 519 | Stocks | adapted: volatility-targeted long book | k151_distress_voltgt |
| 16.2 | Mixed-asset diversification with real estate | real-estate allocation and returns, eq. 520 | Real estate | infeasible: physical property or too few REITs | none |
| 16.3 | Intra-asset diversification within real estate | real-estate allocation and returns, eq. 520 | Real estate | infeasible: physical property or too few REITs | none |
| 16.3.1 | Property type diversification | real-estate allocation and returns, eq. 520 | Real estate | infeasible: physical property or too few REITs | none |
| 16.3.2 | Economic diversification | real-estate allocation and returns, eq. 520 | Real estate | infeasible: physical property or too few REITs | none |
| 16.3.3 | Property type and geographic diversification | real-estate allocation and returns, eq. 520 | Real estate | infeasible: physical property or too few REITs | none |
| 16.4 | Real estate momentum - regional approach | real-estate allocation and returns, eq. 520 | Real estate | infeasible: physical property or too few REITs | none |
| 16.5 | Inflation hedging with real estate | real-estate allocation and returns, eq. 520 | Real estate | infeasible: physical property or too few REITs | none |
| 16.6 | Fix-and-flip | real-estate allocation and returns, eq. 520 | Real estate | infeasible: physical property or too few REITs | none |
| 17.2 | Money laundering - the dark side of cash | illegal activity | Cash | infeasible: illegal, excluded on principle | none |
| 17.3 | Liquidity management | cash management | Cash | infeasible: cash management, not tradable alpha | none |
| 17.4 | Repurchase agreement (REPO) | cash management | Cash | infeasible: cash management, not tradable alpha | none |
| 17.5 | Pawnbroking | cash management | Cash | infeasible: cash management, not tradable alpha | none |
| 17.6 | Loan sharking | illegal activity | Cash | infeasible: illegal, excluded on principle | none |
| 18.2 | Artificial neural network (ANN) | ANN with ReLU hidden layers and softmax output, eqs. 521-537 | Stocks (from crypto) | adapted: 15-minute bars of the u200, 5 quantiles, walk-forward by month | k151_ann_15m |
| 18.3 | Sentiment analysis - na¨ıve Bayes Bernoulli | Bernoulli naive Bayes on keywords, eqs. 538-546 | Stocks (from crypto) | await_data: headlines must be stored for 250 sessions first | k151_nb_news |
| 19.2 | Fundamental macro momentum | long top decile by 4 state variables | Index | adapted: single-country version of 3 of the 4 variables | k151_macro_mom |
| 19.3 | Global macro inflation hedge | CA = max(0, min((HI - CI)/HI, 1)), eq. 547 | Macro | infeasible: no CPI series | none |
| 19.4 | Global fixed-income strategy | rank government bonds on macro variables | Bonds | infeasible: no multi-country bond data | none |
| 19.5 | Trading on economic announcements | long equities on announcement days only | Index | adapted: RBI MPC and FOMC dates in config/macro_events.toml | k151_announcement |
| A.d0 | Appendix A delay-0 mean reversion (R code) | E = ln(P_open(t-1)/P_close(t)), mean-variance holdings with cost haircut, eqs. 548-557 | Stocks | adapted: top 200 by ADDV, PCA risk model, open to close | k151_appA_d0_mr, k151_appA_d0_mr_5m |
| A.d1 | Appendix A delay-1 momentum (R code) | E = -ln(close/open), delay 1 | Stocks | adapted: same machinery, delay-1 momentum | k151_appA_d1_mom |
