"""Sizing tests.

The first one exists to settle an argument numerically rather than rhetorically,
and it is the most important test in this file: buying more shares of a cheaper
stock is not leverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from stocksense.simulation.sizing import (
    breakeven_bps,
    capital_efficiency,
    fractional_kelly,
    probability_of_ruin,
    tick_drag_bps,
    tradeable_price_band,
    whole_share_quantity,
)


def test_share_count_is_not_leverage():
    """THE test. Same capital, same % move, same profit -- regardless of price.

    The intuition this refutes: "with 25,000 buy the 88-rupee stock, not the
    1,000-rupee one -- 284 shares instead of 25, so a 1-rupee move pays 284
    instead of 25." That compares a 1.14% move to a 0.10% move. Held at equal
    RETURN, the share count cancels exactly, because P&L = capital x return.
    """
    capital = 25_000.0
    move_pct = 0.01

    qty_cheap, _ = whole_share_quantity(capital, 88.0)
    qty_dear, _ = whole_share_quantity(capital, 1000.0)
    assert qty_cheap == 284 and qty_dear == 25

    pnl_cheap = qty_cheap * 88.0 * move_pct
    pnl_dear = qty_dear * 1000.0 * move_pct

    # Equal to within whole-share rounding (~0.5% of a position), not by luck.
    assert pnl_cheap == pytest.approx(pnl_dear, rel=0.01)
    assert pnl_cheap == pytest.approx(250.0, rel=0.01)


def test_the_1_rupee_move_intuition_is_a_percentage_illusion():
    """Why the fallacy feels true: 1 rupee is 11x bigger a move on an 88-rupee
    stock than on a 1,000-rupee one."""
    assert (1.0 / 88.0) / (1.0 / 1000.0) == pytest.approx(11.36, rel=0.01)


# ------------------------------------------------------------------ tick drag
def test_tick_drag_penalises_cheap_stocks():
    """NSE's 0.05 tick is a far larger fraction of a cheap stock's price, and it
    is a FLOOR on cost -- no trade can be cheaper than one tick."""
    assert tick_drag_bps(88.0) == pytest.approx(5.68, rel=0.01)
    assert tick_drag_bps(1000.0) == pytest.approx(0.5, rel=0.01)
    assert tick_drag_bps(88.0) > 10 * tick_drag_bps(1000.0)


def test_cheap_stocks_need_a_bigger_move_to_break_even():
    """The practical consequence: the 88-rupee stock must move more than twice
    as far, in percent, before it earns anything."""
    cheap = breakeven_bps(88.0)
    dear = breakeven_bps(1000.0)
    assert cheap == pytest.approx(19.7, abs=0.3)
    assert dear == pytest.approx(9.3, abs=0.3)
    assert cheap > 2 * dear


# --------------------------------------------------------------- divisibility
def test_expensive_stocks_strand_capital_at_a_small_account():
    """The real reason price level matters here -- and it argues the OPPOSITE
    way to tick drag, which is why there is a band rather than a rule."""
    per_position = 87_500.0 / 2  # 2 concentrated names at 5x on 17,500

    qty, stranded = whole_share_quantity(per_position, 14_850.0)  # DIXON-priced
    assert qty == 2
    assert stranded > 14_000
    assert capital_efficiency(per_position, 14_850.0) < 0.70

    # Efficiency degrades monotonically with price -- and note that even a
    # mid-priced 900-rupee name strands ~1.3%, because 43,750/900 = 48.6 shares.
    # Whole-share rounding is never free; it just stops mattering below ~2%.
    eff = {p: capital_efficiency(per_position, p) for p in (250.0, 900.0, 4000.0, 14_850.0)}
    assert eff[250.0] > eff[900.0] > eff[4000.0] > eff[14_850.0]
    assert eff[900.0] > 0.98
    assert eff[14_850.0] < 0.70


def test_price_band_brackets_the_two_opposing_forces(monkeypatch):
    band = tradeable_price_band(equity_inr=17_500, leverage=5.0, max_positions=2)

    # Lower bound from tick drag: 0.05 / 0.0002 = 250
    assert band.min_price_inr == pytest.approx(250.0)
    # Upper bound from divisibility: (87,500 / 2) / 20 shares
    assert band.max_price_inr == pytest.approx(2187.5)

    assert band.contains(900.0)
    assert not band.contains(88.0), "tick drag makes this uneconomic"
    assert not band.contains(14_850.0), "cannot be sized at this account"


def test_band_widens_with_capital():
    """As the account grows the upper bound rises -- expensive names become
    tradeable. The lower bound does not move, because tick drag is a property of
    the market, not of the account."""
    small = tradeable_price_band(17_500)
    large = tradeable_price_band(200_000)
    assert large.max_price_inr > small.max_price_inr
    assert large.min_price_inr == small.min_price_inr


# --------------------------------------------------------------------- Kelly
def test_kelly_is_zero_without_an_edge():
    """No edge, no position. A coin flip at even money must size to nothing."""
    assert fractional_kelly(win_prob=0.5, win_loss_ratio=1.0) == 0.0
    assert fractional_kelly(win_prob=0.4, win_loss_ratio=1.0) == 0.0


def test_fractional_kelly_scales_down_full_kelly():
    full = fractional_kelly(0.6, 1.5, fraction=1.0)
    quarter = fractional_kelly(0.6, 1.5, fraction=0.25)
    assert quarter == pytest.approx(full * 0.25)
    # p - (1-p)/b = 0.6 - 0.4/1.5
    assert full == pytest.approx(0.6 - 0.4 / 1.5)


# ----------------------------------------------------------------------- ruin
def test_probability_of_ruin_rises_with_leverage():
    """The number that should decide leverage on 1-2 concentrated names.

    Same underlying edge, scaled up: risk of ruin must increase, and it does so
    faster than linearly.
    """
    rng = np.random.default_rng(0)
    base = rng.normal(0.0005, 0.02, 2000)  # slight positive drift, 2% daily vol

    p1 = probability_of_ruin(base, 17_500, n_paths=5_000)
    p5 = probability_of_ruin(base * 5, 17_500, n_paths=5_000)
    assert p5 > p1


def test_ruin_uses_the_empirical_distribution_including_its_tails():
    """Bootstrapping the observed returns rather than assuming normality is the
    entire point: a fat left tail must show up as ruin risk."""
    rng = np.random.default_rng(1)
    benign = rng.normal(0.001, 0.01, 2000)
    with_crash = np.concatenate([benign, np.full(40, -0.18)])

    assert probability_of_ruin(with_crash, 17_500, n_paths=5_000) > probability_of_ruin(
        benign, 17_500, n_paths=5_000
    )
