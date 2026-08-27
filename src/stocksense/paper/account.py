"""
Phase J2.a: paper account lifecycle. A paper account is a UNIT book --
see store.py's SCHEMA comment above `paper_accounts` for why no capital
figure lives anywhere in it. Opening an account just fixes WHICH live
model it tracks and HOW it fills; nothing about size.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class PaperAccount:
    account_id: str
    name: str
    model_id: str
    model_type: str
    horizon_bars: int
    top_n: int
    cap_band: str | None
    fill_rule: str
    no_trade_band: float
    status: str


REBALANCE_DATE_CLOSE = "rebalance_date_close"
"""The only fill rule built so far: fills at the rebalance date's own
close price, no next-bar-open/half-spread adjustment. Deliberately
matches evaluation/backtest.simulate_portfolio's own implicit price
basis (it applies raw_actual -- a close-to-close relative return -- to
held weight, never modeling an explicit next-bar fill) so the paper
book's measured alpha stays directly comparable to the number that
actually cleared the gate. A validated, more realistic entry-timing
rule (Phase J4a, execution/fill_model.py against intraday bars) is a
NEW fill_rule value adopted later, once proven as an ablation -- never
a silent redefinition of this one."""


def open_paper_account(
    store, *, name: str, model_id: str, model_type: str, horizon_bars: int, top_n: int,
    cap_band: str | None = None, fill_rule: str = REBALANCE_DATE_CLOSE, no_trade_band: float = 0.02,
    notes: str | None = None,
) -> PaperAccount:
    """Refuses to open against a model that isn't actually registered
    with matching (model_type, horizon_bars) -- an account tracking a
    model_id that doesn't exist would silently generate zero orders
    forever, which is a much more confusing failure than an upfront
    error."""
    model_row = store.con.execute(
        "SELECT model_id FROM model_registry WHERE model_id = ? AND model_type = ? AND horizon_bars = ?",
        [model_id, model_type, horizon_bars],
    ).fetchone()
    if model_row is None:
        raise ValueError(f"no registered model {model_id!r} with model_type={model_type!r}, horizon_bars={horizon_bars}")

    account_id = str(uuid.uuid4())[:12]
    now = datetime.now(timezone.utc)
    store.insert_paper_account({
        "account_id": account_id, "name": name, "model_id": model_id, "model_type": model_type,
        "horizon_bars": horizon_bars, "top_n": top_n, "cap_band": cap_band, "fill_rule": fill_rule,
        "no_trade_band": no_trade_band, "created_at": now, "closed_at": None, "status": "active",
        "notes": notes,
    })
    return PaperAccount(
        account_id=account_id, name=name, model_id=model_id, model_type=model_type,
        horizon_bars=horizon_bars, top_n=top_n, cap_band=cap_band, fill_rule=fill_rule,
        no_trade_band=no_trade_band, status="active",
    )


def get_account(store, account_id: str) -> PaperAccount:
    row = store.get_paper_account(account_id)
    if row.empty:
        raise ValueError(f"no paper account {account_id!r}")
    r = row.iloc[0]
    return PaperAccount(
        account_id=r["account_id"], name=r["name"], model_id=r["model_id"], model_type=r["model_type"],
        horizon_bars=int(r["horizon_bars"]), top_n=int(r["top_n"]),
        cap_band=r["cap_band"] if r["cap_band"] else None, fill_rule=r["fill_rule"],
        no_trade_band=float(r["no_trade_band"]), status=r["status"],
    )


def list_accounts(store):
    return store.read_paper_accounts()


def close_account(store, account_id: str) -> None:
    store.close_paper_account(account_id, datetime.now(timezone.utc))
