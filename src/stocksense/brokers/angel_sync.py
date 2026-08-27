"""
Phase J1.3: Angel One read-only sync -- holdings and positions.

Field names below are taken from REAL live API responses captured
against the user's own account before writing this (never guessed from
docs, which drift): `position()` returns snake-less lowercase keys like
`netqty`/`buyavgprice`/`realised`; `holding()` and `allholding()` return
a similar flat shape. Both endpoints return numeric fields AS STRINGS in
places (`"netqty": "0"`) -- coerced explicitly below rather than trusted.

Scope of this pass: holdings + positions snapshots only, into
broker_holdings/broker_positions_snapshot (point-in-time, upserted per
day -- re-syncing today overwrites today's row, never accumulates
duplicates). Trade/order-level ingestion into the canonical `trades`
table (so kundli/tax pick it up automatically) is a deliberate follow-up,
not built here: it must dedupe byte-for-byte against
statements/parsers/angel.py's own trade_id formula, which needs its own
careful golden-file test against a real XLSX + a real API tradeBook
response for the SAME fills, not assumed to line up.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone

import pandas as pd

from stocksense.brokers.angel_readonly import ReadOnlyBrokerClient
from stocksense.brokers.angel_session import BrokerAuthError, TransientBrokerError, login

BROKER = "angelone"


@dataclass(frozen=True)
class SyncResult:
    sync_id: str
    status: str  # 'ok' | 'partial' | 'transient_failure' | 'auth_failure'
    n_holdings: int
    n_positions: int
    error: str | None


def _f(row: dict, key: str, default: float = 0.0) -> float:
    """Coerces a possibly-string numeric field to float, tolerating the
    empty string this API sometimes returns for an unset field."""
    v = row.get(key)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def normalize_holdings(rows: list[dict], as_of_date: date) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    out = []
    for r in rows:
        out.append({
            "broker": BROKER, "as_of_date": as_of_date,
            "symbol": r.get("tradingsymbol", "").replace("-EQ", ""),
            "exchange": r.get("exchange"), "isin": r.get("isin"),
            "quantity": _f(r, "quantity"), "t1_quantity": _f(r, "t1quantity"),
            "avg_price": _f(r, "averageprice"), "ltp": _f(r, "ltp"),
            "close_price": _f(r, "close"), "pnl": _f(r, "profitandloss"),
            "synced_at": now,
        })
    return pd.DataFrame(out, columns=[
        "broker", "as_of_date", "symbol", "exchange", "isin", "quantity", "t1_quantity",
        "avg_price", "ltp", "close_price", "pnl", "synced_at",
    ])


def normalize_positions(rows: list[dict], as_of_date: date) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    out = []
    for r in rows:
        out.append({
            "broker": BROKER, "as_of_date": as_of_date,
            "symbol": r.get("tradingsymbol", "").replace("-EQ", ""),
            "exchange": r.get("exchange"), "product": r.get("producttype", ""),
            "net_qty": _f(r, "netqty"), "buy_qty": _f(r, "buyqty"), "buy_avg": _f(r, "buyavgprice"),
            "sell_qty": _f(r, "sellqty"), "sell_avg": _f(r, "sellavgprice"),
            "ltp": _f(r, "ltp"), "realised": _f(r, "realised"), "unrealised": _f(r, "unrealised"),
            "synced_at": now,
        })
    return pd.DataFrame(out, columns=[
        "broker", "as_of_date", "symbol", "exchange", "product", "net_qty", "buy_qty",
        "buy_avg", "sell_qty", "sell_avg", "ltp", "realised", "unrealised", "synced_at",
    ])


def sync_angel(store, settings, scopes: tuple[str, ...] = ("holdings", "positions")) -> SyncResult:
    """Logs in fresh (angel_session.login), fetches the requested scopes
    through a ReadOnlyBrokerClient, normalizes, and upserts. Never
    raises on a transient/auth failure -- returns a SyncResult with the
    failure recorded, so a caller (the nightly graph) can decide whether
    to retry rather than crash the whole run."""
    sync_id = str(uuid.uuid4())[:12]
    started_at = datetime.now(timezone.utc)
    today = date.today()

    try:
        _, client = login(settings)
    except BrokerAuthError as e:
        result = SyncResult(sync_id=sync_id, status="auth_failure", n_holdings=0, n_positions=0, error=str(e))
        _record_run(store, sync_id, started_at, scopes, result)
        return result
    except TransientBrokerError as e:
        result = SyncResult(sync_id=sync_id, status="transient_failure", n_holdings=0, n_positions=0, error=str(e))
        _record_run(store, sync_id, started_at, scopes, result)
        return result

    ro = ReadOnlyBrokerClient(client)
    n_holdings = n_positions = 0
    errors = []

    if "holdings" in scopes:
        try:
            resp = ro.holding()
            if resp and resp.get("status"):
                df = normalize_holdings(resp.get("data") or [], today)
                n_holdings = store.upsert_broker_holdings(df)
            else:
                errors.append(f"holdings: {resp.get('message') if resp else 'no response'}")
        except Exception as e:  # noqa: BLE001 -- one scope's failure must not abort the others
            errors.append(f"holdings: {e}")

    if "positions" in scopes:
        try:
            resp = ro.position()
            if resp and resp.get("status"):
                df = normalize_positions(resp.get("data") or [], today)
                n_positions = store.upsert_broker_positions_snapshot(df)
            else:
                errors.append(f"positions: {resp.get('message') if resp else 'no response'}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"positions: {e}")

    status = "ok" if not errors else ("partial" if (n_holdings or n_positions) else "transient_failure")
    result = SyncResult(
        sync_id=sync_id, status=status, n_holdings=n_holdings, n_positions=n_positions,
        error="; ".join(errors) if errors else None,
    )
    _record_run(store, sync_id, started_at, scopes, result, session_source="fresh_login")
    return result


def _record_run(store, sync_id: str, started_at, scopes: tuple[str, ...], result: SyncResult, session_source: str | None = None) -> None:
    import json

    store.insert_broker_sync_run({
        "sync_id": sync_id, "broker": BROKER, "started_at": started_at,
        "finished_at": datetime.now(timezone.utc), "status": result.status,
        "scopes_json": json.dumps(list(scopes)), "n_holdings": result.n_holdings,
        "n_positions": result.n_positions, "session_source": session_source, "error": result.error,
    })
