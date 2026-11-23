"""The daily report (PLAN-G2 card G4): `reports/<d>.md` and `.json`, written by the evening `report` stage.

Rupee amounts appear only in the report file (never in logs or job details). Every number in the markdown
comes from the JSON body built here.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import polars as pl

SINCE_START_NAV = 100.0


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:+.2f}%"


def _nav_stats(nav: pl.DataFrame, d: date, calendar: Any) -> dict[str, Any] | None:
    row = nav.filter(pl.col("date") == d)
    if row.is_empty():
        return None
    r = row.row(0, named=True)
    try:
        back = calendar.shift(d, -5)
    except Exception:  # noqa: BLE001 - a calendar too short for five sessions back
        back = None
    base5 = SINCE_START_NAV
    if back is not None:
        prior = nav.filter(pl.col("date") <= back)
        if prior.height:
            base5 = float(prior["nav"][-1])
    return {
        "nav": float(r["nav"]), "day_ret": float(r["day_ret"]), "n_pos": int(r["n_pos"]),
        "gross": float(r["gross"]), "ret_5": float(r["nav"]) / base5 - 1.0,
        "ret_since_start": float(r["nav"]) / SINCE_START_NAV - 1.0,
    }  # fmt: skip


def _fills(ctx: Any, d: date) -> list[dict[str, Any]]:
    con = ctx.state.connect()
    try:
        rows = con.execute(
            "SELECT o.book, o.sid, o.symbol, o.side, f.qty, f.price, o.reason FROM fills f "
            "JOIN orders o ON o.order_id=f.order_id WHERE o.trade_date=? ORDER BY o.book, o.symbol",
            (d.isoformat(),),
        ).fetchall()
    finally:
        con.close()
    keys = ("book", "sid", "symbol", "side", "qty", "price", "reason")
    return [dict(zip(keys, r, strict=True)) for r in rows]


def _held_days(ctx: Any, book: str, d: date) -> dict[int, int]:
    con = ctx.state.connect()
    try:
        rows = con.execute(
            "SELECT o.sid, MAX(o.trade_date) FROM fills f JOIN orders o ON o.order_id=f.order_id "
            "WHERE o.book=? AND o.side='BUY' GROUP BY o.sid",
            (book,),
        ).fetchall()
    finally:
        con.close()
    out: dict[int, int] = {}
    for sid, td in rows:
        e = date.fromisoformat(td)
        try:
            out[int(sid)] = ctx.calendar.session_index(d) - ctx.calendar.session_index(e)
        except Exception:  # noqa: BLE001
            out[int(sid)] = 0
    return out


def _live_index(ctx: Any, d: date) -> dict[str, Any]:
    """{hhmm: {index name: values}} of the live index captures of day d (G2-H4); empty when none."""
    con = ctx.state.connect()
    try:
        rows = con.execute(
            "SELECT key, value FROM kv WHERE key LIKE ? ORDER BY key", (f"live_index:{d.isoformat()}:%",)
        ).fetchall()
    finally:
        con.close()
    out: dict[str, Any] = {}
    for k, v in rows:
        try:
            out[k.rsplit(":", 1)[-1]] = json.loads(v)
        except ValueError:
            continue
    return out


def _data_checks(ctx: Any, d: date) -> dict[str, Any] | None:
    """The nightly MCP verification result of day d (G2-H5), or None when the stage did not store one."""
    raw = ctx.state.kv_get(f"mcp_verify:{d.isoformat()}")
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None


def _live_ic(ctx: Any) -> dict[str, Any]:
    mid = ctx.state.kv_get("live_model:s1")
    con = ctx.state.connect()
    try:
        ics = [
            r[0]
            for r in con.execute(
                "SELECT ic FROM live_ic WHERE model_id=? AND ic IS NOT NULL ORDER BY decision_date DESC LIMIT 60",  # noqa: E501
                (mid,),
            )
        ]
        validated = con.execute("SELECT validated FROM models WHERE model_id=?", (mid,)).fetchone()
    finally:
        con.close()
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731
    return {
        "model": mid,
        "n": len(ics),
        "ic_20": mean(ics[:20]),
        "ic_60": mean(ics),
        "validated": bool(validated and validated[0]),
    }


def build_report(ctx: Any, d: date, catchup: bool = False) -> tuple[dict[str, Any], str]:
    """(JSON body, markdown) of the report of session d."""
    from stocksense.live.jobs import active_books

    con = ctx.state.connect()
    try:
        jobs = con.execute(
            "SELECT job, status, COALESCE(detail,'') FROM job_runs WHERE trade_date=? AND attempt=("
            "SELECT MAX(attempt) FROM job_runs j2 WHERE j2.job=job_runs.job AND j2.trade_date=job_runs.trade_date)"  # noqa: E501
            " ORDER BY job",
            (d.isoformat(),),
        ).fetchall()
        failed = [
            r[0]
            for r in con.execute(
                "SELECT DISTINCT job FROM job_runs WHERE trade_date=? AND status='failed' "
                "AND COALESCE(detail,'') NOT LIKE 'retry:%'",
                (d.isoformat(),),
            )
        ]
    finally:
        con.close()
    books: dict[str, Any] = {}
    for b in ctx.ledger.books():
        st = _nav_stats(ctx.ledger.nav_series(b), d, ctx.calendar)
        if st is not None:
            books[b] = st
    fills = ctx.ledger.fills_for_date(d)
    cap_raw = ctx.state.kv_get("capital_ref_inr")
    cap = float(cap_raw) if cap_raw else None
    held = _held_days(ctx, "main", d)
    positions = [
        {"book": b, "symbol": p.symbol, "qty": p.qty, "weight": p.weight, "days_held": held.get(p.sid)}
        for b in active_books(ctx, d)
        for p in ctx.ledger.positions(b)
    ]
    plan_raw = ctx.state.kv_get(f"s1_plan:{d.isoformat()}")
    plan = json.loads(plan_raw) if plan_raw else None
    ic = _live_ic(ctx)
    main = books.get("main")
    healthy = not failed
    first = "System healthy" if healthy else f"System has {len(failed)} failed job(s)"
    if main is not None:
        first += f"; main book {_pct(main['day_ret'])} today, {_pct(main['ret_since_start'])} since start"
    else:
        first += "; main book has no marks yet"
    first += "; model validated." if ic["validated"] else "; model not yet validated."
    body = {
        "date": d.isoformat(), "catch_up": catchup, "summary": first, "books": books, "fills": len(fills),
        "reconciled": sum(f.reconciled for f in fills), "failed_jobs": failed,
        "degraded": ctx.state.kv_get(f"quotes_degraded:{d.isoformat()}") is not None,
        "jobs": [{"job": j, "status": s, "detail": x[:120]} for j, s, x in jobs],
        "capital": {"inr": cap, "stale": ctx.state.kv_get("capital_stale") == "1"},
        "trades": _fills(ctx, d), "positions": positions, "live_ic": ic,
        "decay_scale": float(ctx.state.kv_get("decay_scale:main") or 1.0),
        "live_index": _live_index(ctx, d),
        "data_checks": _data_checks(ctx, d),
        "plan": None if plan is None else {
            "exits": plan.get("sells", []), "entries": plan.get("entries", []), "note": plan.get("note", ""),
            "allow_entries": plan.get("allow_entries"),
        },
    }  # fmt: skip
    return body, render_markdown(body)


def render_markdown(b: dict[str, Any]) -> str:
    L = [f"# StockSense report {b['date']}" + (" (catch-up)" if b["catch_up"] else ""), "", b["summary"], ""]
    L.append("## Health")
    L.append(f"- failed jobs: {b['failed_jobs'] or 'none'}")
    L.append(f"- fills {b['fills']} (reconciled {b['reconciled']}); quotes degraded: {b['degraded']}")
    for j in b["jobs"]:
        L.append(f"- {j['job']}: {j['status']} {j['detail']}".rstrip())
    L += ["", "## Capital"]
    cap = b["capital"]
    L.append(
        "- capital read: not available" if cap["inr"] is None
        else f"- capital read: Rs {cap['inr']:,.0f} (stale: {'yes' if cap['stale'] else 'no'})"
    )  # fmt: skip
    L += ["", "## Books"]
    if not b["books"]:
        L.append("- no books yet")
    for name, v in b["books"].items():
        inr = ""
        if cap["inr"] is not None:
            inr = f", Rs {v['ret_since_start'] * cap['inr']:+,.0f} since start at today's capital"
        L.append(
            f"- {name}: NAV {v['nav']:.4f}, day {_pct(v['day_ret'])}, 5-session {_pct(v['ret_5'])}, "
            f"since start {_pct(v['ret_since_start'])}, positions {v['n_pos']}{inr}"
        )
    L += ["", "## Trades today"]
    if not b["trades"]:
        L.append("- none")
    for t in b["trades"]:
        L.append(f"- {t['book']}: {t['side']} {t['qty']} {t['symbol']} @ {t['price']:.2f} ({t['reason']})")
    L += ["", "## Open positions"]
    if not b["positions"]:
        L.append("- none")
    for p in b["positions"]:
        dh = "n/a" if p["days_held"] is None else str(p["days_held"])
        L.append(
            f"- {p['book']}: {p['symbol']} qty {p['qty']:g}, weight {p['weight'] * 100:.1f}%, days held {dh}"
        )
    ic = b["live_ic"]
    f = lambda x: "n/a" if x is None else f"{x:.4f}"  # noqa: E731
    L += ["", "## Model", f"- live model {ic['model']}", f"- live IC last 20: {f(ic['ic_20'])}, last 60: {f(ic['ic_60'])} (n={ic['n']})",  # noqa: E501
          f"- decay scale: {b['decay_scale']}", "", "## Tomorrow's plan"]  # fmt: skip
    p = b["plan"]
    if p is None:
        L.append("- no plan stored for this date")
    else:
        L.append(f"- entries allowed: {p['allow_entries']}; {p['note'] or 'no note'}")
        L += [f"- exit {x['symbol']} qty {x['qty']} ({x['reason']})" for x in p["exits"]]
        L += [
            f"- entry {x['symbol']} weight {x['weight'] * 100:.1f}% (rank {x['rank']})" for x in p["entries"]
        ]
    if b.get("live_index"):
        L += ["", "## Live indices"]
        for hhmm, snap in b["live_index"].items():
            parts = [
                f"{n} {v['last']:.2f} ({_pct((v.get('change_pct') or 0) / 100)})" for n, v in snap.items()
            ]
            L.append(f"- {hhmm[:2]}:{hhmm[2:]} " + ", ".join(parts))
    dc = b.get("data_checks")
    if dc:
        L += ["", "## Data checks"]
        L.append(f"- NSE MCP verification: {dc.get('status')} (mismatch={dc.get('mismatch', 0)})")
        L += [f"- {line}" for line in dc.get("lines", [])]
    return "\n".join(L) + "\n"
