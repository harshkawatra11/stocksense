"""G2-H5: nightly NSE MCP verification (fake bhavcopy MCP answers shaped like the recorded probe)."""

from __future__ import annotations

import json
from datetime import date, datetime

import polars as pl
import pytest

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, SteppingClock
from stocksense.core.state import State
from stocksense.data.io import upsert
from stocksense.data.mcp_verify import verify
from stocksense.live import jobs as J
from stocksense.live.jobs import JobContext

D = date(2026, 11, 19)
SYMS = [f"S{i:02d}" for i in range(30)]


@pytest.fixture
def data(tmp_data_root):
    s = tmp_data_root
    upsert(
        s,
        "bhavcopy_eq",
        pl.DataFrame(
            {
                "symbol": SYMS,
                "series": "EQ",
                "date": D,
                "close": [100.0 + i for i in range(30)],
                "volume": 1000,
            }
        ),
    )
    upsert(
        s,
        "corporate_actions",
        pl.DataFrame(
            {
                "symbol": ["S01", "S02", "S03", "S04"],
                "ex_date": [date(2026, 11, 5), date(2026, 11, 10), date(2026, 11, 12), date(2026, 9, 1)],
                "action_type": ["split", "dividend", "rights", "dividend"],
                "subject_raw": ["a", "b", "c", "old"],
            }
        ),
    )
    idx = [date(2026, 11, x) for x in (13, 16, 17, 18, 19)]
    upsert(
        s,
        "index_daily",
        pl.DataFrame({"index_name": "NIFTY 50", "date": idx, "close": [25000.0 + i for i in range(5)]}),
    )
    return s


class Fake:
    """Answers like the recorded nse-bhavcopy tools; `wrong` perturbs one close, `down` raises."""

    def __init__(self, settings, *, wrong=None, down=False, actions=None, index_shift=0.0):
        self.s, self.wrong, self.down, self.index_shift = settings, wrong, down, index_shift
        self.actions = {
            "S01": [{"exDate": "2026-11-05", "actionType": "SPLIT"}],
            "S02": [{"exDate": "2026-11-10", "actionType": "DIVIDEND"}],
            "S03": [{"exDate": "2026-11-12", "actionType": "OTHER"}],  # rights are OTHER on the server
        }
        self.actions.update(actions or {})
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, calls):
        if self.down:
            raise ConnectionError("down")
        return [self.one(t, a) for t, a in calls]

    def one(self, tool, args):
        self.calls.append((tool, args))
        if tool == "get_ltp_by_date":
            i = SYMS.index(args["symbol"])
            return json.dumps(100.0 + i + (5.0 if args["symbol"] == self.wrong else 0.0))
        if tool == "get_corporate_actions":
            acts = self.actions.get(args["symbol"], [])
            return json.dumps({"symbol": args["symbol"], "count": len(acts), "actions": acts})
        if tool == "get_index_history":
            rows = [
                {"date": f"2026-11-{x}", "close": 25000.0 + i + self.index_shift}
                for i, x in enumerate((13, 16, 17, 18, 19))
            ]
            return json.dumps({"index": "Nifty 50", "data": rows})
        raise AssertionError(tool)


def test_all_match(data):
    r = verify(data, D, Fake(data), n=20)
    assert r.status == "ok" and r.mismatch == 0 and r.closes_checked == 20
    assert r.actions_checked == 3 and r.index_checked == 5


def test_one_wrong_close_is_one_mismatch_and_a_report_line(data):
    fake = Fake(data)
    probe = verify(data, D, fake, n=20)
    assert probe.mismatch == 0
    victim = next(a["symbol"] for t, a in fake.calls if t == "get_ltp_by_date")
    r = verify(data, D, Fake(data, wrong=victim), n=20)
    assert r.status == "mismatch" and r.mismatch == 1
    assert any(victim in line and "close" in line for line in r.lines)


def test_corporate_action_missing_on_the_server_or_wrong_type_is_a_mismatch(data):
    r = verify(data, D, Fake(data, actions={"S01": []}), n=20)
    assert r.mismatch == 1 and any("S01" in line for line in r.lines)
    r = verify(data, D, Fake(data, actions={"S02": [{"exDate": "2026-11-10", "actionType": "SPLIT"}]}), n=20)
    assert r.mismatch == 1


def test_server_only_price_action_is_flagged(data):
    extra = {"S05": [{"exDate": "2026-11-11", "actionType": "BONUS"}]}
    fake = Fake(data, actions=extra)
    # S05 has no ours-side action, so it is only reached when it is asked about; ask for all symbols
    r = verify(data, D, fake, n=20, extra_symbols=["S05"])
    assert r.mismatch == 1 and any("S05" in line for line in r.lines)


def test_old_actions_outside_30_days_are_not_asked_about(data):
    fake = Fake(data)
    verify(data, D, fake, n=20)
    asked = {a["symbol"] for t, a in fake.calls if t == "get_corporate_actions"}
    assert asked == {"S01", "S02", "S03"}


def test_index_history_difference_is_a_mismatch(data):
    r = verify(data, D, Fake(data, index_shift=0.5), n=20)
    assert r.mismatch == 5


def test_unreachable_is_skipped_not_mismatch(data):
    r = verify(data, D, Fake(data, down=True), n=20)
    assert r.status == "unreachable" and r.mismatch == 0


# ---- the stage ------------------------------------------------------------------------------------------


@pytest.fixture
def ctx_of(data):
    def make(batch):
        clock = SteppingClock(datetime(2026, 11, 19, 19, 0, tzinfo=IST))
        st = State(data, clock)
        st.migrate()
        cal = TradingCalendar.from_dates([date(2026, 11, 18), D])

        class Spy:
            enabled = True

            def __init__(self):
                self.msgs = []

            def send(self, text):
                self.msgs.append(text)
                return True

        spy = Spy()
        return JobContext(
            data, clock, st, cal, steps=[], nse=object(), notifier=spy, hooks={"bhavcopy_batch": batch}
        ), spy

    return make


def test_stage_is_in_stages_right_after_build_panel():
    names = [n for n, _ in J.STAGES]
    assert names[:2] == ["build_panel", "mcp_verify"]


def test_stage_done_with_mismatch_detail_stores_kv_and_alerts(data, ctx_of):
    ctx, spy = ctx_of(Fake(data, wrong="S00", index_shift=0.0))
    out = J._stage_mcp_verify(ctx, D)
    assert not out.startswith("skipped") and out.startswith("mismatch=")
    stored = json.loads(ctx.state.kv_get("mcp_verify:2026-11-19"))
    assert stored["mismatch"] == int(out.split()[0].split("=")[1])
    if stored["mismatch"]:
        assert spy.msgs and "mismatch" in spy.msgs[0]


def test_stage_clean_run_has_mismatch_zero_and_no_alert(data, ctx_of):
    ctx, spy = ctx_of(Fake(data))
    assert J._stage_mcp_verify(ctx, D).startswith("mismatch=0")
    assert spy.msgs == []


def test_stage_unreachable_is_skipped(data, ctx_of):
    ctx, spy = ctx_of(Fake(data, down=True))
    assert J._stage_mcp_verify(ctx, D) == "skipped: unreachable"
    assert spy.msgs == []


def test_stage_never_raises_on_a_broken_batch(data, ctx_of):
    def broken(calls):
        raise ValueError("boom")

    ctx, _ = ctx_of(broken)
    assert J._stage_mcp_verify(ctx, D).startswith("skipped")
