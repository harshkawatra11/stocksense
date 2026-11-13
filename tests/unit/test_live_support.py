from __future__ import annotations

import json
from datetime import datetime

import httpx

from stocksense.core.clock import IST
from stocksense.core.config import Secrets
from stocksense.live import keepawake
from stocksense.live.notify import Notifier
from stocksense.live.preflight import FileLock, daemon_lock, gpu_lock, read_heartbeat, write_heartbeat


def test_keepawake_sets_continuous_system_required_then_releases():
    calls: list[int] = []
    assert keepawake.on(calls.append) is True
    assert calls == [0x80000000 | 0x00000001]
    assert keepawake.off(calls.append) is True
    assert calls[-1] == 0x80000000
    with keepawake.keep_awake(calls.append):
        assert calls[-1] == 0x80000001
    assert calls[-1] == 0x80000000


def test_keepawake_without_setter_is_a_noop_returning_false():
    assert keepawake.on(lambda flags: (_ for _ in ()).throw(OSError("nope"))) is False


def test_file_lock_is_exclusive_and_releasable(tmp_path):
    a, b = FileLock(tmp_path / "x.lock"), FileLock(tmp_path / "x.lock")
    assert a.acquire() is True
    assert b.acquire() is False
    a.release()
    assert b.acquire() is True
    b.release()


def test_daemon_and_gpu_locks_live_under_state(tmp_data_root):
    lk = daemon_lock(tmp_data_root)
    assert lk.path.name == "daemon.lock" and lk.path.parent.name == "state"
    assert gpu_lock(tmp_data_root).path.name == "gpu.lock"


def test_heartbeat_roundtrip_is_atomic(tmp_path):
    p = tmp_path / "state" / "heartbeat.json"
    ts = datetime(2026, 11, 13, 9, 0, tzinfo=IST)
    write_heartbeat(p, ts=ts, pid=7, current_job="market_loop", last_ok_job="open_fills", degraded=True)
    hb = read_heartbeat(p)
    assert hb == {
        "ts": ts.isoformat(), "pid": 7, "current_job": "market_loop", "last_ok_job": "open_fills",
        "degraded": True,
    }  # fmt: skip
    assert not list(p.parent.glob("*.tmp"))
    assert read_heartbeat(tmp_path / "missing.json") is None


def _settings(tmp_data_root, **secrets):
    return tmp_data_root.model_copy(update={"secrets": Secrets(**secrets)})


def test_notifier_is_a_noop_without_telegram_secrets(tmp_data_root):
    seen: list[httpx.Request] = []
    n = Notifier(_settings(tmp_data_root), client=httpx.Client(transport=httpx.MockTransport(
        lambda r: seen.append(r) or httpx.Response(200))))  # fmt: skip
    assert n.enabled is False and n.send("hi") is False and seen == []


def test_notifier_posts_when_configured_and_never_raises(tmp_data_root):
    seen: list[httpx.Request] = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json={"ok": True})

    s = _settings(
        tmp_data_root, STOCKSENSE_TELEGRAM_BOT_TOKEN="T0KEN", STOCKSENSE_TELEGRAM_ALLOWED_CHAT_ID="42"
    )
    n = Notifier(s, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert n.enabled and n.send("hello") is True
    assert json.loads(seen[0].content) == {"chat_id": "42", "text": "hello"}

    def boom(req):
        raise httpx.ConnectError("down", request=req)

    n2 = Notifier(s, client=httpx.Client(transport=httpx.MockTransport(boom)))
    assert n2.send("x") is False
