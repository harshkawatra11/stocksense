from __future__ import annotations

import json
import logging

import pytest

from stocksense.core.config import Secrets
from stocksense.core.log import RedactFilter, configure_logging, get_logger
from stocksense.core.paths import Paths


@pytest.fixture(autouse=True)
def _reset_logging():
    yield
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()


def _secrets_settings(tmp_data_root):
    s = Secrets(STOCKSENSE_ANGEL_PASSWORD="hunter2secret", STOCKSENSE_UPSTOX_ANALYTICS_TOKEN="tok-abcdef-123")
    return tmp_data_root.model_copy(update={"secrets": s})


def test_secret_values_method_returns_only_set(tmp_data_root):
    s = _secrets_settings(tmp_data_root)
    assert sorted(s.secrets.secret_values()) == ["hunter2secret", "tok-abcdef-123"]
    assert "hunter2secret" not in repr(s.secrets)


def test_json_line_with_fields(tmp_data_root, fixed_clock):
    configure_logging(tmp_data_root, fixed_clock, console=False)
    get_logger("stocksense.test").info("ingest_done", dataset="bhavcopy_eq", rows=5)
    path = Paths(tmp_data_root).logs / "daemon-2026-11-12.log"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    # event, fields flattened next to ts/level/logger
    assert rec["event"] == "ingest_done" and rec["dataset"] == "bhavcopy_eq" and rec["rows"] == 5
    assert rec["level"] == "INFO" and rec["logger"] == "stocksense.test" and "ts" in rec


def test_redacts_field_by_key_name(tmp_data_root, fixed_clock):
    configure_logging(tmp_data_root, fixed_clock, console=False)
    get_logger("stocksense.test").info("login", api_key="abc", Password="p", user="bob")
    path = Paths(tmp_data_root).logs / "daemon-2026-11-12.log"
    rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert rec["api_key"] == "***" and rec["Password"] == "***" and rec["user"] == "bob"


def test_scrubs_known_secret_in_message_and_fields(tmp_data_root, fixed_clock):
    configure_logging(_secrets_settings(tmp_data_root), fixed_clock, console=False)
    get_logger("stocksense.test").info("got hunter2secret ok", note="header tok-abcdef-123 end")
    text = (Paths(tmp_data_root).logs / "daemon-2026-11-12.log").read_text(encoding="utf-8")
    assert "hunter2secret" not in text and "tok-abcdef-123" not in text
    rec = json.loads(text.splitlines()[0])
    assert rec["event"] == "got *** ok" and rec["note"] == "header *** end"


def test_redact_filter_directly():
    f = RedactFilter(["s3cr3t-value"])
    rec = logging.LogRecord("x", logging.INFO, "p", 1, "has s3cr3t-value", None, None)
    rec.fields = {"token": "t", "k": "a s3cr3t-value b", "n": 3}
    assert f.filter(rec) is True
    assert rec.getMessage() == "has ***"
    assert rec.fields == {"token": "***", "k": "a *** b", "n": 3}
