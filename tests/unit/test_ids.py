from __future__ import annotations

import hashlib
import re
from datetime import date

from stocksense.core import ids


def test_canonical_json_sorted_no_whitespace():
    assert ids.canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_decision_id_deterministic_16_hex():
    a = ids.decision_id("paper", "s1", date(2026, 11, 12))
    b = ids.decision_id("paper", "s1", date(2026, 11, 12))
    assert a == b and re.fullmatch(r"[0-9a-f]{16}", a)
    assert a == hashlib.sha1(b"paper|s1|2026-11-12").hexdigest()[:16]
    assert a != ids.decision_id("paper", "s2", date(2026, 11, 12))


def test_order_id_deterministic_16_hex():
    d = ids.decision_id("paper", "s1", date(2026, 11, 12))
    o = ids.order_id(d, 7, "BUY", "close")
    assert o == hashlib.sha1(f"{d}|7|BUY|close".encode()).hexdigest()[:16]
    assert re.fullmatch(r"[0-9a-f]{16}", o)


def test_spec_hash8_and_model_id():
    spec = {"b": 1, "a": [1, 2]}
    assert ids.spec_hash8(spec) == hashlib.sha1(b'{"a":[1,2],"b":1}').hexdigest()[:8]
    assert ids.spec_hash8(spec) == ids.spec_hash8({"a": [1, 2], "b": 1})
    mid = ids.model_id("lgbm", "y_c2o_5", spec, date(2025, 9, 30))
    assert mid == f"lgbm-y_c2o_5-{ids.spec_hash8(spec)}-20250930"
