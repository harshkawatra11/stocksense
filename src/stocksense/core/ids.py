from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any


def canonical_json(obj: Any) -> str:
    """JSON text with sorted keys and no whitespace (stable across runs)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _sha1_16(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def decision_id(book: str, strategy: str, d: date | str) -> str:
    """16-hex id of one (book, strategy, trade date) decision."""
    return _sha1_16(f"{book}|{strategy}|{d}")


def order_id(decision: str, sid: int, side: str, when: str) -> str:
    """16-hex id of one order within a decision."""
    return _sha1_16(f"{decision}|{sid}|{side}|{when}")


def spec_hash8(spec: Any) -> str:
    """First 8 hex chars of sha1 over the canonical JSON of a model/trial spec."""
    return hashlib.sha1(canonical_json(spec).encode("utf-8")).hexdigest()[:8]


def model_id(family: str, target: str, spec: Any, train_end: date) -> str:
    """`family-target-spec_hash8-YYYYMMDD` identifier of a trained model."""
    return f"{family}-{target}-{spec_hash8(spec)}-{train_end:%Y%m%d}"
